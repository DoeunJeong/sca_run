import os
import sys
import time
import argparse
import asyncio  # Asyncio 사용
import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import BitsAndBytesConfig

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 패키지 경로
from src.inference import Qwen3OmniFullDuplexEngine, EngineConfig
from src.client_utils import log # 로그 유틸

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# 메모리 단편화 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def load_audio_file(file_path, target_sr=16000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    log("info", f"Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

# -----------------------------------------------------------------------------
# [Async Receiver] 엔진 출력을 비동기로 수거 (잔반 처리 로직 포함)
# -----------------------------------------------------------------------------
async def receiver_loop(engine, collected_list):
    log("info", "[Receiver] Listening for output...")
    while True:
        out_bytes = await engine.get_audio_output()
        if out_bytes:
            out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            collected_list.append(out_np)
        else:
            # 데이터가 없는데 엔진도 멈췄다면? -> 할 일 다 했으니 종료
            if not engine.is_running:
                break
            # 데이터는 없지만 엔진은 돌고 있다면? -> 대기
            await asyncio.sleep(0.001)

# -----------------------------------------------------------------------------
# [Async Sender] 오디오를 0.32초 간격으로 투입
# -----------------------------------------------------------------------------
async def sender_loop(engine, chunks, processor, model, device):
    log("info", "[Sender] Streaming audio chunks...")
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < 5120: # 16000 * 0.32
            chunk = np.pad(chunk, (0, 5120 - len(chunk)))
        
        # 전처리 (Blocking이지만 짧음)
        features = processor.feature_extractor(
            [chunk], return_tensors="pt", sampling_rate=16000,padding=False,
        )
        input_features = features.input_features.to(device).to(model.dtype)
        
        if torch.isnan(input_features).any() or torch.isinf(input_features).any():
            log("warning", f"⚠️ Chunk {i}: NaN/Inf detected! Replacing with zeros.")
            input_features = torch.nan_to_num(input_features, nan=0.0, posinf=0.0, neginf=0.0)

        # 비동기 투입
        await engine.push_audio(input_features)
        
        # ★ [수정] 실시간 시뮬레이션 복구 (0.32초 대기)
        # 이제 5분치 데이터가 5분 동안 천천히 들어갑니다.
        await asyncio.sleep(0.32) 
        
        if i % 10 == 0:
            log("info", f"Sent chunk {i}/{len(chunks)}")

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="output_response.wav")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()


    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"
    # )

    # 2. 깔끔한 Device Map (레이어 쪼개기 금지)
    # - Thinker 전체를 GPU 0에 할당 (약 20GB 소모)
    # - Talker/Audio 전체를 GPU 1에 할당 (약 5GB 소모)
    # -> 서로 간섭 없이 독립적으로 돌아가므로 에러가 날 수 없음
    device_map = {
        "thinker": 0,
        "talker": 1,
        "code2wav": 1
    }


    log("info", f"Loading Model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto", 
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, # BF16 원본 로드 (양자화 X)
        #attn_implementation='flash_attention_2',
        attn_implementation='sdpa',
        trust_remote_code=True
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. 엔진 초기화
    config = EngineConfig()
    engine = Qwen3OmniFullDuplexEngine(model, processor.tokenizer, config)
    
    # 3. 오디오 로드 및 5분 컷
    full_audio, sr = load_audio_file(args.input_file, target_sr=16000)
    
    # ★ [수정] 5분(300초)까지만 자르기
    MAX_DURATION_SEC = 0.32 # 300초
    max_samples = int(MAX_DURATION_SEC * sr)
    
    if len(full_audio) > max_samples:
        log("info", f"✂️ Cutting audio to first 5 minutes ({max_samples} samples)")
        full_audio = full_audio[:max_samples]
    
    chunk_size = int(sr * 0.32)
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    log("info", f"Chunks to process: {len(chunks)} (approx {len(chunks)*0.32/60:.1f} mins)")

    # 4. 엔진 시작 (Task 생성)
    await engine.start()
    
    collected_output_audio = []
    
    # 5. Receiver & Sender 동시 실행
    recv_task = asyncio.create_task(receiver_loop(engine, collected_output_audio))
    
    start_time = time.time()
    try:
        await sender_loop(engine, chunks, processor, model, args.device)
        
        log("info", "All chunks sent. Waiting for trailing response...")
        await asyncio.sleep(60.0) # 잔여 응답 대기

    except asyncio.CancelledError:
        pass
    except Exception as e:
        log("error", f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 종료 절차
        log("info", "Stopping engine...")
        await engine.stop() # 1. 엔진 멈춤 신호 발생
        
        log("info", "Waiting for receiver to drain queue...")
        await recv_task     # 2. Receiver가 남은 데이터를 다 꺼낼 때까지 대기
    
    # 6. 저장
    if collected_output_audio:
        final_audio = np.concatenate(collected_output_audio)
        OUTPUT_SR = 24000
        log("info", f"Saving {len(final_audio)} samples ({len(final_audio)/OUTPUT_SR:.1f}s) to {args.output_file}")
        sf.write(args.output_file, final_audio, OUTPUT_SR)
    else:
        log("warning", "No output received!")

    log("info", f"Total Time: {time.time() - start_time:.2f}s")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
