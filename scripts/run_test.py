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

# [Latency 측정용] 입력 시간을 기록하는 큐
timestamp_queue = asyncio.Queue()

# -----------------------------------------------------------------------------
# [Async Receiver] 엔진 출력을 비동기로 수거 (Latency 측정 추가)
# -----------------------------------------------------------------------------
async def receiver_loop(engine, collected_list):
    log("info", "[Receiver] Listening for output...")
    first_packet_received = False
    
    while True:
        out_bytes = await engine.get_audio_output()
        if out_bytes:
            current_time = time.time()
            
            # [Latency Check] 큐에서 가장 오래된 입력 시간 꺼내기
            # (Talker는 입력 순서대로 출력한다고 가정)
            if not timestamp_queue.empty():
                input_time = await timestamp_queue.get()
                latency = current_time - input_time
                log("info", f"⏱️ Latency: {latency*1000:.1f}ms (Input -> Output)")
            
            out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            collected_list.append(out_np)
        else:
            if not engine.is_running:
                break
            await asyncio.sleep(0.001)

# -----------------------------------------------------------------------------
# [Async Sender] 오디오를 0.32초 간격으로 투입 (Timestamp 기록 추가)
# -----------------------------------------------------------------------------
async def sender_loop(engine, chunks, processor, model, device):
    log("info", "[Sender] Streaming audio chunks...")
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < 5120: # 16000 * 0.32
            chunk = np.pad(chunk, (0, 5120 - len(chunk)))
        
        features = processor.feature_extractor(
            [chunk], return_tensors="pt", sampling_rate=16000,padding=False,
        )
        input_features = features.input_features.to(device).to(model.dtype)
        
        if torch.isnan(input_features).any() or torch.isinf(input_features).any():
            log("warning", f"⚠️ Chunk {i}: NaN/Inf detected! Replacing with zeros.")
            input_features = torch.nan_to_num(input_features, nan=0.0, posinf=0.0, neginf=0.0)

        # [Timestamp] 입력 직전 시간 기록
        await timestamp_queue.put(time.time())

        # 비동기 투입
        await engine.push_audio(input_features)
        
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

    log("info", f"Loading Model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16, 
        attn_implementation='sdpa',
        trust_remote_code=True
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    config = EngineConfig()
    engine = Qwen3OmniFullDuplexEngine(model, processor.tokenizer, config)
    
    full_audio, sr = load_audio_file(args.input_file, target_sr=16000)
    
    MAX_DURATION_SEC = 10 # 300초
    max_samples = int(MAX_DURATION_SEC * sr)
    
    if len(full_audio) > max_samples:
        log("info", f"✂️ Cutting audio to first {MAX_DURATION_SEC} seconds ({max_samples} samples)")
        full_audio = full_audio[:max_samples]
    
    chunk_size = int(sr * 0.32)
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    log("info", f"Chunks to process: {len(chunks)} (approx {len(chunks)*0.32/60:.1f} mins)")

    await engine.start()
    
    collected_output_audio = []
    
    recv_task = asyncio.create_task(receiver_loop(engine, collected_output_audio))
    
    start_time = time.time()
    try:
        await sender_loop(engine, chunks, processor, model, args.device)
        
        log("info", "All chunks sent. Waiting for trailing response...")
        await asyncio.sleep(30.0) 

    except asyncio.CancelledError:
        pass
    except Exception as e:
        log("error", f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log("info", "Stopping engine...")
        await engine.stop() 
        
        log("info", "Waiting for receiver to drain queue...")
        await recv_task     
    
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