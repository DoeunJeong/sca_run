# scripts/run_test.py
import os
import sys
import time
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
# 현재 파일 위치: scripts/run_test.py -> 상위(root) -> src 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# [수정 1] 패키지 경로를 실제 프로젝트 구조(sca_core)에 맞게 수정
from src.inference import Qwen3OmniFullDuplexEngine, EngineConfig
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

def load_audio_file(file_path, target_sr=24000):
    """오디오 파일을 로드하고 리샘플링함"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    print(f"📂 Loading audio file: {file_path}")
    # librosa는 float32 [-1, 1]로 로드함
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def main():
    parser = argparse.ArgumentParser(description="Test Full-Duplex Engine with Audio File")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to model")
    parser.add_argument("--input-file", type=str, required=True, help="Input audio file (e.g. 3min_noisy.wav)")
    parser.add_argument("--output-file", type=str, default="output_response.wav", help="Output audio file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run")
    args = parser.parse_args()

    # 1. 모델 로드
    print(f"🔥 Loading Model from {args.model_path}...")
    
    # [수정 2] args.model_path 사용 및 device_map 명시적 설정
    # (A40 2장이면 아래처럼 분산 설정, 1장이면 "auto")
    device_map = "auto" 
    
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map=device_map, 
        dtype='auto',          
        attn_implementation='flash_attention_2', 
        trust_remote_code=True
    )
    
    # 3. 프로세서 로드
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. 엔진 초기화
    config = EngineConfig(audio_input_tokens=4, text_output_tokens=2, audio_output_tokens=4)
    
    # [수정 3] processor 자체가 아니라 processor.tokenizer를 전달해야 함
    engine = Qwen3OmniFullDuplexEngine(model, processor.tokenizer, config)
    
    # 3. 오디오 준비 (Chunking)
    full_audio, sr = load_audio_file(args.input_file, target_sr=24000)
    
    # 4토큰 분량의 오디오 길이 계산 (0.32초)
    # 24000 * 0.32 = 7680 samples
    chunk_size = int(sr * 0.32) 
    
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    print(f"📦 Audio split into {len(chunks)} chunks (Chunk size: {chunk_size} samples)")

    # 4. 테스트 시작 (쓰레드 가동)
    engine.start()
    
    collected_output_audio = []
    start_time = time.time()
    
    try:
        # -- [Sender Loop] 오디오를 실시간처럼 조금씩 밀어넣음 --
        for i, chunk in enumerate(chunks):
            # 마지막 짜투리 패딩 (필요시)
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # [중요 수정] Raw Audio -> Processor -> Mel Spectrogram -> Audio Tower -> Embeddings
            # Engine은 '이미 인코딩된 Feature(Embeddings)'를 받도록 설계했으므로 여기서 전처리 수행
            
            with torch.no_grad():
                # 1. Processor를 통해 Mel Spectrogram 변환 (Input Features)
                # sampling_rate 필수 지정
                processed_inputs = processor(
                    audios=[chunk], 
                    return_tensors="pt", 
                    sampling_rate=24000
                )
                
                # GPU로 이동 및 형변환
                input_features = processed_inputs.input_features.to(args.device).to(model.dtype)
                feature_lens = processed_inputs.feature_attention_mask.sum(1).to(args.device)

                # 2. Audio Tower(Encoder) 통과 -> Embeddings 추출
                # Qwen3-Omni Audio Tower는 (input_features, feature_lens)를 받음
                audio_embeds = model.audio_tower(
                    input_features, 
                    feature_lens=feature_lens
                ).last_hidden_state # [1, Seq, Dim]
            
            # 엔진에 투입 (Non-blocking)
            engine.push_audio(audio_embeds)
            
            # 실시간성 시뮬레이션 (0.32초 대기)
            # 너무 빨리 넣으면 큐가 넘칠 수 있고, 너무 느리면 끊김.
            # 테스트를 위해 약간 빠르게(0.1초) 넣거나 주석 처리 가능
            # time.sleep(0.1) 
            
            # -- [Receiver Loop] 생성된 오디오 수거 --
            # Sender 루프 한 번 돌 때마다 출력 큐를 비울 때까지 확인
            while True:
                out_bytes = engine.get_audio_output()
                if out_bytes is None:
                    break
                
                # Bytes -> Numpy 변환 (float32 [-1, 1]로 변환 가정)
                # Code2Wav 출력이 int16 변환된 bytes라면 아래처럼 복원
                out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                collected_output_audio.append(out_np)
                print(f"🔊 Received output chunk ({len(out_np)} samples) at step {i}")

        # 모든 입력 전송 후 잠시 대기 (잔여 출력 수거)
        print("⏳ Waiting for remaining outputs...")
        time.sleep(3.0) # 충분히 기다려줌
        
        # 남은거 싹 긁어모으기
        while True:
            out_bytes = engine.get_audio_output()
            if out_bytes is None: break
            out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            collected_output_audio.append(out_np)

    except KeyboardInterrupt:
        print("🛑 Test interrupted")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        engine.stop()
    
    # 5. 결과 저장
    if collected_output_audio:
        final_audio = np.concatenate(collected_output_audio)
        print(f"💾 Saving {len(final_audio)} samples ({len(final_audio)/24000:.1f}s) to {args.output_file}")
        sf.write(args.output_file, final_audio, 24000)
    else:
        print("⚠️ No audio generated! (Check if silence token logic is working too strictly)")

    print(f"✅ Test Finished. Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()