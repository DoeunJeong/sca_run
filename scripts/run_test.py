# scripts/run_test.py
import os
import sys
import time
import argparse
import asyncio
import queue
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


# 실제 모델 클래스 (경로는 자네 프로젝트 구조에 맞게)
from src.inference import Qwen3OmniFullDuplexEngine, EngineConfig
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers import Qwen3OmniMoeProcessor

def load_audio_file(file_path, target_sr=24000):
    """오디오 파일을 로드하고 리샘플링함"""
    print(f"📂 Loading audio file: {file_path}")
    # librosa는 float32 [-1, 1]로 로드함
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def main():
    parser = argparse.ArgumentParser(description="Test Full-Duplex Engine with Audio File")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Path to model")
    parser.add_argument("--input-file", type=str, required=True, help="Input audio file (e.g. 3min_noisy.wav)")
    parser.add_argument("--output-file", type=str, default="output_response.wav", help="Output audio file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run")
    args = parser.parse_args()

    # 1. 모델 로드 (run.py와 동일한 방식)
    print("🔥 Loading Model...")
    
    # (선택) Multi-GPU 로드 로직이 필요하면 여기에 추가
    # device_map = load_distributed_map() 
    
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map, # 또는 "auto"
        dtype='auto',          # torch.float16 또는 bfloat16 자동 선택
        attn_implementation='flash_attention_2', 
        trust_remote_code=True
    )
    
    # 3. 프로세서 로드
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)
    
 
    # 2. 엔진 초기화
    config = EngineConfig(audio_input_tokens=4, text_output_tokens=2, audio_output_tokens=4)
    engine = Qwen3OmniFullDuplexEngine(model, processor, config)
    
    # 3. 오디오 준비 (Chunking)
    full_audio, sr = load_audio_file(args.input_file, target_sr=24000)
    
    # 4토큰 분량의 오디오 길이 계산 (예: 0.32초)
    # Qwen3-Omni의 프레임 속도에 맞춰야 함. (가정: 12.5Hz -> 1프레임당 0.08초)
    # 4토큰 = 0.32초 = 24000 * 0.32 = 7680 샘플
    chunk_size = int(sr * 0.32) 
    
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    print(f"📦 Audio split into {len(chunks)} chunks (Chunk size: {chunk_size} samples)")

    # 4. 테스트 시작
    engine.start()
    
    collected_output_audio = []
    start_time = time.time()
    
    try:
        # -- [Sender Loop] 오디오를 실시간처럼 조금씩 밀어넣음 --
        for i, chunk in enumerate(chunks):
            # 마지막 짜투리 패딩 (필요시)
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Tensor 변환 [1, T, D] 등 모델 인풋 형태에 맞게 (Audio Encoder가 있다고 가정)
            # 여기서는 Raw Audio를 Encoder에 넣기 전 단계라고 가정하고 Tensor로만 변환
            # 실제로는 model.audio_encoder(chunk)를 호출하거나, 엔진 내부에서 처리해야 함.
            # 엔진 코드의 push_audio는 "Audio Features"를 받으므로, 여기서 인코딩을 해줘야 함.
            
            audio_tensor = torch.from_numpy(chunk).float().to(args.device)
            
            # [중요] Audio Encoder 통과 (Engine 외부에서 할지 내부에서 할지 결정 필요)
            # Moshi 테스트 코드처럼 여기서 인코딩해서 'Feature'를 넘기는 게 정석
            with torch.no_grad():
                # Qwen3 Audio Encoder 호출 (가정: input_values=[1, len])
                # 실제 모델의 processor나 encoder 메서드 확인 필요
                # 예시: audio_features = model.audio_tower(audio_tensor.unsqueeze(0))
                
                # 임시: 단순히 차원만 맞춰서 보냄 (실제 환경에선 Encoder 호출 필수!)
                # audio_features = audio_tensor.view(1, 1, -1) 
                
                # [수정] 자네 모델의 Audio Encoder 사용
                # Qwen3-Omni Audio Encoder가 mel-spectrogram을 받는지, raw wave를 받는지 확인
                # 여기서는 'audio_tower'가 feature를 뽑아준다고 가정
                audio_features = model.audio_tower(audio_tensor.unsqueeze(0)) # [1, 4, Dim]
            
            # 엔진에 투입
            engine.push_audio(audio_features)
            
            # 실시간성 시뮬레이션 (0.32초 대기)
            # 실제 스트리밍처럼 천천히 넣음 (테스트 속도 높이려면 주석 처리)
            # time.sleep(0.32) 
            
            # -- [Receiver Loop] 생성된 오디오 수거 --
            # 논블로킹으로 확인
            while True:
                out_bytes = engine.get_audio_output()
                if out_bytes is None:
                    break
                
                # Bytes -> Numpy 변환
                out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                collected_output_audio.append(out_np)
                print(f"🔊 Received output chunk ({len(out_np)} samples)")

        # 모든 입력 전송 후 잠시 대기 (잔여 출력 수거)
        print("⏳ Waiting for remaining outputs...")
        time.sleep(2.0) 
        
        # 남은거 싹 긁어모으기
        while True:
            out_bytes = engine.get_audio_output()
            if out_bytes is None: break
            out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            collected_output_audio.append(out_np)

    except KeyboardInterrupt:
        print("🛑 Test interrupted")
    finally:
        engine.stop()
    
    # 5. 결과 저장
    if collected_output_audio:
        final_audio = np.concatenate(collected_output_audio)
        print(f"💾 Saving {len(final_audio)} samples to {args.output_file}")
        sf.write(args.output_file, final_audio, 24000)
    else:
        print("⚠️ No audio generated!")

    print(f"✅ Test Finished. Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()