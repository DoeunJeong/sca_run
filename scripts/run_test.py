import os
import sys
import time
import argparse
import threading  # 멀티쓰레드용
import torch
import numpy as np
import librosa
import soundfile as sf

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 패키지 경로 (sca_core)
from src.inference import Qwen3OmniFullDuplexEngine, EngineConfig
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

def load_audio_file(file_path, target_sr=16000):
    """오디오 파일을 로드하고 리샘플링함 (Whisper 입력용 16kHz)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    print(f"📂 Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

# -----------------------------------------------------------------------------
# [Receiver Thread] 엔진에서 나오는 오디오를 별도 쓰레드로 계속 수거
# -----------------------------------------------------------------------------
def audio_receiver_loop(engine, collected_list, stop_event):
    print("🎧 [Receiver] Listening for output...")
    while not stop_event.is_set():
        # Non-blocking으로 확인
        out_bytes = engine.get_audio_output()
        if out_bytes:
            # Bytes -> Float32 변환
            out_np = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            collected_list.append(out_np)
            print(".", end="", flush=True) # 진행 상황 표시
        else:
            time.sleep(0.001) # CPU 양보

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="output_response.wav")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 모델 로드
    print(f"🔥 Loading Model from {args.model_path}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype='auto',
        attn_implementation='flash_attention_2',
        trust_remote_code=True
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. 엔진 초기화
    config = EngineConfig(audio_input_tokens=4, text_output_tokens=2, audio_output_tokens=4)
    engine = Qwen3OmniFullDuplexEngine(model, processor.tokenizer, config)
    
    # 3. 오디오 준비 (16kHz)
    full_audio, sr = load_audio_file(args.input_file, target_sr=16000)
    chunk_size = int(sr * 0.32) # 0.32초 단위
    chunks = [full_audio[i:i + chunk_size] for i in range(0, len(full_audio), chunk_size)]
    print(f"📦 Input Audio Split: {len(chunks)} chunks (0.32s each)")

    # 4. 엔진 시작
    engine.start()
    
    # 5. [Receiver Thread] 시작 (비동기 수신)
    collected_output_audio = []
    stop_receiver = threading.Event()
    receiver_thread = threading.Thread(
        target=audio_receiver_loop, 
        args=(engine, collected_output_audio, stop_receiver),
        daemon=True
    )
    receiver_thread.start()
    
    start_time = time.time()
    
    try:
        # 6. [Sender Loop] 메인 쓰레드는 오디오 밀어넣기만 수행
        print("🎙️ [Sender] Streaming audio chunks...")
        for i, chunk in enumerate(chunks):
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            with torch.no_grad():
                # ★ [수정] Audio Tower 직접 호출 X -> Feature Extractor 사용
                # Raw Audio(16k) -> Mel Spectrogram 변환
                features = processor.feature_extractor(
                    [chunk], 
                    return_tensors="pt", 
                    sampling_rate=16000
                )
                # [Batch, Mel, Time] -> GPU 이동
                input_features = features.input_features.to(args.device).to(model.dtype)
            
            # 엔진에 Feature 투입 (Non-blocking)
            engine.push_audio(input_features)
            
            # (옵션) 실시간성 시뮬레이션: 0.32초 대기
            # time.sleep(0.32) 

        print("\n✅ [Sender] All chunks sent. Waiting for trailing response...")
        
        # 7. 잔여 응답 대기 (3초)
        time.sleep(3.0)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 종료 처리
        stop_receiver.set()
        receiver_thread.join()
        engine.stop()
    
    # 8. 결과 저장
    if collected_output_audio:
        final_audio = np.concatenate(collected_output_audio)
        OUTPUT_SR = 24000 
        print(f"💾 Saving {len(final_audio)} samples ({len(final_audio)/OUTPUT_SR:.1f}s) to {args.output_file}")
        sf.write(args.output_file, final_audio, OUTPUT_SR)
    else:
        print("⚠️ No output received!")

    print(f"⏱️ Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()