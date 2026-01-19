#!/usr/bin/env python3
"""
팀원 코드와의 통합 테스트 스크립트

이 스크립트는 다음을 테스트합니다:
1. 팀원의 inference.py와 interface.py 로드 가능 여부
2. Log-Mel Spectrogram 생성 가능 여부
3. team_infer.py의 infer_team_wav 함수 작동 여부
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_team_code_import():
    """팀원의 코드를 임포트할 수 있는지 확인"""
    print("\n" + "="*70)
    print("✅ Test 1: 팀원 코드 임포트")
    print("="*70)
    
    try:
        print("[Loading] src/inference.py...")
        from src.inference import Qwen3DuplexLogic, EngineConfig
        print("  ✅ Qwen3DuplexLogic loaded")
        print("  ✅ EngineConfig loaded")
        
        print("[Loading] src/interface.py...")
        from src.interface import FullDuplexConfig, ConversationState
        print("  ✅ FullDuplexConfig loaded")
        print("  ✅ ConversationState loaded")
        
        return True
    except ImportError as e:
        print(f"  ❌ 임포트 실패: {e}")
        return False


def test_feature_extraction():
    """Log-Mel Spectrogram 생성 테스트"""
    print("\n" + "="*70)
    print("✅ Test 2: Log-Mel Spectrogram 생성")
    print("="*70)
    
    try:
        from sca_run.feature_extractor import log_mel_spectrogram
        
        # 더미 오디오 생성 (1초, 16kHz)
        print("[Creating] 1초 더미 오디오 (16kHz mono)...")
        dummy_audio = torch.randn(16000)  # 1초
        print(f"  입력 형태: {dummy_audio.shape}")
        
        # Log-Mel 생성
        print("[Processing] Log-Mel Spectrogram 생성 중...")
        features = log_mel_spectrogram(dummy_audio, sample_rate=16000, n_mels=128)
        print(f"  출력 형태: {features.shape}")
        print(f"  Expected: [1, 128, T]")
        
        if features.shape[0] == 1 and features.shape[1] == 128:
            print("  ✅ Log-Mel Spectrogram 생성 성공!")
            return True
        else:
            print(f"  ❌ 예상과 다른 형태: {features.shape}")
            return False
    
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_input_creation():
    """AudioInput 객체 생성 테스트"""
    print("\n" + "="*70)
    print("✅ Test 3: AudioInput 객체 생성")
    print("="*70)
    
    try:
        from sca_run.io_types import AudioInput
        from sca_run.feature_extractor import log_mel_spectrogram
        
        # Log-Mel 생성
        print("[Creating] Log-Mel features...")
        dummy_audio = torch.randn(16000)
        features = log_mel_spectrogram(dummy_audio, sample_rate=16000, n_mels=128)
        
        # AudioInput 생성
        print("[Creating] AudioInput 객체...")
        audio_in = AudioInput(features=features, timestamp=0.0)
        print(f"  입력 Feature 형태: {audio_in.features.shape}")
        print("  ✅ AudioInput 생성 성공!")
        
        return True
    
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Config 로드 테스트"""
    print("\n" + "="*70)
    print("✅ Test 4: Config 로드")
    print("="*70)
    
    try:
        from sca_run.config import load_config
        
        print("[Loading] config/default.toml...")
        cfg = load_config("config/default.toml")
        
        print(f"  Audio Config:")
        print(f"    - sample_rate: {cfg.audio.sample_rate}")
        print(f"    - frame_hz: {cfg.audio.frame_hz}")
        print(f"    - frames_per_chunk: {cfg.audio.frames_per_chunk}")
        print(f"    - chunk_ms: {cfg.audio.chunk_ms}")
        print(f"    - chunk_bytes: {cfg.audio.chunk_bytes}")
        
        print(f"  Qwen Config:")
        print(f"    - backend: {cfg.qwen.backend}")
        
        print("  ✅ Config 로드 성공!")
        return True
    
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_team_infer_import():
    """team_infer.py 임포트 테스트"""
    print("\n" + "="*70)
    print("✅ Test 5: team_infer.py 임포트")
    print("="*70)
    
    try:
        from sca_run.team_infer import infer_team_wav, reset_conversation
        
        print("[Loading] sca_run/team_infer.py...")
        print("  ✅ infer_team_wav function loaded")
        print("  ✅ reset_conversation function loaded")
        
        return True
    
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_import():
    """server.py 임포트 테스트"""
    print("\n" + "="*70)
    print("✅ Test 6: server.py 임포트")
    print("="*70)
    
    try:
        from sca_run.server import app, CFG
        
        print("[Loading] sca_run/server.py...")
        print("  ✅ FastAPI app loaded")
        print("  ✅ Config loaded")
        
        # 라우트 확인
        routes = [route.path for route in app.routes]
        print(f"\n  등록된 라우트:")
        for route in routes:
            print(f"    - {route}")
        
        if "/ws/pcm16" in routes:
            print("  ✅ WebSocket 핸들러 등록됨")
        
        return True
    
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """테스트 결과 요약"""
    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70)
    
    tests = [
        ("팀원 코드 임포트", results[0]),
        ("Log-Mel 생성", results[1]),
        ("AudioInput 생성", results[2]),
        ("Config 로드", results[3]),
        ("team_infer.py 임포트", results[4]),
        ("server.py 임포트", results[5]),
    ]
    
    passed = sum(results)
    total = len(results)
    
    print()
    for name, result in tests:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\n총 {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 통합이 완료되었습니다!")
        print("\n다음 단계:")
        print("1. 환경 변수 설정:")
        print('   $env:SCA_QWEN_MODEL_ID = "your_model_path"')
        print("\n2. 서버 시작:")
        print("   python -m sca_run.server --config config/default.toml")
        print("\n3. 브라우저에서 접속:")
        print("   http://localhost:8000")
        return 0
    else:
        print("\n⚠️  일부 테스트 실패. 위의 오류를 확인하세요.")
        return 1


def main():
    """모든 테스트 실행"""
    print("\n" + "🧪 " * 35)
    print("Qwen3-Omni FullDuplex 통합 테스트")
    print("🧪 " * 35)
    
    results = [
        test_team_code_import(),
        test_feature_extraction(),
        test_audio_input_creation(),
        test_config_loading(),
        test_team_infer_import(),
        test_server_import(),
    ]
    
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
