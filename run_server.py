#!/usr/bin/env python3
"""
빠른 시작 스크립트

이 파일을 실행하면 Qwen3-Omni FullDuplex 서버를 빠르게 시작할 수 있습니다.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header():
    """헤더 출력"""
    print("\n" + "🚀 " * 35)
    print("Qwen3-Omni FullDuplex 서버 시작")
    print("🚀 " * 35 + "\n")


def check_requirements():
    """필수 요구사항 확인"""
    print("📋 필수 요구사항 확인 중...\n")
    
    # 파일 확인
    required_files = [
        "sca_run/team_infer.py",
        "sca_run/server.py",
        "config/default.toml",
        "src/inference.py",
        "src/interface.py",
    ]
    
    missing = []
    for f in required_files:
        if not Path(f).exists():
            print(f"❌ {f} 없음")
            missing.append(f)
        else:
            print(f"✅ {f}")
    
    if missing:
        print(f"\n⚠️  {len(missing)}개의 파일이 없습니다!")
        return False
    
    print("\n✅ 모든 파일이 있습니다!")
    return True


def check_env_vars():
    """환경 변수 확인"""
    print("\n📝 환경 변수 확인\n")
    
    model_id = os.getenv("SCA_QWEN_MODEL_ID", "")
    device_map = os.getenv("SCA_QWEN_DEVICE_MAP", "auto")
    torch_dtype = os.getenv("SCA_QWEN_TORCH_DTYPE", "auto")
    
    if not model_id:
        print("⚠️  SCA_QWEN_MODEL_ID이 설정되지 않았습니다!")
        print("\n   설정 방법 (PowerShell):")
        print('   $env:SCA_QWEN_MODEL_ID = "path/to/your/finetuned/model"')
        print("\n   또는 (Linux/Mac):")
        print('   export SCA_QWEN_MODEL_ID="path/to/your/finetuned/model"')
        return False
    
    print(f"✅ SCA_QWEN_MODEL_ID: {model_id}")
    print(f"✅ SCA_QWEN_DEVICE_MAP: {device_map}")
    print(f"✅ SCA_QWEN_TORCH_DTYPE: {torch_dtype}")
    
    return True


def start_server():
    """서버 시작"""
    print("\n" + "="*70)
    print("🌐 서버 시작 중...")
    print("="*70 + "\n")
    
    print("💡 팁:")
    print("   - 첫 호출: 모델 로드 때문에 5~30초 소요")
    print("   - 이후: 실시간 처리 (1~2초)")
    print("   - 웹 브라우저: http://localhost:8000\n")
    
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "sca_run.server",
            "--config",
            "config/default.toml",
            "--host",
            "0.0.0.0",
            "--port",
            "8000"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 서버 중지됨")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)


def main():
    """메인"""
    print_header()
    
    # 1. 파일 확인
    if not check_requirements():
        print("\n❌ 필수 파일이 부족합니다.")
        print("다음을 확인하세요:")
        print("  1. src/inference.py와 src/interface.py가 있나요?")
        print("  2. sca_run/team_infer.py가 수정되었나요?")
        return 1
    
    # 2. 환경 변수 확인
    if not check_env_vars():
        return 1
    
    # 3. 서버 시작
    start_server()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
