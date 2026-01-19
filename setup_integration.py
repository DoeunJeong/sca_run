#!/usr/bin/env python3
"""
통합 설정 및 테스트 스크립트

이 스크립트는:
1. 팀원의 코드(src/inference.py, src/interface.py)를 확인
2. 모델 경로 설정
3. 통합이 제대로 되었는지 테스트
"""

import os
import sys
from pathlib import Path

def check_team_code():
    """팀원의 코드 파일이 있는지 확인"""
    print("\n" + "="*70)
    print("📋 팀원 코드 확인")
    print("="*70)
    
    required_files = {
        "src/inference.py": "Qwen3DuplexLogic 클래스",
        "src/interface.py": "FullDuplexConfig, ConversationState 클래스",
    }
    
    all_exist = True
    for filepath, description in required_files.items():
        full_path = Path(filepath)
        if full_path.exists():
            print(f"✅ {filepath} ({description})")
        else:
            print(f"❌ {filepath} (없음!) - {description}")
            all_exist = False
    
    return all_exist


def check_main_files():
    """메인 파일들이 수정되었는지 확인"""
    print("\n" + "="*70)
    print("📋 메인 파일 확인")
    print("="*70)
    
    files_to_check = {
        "sca_run/team_infer.py": "Qwen3DuplexLogic 임포트",
        "sca_run/feature_extractor.py": "Log-Mel 생성",
        "sca_run/server.py": "WebSocket 핸들러",
    }
    
    all_exist = True
    for filepath, description in files_to_check.items():
        full_path = Path(filepath)
        if full_path.exists():
            print(f"✅ {filepath} ({description})")
        else:
            print(f"❌ {filepath} (없음!)")
            all_exist = False
    
    return all_exist


def suggest_env_vars():
    """환경 변수 설정 제안"""
    print("\n" + "="*70)
    print("🔧 환경 변수 설정")
    print("="*70)
    print("\n다음 환경 변수를 설정하세요 (PowerShell):")
    print("\n# 파인튜닝된 모델 경로 (HuggingFace 모델 ID)")
    print('$env:SCA_QWEN_MODEL_ID = "path/to/finetuned/model"')
    print('# 또는: $env:SCA_QWEN_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"')
    print("\n# GPU 사용 설정")
    print('$env:SCA_QWEN_DEVICE_MAP = "auto"')
    print('# 또는: $env:SCA_QWEN_DEVICE_MAP = "cuda:0"')
    print("\n# 데이터 타입 설정 (메모리 절약)")
    print('$env:SCA_QWEN_TORCH_DTYPE = "float16"')
    print('# 또는: $env:SCA_QWEN_TORCH_DTYPE = "auto"')
    
    print("\n또는 Linux/Mac에서:")
    print("\nexport SCA_QWEN_MODEL_ID=\"path/to/finetuned/model\"")
    print("export SCA_QWEN_DEVICE_MAP=\"auto\"")
    print("export SCA_QWEN_TORCH_DTYPE=\"float16\"")


def test_imports():
    """필요한 패키지들이 설치되었는지 확인"""
    print("\n" + "="*70)
    print("📦 패키지 확인")
    print("="*70)
    
    packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "transformers": "Hugging Face Transformers",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
    }
    
    all_installed = True
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {package} ({description})")
        except ImportError:
            print(f"❌ {package} (설치 필요!) - {description}")
            all_installed = False
    
    return all_installed


def print_next_steps():
    """다음 단계"""
    print("\n" + "="*70)
    print("🚀 다음 단계")
    print("="*70)
    print("\n1. 팀원의 코드 확인:")
    print("   - src/inference.py, src/interface.py가 sca_run 폴더와 같은 레벨에 있는지 확인")
    print("\n2. 모델 설정:")
    print("   - 환경 변수에서 제시된 파인튜닝 모델 경로를 설정")
    print("   - 또는 config/default.toml을 수정")
    print("\n3. 필요한 패키지 설치 (필요시):")
    print("   pip install transformers accelerate qwen-omni-utils")
    print("\n4. 서버 시작:")
    print("   python -m sca_run.server --config config/default.toml")
    print("\n5. 웹 브라우저에서:")
    print("   http://localhost:8000")


def main():
    """통합 체크"""
    print("\n" + "🔗 " * 35)
    print("Qwen3-Omni FullDuplex 통합 확인 스크립트")
    print("🔗 " * 35)
    
    team_code_ok = check_team_code()
    main_files_ok = check_main_files()
    imports_ok = test_imports()
    
    suggest_env_vars()
    print_next_steps()
    
    print("\n" + "="*70)
    print("📊 통합 상태")
    print("="*70)
    print(f"팀원 코드: {'✅ 준비됨' if team_code_ok else '❌ 확인 필요'}")
    print(f"메인 파일: {'✅ 준비됨' if main_files_ok else '❌ 확인 필요'}")
    print(f"필수 패키지: {'✅ 설치됨' if imports_ok else '❌ 설치 필요'}")
    
    print("\n")
    
    if not team_code_ok:
        print("⚠️  팀원 코드(src/inference.py, src/interface.py)를 확인하세요!")
        return 1
    
    if not imports_ok:
        print("⚠️  필수 패키지를 설치하세요!")
        print("    pip install -r requirements.txt")
        return 1
    
    print("✅ 모든 파일이 준비되었습니다!")
    print("서버를 시작하세요: python -m sca_run.server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
