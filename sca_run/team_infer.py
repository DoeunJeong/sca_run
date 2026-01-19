from __future__ import annotations

"""Team inference integration point - Qwen3-Omni FullDuplex 결합 버전.

이 모듈은 다음을 수행합니다:
  1. feature_extractor.py가 생성한 Log-Mel Spectrogram [1, 128, T]을 받음
  2. 팀원의 Qwen3DuplexLogic(src/inference.py)에 전달
  3. Thinker(이해) → Talker(대답) → Code2Wav(음성 생성) 처리
  4. 생성된 음성을 TeamAudioReturn으로 반환

데이터 흐름:
  PCM16 음성 → feature_extractor → Log-Mel [1,128,T]
           → team_infer.py (이 파일)
           → Qwen3DuplexLogic
           → 음성 생성 [T]
           → WebSocket으로 클라이언트에 전달
"""

import os
import threading
import queue
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from .config import AppConfig
from .io_types import AudioInput, TeamAudioReturn

# 팀원의 코드 임포트
try:
    from src.inference import Qwen3DuplexLogic, EngineConfig
    TEAM_CODE_AVAILABLE = True
except ImportError:
    TEAM_CODE_AVAILABLE = False
    print("[Warning] 팀원의 inference.py를 찾을 수 없습니다. src/inference.py 경로를 확인하세요.")


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return default if v is None else v


# ============================================================================
# 전역 상태 관리 (대화 지속을 위해)
# ============================================================================

_model_lock = threading.Lock()
_qwen_model = None
_duplex_logic: Optional[Qwen3DuplexLogic] = None
_audio_output_queue: Optional[queue.Queue] = None

# 추론 단계 카운터
_step_count = 0


def _load_qwen_model(cfg: AppConfig):
    """팀원의 파인튜닝된 Qwen3-Omni 모델을 로드합니다."""
    global _qwen_model
    
    if _qwen_model is not None:
        return _qwen_model
    
    print("[Team Inference] 🔄 Qwen3-Omni 모델 로딩 중...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = _env(
            "SCA_QWEN_MODEL_ID",
            "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        )
        device_map = _env("SCA_QWEN_DEVICE_MAP", "auto")
        torch_dtype = _env("SCA_QWEN_TORCH_DTYPE", "auto")
        
        print(f"[Team Inference] Model ID: {model_id}")
        print(f"[Team Inference] Device Map: {device_map}")
        print(f"[Team Inference] Torch Dtype: {torch_dtype}")
        
        # 모델 로드
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype if torch_dtype != "auto" else None,
            trust_remote_code=True,
        )
        
        print("[Team Inference] ✅ 모델 로드 완료!")
        return _qwen_model
    
    except Exception as e:
        print(f"[Team Inference] ❌ 모델 로드 실패: {e}")
        raise


def _init_duplex_logic(cfg: AppConfig) -> Qwen3DuplexLogic:
    """Qwen3DuplexLogic을 초기화합니다."""
    global _duplex_logic
    
    with _model_lock:
        if _duplex_logic is not None:
            return _duplex_logic
        
        model = _load_qwen_model(cfg)
        
        # EngineConfig 설정
        engine_cfg = EngineConfig(
            audio_input_tokens=4,   # 0.32초 (4 × 80ms)
            text_output_tokens=2,   # Thinker가 생성할 텍스트 토큰
            audio_output_tokens=4,  # Talker가 생성할 오디오 토큰
            silence_token_id=151646,  # Qwen3 Silence 토큰
            system_prompt_text=(
                "<|im_start|>system\n"
                "You are a helpful AI assistant using Qwen3-Omni.\n"
                "<|im_end|>\n"
            )
        )
        
        _duplex_logic = Qwen3DuplexLogic(model)
        
        print("[Team Inference] ✅ Qwen3DuplexLogic 초기화 완료!")
        return _duplex_logic


def infer_team_wav(cfg: AppConfig, audio_in: AudioInput) -> Optional[TeamAudioReturn]:
    """
    팀원의 Qwen3-Omni FullDuplex 모델로 추론합니다.
    
    입력:
        cfg: AppConfig (설정)
        audio_in: AudioInput (Log-Mel features [1, 128, T])
    
    출력:
        TeamAudioReturn (wav float32, sample_rate=24000)
    """
    
    if not TEAM_CODE_AVAILABLE:
        print("[Team Inference] ⚠️ 팀원의 inference.py를 찾을 수 없습니다.")
        return None
    
    try:
        global _duplex_logic, _step_count
        
        # 1. Duplex Logic 초기화 (처음 한 번만)
        logic = _init_duplex_logic(cfg)
        
        # 2. 입력 데이터 준비
        features = audio_in.features
        
        # CPU에 있으면 유지, GPU에 있으면 그대로
        if isinstance(features, torch.Tensor):
            features = features.float()
        else:
            features = torch.from_numpy(features).float()
        
        print(f"[Team Inference] 입력 Feature 형태: {features.shape}")
        
        # 3. Thinker 단계: 오디오 이해하기
        print("[Team Inference] 🧠 Thinker 처리 중...")
        
        # Feature Attention Mask 생성
        time_len = features.shape[2] if features.dim() == 3 else features.shape[1]
        feature_mask = torch.ones((1, time_len), dtype=torch.long)
        
        with torch.no_grad():
            # Thinker Step
            thinker_out = logic.thinker_step(
                input_ids=None,
                input_features=features,
                feature_attention_mask=feature_mask,
                past_key_values=None,
                step_idx=_step_count
            )
            
            _step_count += 1
            
            # 첫 토큰 예측
            next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            print(f"[Team Inference] Thinker 예측 토큰: {next_token.item()}")
            
            # 4. Talker 단계: 답변 생성하기
            print("[Team Inference] 👄 Talker 처리 중...")
            
            # Thinker의 hidden state를 가져오기
            thinker_hidden = thinker_out.hidden_states[-1]
            
            # Talker Step
            audio_codes, talker_kv = logic.talker_step(
                thinker_hidden=thinker_hidden,
                past_key_values=None,
                step_idx=_step_count,
                input_ids=None
            )
            
            _step_count += 1
            
            print(f"[Team Inference] 생성된 오디오 코드 형태: {audio_codes.shape}")
            
            # 5. Code2Wav 단계: 음성 생성하기
            print("[Team Inference] 🎵 Code2Wav 처리 중...")
            
            wav_bytes = logic.decode_audio(audio_codes)
            
            # 바이트를 float32 배열로 변환
            wav_int16 = np.frombuffer(wav_bytes, dtype=np.int16)
            wav_float = wav_int16.astype(np.float32) / 32768.0
            wav_float = np.clip(wav_float, -1.0, 1.0)
            
            print(f"[Team Inference] ✅ 생성된 음성 길이: {len(wav_float)} samples ({len(wav_float)/24000:.2f}초)")
            
            # 6. TeamAudioReturn으로 반환
            return TeamAudioReturn(
                wav=wav_float,
                sample_rate=24000,  # Qwen3-Omni의 기본 샘플레이트
                channels=1,
                text_log=None
            )
    
    except Exception as e:
        import traceback
        print(f"[Team Inference] ❌ 추론 실패: {e}")
        traceback.print_exc()
        return None


def reset_conversation():
    """대화 상태를 초기화합니다 (새로운 대화 시작)."""
    global _duplex_logic, _step_count
    
    with _model_lock:
        _duplex_logic = None
        _step_count = 0
        print("[Team Inference] 🔄 대화 상태 초기화됨")
