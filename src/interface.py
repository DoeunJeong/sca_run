# src/sca_core/inference/interface.py
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Any, Deque
from collections import deque

@dataclass
class FullDuplexConfig:
    """
    파인튜닝된 4:2:4 패턴을 동적으로 조절하는 설정 클래스
    """
    audio_input_tokens: int = 4   # 한 번에 입력받는 오디오 토큰 수 (User Audio)
    text_output_tokens: int = 2   # Thinker가 생성할 텍스트 토큰 수
    audio_output_tokens: int = 4  # Talker가 생성할 오디오 토큰 수 (AI Audio)
    
    # 모델 관련 상수 (Qwen3-Omni 스펙 참조)
    audio_sample_rate: int = 24000
    chunk_duration: float = 0.08 * 4  # 12.5Hz * 4 tokens = 0.32s (예시)

@dataclass
class OmniModelContext:
    """모델 덩어리 (불변)"""
    audio_encoder: Any
    thinker: Any
    talker: Any
    code2wav: Any
    config: FullDuplexConfig

@dataclass
class ConversationState:
    """
    대화 상태 및 파이프라인 버퍼 (가변)
    Thinker와 Talker 사이의 'Hidden State'를 큐로 관리하여 비동기 효과를 냄.
    """
    # 1. KV Cache (과거 기억)
    past_key_values_thinker: Optional[List[torch.Tensor]] = None
    past_key_values_talker: Optional[List[torch.Tensor]] = None
    
    # 2. 텍스트 히스토리 (System Prompt + 대화)
    text_history_ids: Optional[torch.Tensor] = None
    
    # 3. ★ 핵심: Thinker와 Talker 사이의 비동기 파이프라인 버퍼
    # Thinker가 생각(Hidden State)을 마치면 여기에 쌓아두고 바로 다음 듣기로 넘어감.
    # Talker는 여기서 하나씩 꺼내서 말함.
    thinker_output_queue: Deque[torch.Tensor] = field(default_factory=deque)

@dataclass
class InferenceStepInput:
    new_audio_features: Optional[torch.Tensor] # 전처리된 오디오 텐서
    state: ConversationState

@dataclass
class InferenceStepOutput:
    generated_audio_bytes: bytes # 생성된 오디오 (없으면 b'')
    updated_state: ConversationState