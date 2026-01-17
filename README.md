import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioInput:
    """
    사용자의 음성을 엔진에 투입할 때 사용하는 패킷
    """
    # 전처리된 Mel-Spectrogram Feature [1, 128, T]
    features: torch.Tensor 
    
    # (선택) 디버깅용 타임스탬프 (Lag 측정용)
    timestamp: float = 0.0

@dataclass
class ThoughtPacket:
    """
    Thinker가 생각을 마치고 Talker에게 넘겨주는 패킷
    """
    # Talker의 입력이 될 Hidden States [1, Seq, Dim]
    hidden_states: torch.Tensor
    
    # 예: "음...", "반갑", "습니다"
    text_token_str: Optional[str] = None

@dataclass
class AudioOutput:
    """
    엔진이 생성한 최종 결과물 패킷
    """
    # 스피커로 재생할 PCM Audio Bytes (Int16)
    audio_bytes: bytes
    
    # (선택) 이 오디오가 어떤 텍스트에서 나왔는지 (자막용)
    text_log: Optional[str] = None
