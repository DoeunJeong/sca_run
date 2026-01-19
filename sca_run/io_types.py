from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AudioInput:
    features: torch.Tensor
    timestamp: float = 0.0


@dataclass
class ThoughtPacket:
    hidden_states: torch.Tensor
    text_token_str: Optional[str] = None


@dataclass
class AudioOutput:
    audio_bytes: bytes
    text_log: Optional[str] = None
