"""Team-owned inference hook.

Single interface to call teammate's Qwen3-Omni inference pipeline.
Handles both direct waveform and codec-to-waveform decoding transparently.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional


_PIPELINE: Any = None


class TeamPipeline:
    """Placeholder for teammate's inference pipeline."""

    def __init__(self, cfg):
        self.cfg = cfg

    def infer(self, audio_in, *, user_text: str = "") -> Dict[str, Any]:
        """Run inference and return dict with 'text', 'wav_f32' or 'audio_codes', 'sr'."""
        raise NotImplementedError(
            "TeamPipeline.infer() is not implemented. "
            "Either (a) implement it in sca_run/team_infer.py, "
            "or (b) set SCA_TEAM_INFER_MODULE and SCA_TEAM_INFER_CLASS to load your teammate's pipeline."
        )


def _get_pipeline(cfg) -> Any:
    """Load and cache teammate's pipeline from environment vars or use placeholder."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    mod_name = os.getenv("SCA_TEAM_INFER_MODULE", "")
    cls_name = os.getenv("SCA_TEAM_INFER_CLASS", "")

    if mod_name and cls_name:
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        _PIPELINE = cls(cfg)
        return _PIPELINE

    _PIPELINE = TeamPipeline(cfg)
    return _PIPELINE


def _to_numpy_f32(wav_obj: Any):
    """Convert tensor or array to float32 numpy array."""
    import numpy as np
    try:
        import torch
        if torch.is_tensor(wav_obj):
            return wav_obj.to("cpu").float().numpy()
    except Exception:
        pass
    return np.asarray(wav_obj, dtype=np.float32)


def infer_audio_input_once_result(cfg, audio_in, user_text: str = "") -> Dict[str, Any]:
    """Run inference pipeline and return normalized audio dict.
    
    Handles both output types:
    - Direct waveform: returns wav_f32 directly
    - Codec codes: decodes via pipeline.decode_audio()
    
    Returns dict with keys: text, wav_f32 (np.ndarray), sr (sample rate)
    """
    pipeline = _get_pipeline(cfg)
    out = pipeline.infer(audio_in, user_text=user_text)

    if not isinstance(out, dict):
        raise TypeError("Team pipeline must return a dict.")

    text = (out.get("text") or "")
    sr = int(out.get("sr") or getattr(cfg.qwen, "talker_sample_rate", 24000))

    # Return waveform if already decoded
    wav = out.get("wav_f32")
    if wav is not None:
        return {"text": text, "wav_f32": _to_numpy_f32(wav), "sr": sr}

    # Decode from audio codes if provided
    codes = out.get("audio_codes")
    if codes is not None:
        wav = pipeline.decode_audio(codes)
        return {"text": text, "wav_f32": _to_numpy_f32(wav), "sr": sr}

    return {"text": text, "wav_f32": None, "sr": sr}
