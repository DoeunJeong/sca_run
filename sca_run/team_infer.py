"""Team-owned inference hook.

Why this file exists
--------------------
Your team is splitting the system into:
  - *this repo*: server + mic UI + streaming + feature extraction
  - *teammate code*: Qwen3-Omni thinker/talker inference on GPU

In that setup, the server should not need to know whether the model returns:
  - waveform directly (float32), OR
  - codec / RVQ codes that need to be decoded via talker's `code2wav`.

So this module provides a single stable function:
    infer_audio_input_once_result(cfg, audio_in, user_text="")

Your teammate can implement their pipeline either by:
  1) Editing `TeamPipeline.infer()` below, or
  2) Pointing env vars to their own module/class:
        SCA_TEAM_INFER_MODULE=some_package.some_module
        SCA_TEAM_INFER_CLASS=Pipeline

Expected return contract from the teammate pipeline
--------------------------------------------------
Return a dict with any of the following keys:
  - "text": optional string
  - "wav_f32": optional waveform as numpy array or torch tensor
  - "audio_codes": optional codec codes as torch tensor
  - "sr": optional sample rate (defaults to cfg.qwen.talker_sample_rate)

If "audio_codes" is returned and the pipeline exposes `decode_audio(audio_codes)`
(as in your teammate's snippet), we will decode it to waveform.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Optional


_PIPELINE: Any = None


class TeamPipeline:
    """Placeholder pipeline.

    Replace this class with your teammate's inference code, or set
    SCA_TEAM_INFER_MODULE / SCA_TEAM_INFER_CLASS to load an external class.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def infer(self, audio_in, *, user_text: str = "") -> Dict[str, Any]:
        raise NotImplementedError(
            "TeamPipeline.infer() is not implemented. "
            "Either (a) implement it in sca_run/team_infer.py, "
            "or (b) set SCA_TEAM_INFER_MODULE and SCA_TEAM_INFER_CLASS to load your teammate's pipeline."
        )

    # Optional: if your pipeline returns audio_codes, expose decode_audio(codes)->np.ndarray
    # def decode_audio(self, audio_codes):
    #     ...


def _get_pipeline(cfg) -> Any:
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

    # Fallback to the placeholder class (expected to be edited in-repo).
    _PIPELINE = TeamPipeline(cfg)
    return _PIPELINE


def _to_numpy_f32(wav_obj: Any):
    import numpy as np
    try:
        import torch
        if torch.is_tensor(wav_obj):
            return wav_obj.to("cpu").float().numpy()
    except Exception:
        pass
    return np.asarray(wav_obj, dtype=np.float32)


def infer_audio_input_once_result(cfg, audio_in, user_text: str = "") -> Dict[str, Any]:
    """Run teammate pipeline and normalize output to server-friendly dict."""
    pipeline = _get_pipeline(cfg)

    # Call convention: prefer .infer(audio_in, user_text=..)
    if hasattr(pipeline, "infer") and callable(getattr(pipeline, "infer")):
        out = pipeline.infer(audio_in, user_text=user_text)
    elif callable(pipeline):
        out = pipeline(audio_in)
    else:
        raise RuntimeError("Team pipeline is not callable and has no .infer().")

    if out is None:
        out = {}
    if not isinstance(out, dict):
        raise TypeError("Team pipeline must return a dict.")

    text = (out.get("text") or "")
    sr = int(out.get("sr") or getattr(cfg.qwen, "talker_sample_rate", 24000))

    wav = out.get("wav_f32")
    if wav is not None:
        return {"text": text, "wav_f32": _to_numpy_f32(wav), "sr": sr}

    # If teammate returns codec codes, decode with their decoder if available.
    codes = out.get("audio_codes") or out.get("codec_codes") or out.get("codes")
    if codes is None:
        return {"text": text, "wav_f32": None, "sr": None}

    if not hasattr(pipeline, "decode_audio") or not callable(getattr(pipeline, "decode_audio")):
        raise RuntimeError(
            "Team pipeline returned audio_codes but has no decode_audio(codes) method. "
            "Either return wav_f32 directly, or implement decode_audio(codes)->np.ndarray."
        )

    wav = pipeline.decode_audio(codes)
    return {"text": text, "wav_f32": _to_numpy_f32(wav), "sr": sr}
