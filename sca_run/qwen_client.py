from __future__ import annotations

import array
import io
import threading
import wave
from typing import Optional, Tuple

from .config import AppConfig
from .types import AudioInput


def wav_bytes_to_pcm16le(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    """Decode simple PCM WAV bytes into raw PCM16LE frames.

    This uses Python's built-in `wave` module, so it supports *uncompressed* WAV.
    If your clients may upload compressed WAV, decode it on the client side first.

    Returns:
        pcm16le_bytes, sample_rate, channels
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_rate = int(wf.getframerate())
        sample_width = int(wf.getsampwidth())
        n_frames = int(wf.getnframes())
        frames = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported (got sample_width={sample_width}).")
    return frames, sample_rate, channels


def _pcm16le_to_float_mono(pcm16le: bytes, channels: int) -> list[float]:
    """Convert PCM16LE bytes to a mono float waveform in [-1, 1].

    - If channels==2, averages L/R into mono.
    - If channels>2, averages all channels.
    """
    if channels <= 0:
        raise ValueError("channels must be positive")

    # array('h') reads native-endian int16. Most machines are little-endian.
    a = array.array("h")
    a.frombytes(pcm16le)

    if channels == 1:
        return [float(x) / 32768.0 for x in a]

    # Multi-channel: average per frame.
    n = len(a) // channels
    out: list[float] = []
    out_extend = out.append
    idx = 0
    for _ in range(n):
        s = 0
        for _c in range(channels):
            s += a[idx]
            idx += 1
        out_extend((float(s) / float(channels)) / 32768.0)
    return out


# -----------------------------
# Transformers backend (local)
# -----------------------------

# Lazily loaded singleton (model + processor)
_LOCK = threading.Lock()
_MODEL = None
_PROCESSOR = None


def _load_transformers_backend(cfg: AppConfig):
    """Load Qwen3-Omni model + processor once.

    This is intentionally lazy to keep import time fast.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    with _LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return _MODEL, _PROCESSOR

        # Heavy imports live here
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        model_kwargs = {
            "device_map": cfg.qwen.device_map,
        }

        # torch_dtype is stable; some examples also use dtype="auto".
        # Try torch_dtype first, fall back to dtype for compatibility.
        try:
            model_kwargs["torch_dtype"] = cfg.qwen.torch_dtype
            if cfg.qwen.attn_implementation:
                model_kwargs["attn_implementation"] = cfg.qwen.attn_implementation
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(cfg.qwen.model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("torch_dtype", None)
            model_kwargs["dtype"] = cfg.qwen.torch_dtype
            if cfg.qwen.attn_implementation:
                model_kwargs["attn_implementation"] = cfg.qwen.attn_implementation
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(cfg.qwen.model_id, **model_kwargs)

        processor = Qwen3OmniMoeProcessor.from_pretrained(cfg.qwen.model_id)

        model.eval()
        _MODEL, _PROCESSOR = model, processor
        return _MODEL, _PROCESSOR


def extract_audio_input_from_pcm16le(
    cfg: AppConfig,
    pcm16le: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    timestamp: float = 0.0,
) -> AudioInput:
    """CPU-side feature extraction: PCM16LE -> AudioInput(features).

    This produces the mel-spectrogram-style features that Qwen3-Omni expects
    (typically shaped [1, 128, T]).

    Notes:
    - This uses processor.feature_extractor if available.
    - The returned tensor is kept on CPU; the caller (inference) can move it to
      GPU if desired.
    """
    if cfg.qwen.backend.lower() != "transformers":
        raise ValueError(f"Unsupported backend: {cfg.qwen.backend!r}. Expected 'transformers'.")

    _model, processor = _load_transformers_backend(cfg)

    fe = getattr(processor, "feature_extractor", None)
    if fe is None:
        raise RuntimeError("processor.feature_extractor is not available; cannot precompute features")

    # Convert PCM bytes to mono float waveform.
    waveform = _pcm16le_to_float_mono(pcm16le, channels=channels)

    # Heavy imports live here
    import torch

    fe_out = fe(waveform, sampling_rate=sample_rate, return_tensors="pt")
    if not hasattr(fe_out, "input_features"):
        raise RuntimeError("feature_extractor output missing input_features")

    features: torch.Tensor = fe_out.input_features
    return AudioInput(features=features.cpu(), timestamp=float(timestamp))


def infer_audio_input_once(cfg: AppConfig, audio_in: AudioInput, user_text: str) -> str:
    """Local inference: precomputed AudioInput(features) -> text.

    Use this when the *caller* has already computed audio features and wants
    to pass them into the inference layer.
    """
    if cfg.qwen.backend.lower() != "transformers":
        raise ValueError(f"Unsupported backend: {cfg.qwen.backend!r}. Expected 'transformers'.")

    model, processor = _load_transformers_backend(cfg)

    # Build a conversation with an audio placeholder so the chat template inserts the right tokens.
    conversation = [
        {"role": "system", "content": cfg.qwen.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "<features>"},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Heavy imports live here
    import torch

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Tokenize text only; attach precomputed audio features.
    text_inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = dict(text_inputs)
    inputs["input_features"] = audio_in.features

    # Move inputs to model device.
    try:
        device = next(model.parameters()).device
    except Exception:
        device = getattr(model, "device", None)

    if device is not None and str(device) != "meta":
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                # Keep audio features as float32 unless you *know* the model supports fp16 here.
                if k == "input_features":
                    inputs[k] = v.to(device=device, dtype=torch.float32)
                else:
                    inputs[k] = v.to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(cfg.qwen.max_new_tokens))

    # Decode only newly generated tokens when possible.
    try:
        input_len = inputs["input_ids"].shape[1]
        gen = out[:, input_len:]
        if gen.numel() == 0:
            gen = out
    except Exception:
        gen = out

    return processor.batch_decode(gen, skip_special_tokens=True)[0]


def infer_pcm16le_once(
    cfg: AppConfig,
    pcm16le: bytes,
    user_text: str,
    *,
    sample_rate: int,
    channels: int = 1,
    timestamp: float = 0.0,
) -> str:
    """Local inference: PCM16LE chunk -> text using Transformers.

    Design choice (per team discussion):
    - The *streaming layer* does NOT wrap PCM into WAV.
    - Feature extraction happens here (in the inference layer), not in the audio-input layer.

    Args:
        pcm16le: raw PCM16LE bytes.
        user_text: prompt text.
        sample_rate: PCM sample rate.
        channels: number of channels in pcm16le.
        timestamp: optional, useful for lag measurement.
    """
    if cfg.qwen.backend.lower() != "transformers":
        raise ValueError(f"Unsupported backend: {cfg.qwen.backend!r}. Expected 'transformers'.")

    # Two-step path: precompute features, then run inference.
    audio_in = extract_audio_input_from_pcm16le(
        cfg,
        pcm16le,
        sample_rate=sample_rate,
        channels=channels,
        timestamp=timestamp,
    )
    return infer_audio_input_once(cfg, audio_in, user_text)


def infer_wav_once(cfg: AppConfig, wav_bytes: bytes, user_text: str) -> str:
    """Convenience wrapper: WAV upload -> PCM16LE -> inference."""
    pcm16le, sr, ch = wav_bytes_to_pcm16le(wav_bytes)
    return infer_pcm16le_once(cfg, pcm16le, user_text, sample_rate=sr, channels=ch)


# Backward compatible name (older scaffolds used infer_audio_once on WAV bytes)
def infer_audio_once(cfg: AppConfig, wav_bytes: bytes, user_text: str) -> str:  # pragma: no cover
    return infer_wav_once(cfg, wav_bytes, user_text)
