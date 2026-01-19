from __future__ import annotations

import array
import io
import threading
import wave
from typing import Any, Optional, Tuple, Callable

from .config import AppConfig
from .io_types import AudioInput


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
    # Feature extraction is needed for both:
    # - backend="transformers" (end-to-end)
    # - backend="team" (teammate pipeline still needs the same features)
    if cfg.qwen.backend.lower() not in ("transformers", "team"):
        raise ValueError(
            f"Unsupported backend: {cfg.qwen.backend!r}. Expected 'transformers' or 'team'."
        )

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


def _extract_sequences_and_audio(generate_out: Any):
    """Best-effort extraction of (sequences, audio_tensor) from model.generate output.

    Qwen3-Omni may return:
    - a Tensor of token ids
    - a tuple (token_ids, audio)
    - a GenerateOutput-like object with .sequences and maybe .audio/.audios
    """
    seq = None
    audio = None

    # Tuple style: (text_ids, audio)
    if isinstance(generate_out, tuple) and len(generate_out) >= 1:
        seq = generate_out[0]
        if len(generate_out) >= 2:
            audio = generate_out[1]
        return seq, audio

    # GenerateOutput style
    if hasattr(generate_out, "sequences"):
        seq = getattr(generate_out, "sequences")
        for k in (
            "audio",
            "audios",
            "audio_values",
            "audio_waveform",
            "waveform",
            # Some implementations return codec/codes rather than waveform
            "audio_codes",
            "codec_codes",
            "codes",
        ):
            if hasattr(generate_out, k):
                audio = getattr(generate_out, k)
                break
        return seq, audio

    # Dict style
    if isinstance(generate_out, dict):
        seq = generate_out.get("sequences") or generate_out.get("sequence") or generate_out.get("ids")
        for k in (
            "audio",
            "audios",
            "audio_values",
            "audio_waveform",
            "waveform",
            "audio_codes",
            "codec_codes",
            "codes",
        ):
            if k in generate_out:
                audio = generate_out[k]
                break
        return seq, audio

    # Plain tensor
    return generate_out, None


def _find_code2wav(model: Any) -> Optional[Callable[[Any], Any]]:
    """Find a `code2wav` callable on common Qwen3-Omni model layouts."""
    candidates = []
    for obj in (
        model,
        getattr(model, "model", None),
        getattr(model, "talker", None),
        getattr(getattr(model, "talker", None), "model", None),
    ):
        if obj is None:
            continue
        fn = getattr(obj, "code2wav", None)
        if callable(fn):
            candidates.append(fn)
    return candidates[0] if candidates else None


def _maybe_decode_codec_to_wav(cfg: AppConfig, model: Any, audio_obj: Any):
    """Best-effort: if `audio_obj` looks like RVQ codec codes, decode via code2wav.

    Returns:
        (wav_tensor_or_ndarray, did_decode: bool)
    """
    import torch

    if audio_obj is None:
        return None, False

    # Only attempt if it is a Tensor and looks *discrete*.
    if not torch.is_tensor(audio_obj):
        return None, False

    # Heuristic: discrete dtypes, or integer-like values.
    is_discrete = audio_obj.dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    )
    if not is_discrete:
        # If float but values are large integers, treat as codes.
        try:
            if audio_obj.numel() > 0:
                mx = float(audio_obj.detach().abs().max().item())
                is_discrete = mx > 2.0
        except Exception:
            is_discrete = False

    if not is_discrete:
        return None, False

    code2wav = _find_code2wav(model)
    if code2wav is None:
        return None, False

    codes = audio_obj
    # Teammate's decoder expects an extra trailing dim sometimes.
    if codes.dim() == 2:
        codes = codes.unsqueeze(-1)

    try:
        wav = code2wav(codes)
        return wav, True
    except Exception:
        return None, False


def _wav_to_numpy_f32(wav_obj: Any):
    """Convert a torch tensor / ndarray into numpy float32."""
    import numpy as np
    import torch

    if wav_obj is None:
        return None
    if torch.is_tensor(wav_obj):
        return wav_obj.to("cpu").float().numpy()
    return np.asarray(wav_obj, dtype=np.float32)


def infer_audio_input_once_result(cfg: AppConfig, audio_in: AudioInput, user_text: str = "") -> dict:
    """Local inference: precomputed AudioInput(features) -> result dict.

    Returns:
        {
          "text": str,
          "wav_f32": Optional[np.ndarray],  # mono float32 waveform in [-1,1]
          "sr": Optional[int],              # sample rate for wav_f32
        }

    Notes:
    - Audio output is best-effort; depending on the model/transformers version,
      generate() may or may not return audio.
    """
    backend = cfg.qwen.backend.lower()
    if backend == "team":
        # Teammate-owned inference pipeline. This keeps the server/UI stable while
        # your teammate iterates on model internals.
        from . import team_infer

        return team_infer.infer_audio_input_once_result(cfg, audio_in, user_text=user_text)

    if backend != "transformers":
        raise ValueError(f"Unsupported backend: {cfg.qwen.backend!r}. Expected 'transformers' or 'team'.")

    model, processor = _load_transformers_backend(cfg)

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

    import torch

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    text_inputs = processor(text=text_prompt, return_tensors="pt", padding=True)

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
                if k == "input_features":
                    inputs[k] = v.to(device=device, dtype=torch.float32)
                else:
                    inputs[k] = v.to(device)

    # Run generation (audio output is model/version dependent).
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": int(cfg.qwen.max_new_tokens),
            # These flags are used in HF docs/examples for Qwen3-Omni.
            # If unsupported by the installed transformers version, we fall back.
            "thinker_do_sample": False,
            "talker_do_sample": True,
            "return_dict_in_generate": True,
        }
        if bool(getattr(cfg.qwen, "return_audio", True)):
            gen_kwargs["return_audio"] = True

        # Best-effort compatibility ladder
        try:
            gen_out = model.generate(**inputs, **gen_kwargs)
        except TypeError:
            gen_kwargs.pop("return_audio", None)
            try:
                gen_out = model.generate(**inputs, **gen_kwargs)
            except TypeError:
                gen_kwargs.pop("thinker_do_sample", None)
                gen_kwargs.pop("talker_do_sample", None)
                try:
                    gen_out = model.generate(**inputs, **gen_kwargs)
                except TypeError:
                    gen_kwargs.pop("return_dict_in_generate", None)
                    gen_out = model.generate(**inputs, **gen_kwargs)

    seq, audio = _extract_sequences_and_audio(gen_out)

    # Decode tokens (new tokens only when possible).
    gen_tokens = seq
    try:
        input_len = inputs["input_ids"].shape[1]
        if hasattr(seq, "__getitem__"):
            sliced = seq[:, input_len:]
            if getattr(sliced, "numel", lambda: 0)() != 0:
                gen_tokens = sliced
    except Exception:
        pass

    text_out = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    wav_np = None
    sr = None
    if audio is not None:
        # Some pipelines return waveform directly, others return RVQ codec codes.
        decoded_wav, did_decode = _maybe_decode_codec_to_wav(cfg, model, audio)
        if did_decode:
            wav_np = _wav_to_numpy_f32(decoded_wav)
            sr = int(getattr(cfg.qwen, "talker_sample_rate", 24000))
        else:
            wav_np = _wav_to_numpy_f32(audio)
            sr = int(getattr(cfg.qwen, "talker_sample_rate", 24000)) if wav_np is not None else None

    return {"text": text_out, "wav_f32": wav_np, "sr": sr}


def infer_audio_input_once(cfg: AppConfig, audio_in: AudioInput, user_text: str) -> str:
    """Backward-compatible helper: returns text only."""
    return infer_audio_input_once_result(cfg, audio_in, user_text).get("text", "")


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
