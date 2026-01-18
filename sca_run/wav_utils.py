from __future__ import annotations

import io
import wave


def pcm16_mono_to_wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16LE mono bytes into a WAV container."""
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return bio.getvalue()
