from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.responses import Response

from .audio_chunker import PCMChunker
from .config import AppConfig, load_config
from .qwen_client import (
    extract_audio_input_from_pcm16le,
    wav_bytes_to_pcm16le,
)

# New: best-effort text+audio result
from .qwen_client import infer_audio_input_once_result

app = FastAPI(title="sca_run")

# Load default config at import time; CLI can override
_CFG_PATH = os.getenv("SCA_CONFIG")
CFG: AppConfig = load_config(_CFG_PATH)


# UI is served from sca_run/static/index.html
from importlib import resources
from pathlib import Path

def _load_index_html() -> str:
    """Load the minimal mic UI HTML from package data.

    Kept in a separate file for readability.
    """
    try:
        return resources.files("sca_run").joinpath("static", "index.html").read_text(encoding="utf-8")
    except Exception:
        # Fallback for editable/dev runs when package data isn't included
        here = Path(__file__).resolve().parent
        return (here / "static" / "index.html").read_text(encoding="utf-8")

INDEX_HTML = _load_index_html()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/favicon.ico")
def favicon():
    # Browsers request this automatically; returning an empty response avoids noisy 404s.
    return Response(status_code=204)


def _wav_np_to_mono_f32(wav: np.ndarray) -> np.ndarray:
    """Normalize various wav array shapes to mono float32 [T]."""
    w = wav
    if w is None:
        return np.zeros((0,), dtype=np.float32)

    # Common shapes: [T], [1,T], [B,T], [B,T,1], [B,T,C]
    if w.ndim == 3:
        # [B, T, C] -> first batch, first channel
        w = w[0, :, 0]
    elif w.ndim == 2:
        # [B, T] -> first batch
        w = w[0]

    w = np.asarray(w, dtype=np.float32).reshape(-1)
    return w


def _f32_to_pcm16le_bytes(w: np.ndarray) -> bytes:
    """Convert float32 waveform in [-1,1] to PCM16LE bytes."""
    if w.size == 0:
        return b""
    w = np.clip(w, -1.0, 1.0)
    pcm = (w * 32767.0).astype(np.int16)
    return pcm.tobytes()


@app.get("/health")
def health() -> str:
    return "ok"


@app.post("/infer_wav")
async def infer_wav(file: UploadFile = File(...)):
    """One-shot inference endpoint.

    This endpoint accepts a WAV file upload and runs inference.

    NOTE: For streaming use-cases, prefer /ws/pcm16.
    """
    wav = await file.read()
    pcm16le, sr, ch = wav_bytes_to_pcm16le(wav)

    # 1) Audio -> features (AudioInput)
    audio_in = extract_audio_input_from_pcm16le(CFG, pcm16le, sample_rate=sr, channels=ch)

    # 2) Features -> inference (no extra prompt text)
    result = infer_audio_input_once_result(CFG, audio_in, "")
    return {
        "text": result.get("text", ""),
        "sample_rate": sr,
        "channels": ch,
        "has_audio": bool(result.get("wav_f32") is not None),
    }


@app.websocket("/ws/pcm16")
async def ws_pcm16(websocket: WebSocket):
    """WebSocket streaming: client sends PCM16LE bytes; server processes per chunk.

    Client:
    - send binary frames only (ArrayBuffer)

    Server:
    - accumulates bytes
    - every (frames_per_chunk) frames, extracts features and runs inference

    NOTE: No prompt / no runtime overrides.
    """

    await websocket.accept()

    session_cfg = CFG
    chunker = PCMChunker(chunk_bytes=session_cfg.audio.chunk_bytes)
    chunks = 0

    try:
        while True:
            msg = await websocket.receive()

            # Ignore any text frames.
            data = msg.get("bytes")
            if not data:
                continue

            for chunk in chunker.feed(data):
                chunks += 1
                try:
                    audio_in = extract_audio_input_from_pcm16le(
                        session_cfg,
                        chunk,
                        sample_rate=session_cfg.audio.sample_rate,
                        channels=session_cfg.audio.channels,
                    )

                    # Run inference (no prompt text). Keep a short preview for UI debug.
                    res = infer_audio_input_once_result(session_cfg, audio_in, "")
                    text = (res.get("text") or "").strip()
                    preview = text.replace("\n", " ")
                    if len(preview) > 120:
                        preview = preview[:120] + "..."

                    # 1) Always send a chunk status message
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "chunk_result",
                                "chunks": chunks,
                                "chunk_ms": session_cfg.audio.chunk_ms,
                                "frames_per_chunk": session_cfg.audio.frames_per_chunk,
                                "frame_hz": session_cfg.audio.frame_hz,
                                "text_preview": preview,
                            }
                        )
                    )

                    # 2) If text exists, send it as a separate event for the UI log
                    if text:
                        await websocket.send_text(json.dumps({"type": "talker_text", "text": text}))

                    # 3) If the model returned audio, stream it as:
                    #    JSON meta (talker_audio) -> binary PCM16LE bytes
                    wav = res.get("wav_f32")
                    sr = res.get("sr") or 24000
                    if wav is not None:
                        mono = _wav_np_to_mono_f32(wav)
                        pcm_bytes = _f32_to_pcm16le_bytes(mono)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "talker_audio",
                                    "sr": int(sr),
                                    "fmt": "pcm16le",
                                    "channels": 1,
                                }
                            )
                        )
                        await websocket.send_bytes(pcm_bytes)
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "error": str(e), "chunks": chunks}))

    except WebSocketDisconnect:
        return


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    global CFG
    CFG = load_config(args.config)

    import uvicorn

    uvicorn.run("sca_run.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
