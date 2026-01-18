from __future__ import annotations

import argparse
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .audio_chunker import PCMChunker
from .config import AppConfig, load_config
from .qwen_client import (
    extract_audio_input_from_pcm16le,
    infer_audio_input_once,
    wav_bytes_to_pcm16le,
)

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
    text = infer_audio_input_once(CFG, audio_in, "")
    return {"text": text, "sample_rate": sr, "channels": ch}


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
                    text = infer_audio_input_once(session_cfg, audio_in, "")
                    preview = (text or "").strip().replace("\n", " ")
                    if len(preview) > 120:
                        preview = preview[:120] + "..."

                    await websocket.send_json(
                        {
                            "chunks": chunks,
                            "chunk_ms": session_cfg.audio.chunk_ms,
                            "frames_per_chunk": session_cfg.audio.frames_per_chunk,
                            "frame_hz": session_cfg.audio.frame_hz,
                            "text_preview": preview,
                        }
                    )
                except Exception as e:
                    await websocket.send_json({"error": str(e), "chunks": chunks})

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
