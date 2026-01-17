from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File

from .config import AppConfig, load_config
from .audio_chunker import PCMChunker
from .qwen_client import (
    extract_audio_input_from_pcm16le,
    infer_audio_input_once,
    wav_bytes_to_pcm16le,
)

app = FastAPI(title="sca_run")

# Load default config at import time; CLI can override
_CFG_PATH = os.getenv("SCA_CONFIG")
CFG: AppConfig = load_config(_CFG_PATH)


@app.get("/health")
def health() -> str:
    return "ok"


@app.post("/infer_wav")
async def infer_wav(file: UploadFile = File(...), prompt: str = "Transcribe the audio."):
    """One-shot inference endpoint.

    This endpoint accepts a WAV file upload and runs inference.

    NOTE: For streaming use-cases, prefer /ws/pcm16.
    """
    wav = await file.read()
    pcm16le, sr, ch = wav_bytes_to_pcm16le(wav)
    # 1) Audio -> features (AudioInput)
    audio_in = extract_audio_input_from_pcm16le(CFG, pcm16le, sample_rate=sr, channels=ch)
    # 2) Features -> inference
    text = infer_audio_input_once(CFG, audio_in, prompt)
    return {"text": text, "sample_rate": sr, "channels": ch}


@app.websocket("/ws/pcm16")
async def ws_pcm16(websocket: WebSocket):
    """WebSocket streaming: client sends PCM16LE bytes; server returns text per chunk.

    Protocol:
    - optional first text message: JSON dict with overrides, e.g.
      {"prompt": "...", "frames_per_chunk": 6}
    - subsequent messages: binary PCM16LE frames

    Important design choice:
    - Server does NOT package PCM into WAV.
    - Server computes **audio features** (via processor.feature_extractor) and
      forwards those features to the inference step.
    """

    await websocket.accept()
    session_cfg = CFG
    prompt = "Transcribe the audio."

    chunker = PCMChunker(chunk_bytes=session_cfg.audio.chunk_bytes)

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("text"):
                # Allow runtime overrides
                try:
                    obj = json.loads(msg["text"])
                except Exception:
                    await websocket.send_json({"error": "invalid JSON"})
                    continue

                if isinstance(obj, dict):
                    if "prompt" in obj and isinstance(obj["prompt"], str):
                        prompt = obj["prompt"]

                    if "frames_per_chunk" in obj:
                        try:
                            fpc = int(obj["frames_per_chunk"])
                            if fpc <= 0:
                                raise ValueError
                            session_cfg = AppConfig(
                                audio=replace(session_cfg.audio, frames_per_chunk=fpc),
                                qwen=session_cfg.qwen,
                            )
                            chunker = PCMChunker(chunk_bytes=session_cfg.audio.chunk_bytes)
                            await websocket.send_json(
                                {
                                    "info": "updated frames_per_chunk",
                                    "frames_per_chunk": fpc,
                                    "chunk_ms": session_cfg.audio.chunk_ms,
                                }
                            )
                        except Exception:
                            await websocket.send_json({"error": "frames_per_chunk must be positive int"})

                    if "frame_hz" in obj:
                        try:
                            fhz = float(obj["frame_hz"])
                            if fhz <= 0:
                                raise ValueError
                            session_cfg = AppConfig(
                                audio=replace(session_cfg.audio, frame_hz=fhz),
                                qwen=session_cfg.qwen,
                            )
                            chunker = PCMChunker(chunk_bytes=session_cfg.audio.chunk_bytes)
                            await websocket.send_json(
                                {
                                    "info": "updated frame_hz",
                                    "frame_hz": fhz,
                                    "chunk_ms": session_cfg.audio.chunk_ms,
                                }
                            )
                        except Exception:
                            await websocket.send_json({"error": "frame_hz must be positive number"})
                continue

            data = msg.get("bytes")
            if not data:
                continue

            for chunk in chunker.feed(data):
                try:
                    # 1) Audio -> features (AudioInput)
                    audio_in = extract_audio_input_from_pcm16le(
                        session_cfg,
                        chunk,
                        sample_rate=session_cfg.audio.sample_rate,
                        channels=session_cfg.audio.channels,
                    )
                    # 2) Features -> inference
                    text = infer_audio_input_once(session_cfg, audio_in, prompt)
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
                    continue

                await websocket.send_json(
                    {
                        "text": text,
                        "chunk_ms": session_cfg.audio.chunk_ms,
                        "frames_per_chunk": session_cfg.audio.frames_per_chunk,
                        "frame_hz": session_cfg.audio.frame_hz,
                    }
                )

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
