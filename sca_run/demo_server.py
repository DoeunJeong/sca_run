"""demo_server

Kept for backwards-compatibility with earlier scaffolds.

This module simply re-exports the main FastAPI app from `sca_run.server`.
Use:
  uvicorn sca_run.demo_server:app --host 0.0.0.0 --port 8080

The app exposes:
- GET /health
- POST /infer_wav
- WS  /ws/pcm16
"""

from .server import app
