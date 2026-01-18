#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   ./run_demo_server.sh

export PYTHONPATH="$(pwd)"
uvicorn sca_run.demo_server:app --host 0.0.0.0 --port 8080
