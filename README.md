# sca_run

Minimal scaffold for **streaming audio -> features -> Qwen3-Omni (Transformers) -> text**.

Team request addressed:
- remove hardcoded `12.5Hz / 4` audio splitting
- make audio chunking configurable (TOML + env overrides)

Architecture note:
- The server can precompute audio features (mel-style `input_features`) and pass
  them into the inference step via `AudioInput`. This makes it easy to plug in a
  separate "thinker/talker" module that expects features instead of raw audio.

## 1) Config

Edit: `config/default.toml`

Key knobs:
- `audio.frame_hz` (default `12.5`)  
  - 12.5Hz => 80ms per frame
- `audio.frames_per_chunk` (default `4`)  
  - 4 frames => 320ms per request

You can override without touching code:
- `SCA_FRAME_HZ`
- `SCA_FRAMES_PER_CHUNK`

Qwen (Transformers) settings:
- `qwen.model_id` / `SCA_QWEN_MODEL_ID`
- `qwen.device_map` / `SCA_QWEN_DEVICE_MAP`
- `qwen.torch_dtype` / `SCA_QWEN_TORCH_DTYPE`
- `qwen.attn_implementation` / `SCA_QWEN_ATTN_IMPL`
- `qwen.max_new_tokens` / `SCA_QWEN_MAX_NEW_TOKENS`

> Note: /infer_wav uses Python's built-in WAV decoder, so it supports *uncompressed* 16-bit PCM WAV. For other formats, decode on the client side first.

## 2) Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# example: local HF model id
export SCA_QWEN_MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
export SCA_QWEN_DEVICE_MAP="auto"      # or "cuda:0"
export SCA_QWEN_TORCH_DTYPE="auto"     # or "float16"

python -m sca_run.server --config config/default.toml --host 0.0.0.0 --port 8000
```

## 3) Endpoints

### Health
- `GET /health` => "ok"

### One-shot WAV inference
- `POST /infer_wav?prompt=...` with multipart form field `file` (wav)

### Streaming PCM16 WebSocket
- `WS /ws/pcm16`

Protocol:
1) (optional) first message: **text JSON** for session overrides
   - `{"prompt": "..."}`
   - `{"frames_per_chunk": 6}`
   - `{"frame_hz": 12.5}`
2) then send **binary** messages: little-endian **mono PCM16** bytes
3) server converts PCM -> features (AudioInput) -> inference, and replies **JSON** per chunk:
   - `{ "text": "...", "chunk_ms": 320.0, "frames_per_chunk": 4, "frame_hz": 12.5 }`
