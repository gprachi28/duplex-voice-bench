# Production AI Voice Agent

Voice pipeline over LiveKit WebRTC. See [design.md](design.md) for the full
architecture spec; this README documents only what is built and verified.

## What works today

An echo loop with Silero VAD and Smart Turn v3 both observing the
normalised buffer:

- Browser mic → LiveKit Cloud → `livekit-agents` worker (Python, M4 Pro)
- Every incoming audio frame is normalised at ingress to **16 kHz mono
  float32** (the format contract every downstream ML stage will consume)
- The worker republishes the normalised audio back into the room (echo)
- **Silero VAD** runs alongside on the same normalised buffer and logs
  `SPEECH_START` / `SPEECH_END` events during utterances
- **Smart Turn v3** (ONNX, CPU) runs on a background thread, re-scoring an
  8 s rolling window every 100ms. Its end-of-turn completion probability is
  logged alongside every `SPEECH_END` event
  (`smart_turn_prob=0.95` = confident the turn is complete)
- Browser plays the echo through the same LiveKit connection

Verified: unit tests against two recorded fixtures (a complete and an
incomplete utterance) confirm the ONNX inference + feature extraction
pipeline classifies correctly, and a live LiveKit session confirmed the
worker logs a varied, non-constant probability per turn (`0.01`–`0.95`
across 5 turns in one run).

No STT, LLM, or TTS yet — those plug into the normalised buffer. Smart Turn
is an observer only: nothing downstream consumes its probability yet.

## Layout

```
agent/
  __init__.py
  audio.py             # to_16k_mono_f32 — the ingress format contract
  vad.py               # Silero VAD config + process-wide singleton
  smart_turn.py        # ring buffer, ONNX scorer, background observer
  whisper_features.py  # vendored numpy-only log-mel feature extraction
  worker.py            # livekit-agents echo-loop worker
client/
  index.html          # LiveKit Web SDK demo client
server/
  main.py             # FastAPI sidecar: /token, /health, GET / serves the client
tests/
  test_audio.py               # 5 contract tests for the ingress function
  test_smart_turn_buffer.py   # ring buffer unit tests
  test_smart_turn_model.py    # ONNX inference against recorded fixtures
  test_smart_turn_observer.py # background-thread scoring behaviour
  fixtures/smart_turn/        # complete.wav / incomplete.wav
design.md             # architecture spec (source of truth)
requirements.txt      # runtime deps (livekit-agents, fastapi, etc.)
requirements-dev.in   # dev deps (pytest)
```

## Requirements

- macOS with Apple Silicon (dev tested on M4 Pro)
- Python 3.12
- A LiveKit Cloud project (free tier is sufficient)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest        # dev-only, see requirements-dev.in

cp .env.example .env
# Edit .env with your LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
```

If you hit `SSLCertVerificationError` on the first LiveKit connect (common
on python.org's macOS installer), run:

```bash
"/Applications/Python 3.12/Install Certificates.command"
```

### Smart Turn v3 model weights

Weights are gitignored (`models/`), not committed, and `huggingface_hub` is
not a runtime dependency — install it once to fetch the ONNX build, then
point `SMART_TURN_MODEL_PATH` (in `.env`) at it:

```bash
pip install huggingface_hub
huggingface-cli download onnx-community/smart-turn-v3-ONNX \
  --include "onnx/model.onnx" --local-dir models/smart-turn-v3
```

## Run the echo loop

Two processes, one machine.

Terminal 1 — sidecar (serves the client + mints room tokens):
```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — worker (joins dispatched rooms, does the echo):
```bash
python -m agent.worker dev
```

Then open **http://127.0.0.1:8000**, click **Connect**, allow mic, and speak.
You should hear yourself echoed back with 200–400 ms of round-trip latency.

## Run the tests

```bash
python -m pytest tests/
```

`test_smart_turn_model.py` skips unless `SMART_TURN_MODEL_PATH` is set (see
above):

```bash
SMART_TURN_MODEL_PATH=models/smart-turn-v3/onnx/model.onnx python -m pytest tests/
```
