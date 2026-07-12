# Production AI Voice Agent

Voice pipeline over LiveKit WebRTC. See [design.md](design.md) for the full
architecture spec; this README documents only what is built and verified.

## What works today

An echo loop with Silero VAD observation on the normalised buffer:

- Browser mic → LiveKit Cloud → `livekit-agents` worker (Python, M4 Pro)
- Every incoming audio frame is normalised at ingress to **16 kHz mono
  float32** (the format contract every downstream ML stage will consume)
- The worker republishes the normalised audio back into the room (echo)
- **Silero VAD** runs alongside on the same normalised buffer and logs
  `SPEECH_START` / `SPEECH_END` events during utterances
- Browser plays the echo through the same LiveKit connection

No STT, LLM, or TTS yet — those plug into the normalised buffer.

## Layout

```
agent/
  __init__.py
  audio.py            # to_16k_mono_f32 — the ingress format contract
  vad.py              # Silero VAD config + process-wide singleton
  worker.py           # livekit-agents echo-loop worker
client/
  index.html          # LiveKit Web SDK demo client
server/
  main.py             # FastAPI sidecar: /token, /health, GET / serves the client
tests/
  test_audio.py       # 5 contract tests for the ingress function
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
