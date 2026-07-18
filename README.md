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

- **mlx-whisper STT** now consumes the gate's decision: on `SPEECH_END`,
  if Smart Turn's completion probability is above threshold the
  accumulated utterance is transcribed (`large-v3`, Metal-accelerated);
  if it's below threshold, the pipeline logs "turn incomplete, continuing
  to listen" and keeps accumulating audio across the next speech segment
  instead of firing early. A 30s safety valve force-fires STT if Smart
  Turn never confirms completion, so one confused turn can't hang the
  pipeline

Verified live: a sentence spoken with a mid-sentence pause was correctly
suppressed and stitched across three VAD segments into one accurate
transcript, and continuous unbroken speech past the safety-valve cap
correctly force-fired STT with a logged warning.

- **Ollama LLM** (`llama3.2:3b`, local, streaming) now consumes the
  transcript: each confirmed or forced transcript is appended to a
  per-room conversation history and sent to Ollama's `/api/chat`, and the
  streamed reply is logged token-by-token alongside a TTFT (time to first
  token) metric. History is multi-turn (the model sees prior turns in the
  same room session) and resets automatically when the room session ends.

Verified live: a multi-turn conversation confirmed the model correctly
recalled an earlier turn's content when answering a later turn, proving
history is actually threaded through the LLM call rather than only
logged. The same run also exercised the `asyncio.Lock` concurrency guard
against a real race — a spurious empty-transcript dispatch overlapped
with a genuine follow-up question, logged `overlapping turn: waiting
for prior LLM reply to finish`, and both turns still completed and
appended to history in the correct order.

No TTS yet — the LLM's streamed reply is log-only for now (no
client-visible voice response).

## Layout

```
agent/
  __init__.py
  audio.py              # to_16k_mono_f32 — the ingress format contract
  vad.py                # Silero VAD config + process-wide singleton
  smart_turn.py         # ring buffer, ONNX scorer, background observer
  turn_gate.py          # utterance accumulation + Smart Turn-gated firing decision
  stt.py                # STTBackend protocol + mlx-whisper implementation
  llm.py                # LLMBackend protocol + streaming Ollama implementation
  whisper_features.py   # vendored numpy-only log-mel feature extraction
  worker.py             # livekit-agents worker: echo, VAD, Smart Turn, gated STT + LLM
client/
  index.html          # LiveKit Web SDK demo client
server/
  main.py             # FastAPI sidecar: /token, /health, GET / serves the client
tests/
  test_audio.py                    # 5 contract tests for the ingress function
  test_smart_turn_buffer.py        # ring buffer unit tests
  test_smart_turn_model.py         # ONNX inference against recorded fixtures
  test_smart_turn_observer.py      # background-thread scoring behaviour
  test_turn_gate.py                # utterance accumulation + gating decision
  test_turn_gate_smart_turn.py     # gate decisions against real Smart Turn scores
  test_stt_backend.py              # mlx-whisper transcription against recorded fixtures
  test_llm_backend.py              # Ollama streaming chat against a real local server
  test_worker_dispatch.py          # history + lock bookkeeping against fake backends
  fixtures/smart_turn/             # complete.wav / incomplete.wav
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

### mlx-whisper STT weights

No manual download step — `mlx-whisper` resolves and caches
`mlx-community/whisper-large-v3-mlx` from the HF Hub itself on first use
(a few GB, cached under `~/.cache/huggingface`, not `models/`). The first
transcription after a fresh install or cache clear will be slow while it
downloads; subsequent runs are fast.

### Ollama LLM setup

Install Ollama and pull the model this project uses:

```bash
brew install ollama
ollama serve &
ollama pull llama3.2:3b
```

The worker talks to Ollama over HTTP at `OLLAMA_HOST` (defaults to
`http://localhost:11434` if unset in `.env`). If Ollama isn't running,
the worker still starts — the LLM call simply fails and is logged as an
exception (`LLM streaming failed`) rather than crashing the pipeline.

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

`test_smart_turn_model.py` and `test_turn_gate_smart_turn.py` skip unless
`SMART_TURN_MODEL_PATH` is set (see above):

```bash
SMART_TURN_MODEL_PATH=models/smart-turn-v3/onnx/model.onnx python -m pytest tests/
```

`test_stt_backend.py` skips on non-Apple-Silicon machines and downloads the
`large-v3` weights on first run (see "mlx-whisper STT weights" above).

`test_llm_backend.py` skips unless Ollama is reachable at `OLLAMA_HOST`
(see "Ollama LLM setup" above).
