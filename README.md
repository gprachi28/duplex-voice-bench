# Production AI Voice Agent

Voice pipeline over LiveKit WebRTC. See [design.md](design.md) for the full
architecture spec; this README documents only what is built and verified.

## What works today

A full-local voice agent loop: mic in, Silero VAD + Smart Turn v3 gate the
turn, `mlx-whisper` transcribes, Ollama replies, and Kokoro speaks the
reply back — no echo, no cloud LLM/TTS.

- Browser mic → LiveKit Cloud → `livekit-agents` worker (Python, M4 Pro)
- Every incoming audio frame is normalised at ingress to **16 kHz mono
  float32** (the format contract every downstream ML stage consumes)
- **Silero VAD** runs on the normalised buffer and logs
  `SPEECH_START` / `SPEECH_END` events during utterances
- **Smart Turn v3** (ONNX, CPU) runs on a background thread, re-scoring an
  8 s rolling window every 100ms. Its end-of-turn completion probability is
  logged alongside every `SPEECH_END` event
  (`smart_turn_prob=0.95` = confident the turn is complete)

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

- A **sentence buffer** sits between the LLM stream and TTS: it
  accumulates streamed tokens and flushes a segment once it both hits a
  sentence boundary (`[. ? ! ,]`) and reaches the 20-char ElevenLabs-
  recommended minimum, so TTS always gets full-sentence context instead
  of one token at a time.
- **Kokoro TTS** (`prince-canuma/Kokoro-82M`, MLX-accelerated via
  `mlx-audio`) synthesises each flushed segment and the worker publishes
  the resulting audio back into the room as the agent's spoken reply.
  Any text still buffered when the LLM stream ends is flushed and
  spoken too. The outbound track now carries this reply instead of an
  echo of the user's own voice — the earlier echo-loop scaffolding is
  gone.

Verified: unit tests confirm the sentence buffer's flush-on-boundary
logic and that Kokoro produces non-empty float32 audio at its native
24 kHz. Verified live: dispatching a real LiveKit room join showed the
worker preloading all five backends (VAD, Smart Turn, STT, LLM, TTS) and
publishing the `agent-reply` track without error.

- **Barge-in** reacts to Silero VAD `SPEECH_START` directly instead of
  waiting for Smart Turn to confirm the interrupting utterance is a
  complete turn. Smart Turn's completion probability is inherently
  variable for short interjections ("stop", "wait"), so gating barge-in
  on it meant real interruptions were silently absorbed by `TurnGate`'s
  `Continue` branch while the agent kept talking (confirmed live: a real
  `SPEECH_END` scored `smart_turn_prob=0.02` and didn't interrupt at
  all). Now any detected speech immediately pauses playback — cooperative,
  via a checked flag, not `asyncio.Task.cancel()` (an earlier cancel-based
  version corrupted LiveKit's `AudioSource` when it landed mid-
  `capture_frame()`, reproduced live). `PlaybackPump` submits TTS audio in
  small real-time-paced frames instead of one call per segment, so
  pausing mid-segment and resuming later is possible at all. If speech is
  sustained past a 0.7s wall-clock timer (independent of waiting for
  Smart Turn; must exceed Silero's own 0.3s trailing-silence window or
  every interruption wins the race and the pause never gets a chance to
  resolve as a false alarm), the pause escalates into a real interrupt.
  If speech turns out to be brief instead, playback resumes from a
  rewound word boundary — computed from Kokoro's own word-level alignment
  (start/end timestamps per word, already produced during synthesis)
  rather than a fixed time guess, so it never resumes mid-syllable. TTS
  synthesis is deferred entirely while paused (checked both before
  starting and again after, since synthesis itself can't be interrupted
  mid-call), so a sentence generated during a pause that ends up
  escalating never costs real Kokoro/MLX compute for audio nobody hears.

Verified live: repeated barge-in — both brief (resume) and sustained
(escalate) — during active replies muted the agent within one frame,
with no `RtcError` and no dropped session. The `smart_turn_prob=0.02`
scenario that previously went unnoticed now escalates correctly. A
resumed reply continued from a rewound word boundary instead of
mid-syllable; an escalated reply discarded only the small in-flight
remainder rather than multiple full buffered-but-unplayed segments.

- Conversation history preserves a **truncated prefix of an interrupted
  reply** — not the full generated text, and not nothing — using
  Kokoro's own word-level alignment correlated against the wall-clock
  moment the interrupt landed. On a pause that resumes instead of
  escalating, the same alignment reconciles the truncation point back to
  the rewind, so words about to be re-heard aren't double-counted.

Verified live: an escalated reply's history entry stopped at the exact
word whose modeled playback preceded the cut, matching the audio
actually heard — not the full sentence, not an empty turn, and not
words spoken after the interruption began.

- `TurnGate` also force-fires after a **15s wall-clock deadline**
  measured from a turn's first `SPEECH_START`, independent of the
  existing 30s accumulated-speech-sample safety valve. Smart Turn's
  completion probability can stay persistently low across several
  short, fragmented speech bursts (observed live: `0.07`/`0.01`/`0.01`/
  `0.05` across four bursts spanning 47s of wall clock) — since the
  sample-count valve only counts actual voiced audio, fragmented bursts
  rarely add up to 30s and the turn could otherwise hang with no reply.

Verified: unit tests confirm the deadline fires despite low probability
and low sample count, is measured from the turn's original start rather
than reset by each resumed `begin()`, and resets correctly after a
`Fire`. Live verification against the original hang scenario is still
pending.

- A **`/metrics` endpoint** on the FastAPI sidecar exposes
  Prometheus-compatible p50/p95 latency per pipeline stage
  (`end_of_turn`, `transcription`, `llm_ttft`, `sentence_buffer`,
  `tts_first_chunk`, `ttfa`) plus a per-combination turns-processed
  counter. The worker (and each per-room job subprocess it spawns)
  appends one structured JSON record per completed turn to a shared
  file (`agent/metrics.py`) — the sidecar is a separate OS process with
  no other access to that data, so the file is the IPC. The sidecar
  reads only newly-appended lines on each scrape and keeps a bounded
  in-memory sliding window (most recent turns) to compute exact
  nearest-rank percentiles — no persistence across sidecar restarts, no
  cross-instance aggregation, matching this project's single-instance,
  one-active-room design.

Verified live: a real conversation session produced correct `/metrics`
output — all six stage latencies and a `combination_id` reflecting the
actual running STT/LLM/TTS models populated correctly for every turn,
and each turn's `interrupted` flag cross-checked exactly (down to the
millisecond) against the worker's own barge-in log lines.

## Layout

```
agent/
  __init__.py
  audio.py              # to_16k_mono_f32 — the ingress format contract
  vad.py                # Silero VAD config + process-wide singleton
  smart_turn.py         # ring buffer, ONNX scorer, background observer
  turn_gate.py          # utterance accumulation + Smart Turn-gated + wall-clock-deadline firing decision
  stt.py                # STTBackend protocol + mlx-whisper implementation
  llm.py                # LLMBackend protocol + streaming Ollama implementation
  playback.py           # PlaybackPump: real-time-paced audio submission, pause/resume-with-rewind
  sentence_buffer.py    # buffers LLM tokens, flushes full sentences to TTS
  tts.py                # TTSBackend protocol + Kokoro (MLX), incl. word-level alignment
  whisper_features.py   # vendored numpy-only log-mel feature extraction
  metrics.py            # TurnMetrics per-turn latency record + JSONL append sink
  worker.py             # livekit-agents worker: VAD, Smart Turn, gated STT, streaming LLM, sentence-buffered TTS reply, barge-in, per-turn metrics
client/
  index.html          # LiveKit Web SDK demo client
server/
  main.py             # FastAPI sidecar: /token, /health, /metrics, GET / serves the client
  metrics.py          # MetricsAggregator: reads the turn JSONL, renders Prometheus text
tests/
  test_audio.py                    # 5 contract tests for the ingress function
  test_smart_turn_buffer.py        # ring buffer unit tests
  test_smart_turn_model.py         # ONNX inference against recorded fixtures
  test_smart_turn_observer.py      # background-thread scoring behaviour
  test_turn_gate.py                # utterance accumulation + gating decision + wall-clock deadline
  test_turn_gate_smart_turn.py     # gate decisions against real Smart Turn scores
  test_stt_backend.py              # mlx-whisper transcription against recorded fixtures
  test_llm_backend.py              # Ollama streaming chat against a real local server
  test_sentence_buffer.py          # flush-on-boundary + min-length logic
  test_tts_backend.py              # Kokoro synthesis against real inference
  test_playback.py                 # PlaybackPump framing, pause/resume rewind math, discard logging
  test_worker_dispatch.py          # history/lock bookkeeping, TTS handoff, and barge-in against fake backends
  test_metrics.py                  # TurnMetrics delta computation + JSONL sink
  test_server_metrics.py           # MetricsAggregator refresh/percentile/exposition-format
  test_server_main_metrics.py      # GET /metrics wiring in the FastAPI sidecar
  test_worker_metrics.py           # TurnMetrics capture wiring in _dispatch_gate_result/_synthesize_and_play
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

`mlx-whisper` and Kokoro (via `mlx-audio`) both resolve their HF Hub repo
IDs over the network on every load to check for a newer revision, even
once the weights are already cached locally. Seen in practice: that
network call can hang, and `livekit-agents`' job-init timeout then kills
the worker's job process before it ever joins a room. `.env.example`
sets `HF_HUB_OFFLINE=1` to skip that check and load straight from cache.

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

### Kokoro TTS weights

No manual download step — `mlx-audio` resolves and caches
`prince-canuma/Kokoro-82M` from the HF Hub itself on first use, and
`misaki[en]` (the text-to-phoneme frontend Kokoro depends on) downloads
a spaCy English model (`en_core_web_sm`) on its first import. Both are
cached under `~/.cache`, not `models/`. The first synthesis after a
fresh install will be slow while these download and MLX compiles the
model graph; subsequent calls are fast (tens to low hundreds of ms per
sentence-length segment).

## Run the agent

Two processes, one machine.

Terminal 1 — sidecar (serves the client + mints room tokens):
```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — worker (joins dispatched rooms, runs VAD → Smart Turn →
STT → LLM → TTS):
```bash
python -m agent.worker dev
```

Then open **http://127.0.0.1:8000**, click **Connect**, allow mic, and
speak. The worker transcribes your turn, sends it to the LLM, and speaks
the streamed reply back through Kokoro TTS.

Per-stage latency metrics are available at **http://127.0.0.1:8000/metrics**
(Prometheus text format) once at least one turn has completed. Both
processes need to agree on where the turn-metrics file lives —
`METRICS_LOG_PATH` in `.env` (defaults to `/tmp/voice-agent-metrics.jsonl`
if unset, so this works with no config).

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

`test_tts_backend.py` skips on non-Apple-Silicon machines and downloads
the Kokoro-82M weights on first run (see "Kokoro TTS weights" above).
