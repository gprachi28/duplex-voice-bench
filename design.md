# Voice Pipeline — Design Document

## Goal

Build a production-instrumented voice AI pipeline with pluggable STT, LLM, and TTS
backends. The primary deliverable is not just a working demo — it is a **benchmark
report** showing latency, accuracy, and quality tradeoffs across component combinations,
paired with documented production-readiness decisions.

Target metric: **< 1s Time to First Audio (TTFA)** from end of user speech.
Stretch target: **< 500ms TTFA** on cloud stack.

---

## Hardware

**Local dev machine: Apple M4 Pro (unified memory, Metal GPU, Neural Engine)**

This shapes every local stack decision:
- Metal GPU via PyTorch MPS backend accelerates STT and TTS inference
- Unified memory means no CPU↔GPU data transfer overhead — models load fast
- 24–48GB RAM allows running large models locally that would OOM on discrete GPUs
- MLX (Apple's ML framework) is purpose-built for this architecture and outperforms
  CUDA-equivalent stacks on Apple Silicon for the model sizes used here

**VAD and end-of-turn run on CPU by design.** Both are tiny models (Silero ~1MB,
Smart Turn v3 ONNX lightweight classifier). Their combined inference time is < 5ms
per chunk — negligible in the TTFA budget. GPU headroom is reserved entirely for
STT and TTS where inference is actually heavy.

---

## Architecture

Transport is LiveKit WebRTC (Cloud, free tier). Browser connects to a LiveKit
room via the Web SDK; the agent worker joins the same room as a participant
and receives decoded audio frames. FastAPI runs as a sidecar for `/metrics` and
`/health` only — audio does not flow through it.

```
Browser (LiveKit Web SDK, local HTML/JS in repo)
      │
      │  WebRTC (Opus over UDP)
      ▼
┌──────────────────────┐
│    LiveKit Cloud     │  SFU routes audio between participants
└──────────┬───────────┘
           │  Opus @ 48kHz
           ▼
┌─────────────────────────────────────────────────┐
│      livekit-agents worker (M4 Pro, local)      │
│                                                 │
│  Room join → PCM frames @ 48kHz                 │
│       │                                         │
│       ▼                                         │
│  Resample 48kHz → 16kHz mono float32            │
│       │                                         │
│       ▼                                         │
│   Silero VAD  ──── speech / non-speech          │
│       │                                         │
│       ▼                                         │
│  Smart Turn v3 (ONNX)  ── end-of-turn signal   │
│       │                                         │
│       ▼                                         │
│   STT  (pluggable)                              │
│       │                                         │
│       ▼                                         │
│   LLM  (streaming)                              │
│       │                                         │
│       ▼                                         │
│  Sentence buffer  ── flush on [. ? ! ,]         │
│       │                                         │
│       ▼                                         │
│   TTS  (pluggable, streaming)                   │
│       │                                         │
│       ▼                                         │
│  Publish audio track back to room               │
└─────────────────────────────────────────────────┘

┌──────────────────────┐
│  FastAPI sidecar     │  /token, /metrics, /health (localhost bind)
└──────────────────────┘
```

---

## Component Decisions

### Transport — LiveKit WebRTC

**Decision:** LiveKit Cloud for transport, `livekit-agents` Python SDK on the
server, LiveKit Web SDK on the client. Rationale:

- WebRTC gives production-grade audio transport (jitter buffer, packet loss
  recovery, adaptive bitrate) without hand-rolling any of it. Raw WebSocket + PCM
  has TCP head-of-line blocking and no packet loss handling — unusable over real
  networks.
- LiveKit Cloud free tier (5000 connection-minutes/month, 50GB bandwidth) covers
  demo traffic. Same code runs against self-hosted LiveKit if usage grows.
- Room-based JWT tokens provide authentication out of the box, closing the auth
  gap on a public demo.

**Client:** minimal LiveKit Web SDK page shipped as a local HTML/JS file in the
repo. Connects to a room, shows live transcript and per-stage latency HUD,
exposes the backend selector. No Vercel, no Gradio, no HuggingFace Spaces —
the primary demo artifact is a set of recorded videos. Live access is
credential-gated: run the repo locally with your own API keys, or with a
temporary invite code shared out-of-band.

**Server:** `livekit-agents` worker runs on the M4 Pro. LiveKit dispatches an
incoming room to the worker; the worker joins as a participant and receives
audio frames. Metal acceleration is preserved because inference stays local.

### Audio ingress — format normalisation

LiveKit delivers Opus decoded to PCM at 48kHz. The only ingress step is
resampling to 16kHz mono float32 for Whisper. Use `soundfile` or `av`.

No browser-side format handling is required — the LiveKit Web SDK captures the
mic and encodes to Opus internally.

---

### VAD — Silero

Silero VAD segments the audio stream into speech / non-speech regions. It is
lightweight, runs on CPU, and has a well-characterised false-positive rate.

**Tunable:** trailing silence window (default 300ms). Expose as config — too short
cuts users off, too long adds perceived latency.

---

### End-of-turn detection — Pipecat Smart Turn v3

**Problem:** Silence-based VAD cannot distinguish a mid-sentence pause from a
completed turn. This causes either premature cutoff or unacceptable lag.

**Decision:** Use `pipecat-ai/smart-turn-v3` (ONNX, Apache 2.0, weights on
HuggingFace). The model analyses an 8-second rolling mel spectrogram window
(80 mel bins, 800 frames at 10ms hop) and detects prosodic turn-completion cues
from raw audio — no transcription required.

This separates the pipeline into two clear responsibilities:
- Silero VAD: is audio present?
- Smart Turn: is the turn complete?

Smart Turn v3 weights are bundled as ONNX so no framework dependency on Pipecat.

---

### STT — pluggable

Primary local backend is `mlx-whisper` (Apple MLX, Metal-accelerated) rather than
`faster-whisper` (CTranslate2, no Metal support). On M4 Pro, MLX Whisper runs
significantly faster than the CPU-bound CTranslate2 backend for the same model size.

| Backend | Type | WER (en) | Latency (est. M4 Pro) | Notes |
|---|---|---|---|---|
| `mlx-whisper tiny` | Local / Metal | ~12% | 20–50ms | Lowest latency |
| `mlx-whisper large-v3` | Local / Metal | ~6% | 80–200ms | Best local accuracy |
| `whisper.cpp` + Metal | Local / Metal | ~6% | 80–200ms | Alt to MLX, most mature |
| Cohere Transcribe API | Cloud | 5.42% | 200–500ms | SOTA HF Open ASR Leaderboard (Mar 2026) |

All backends implement a common `STTBackend` interface so swapping requires no
pipeline changes.

**Benchmark target:** WER on a fixed evaluation set + p50/p95 latency per backend.

---

### LLM — streaming

OpenAI GPT-4o (cloud) or Ollama with Metal acceleration (local) via streaming API.
On M4 Pro, Ollama runs Llama 3.2 3B at ~80–120 tok/s and 8B at ~40–60 tok/s —
fast enough for real-time conversation. The pipeline consumes the token stream and
does not wait for a full response before forwarding to TTS.

**Metric tracked:** TTFT (time to first token) per request.

---

### LLM → TTS handoff — sentence buffer

**Problem:** Sending one token at a time to TTS produces incorrect prosody and
choppy audio. TTS models need sufficient context to commit to intonation.

**Decision:** Buffer the LLM token stream and flush to TTS on sentence boundaries
(`[. ? ! ,]`). Minimum flush size: 20 characters (ElevenLabs recommendation).

This is the component most voice demos skip. Implementing it correctly is a
visible differentiator.

---

### TTS — pluggable

| Backend | Type | First chunk latency | Quality | Notes |
|---|---|---|---|---|
| ElevenLabs streaming | Cloud | 200–400ms | High | Requires sentence buffer |
| Kokoro | Local | 50–150ms GPU | Good | Fully local, Apache 2.0 |

All backends implement a common `TTSBackend` interface.

**Benchmark target:** time from flush trigger to first audio chunk received.

---

## Deployment Stacks

Two primary stacks targeting different scenarios. Both run on M4 Pro.

### Option 1 — Full local (MLX)
All inference on-device via Apple MLX / Metal. No API costs, best privacy story.

| Component | Tool | Acceleration |
|---|---|---|
| VAD | Silero | CPU |
| End-of-turn | Smart Turn v3 ONNX | CPU |
| STT | `mlx-whisper large-v3` | Metal (MLX) |
| LLM | Ollama + Llama 3.2 8B | Metal |
| TTS | Kokoro | MPS |

**Expected TTFA: 500ms–1.2s**

### Option 2 — Local STT/TTS, cloud LLM (hybrid)
Offload LLM latency to GPT-4o streaming while keeping audio processing local.
Best TTFA while keeping audio data on-device.

| Component | Tool | Acceleration |
|---|---|---|
| VAD | Silero | CPU |
| End-of-turn | Smart Turn v3 ONNX | CPU |
| STT | `mlx-whisper large-v3` | Metal (MLX) |
| LLM | GPT-4o streaming | Cloud |
| TTS | Kokoro | MPS |

**Expected TTFA: 350–700ms**

The demo runs Option 2 by default (best TTFA, most reliable). Option 1 is available
via the backend selector to demonstrate the full-local capability.

---

## Benchmark Combinations

The benchmark suite runs every combination below and records per-stage and
end-to-end TTFA.

| # | STT | LLM | TTS | Stack type |
|---|---|---|---|---|
| 1 | `mlx-whisper tiny` | GPT-4o | Kokoro | Hybrid – speed |
| 2 | `mlx-whisper large-v3` | GPT-4o | Kokoro | Hybrid – accuracy (Option 2) |
| 3 | Cohere Transcribe | GPT-4o | ElevenLabs | Full cloud |
| 4 | Cohere Transcribe | GPT-4o | Kokoro | Cloud STT/LLM, local TTS |
| 5 | `mlx-whisper large-v3` | Ollama 8B | Kokoro | Full local (Option 1) |

**Per-stage metrics logged for every combination:**

- VAD end → Smart Turn confirmation (end-of-turn latency)
- Smart Turn → STT complete (transcription latency)
- STT complete → LLM first token (TTFT)
- LLM first token → first TTS flush (sentence buffer latency)
- TTS flush → first audio chunk (TTS latency)
- **Total TTFA** (primary metric)

Industry baselines for comparison in the report:

| System | TTFA |
|---|---|
| Amazon Alexa | ~1500ms |
| Siri / Google Assistant | ~1000ms |
| Bland.ai / Retell (claimed) | < 500ms |
| **This project target** | **< 1000ms (cloud), < 500ms stretch** |

---

## Production-Readiness Decisions

### Concurrency
**Decision: one active room per agent worker, hard cap.** LiveKit dispatches
incoming rooms to available workers; if the worker is busy, the caller sees a
"busy, try again shortly" message on the client. Rationale: MLX Whisper large-v3
and Kokoro share a single Metal GPU. Two concurrent rooms means contended
inference and non-reproducible latency numbers — which invalidates the benchmark
that is the point of this project. Horizontal scale = more workers, documented
as a paper-only claim, not exercised on a single M4 Pro.

Inside the worker, STT and TTS still run in a thread/process executor to keep
the `livekit-agents` event loop responsive during model inference.

### Authentication
LiveKit room access is gated by short-lived JWT tokens issued from the FastAPI
sidecar. Tokens carry room name, participant identity, and expiry. No
unauthenticated audio path exists.

### Audio format contract
All audio entering the pipeline is normalised to 16kHz mono PCM float32 at ingress.
No component downstream makes assumptions about input format.

### Error recovery
Each stage has an explicit timeout. On timeout or exception, the pipeline sends
a fallback TTS response ("Sorry, I didn't catch that") rather than hanging. The
WebSocket connection remains open.

### Observability
Every pipeline run emits a structured JSON log with per-stage latencies and the
combination ID. A `/metrics` FastAPI endpoint exposes aggregate p50/p95 latencies
compatible with Prometheus scraping.

### Barge-in (known gap)
If the user speaks while TTS is playing, the current design continues playback.
True barge-in requires detecting new VAD activity during TTS output and cancelling
the audio queue. This is documented as a known limitation — not a missing oversight.

---

## Demo

**Primary artifact: recorded video walkthroughs.** Each backend combination gets
its own video showing the same prompts, per-stage latency HUD, and audio quality
side by side. Videos live in `demos/` and are linked from the README + benchmark
report.

**Live access is credential-gated.** No public deploy, no public URL. Someone
who wants to try it live either:
- clones the repo, brings their own OpenAI / ElevenLabs / LiveKit keys, runs
  the stack locally; or
- gets a temporary invite code shared out-of-band, runs the shipped local
  client HTML against a LiveKit room I spin up on demand.

The rationale is cost control: no public `/token` endpoint means no scraper,
no bot, and no accidental $500 OpenAI bill from a viral demo.

The local client HTML (in-repo) uses the LiveKit Web SDK and shows:
- Connect / disconnect from the room
- Live transcript as speech is recognised
- Per-stage latency breakdown in real time (fed via LiveKit data channel from
  the agent worker)
- Active combination selector — swap STT/TTS backends from the UI, mid-session

The combination selector is the thing that makes videos worth watching: same
prompt, different backend, audibly different result.

---

## Known Issues and Mitigations

| Issue | Impact | Mitigation |
|---|---|---|
| CPU-only TTFA > 2s | Not applicable — M4 Pro Metal used for STT/TTS | VAD + Smart Turn on CPU < 5ms, negligible |
| Cohere Transcribe SDK maturity | Unstable API surface | Abstract behind interface; easy to swap |
| Sentence buffer adds latency | First TTS flush delayed | Tune minimum flush size; measure tradeoff |
| Smart Turn false positives | Premature cutoff | Expose confidence threshold as config; benchmark against silence baseline |
| LiveKit Cloud free tier limits | Live demos unavailable if exhausted | 5000 conn-min/month is >20h of talk time; videos are the primary artifact, so exhaustion only blocks credentialed live sessions |
| Agent worker down = live demo down | Local M4 must be running for live access | Worker connects out to LiveKit Cloud; no inbound port needed. Video demos unaffected |
| No public deploy = cannot try before asking | Recruiter friction | Video demos front the README; live access is a one-email ask |

---

## Strengths

- **Benchmark-first** — the project produces a report, not just a demo. Every
  architectural decision is backed by measured data.
- **Pluggable interfaces** — swapping STT or TTS requires no pipeline changes,
  enabling fair A/B comparison.
- **Production turn detection** — Smart Turn v3 ONNX, the same model Pipecat
  ships in production, used independently of the framework.
- **Sentence-buffered handoff** — correctly solves the LLM→TTS prosody problem
  that most demos skip.
- **Instrumented from day one** — per-stage latency logging and a `/metrics`
  endpoint, not bolted on after the fact.
- **Cohere Transcribe** — SOTA WER (5.42%), Apache 2.0, released March 2026.
  Minimal portfolio coverage at this date.
- **Video-first demo, credential-gated live access** — the primary artifact is a
  set of recorded videos comparing backend combinations on identical prompts.
  Live access is invite-code gated to keep API spend bounded.
- **WebRTC transport** — production-grade audio (jitter buffer, loss recovery)
  via LiveKit Cloud, not hand-rolled WebSocket + PCM.

---

## Version 2 — Three-Layer Turn Detection

Based on Pipecat PR by Mark Backman (April 2026). Adds a third layer on top of
VAD + Smart Turn that uses the conversation LLM itself to confirm turn completion.

**The three layers:**
1. VAD (200ms trigger) — is audio present?
2. Smart Turn v3 ONNX — do acoustic features signal completion?
3. LLM single-token tag — does conversation context confirm completion?

**Single-token tagging:** A prompt mixin instructs the LLM to output one of three
tags at the very start of every response:

| Tag | Meaning | Action |
|---|---|---|
| `✓` | Turn complete | Respond immediately |
| `○` | Short incomplete | Wait 5s, re-evaluate |
| `◐` | Long incomplete | Wait 10s, re-evaluate |

Tag is stripped before TTS. Near-zero latency overhead since it's the first token.
Wait times are configurable and can be adjusted in-context mid-conversation
(e.g. "I'm going to give you a phone number").

**Model requirement:** GPT-4.1, Gemini 2.5 Flash, Claude Sonnet 4.5, AWS Nova 2 Pro.
Smaller open-weights models cannot reliably output single-token tags — full-local
Option 1 (Ollama) does not support this layer. V2 requires the hybrid stack (Option 2).
