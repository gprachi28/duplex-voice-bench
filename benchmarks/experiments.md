# Experiments — Fixes Log

Dated, reverse-chronological record of pipeline fixes made in response to
benchmark/live-session findings — root cause, the fix, and how it was
verified. Distinct from [logbook.md](logbook.md), which records benchmark
*run results* (latency/WER numbers); this file records *why the pipeline
changed* between runs, so a later regression can be traced back to a
specific, reasoned intervention rather than an unlogged tweak.

Entry template:

```markdown
## YYYY-MM-DD -- <short title>
- symptom: <what was observed, ideally from a real session>
- root cause: <confirmed cause, not a guess -- cite the file/line>
- fix: <what changed, file(s) touched>
- verification: <how confirmed -- test added, direct-call repro, live re-run>
- status: <fixed | mitigated | monitoring>
```

---

## 2026-07-22 -- Prewarm regression: worker silently killed by livekit-agents' init timeout

- **symptom**: After a network blip forced a worker restart, the new
  process produced *no* log output at all past a certain point -- no
  `preloaded`, no `ready`, nothing -- making it look like "the worker
  isn't working" with no error visible in the terminal being watched
  (compounded by initially checking the wrong, already-dead terminal).
  Once the correct terminal was found, the real error was there:
  `ERROR:livekit.agents:initialization timed out, killing process`,
  logged right after Ollama's warmup call succeeded and Kokoro's pipeline
  creation started.
- **root cause**: A regression from this same session's earlier "STT and
  LLM were never warmed" fix. `agents.WorkerOptions.initialize_process_timeout`
  defaults to **10.0s** (confirmed via `inspect.signature`). Before that
  fix, `_prewarm()` only did TTS's dummy synthesize call; now it also does
  a real STT transcribe and a real LLM round-trip first, back to back.
  Total prewarm time (measured standalone earlier at ~6.8s) was close
  enough to the 10s ceiling that real-world variance pushed it over,
  and livekit-agents kills the whole process before prewarm can finish --
  silently, from the outside: no error reaches the metrics file or the
  visible log, the process just dies mid-startup, every time.
- **fix**: `agents.WorkerOptions(...)` in `agent/worker.py` now passes
  `initialize_process_timeout=60.0`. The prewarm work itself is legitimate
  and worth keeping (that's the point of the earlier fix); the 10s default
  was just never sized for it.
- **verification**: Config value, not new logic -- no new test (consistent
  with `_prewarm` itself not being unit-tested). Full suite: 158/158
  passing, no regressions. Live-confirmed: worker restarted, survived
  prewarm, and processed a real multi-turn session (`echo-test-10` through
  `echo-test-14` in the metrics JSONL) with fast LLM TTFT (~0.15-0.21s)
  and no repetition-loop detections.
- **status**: fixed

## 2026-07-22 -- STT decoder repetition-loop guard + spoken fallback

- **symptom**: Live session log showed
  `TRANSCRIPT (confirmed): 'should should should should...'` (~100
  repeats), and separately 12 repeats later in the same session. Both
  occurred exactly during barge-in on already-playing TTS audio. The
  second instance took ~8s to transcribe a 0.67s audio clip -- an
  anomalous latency consistent with a decoder stuck in a repetition loop
  rather than a clean transcription.
- **root cause**: Confirmed headphones were in use (ruling out acoustic
  echo/TTS bleed-through into the mic). This looks like a genuine Whisper
  decoder repetition-loop failure on short, abrupt barge-in audio -- a
  known ASR failure mode, not traceable to a bug in this pipeline's own
  code. Both times, the LLM happened to handle the garbage input gracefully
  (replying about "a repeat loop") and that reply is what got spoken --
  not a guard catching it. Not something to rely on.
- **fix**: `agent/stt.py` gets `is_repetition_loop(text, min_repeats=5)` --
  a cheap post-hoc filter (same word repeated 5+ times consecutively,
  case-insensitive, punctuation-stripped). `agent/metrics.py` gets a new
  `TurnMetrics.stt_repetition_detected` field so occurrences are tracked
  in the turn JSONL going forward. In `agent/worker.py`'s
  `_dispatch_gate_result`, a detected loop now skips the LLM and history
  entirely (neither the garbage transcript nor a fallback reply is
  appended) and speaks a fixed `FALLBACK_REPLY` ("Sorry, I didn't catch
  that.") via the normal TTS path -- the exact wording design.md's "Error
  recovery" section already specified but this is the first path that
  actually speaks it, rather than silently returning like the existing
  STT-exception and LLM-exception handlers do.
- **verification**: 8 new tests in `tests/test_stt_repetition.py`
  (long/short runs, case-insensitivity, punctuation, natural repetition
  like "no no no" not flagged, custom threshold); 2 new tests in
  `tests/test_metrics.py`; 1 new test in `tests/test_worker_metrics.py`
  asserting the LLM is never called (an `ExplodingLLM` fake that raises if
  invoked) and history stays empty. Full suite: 158/158 passing.
- **status**: fixed

## 2026-07-22 -- Client-side "still listening" indicator for long pauses

- **symptom**: See the previous entry ("No feedback during a long pause").
  User chose the client-side visual indicator option over shortening the
  wall-clock deadline or an audio cue.
- **fix**: `agent/turn_gate.py` gets a new `TurnGate.is_open` read-only
  property (`True` from the first `begin()` of a turn until it resolves
  via `_clear()`), which is the single source of truth for whether a turn
  is pending -- needed because `_ingest`'s own 30s sample-count ForceFire
  path can resolve the same turn independently of `_consume_vad_events`'s
  VAD-event-driven path, so a locally-duplicated boolean flag in each
  function could have drifted out of sync. Both functions now publish
  `{"state": "listening"}` / `{"state": "idle"}` over the LiveKit data
  channel (topic `turn_state`) via a new `_publish_turn_state()` helper,
  fired via `asyncio.create_task` (never awaited inline) so a slow/stalled
  publish can't delay turn dispatch. `client/index.html` listens for
  `RoomEvent.DataReceived` on that topic and toggles a `#turnState` banner
  ("🎤 Still listening — take your time…").
- **verification**: 4 new tests in `tests/test_turn_gate.py` for
  `is_open` (false initially, true across a Continue, false after
  Fire/ForceFire); 2 new tests in `tests/test_turn_state.py` for
  `_publish_turn_state`'s payload/topic. Full suite: 147/147 passing.
  **Still pending**: the client/index.html change itself can only be
  confirmed by watching the banner during a real live session (I can't
  drive a browser + microphone myself) -- specifically, it should appear
  during the mid-sentence-pause stress test (session_script_v1.md §C) and
  stay visible across the whole pause, not just while actively speaking.
- **status**: fixed (pending live visual confirmation)

## 2026-07-22 -- SPEECH_END always logged duration=0.00s

- **symptom**: While diagnosing a "went blank" live-session report, every
  single `SPEECH_END (duration=...)` log line read `0.00s`, including one
  where the actual `SPEECH_START`→`SPEECH_END` gap was 1.4s by wall clock.
  Made root-causing the actual incident (next entry) much harder than it
  should have been.
- **root cause**: Not our code. `agent/worker.py` logs
  `event.speech_duration` from the installed `livekit-plugins-silero`'s
  `VADEvent`. Reading that package's source
  (`site-packages/livekit/plugins/silero/vad.py:486`) shows it sets
  `pub_speech_duration = 0.0` immediately before constructing the
  `END_OF_SPEECH` event at line 489-499, which then reports that same
  now-zeroed value. Every `END_OF_SPEECH` event from this library version
  reports `speech_duration=0.0`, unconditionally.
- **fix**: Added `_speech_duration_since(started_at, now)` in
  `agent/worker.py` -- `_consume_vad_events` now records its own
  `time.monotonic()` at `SPEECH_START` and computes the wall-clock gap
  itself for the `SPEECH_END` log line, instead of trusting the upstream
  field.
- **verification**: 2 new tests in `tests/test_speech_duration.py`. Full
  suite: 141/141 passing.
- **status**: fixed

## 2026-07-22 -- Whisper hallucination on a too-short first utterance

- **symptom**: Live session log showed
  `TRANSCRIPT (confirmed): 'Sous-titrage Société Radio-Canada'` — a
  well-known Whisper near-silence hallucination string — even though the
  user hadn't said anything like that.
- **root cause**: Traced from the actual worker log (`SPEECH_START` /
  `SPEECH_END` / `TRANSCRIPT` lines around `10:45:01`-`10:45:04`). The very
  first `SPEECH_START`→`SPEECH_END` pair of the session lasted only ~0.51s
  wall-clock; Smart Turn scored it `0.75` (>= the `0.5` gate threshold), so
  `TurnGate.evaluate()` (`agent/turn_gate.py`) fired immediately with no
  minimum-duration guard anywhere in the pipeline. A ~0.5s scrap of audio
  went straight to `mlx-whisper`, which hallucinated. The ~2.9s gap before
  the transcript logged is consistent with STT's cold-start cost landing on
  this same turn (see the next entry).
- **fix**: Added `MIN_UTTERANCE_S = 0.3` (matching Silero's own 300ms
  trailing-silence convention already in this codebase) to
  `agent/turn_gate.py`. `TurnGate.evaluate()` now only fires on the
  probability path (`smart_turn_prob >= threshold`) if accumulated audio
  also meets this floor; below it, the turn keeps listening (`Continue`)
  instead. The two `ForceFire` safety valves (30s sample cap, 15s
  wall-clock deadline) are deliberately untouched — they exist to prevent
  hangs, not to judge signal quality, and nothing in the evidence pointed
  at them.
- **verification**: 4 new tests in `tests/test_turn_gate.py` (fire
  suppressed below the floor, fire proceeds once the floor is reached, both
  `ForceFire` paths bypass the floor); 3 existing tests updated where their
  placeholder audio (1-2 samples) was incidentally below the new floor.
  Full suite: 137/137 passing after this change alone.
  **Still pending**: a live re-run reproducing a genuinely short first
  utterance, to confirm the gate now keeps listening instead of firing.
- **status**: fixed (pending live confirmation)

## 2026-07-22 -- STT and LLM were never warmed (only TTS was)

- **symptom**: Users repeatedly said "hello, can you hear me?" because the
  agent's first reply took noticeably longer than later ones, despite the
  "ready — you can start talking now" log line firing immediately after
  prewarm.
- **root cause**: `_prewarm()` (`agent/worker.py`) already gave TTS a real
  dummy `synthesize()` call (with a code comment measuring Kokoro's
  ~2-3s cold vs ~0.1-0.2s warm gap) but only *constructed* the STT and LLM
  backend objects — `MlxWhisperBackend.__init__` stores a model name and
  nothing else (`agent/stt.py`), and `mlx_whisper.transcribe()` doesn't
  load/compile large-v3's weights until the first real call;
  `OllamaBackend.__init__` just builds an `httpx.AsyncClient` (`agent/llm.py`),
  so Ollama doesn't load `llama3.2:3b` into memory until the first real
  `/api/chat`. Both costs landed on the first live turn instead of prewarm.
- **fix**: `_prewarm()` now also calls `create_stt_backend().transcribe()`
  on a throwaway silent buffer, and drives one throwaway round-trip through
  a new `_warm_llm()` helper via `asyncio.run()`. The LLM warmup is wrapped
  in try/except (logs a warning, doesn't crash prewarm) since it depends on
  Ollama actually running, which README.md already documents as
  best-effort; STT/TTS warmup is unguarded, matching TTS's existing
  precedent, since both are purely local with already-cached weights.
- **verification**: 2 new tests in `tests/test_worker_prewarm.py`
  (`_warm_llm` drains the stream without error, sends the expected minimal
  message) against a fake LLM backend. Direct-call measurement against the
  real backends: `_prewarm()` now takes ~6.8s total (previously ~0s was
  spent on STT/LLM), and an immediate real STT + LLM call afterward runs at
  normal steady-state latency (~1.3s / ~0.2s) instead of a multi-second
  cold-load spike. Full suite: 139/139 passing.
- **status**: fixed

## 2026-07-22 -- No system prompt: chatty replies + Hindi language drift

- **symptom**: During the first live benchmark session
  (`benchmarks/session_script_v1.md`), two problems surfaced: (1) replies
  were consistently long/verbose for what should be short conversational
  answers; (2) on the history-reflection question ("What was the first
  thing I asked you today?"), the LLM's transcript showed a correct
  response *in Hindi* — the LLM itself produced Hindi text (confirmed via
  the logged transcript, not a transcription error) — which Kokoro
  (English-only TTS) then rendered as garbled, unintelligible audio.
- **root cause**: `agent/worker.py` initialized the per-room `history` as
  a plain empty list (`history: list[dict[str, str]] = []`) and passed it
  straight to `llm_backend.stream_chat(history)` with no `system` role
  message anywhere in the pipeline. With zero constraints on length,
  tone, or language, `llama3.2:3b`'s default instruct behavior is
  verbose, and it's unconstrained enough to drift language entirely on
  a self-referential prompt.
- **fix**: Added a module-level `SYSTEM_PROMPT` constant in
  `agent/worker.py` instructing short (1-3 sentence) replies and
  English-only output regardless of what the conversation references.
  Added `_new_history()` returning a fresh list seeded with that system
  message, and swapped it in at the one place per-room history is
  constructed. No other behavior changed — no temperature/sampling
  knobs touched, no separate language-detection guard added, per
  systematic-debugging's "one change at a time."
- **verification**:
  - `tests/test_worker_history_seed.py` (5 tests): `_new_history()`
    always starts with the system message, the message instructs
    conciseness and English, and each call returns an independent list
    (no shared mutable state across rooms).
  - Direct-call reproduction against the real local Ollama server
    (bypassing LiveKit/STT, isolating the LLM component): re-ran the
    exact two-turn exchange that triggered the bug — "What is the
    capital of Australia?" followed by "What was the first thing I
    asked you today?" — the second reply now stays in English and both
    replies are shorter than the pre-fix behavior.
  - Full suite: 129/129 passing after the change, no regressions.
  - **Still pending**: a live LiveKit re-run of the full session script
    (real STT + TTS in the loop) to confirm the fix holds end-to-end,
    not just at the direct LLM-call level.
- **status**: fixed (pending live end-to-end confirmation)
