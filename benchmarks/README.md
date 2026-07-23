# Benchmarks

Evaluation harness and results for the voice pipeline. This folder holds
what's actually been measured — see [logbook.md](logbook.md) for the dated
run-by-run narrative, [experiments.md](experiments.md) for pipeline *fixes*
made in response to those runs (root cause, fix, verification),
[prompts.md](prompts.md) for the `SYSTEM_PROMPT` changelog (exact text per
version, paired with the model it was tested against), and
[design.md](../design.md) for the target combinations and metrics this
harness is building toward.

Every turn's metrics record carries both `combination_id` (STT|LLM|TTS) and
`prompt_version` (`agent/worker.py`'s `SYSTEM_PROMPT_VERSION`). A run file's
name carries a third key, `change_tag` (the `__<change-tag>` suffix — see
"Running" below), marking which code-state iteration it was captured
against. `eval_latency.py` groups by all three — so neither a prompt
change nor a same-day fix iteration ever gets silently pooled with a
different one under the same combination.

## Layout

```
datasets/
  manifest.jsonl   ground truth: {audio_path, transcript, notes} -- committed
  audio/           curated eval clips (a few MB) -- committed
  bulk/            large external corpora (e.g. a LibriSpeech slice), if ever
                   added -- gitignored, not committed
results/
  runs/            JSONL snapshots copied out of the live metrics file, one
                   per benchmark run
  wer/             WER/MER summaries per run
plots/             generated PNGs -- committed, these are the artifact
eval_latency.py    offline analysis: groups results/runs/*.jsonl by
                   combination_id, prompt_version, and change_tag (parsed
                   from each filename), computes p50/p95/p99 per stage
eval_stt.py        WER/MER harness: calls STTBackend.transcribe() directly
                   against datasets/manifest.jsonl (no LiveKit transport)
plot.py            reads eval_latency.py's output, writes plots/*.png
```

## Combination IDs

The worker's `COMBINATION_ID` (`agent/worker.py`) is already an honest,
dynamically-built string: `f"{WHISPER_MODEL}|{OLLAMA_MODEL}|{KOKORO_REPO}"`
(or the equivalent for whichever backends are wired at the time). The table
below maps design.md's named combinations to that literal string so logbook
entries, run filenames, and plots all key on the same identifier.

| Slug (used in logbook/filenames) | design.md combo | Wired today? |
|---|---|---|
| `local-lv3-ollama3b-kokoro` | *(not a design.md combo — actual running stack)* | **Yes** |
| `local-lv3-ollama8b-kokoro` | #5, Option 1 (full local) | No — 3B is running, not 8B; not currently prioritized |
| `hybrid-speed` | #1 (tiny + GPT-4o + Kokoro) | No — needs GPT-4o LLM backend |
| `hybrid-accuracy` | #2, Option 2 (large-v3 + GPT-4o + Kokoro) | No — needs GPT-4o LLM backend |
| `cloud-stt-local-tts` | #4 (Cohere + GPT-4o + Kokoro) | No — needs Cohere STT + GPT-4o backends |
| `full-cloud` | #3 (Cohere + GPT-4o + ElevenLabs) | No — needs Cohere STT + GPT-4o + ElevenLabs backends |

**Sequencing:** benchmark the local stack (`local-lv3-ollama3b-kokoro`) first —
latency baseline, parameter sweeps (Smart Turn threshold, sentence-buffer
flush size, tiny-vs-large-v3 STT), then WER. Cloud combinations come later,
one new backend at a time, each unlocking one more Pareto point. The 8B swap
for `local-lv3-ollama8b-kokoro` is deferred, not currently a priority.

## Running

```bash
# after a live session has produced turns in the worker's metrics JSONL:
cp /tmp/voice-agent-metrics.jsonl benchmarks/results/runs/$(date +%Y%m%d_%H%M%S)_local-lv3-ollama3b-kokoro__<change-tag>.jsonl

python -m benchmarks.eval_latency benchmarks/results/runs
python -m benchmarks.plot
```

`<change-tag>` is a short slug for the code state this run was captured
against (e.g. `baseline`, `stt-lang-temp`) — every run file must have one,
`eval_latency.py` raises if it finds one that doesn't. Reuse the same tag
across repeat sessions of the same code state; give it a new tag once a
fix lands so the trend plots (`ttfa_trend.png`, `stage_trend.png`) can
show the before/after.

## Future / not in scope yet

- Headless LiveKit-participant audio injection (direct backend calls are
  cheaper and cleaner for accuracy/inference-latency measurement; live
  sessions already cover true end-to-end TTFA and barge-in)
- Network degradation simulation (macOS Network Link Conditioner isn't
  scriptable; would mostly test LiveKit's jitter buffer, not this pipeline)
- Pareto frontier plot (needs ≥2-3 combinations with both a WER and a
  latency axis — comes after the cloud backends land)
