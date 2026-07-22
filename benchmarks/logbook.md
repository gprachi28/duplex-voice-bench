# Experiments Logbook

Dated, reverse-chronological record of benchmark runs: what was tried, why,
and what the numbers showed. Every entry links a run file under
`results/runs/` — no claim here without a corresponding raw record.

Entry template:

```markdown
## YYYY-MM-DD -- <combination slug> -- <what changed>
- combination_id: <slug from README.md's table>
- run file: results/runs/<filename> (N turns)
- change under test: <e.g. "none -- first baseline" | "sentence-buffer flush size 20 -> 40">
- result: TTFA p50=... p95=... p99=...; transcription p95=...; llm_ttft p95=...
- reading: <1-3 sentences -- what the number means, what was surprising>
- next: <the single next action>
```

---

## 2026-07-22 -- local-lv3-ollama3b-kokoro -- first baseline capture from live dev sessions
- combination_id: mlx-community/whisper-large-v3-mlx|llama3.2:3b|prince-canuma/Kokoro-82M
- run file: results/runs/20260722_214718_local-lv3-ollama3b-kokoro.jsonl (14 turns)
- change under test: none -- first baseline. `/tmp/voice-agent-metrics.jsonl` held 90 turns across 7 separate worker sessions (restarts reset `turn_id` to `echo-test-1`); the first 6 sessions were pipeline-fix iterations (prewarm timeout, STT repetition-loop guard -- see experiments.md), not steady state. This entry keeps only the last, post-fix session (`echo-test-1..14`, all `prompt_version=v1-concise-en`) so the numbers reflect the pipeline as it stands, not mid-fix noise.
- result (n=14): TTFA p50=2.02s p95=3.42s; transcription p50=1.28s p95=1.44s; llm_ttft p50=0.16s p95=0.39s; tts_first_chunk p50=0.43s p95=1.14s
- reading: Transcription (STT) still dominates TTFA -- ~63% of the p50 budget -- but with the fix-iteration noise removed, the tail is far tighter (p95 TTFA 3.42s vs. 3.84s pooled, and no more double-digit-second outliers). STT p50 (1.28s) is still 6-16x design.md's projected 80-200ms for `mlx-whisper large-v3` -- that table entry reads as an unvalidated estimate, not a measurement. `tts_first_chunk_s` (p50 0.43s) is ~3x over the 50-150ms Kokoro estimate. p50 TTFA is 2x the <1s target. 7/14 turns (50%) were barge-in interruptions.
- next: log utterance audio duration per turn (not currently captured) to check whether transcription_s scales with input length -- if so, the fix is capping/chunking audio before STT, not swapping STT models.

<!-- Newest entries go above this line. -->
