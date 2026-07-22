"""MlxWhisperBackend must pass temperature=0.0 (a single value, not
mlx_whisper's default 6-value fallback tuple) to mlx_whisper.transcribe.

mlx_whisper.transcribe() retries the full decode at up to 6 temperatures
whenever compression_ratio_threshold ("too repetitive") or logprob_threshold
fails. Direct-call benchmark: forcing the full 6-temperature fallback costs
4.04s vs 0.64s for a single pass (~6.3x) -- in the same range as the 8-11s
p99 TTFA outliers seen in benchmarks/logbook.md's baseline run. Since
agent/stt.py's is_repetition_loop() already discards repetitive output
regardless of which temperature produced it, the fallback buys this
pipeline nothing on its primary trigger (repetition) while risking exactly
the multi-second latency spikes design.md's <1s TTFA target can't afford.
See benchmarks/experiments.md.
"""

import numpy as np

import agent.stt as stt_module
from agent.stt import MlxWhisperBackend


def test_transcribe_passes_single_temperature_not_fallback_tuple(monkeypatch):
    captured_kwargs = {}

    def fake_transcribe(audio, *, path_or_hf_repo, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "hello"}

    monkeypatch.setattr(stt_module.mlx_whisper, "transcribe", fake_transcribe)

    MlxWhisperBackend().transcribe(np.zeros(16_000, dtype=np.float32))

    assert captured_kwargs.get("temperature") == 0.0
