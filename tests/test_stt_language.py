"""MlxWhisperBackend must pass language="en" to mlx_whisper.transcribe --
this is an English-only pipeline (SYSTEM_PROMPT enforces English replies,
Kokoro TTS is English-only). Leaving language unset makes mlx_whisper run a
full extra encoder forward pass for language auto-detection
(mlx_whisper/transcribe.py's detect_language() call) whose result is
discarded immediately -- confirmed via direct-call benchmark to add ~0.56s
per turn (roughly one full encoder pass) on large-v3. See
benchmarks/experiments.md.

Mocked, not a real Metal call -- this tests the call contract (what we pass
to mlx_whisper), not mlx_whisper's own behavior, which tests/test_stt_backend.py
already covers against real audio fixtures.
"""

import numpy as np

import agent.stt as stt_module
from agent.stt import MlxWhisperBackend


def test_transcribe_passes_language_en_to_skip_autodetection(monkeypatch):
    captured_kwargs = {}

    def fake_transcribe(audio, *, path_or_hf_repo, **kwargs):
        captured_kwargs.update(kwargs)
        return {"text": "hello"}

    monkeypatch.setattr(stt_module.mlx_whisper, "transcribe", fake_transcribe)

    MlxWhisperBackend().transcribe(np.zeros(16_000, dtype=np.float32))

    assert captured_kwargs.get("language") == "en"
