"""Silero VAD configuration. Central place for tuning speech-detection knobs."""

from livekit.plugins import silero

from agent.audio import TARGET_SR

# Design contract: trailing silence window that decides end-of-speech.
# Too short cuts users off; too long adds perceived latency.
MIN_SILENCE_DURATION = 0.3

# Probability threshold above which a frame is treated as speech.
ACTIVATION_THRESHOLD = 0.5

_vad_instance: silero.VAD | None = None


def create_vad() -> silero.VAD:
    """Return the process-wide Silero VAD, loading it on first call."""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = silero.VAD.load(
            sample_rate=TARGET_SR,
            min_silence_duration=MIN_SILENCE_DURATION,
            activation_threshold=ACTIVATION_THRESHOLD,
            force_cpu=True,  # VAD stays on CPU; GPU is reserved for STT/TTS
        )
    return _vad_instance