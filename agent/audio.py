"""Audio ingress format contract: normalise every incoming frame to the shape
every downstream ML stage (VAD, Smart Turn, Whisper) natively consumes.

Input:  interleaved int16 PCM (LiveKit wire format)
Output: 1-D float32, 16 kHz, mono, in [-1, +1]
"""

import numpy as np
from livekit import rtc

TARGET_SR = 16_000


def to_16k_mono_f32(
    samples: np.ndarray, sample_rate: int, channels: int
) -> np.ndarray:
    """Normalise a raw PCM buffer to 16 kHz mono float32 in [-1, +1]."""
    assert samples.dtype == np.int16, "expected int16 (LiveKit wire format)"

    # Collapse to mono in int16 space first — halves samples before resample.
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    # Anti-aliased polyphase resample via LiveKit's own resampler.
    if sample_rate != TARGET_SR:
        resampler = rtc.AudioResampler(sample_rate, TARGET_SR, num_channels=1)
        frame = rtc.AudioFrame(
            data=samples.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(samples),
        )
        out = resampler.push(frame) + resampler.flush()
        samples = np.concatenate(
            [np.frombuffer(f.data, dtype=np.int16) for f in out]
        )

    return samples.astype(np.float32) / 32768.0