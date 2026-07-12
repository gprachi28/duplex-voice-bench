"""Contract tests for agent.audio.to_16k_mono_f32."""

import numpy as np
import pytest

from agent.audio import to_16k_mono_f32


def test_int16_range_maps_to_unit_float():
    x = np.array([-32768, 0, 32767], dtype=np.int16)
    y = to_16k_mono_f32(x, sample_rate=16_000, channels=1)
    assert y.dtype == np.float32
    assert y[0] == pytest.approx(-1.0)
    assert y[1] == pytest.approx(0.0)
    assert y[2] == pytest.approx(1.0, abs=1e-4)


def test_16k_mono_passthrough_preserves_length():
    x = np.random.randint(-1000, 1000, size=1600, dtype=np.int16)
    y = to_16k_mono_f32(x, sample_rate=16_000, channels=1)
    assert y.shape == (1600,)


def test_48k_mono_downsamples_by_3():
    x = np.zeros(48_000, dtype=np.int16)  # 1 s of silence
    y = to_16k_mono_f32(x, sample_rate=48_000, channels=1)
    assert y.shape == (16_000,)
    assert y.dtype == np.float32


def test_stereo_collapses_to_mono():
    # Interleaved [L0, R0, L1, R1]: averaging cancels opposite-phase channels.
    x = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
    y = to_16k_mono_f32(x, sample_rate=16_000, channels=2)
    assert y.shape == (2,)
    assert np.allclose(y, 0.0, atol=1e-3)


def test_1khz_tone_survives_48k_to_16k_resample():
    # A 1 kHz sine at 48 kHz should still peak at bin 1000 (= 1 kHz) at 16 kHz.
    fs_in = 48_000
    t = np.arange(fs_in, dtype=np.float64) / fs_in
    x = (np.sin(2 * np.pi * 1000 * t) * 30_000).astype(np.int16)
    y = to_16k_mono_f32(x, sample_rate=fs_in, channels=1)
    assert y.shape[0] == pytest.approx(16_000, abs=10)
    spectrum = np.abs(np.fft.rfft(y))
    peak_bin = int(np.argmax(spectrum))
    assert peak_bin == pytest.approx(1000, abs=2), f"peak at bin {peak_bin}"