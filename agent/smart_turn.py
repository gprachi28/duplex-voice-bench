"""End-of-turn scorer built on Pipecat's Smart Turn v3 ONNX model.

Passive observer on the same 16 kHz mono float32 buffer Silero VAD sees:
maintains an 8 s rolling window, runs ONNX inference every 100 ms in the
background, exposes the latest completion probability for the pipeline to
consult on Silero END_OF_SPEECH.
"""

from __future__ import annotations

import threading

import numpy as np
import onnxruntime as ort

from agent.whisper_features import compute_whisper_log_mel_features

WINDOW_SAMPLES = 8 * 16_000  # 8 s at 16 kHz, matches the model's rolling window


class _RingBuffer:
    """Fixed-size float32 ring with a single write cursor.

    Writers push mono samples in chronological order; readers get back a
    linearised copy oldest-first via snapshot(). Silence-initialised, so
    an unfilled buffer reads as zero-padded head + real audio tail.
    """

    def __init__(self, size: int) -> None:
        self._buf = np.zeros(size, dtype=np.float32)
        self._cursor = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        n = len(samples)
        if n == 0:
            return
        size = self._buf.size
        # Only the last `size` samples of an oversize push can survive.
        if n >= size:
            with self._lock:
                self._buf[:] = samples[-size:]
                self._cursor = 0
            return
        with self._lock:
            end = self._cursor + n
            if end <= size:
                self._buf[self._cursor:end] = samples
            else:
                split = size - self._cursor
                self._buf[self._cursor:] = samples[:split]
                self._buf[:n - split] = samples[split:]
            self._cursor = end % size

    def snapshot(self) -> np.ndarray:
        with self._lock:
            c = self._cursor
            return np.concatenate([self._buf[c:], self._buf[:c]]).copy()


def _pad_or_truncate_to_window(audio: np.ndarray) -> np.ndarray:
    """Fit audio to exactly WINDOW_SAMPLES, keeping the trailing (most recent)
    audio. Shorter inputs are zero-padded at the head, not the tail — matches
    Pipecat's reference `truncate_audio_to_last_n_seconds` and the ring
    buffer's own head-padding, so real speech always lands at the same
    position the model was trained on regardless of how much history exists.
    """
    if len(audio) > WINDOW_SAMPLES:
        return audio[-WINDOW_SAMPLES:]
    if len(audio) < WINDOW_SAMPLES:
        return np.pad(audio, (WINDOW_SAMPLES - len(audio), 0), mode="constant")
    return audio


class SmartTurnScorer:
    """Scores end-of-turn completion probability via the Smart Turn v3 ONNX model.

    Consumes a 16 kHz mono float32 buffer of any length (fit to the 8 s
    window internally) and returns the completion probability in [0, 1]. The
    exported graph's output is named "logits" and is not sigmoid-activated —
    confirmed by running inference on an all-zero input and observing a
    value outside [0, 1] — so sigmoid is applied here.
    """

    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def score(self, audio: np.ndarray) -> float:
        audio = _pad_or_truncate_to_window(audio)
        features = compute_whisper_log_mel_features(audio)
        input_features = features[np.newaxis, :, :]
        (logits,) = self._session.run(None, {"input_features": input_features})
        return float(1.0 / (1.0 + np.exp(-logits[0, 0])))
