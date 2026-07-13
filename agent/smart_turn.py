"""End-of-turn scorer built on Pipecat's Smart Turn v3 ONNX model.

Passive observer on the same 16 kHz mono float32 buffer Silero VAD sees:
maintains an 8 s rolling window, runs ONNX inference every 100 ms in the
background, exposes the latest completion probability for the pipeline to
consult on Silero END_OF_SPEECH.
"""

from __future__ import annotations

import threading

import numpy as np


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
