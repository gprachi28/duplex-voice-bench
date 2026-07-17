"""Threading behavior of SmartTurnObserver, using a stub scorer (no ONNX)."""

import time

import numpy as np

from agent.smart_turn import SmartTurnObserver


class _StubScorer:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.calls = 0

    def score(self, audio: np.ndarray) -> float:
        self.calls += 1
        return self.probability


def test_background_thread_updates_latest_probability():
    stub = _StubScorer(0.75)
    observer = SmartTurnObserver(stub, interval_s=0.01)
    observer.start()
    try:
        deadline = time.monotonic() + 1.0
        while stub.calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stub.calls > 0
        assert observer.latest_probability == 0.75
    finally:
        observer.stop()


def test_stop_joins_the_background_thread():
    observer = SmartTurnObserver(_StubScorer(0.1), interval_s=0.01)
    observer.start()
    observer.stop()
    assert not observer._thread.is_alive()
