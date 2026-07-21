"""Pure unit tests for TurnGate — no ONNX model, no audio files."""

import numpy as np
import pytest

from agent.turn_gate import Continue, Fire, ForceFire, TurnGate


class FakeClock:
    """Manually-advanced stand-in for time.monotonic, injected into
    TurnGate so wall-clock-deadline tests don't need real sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def gate() -> TurnGate:
    # sample_rate=10, max_duration_s=1.0 -> a 10-sample cap, fast to hit in tests.
    return TurnGate(threshold=0.5, max_duration_s=1.0, sample_rate=10)


def test_fire_on_high_probability_returns_pushed_audio_and_clears_buffer(gate):
    gate.begin()
    gate.push(np.array([1.0, 2.0], dtype=np.float32))
    gate.push(np.array([3.0], dtype=np.float32))
    result = gate.evaluate(0.9)
    assert isinstance(result, Fire)
    assert np.array_equal(result.audio, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    # Buffer cleared: a subsequent turn only contains what's pushed after this point.
    gate.begin()
    gate.push(np.array([9.0], dtype=np.float32))
    result2 = gate.evaluate(0.9)
    assert np.array_equal(result2.audio, np.array([9.0], dtype=np.float32))


def test_continue_on_low_probability_keeps_buffer(gate):
    gate.begin()
    gate.push(np.array([1.0, 2.0], dtype=np.float32))
    result = gate.evaluate(0.1)
    assert isinstance(result, Continue)


def test_resumed_turn_stitches_onto_retained_buffer(gate):
    gate.begin()
    gate.push(np.array([1.0, 2.0], dtype=np.float32))
    gate.evaluate(0.1)  # Continue, buffer retained
    gate.begin()  # resume — must not reset the retained buffer
    gate.push(np.array([3.0, 4.0], dtype=np.float32))
    result = gate.evaluate(0.9)
    assert isinstance(result, Fire)
    assert np.array_equal(
        result.audio, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )


def test_push_before_begin_is_a_noop(gate):
    assert gate.push(np.array([1.0, 2.0, 3.0], dtype=np.float32)) is False
    gate.begin()
    result = gate.evaluate(0.9)
    assert isinstance(result, Fire)
    assert result.audio.shape == (0,)


def test_push_returns_true_exactly_once_when_crossing_max_duration(gate):
    gate.begin()
    assert gate.push(np.zeros(9, dtype=np.float32)) is False
    assert gate.push(np.array([1.0, 2.0], dtype=np.float32)) is True  # crosses 10
    assert gate.push(np.array([3.0], dtype=np.float32)) is False  # dropped, already over budget


def test_force_fire_after_max_duration_ignores_probability_and_clears_buffer(gate):
    gate.begin()
    gate.push(np.zeros(11, dtype=np.float32))  # already past the 10-sample cap
    result = gate.evaluate(0.01)  # low probability would normally Continue
    assert isinstance(result, ForceFire)
    assert result.audio.shape == (11,)

    gate.begin()
    gate.push(np.array([99.0], dtype=np.float32))
    result2 = gate.evaluate(0.9)
    assert np.array_equal(result2.audio, np.array([99.0], dtype=np.float32))


def test_wall_clock_deadline_force_fires_despite_low_probability_and_low_sample_count():
    # max_duration_s=100 -> the sample-count valve can't be what fires here.
    clock = FakeClock()
    gate = TurnGate(
        threshold=0.5,
        max_duration_s=100.0,
        sample_rate=10,
        max_wall_clock_s=15.0,
        clock=clock,
    )
    gate.begin()
    gate.push(np.array([1.0], dtype=np.float32))
    clock.now = 14.9
    assert isinstance(gate.evaluate(0.01), Continue)  # not yet at the deadline

    gate.begin()  # resume after Continue
    clock.now = 15.1
    result = gate.evaluate(0.01)  # still low probability, deadline now passed
    assert isinstance(result, ForceFire)
    assert np.array_equal(result.audio, np.array([1.0], dtype=np.float32))


def test_wall_clock_deadline_is_measured_from_turn_start_not_from_each_resume():
    clock = FakeClock()
    gate = TurnGate(
        threshold=0.5,
        max_duration_s=100.0,
        sample_rate=10,
        max_wall_clock_s=15.0,
        clock=clock,
    )
    gate.begin()
    clock.now = 10.0
    gate.evaluate(0.01)  # Continue, turn "started" at t=0

    clock.now = 12.0
    gate.begin()  # resume — must NOT reset the deadline to 12.0 + 15.0
    clock.now = 15.1  # 15.1s since the ORIGINAL start, only 3.1s since resume
    assert isinstance(gate.evaluate(0.01), ForceFire)


def test_wall_clock_deadline_resets_after_a_fire():
    clock = FakeClock()
    gate = TurnGate(
        threshold=0.5,
        max_duration_s=100.0,
        sample_rate=10,
        max_wall_clock_s=15.0,
        clock=clock,
    )
    gate.begin()
    clock.now = 20.0  # already past what would be a 15s deadline from t=0
    gate.evaluate(0.9)  # Fire on high probability, clears the turn

    gate.begin()  # a fresh turn starting now, at t=20.0
    clock.now = 21.0  # only 1s into the new turn
    assert isinstance(gate.evaluate(0.01), Continue)
