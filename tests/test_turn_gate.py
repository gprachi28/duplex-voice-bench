"""Pure unit tests for TurnGate — no ONNX model, no audio files."""

import numpy as np
import pytest

from agent.turn_gate import Continue, Fire, ForceFire, TurnGate


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
