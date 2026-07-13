"""Pure-numpy tests for the Smart Turn rolling ring buffer.
No ONNX model required."""

import numpy as np

from agent.smart_turn import _RingBuffer


def test_push_less_than_size_pads_head_with_zeros():
    buf = _RingBuffer(size=10)
    buf.push(np.arange(1, 4, dtype=np.float32))  # [1, 2, 3]
    snap = buf.snapshot()
    assert snap.shape == (10,)
    assert snap.dtype == np.float32
    # Oldest first: 7 zeros then [1, 2, 3].
    assert np.array_equal(snap, np.array([0]*7 + [1, 2, 3], dtype=np.float32))


def test_push_exactly_size_returns_pushed_in_order():
    buf = _RingBuffer(size=5)
    buf.push(np.array([1, 2, 3, 4, 5], dtype=np.float32))
    assert np.array_equal(
        buf.snapshot(), np.array([1, 2, 3, 4, 5], dtype=np.float32)
    )


def test_push_more_than_size_keeps_only_the_tail():
    buf = _RingBuffer(size=5)
    buf.push(np.arange(1, 9, dtype=np.float32))  # [1..8], last 5 = [4..8]
    assert np.array_equal(
        buf.snapshot(), np.array([4, 5, 6, 7, 8], dtype=np.float32)
    )


def test_many_small_pushes_across_wrap_boundary():
    buf = _RingBuffer(size=5)
    for v in range(1, 9):
        buf.push(np.array([v], dtype=np.float32))
    # After 8 pushes into a size-5 ring, snapshot = last 5 = [4, 5, 6, 7, 8].
    assert np.array_equal(
        buf.snapshot(), np.array([4, 5, 6, 7, 8], dtype=np.float32)
    )


def test_push_that_straddles_wrap_boundary_is_linearised():
    buf = _RingBuffer(size=5)
    buf.push(np.array([1, 2, 3], dtype=np.float32))       # cursor -> 3
    buf.push(np.array([4, 5, 6, 7], dtype=np.float32))    # wraps, cursor -> 2
    # Ring contents at this point (raw): [6, 7, 3, 4, 5] with cursor=2.
    # Linearised: start at cursor, so [3, 4, 5, 6, 7].
    assert np.array_equal(
        buf.snapshot(), np.array([3, 4, 5, 6, 7], dtype=np.float32)
    )
