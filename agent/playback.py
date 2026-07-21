"""Real-time-paced audio playback with pause/resume-with-rewind, so a
barge-in can mute the agent instantly on any detected speech and, if
speech turns out to be brief (a cough, a line pop), resume without a
mid-syllable jump -- see docs/superpowers/specs/2026-07-20-barge-in_heard_text.md.

play_audio (agent/worker.py, pre-Phase-3) pushed a whole TTS segment into
rtc.AudioSource.capture_frame() in one call. There's no way to pause
mid-segment and resume later that way -- clear_queue() is a one-way
destructive flush, with no seek/resume primitive. PlaybackPump submits
audio in small fixed-size frames instead: "pause" is just "stop
submitting more frames," and "resume" is "keep submitting from a chosen
sample offset," computed from the segment's own word-level alignment
(WordTiming) rather than a fixed time guess.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import Awaitable, Callable

import numpy as np

from agent.tts import WordTiming

logger = logging.getLogger("voice-agent-worker")

FRAME_MS = 20
RESUME_REWIND_WORDS = 2

CaptureFrame = Callable[[np.ndarray], Awaitable[None]]
ClearQueue = Callable[[], None]


class PlaybackState(enum.Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


def _compute_rewind(
    words: list[WordTiming], paused_at_s: float, n_words: int
) -> tuple[float, int]:
    """Given a reply's accumulated word alignment and the position
    (seconds into that reply's audio) playback was paused at, return
    (rewind_to_seconds, words_kept_count) -- n_words back from whichever
    word was playing at the pause point, clamped to the start. Resuming
    a couple of words early is a natural "...as I was saying"; resuming
    later than the true position (from timing-estimate error) would skip
    audio the user never heard, which is the worse failure mode."""
    preceding = [i for i, w in enumerate(words) if w.start <= paused_at_s]
    if not preceding:
        return 0.0, 0
    current_index = preceding[-1]
    rewind_index = max(0, current_index - n_words)
    return words[rewind_index].start, rewind_index


class PlaybackPump:
    """Owns real-time-paced submission of one reply's TTS audio to a
    LiveKit AudioSource (via injected capture_frame/clear_queue
    callables, kept LiveKit-agnostic for testing). Retains the reply's
    audio + word alignment so a pause can be followed by a rewound
    resume instead of losing the segment outright."""

    def __init__(
        self, capture_frame: CaptureFrame, clear_queue: ClearQueue, sample_rate: int
    ) -> None:
        self._capture_frame = capture_frame
        self._clear_queue = clear_queue
        self._sample_rate = sample_rate
        self._frame_samples = sample_rate * FRAME_MS // 1000
        self._audio = np.zeros(0, dtype=np.float32)
        self._words: list[WordTiming] = []
        self._submitted_samples = 0
        self._state = PlaybackState.PLAYING

    @property
    def state(self) -> PlaybackState:
        return self._state

    async def submit(self, audio: np.ndarray, words: list[WordTiming]) -> None:
        """Append one synthesized segment's audio + word alignment
        (offset into this reply's running timeline) and, if playing,
        drain it out in real-time-paced frames."""
        offset_s = len(self._audio) / self._sample_rate
        self._audio = (
            np.concatenate([self._audio, audio]) if self._audio.size else audio.copy()
        )
        for w in words:
            self._words.append(
                WordTiming(w.text, offset_s + w.start, offset_s + w.end)
            )
        await self._drain()

    def pause(self) -> float:
        """Stop submitting further frames and flush whatever's still
        queued ahead-of-realtime in the audio source. Returns the
        estimated position (seconds into this reply's audio) playback
        reached."""
        self._state = PlaybackState.PAUSED
        self._clear_queue()
        return self._submitted_samples / self._sample_rate

    def resume(self, rewind_words: int = RESUME_REWIND_WORDS) -> int:
        """Rewind by rewind_words from the paused position (using the
        retained word alignment) and resume submitting in the
        background. Returns the word count kept "confirmed heard" up to
        the rewind point, for the caller to reconcile against any
        parallel word-indexed bookkeeping (e.g. ActiveReply.heard_timeline)."""
        paused_at_s = self._submitted_samples / self._sample_rate
        rewind_s, keep_count = _compute_rewind(self._words, paused_at_s, rewind_words)
        self._submitted_samples = int(rewind_s * self._sample_rate)
        self._state = PlaybackState.PLAYING
        asyncio.create_task(self._drain())
        return keep_count

    def stop(self) -> None:
        """Hard stop: flush the queue and drop all retained audio --
        there's nothing to resume to. Logs how much synthesized-but-
        never-played audio is being thrown away, if any -- confirmed
        live, segments synthesized while paused sat fully buffered
        (never drained) and got silently wiped here, which read from the
        log as "TTS segment logged" but was never actually heard."""
        unplayed_s = (len(self._audio) - self._submitted_samples) / self._sample_rate
        if unplayed_s > 0:
            logger.info("pump: discarding %.2fs of unplayed audio on stop", unplayed_s)
        self._clear_queue()
        self._drop_audio()
        self._state = PlaybackState.STOPPED

    def reset_for_new_reply(self) -> None:
        """Called when a new reply begins -- drop the previous reply's
        retained audio/words and go back to accepting submissions."""
        self._drop_audio()
        self._state = PlaybackState.PLAYING

    def _drop_audio(self) -> None:
        self._audio = np.zeros(0, dtype=np.float32)
        self._words = []
        self._submitted_samples = 0

    async def _drain(self) -> None:
        """Submit remaining frames from _submitted_samples to the end of
        the retained audio, real-time-paced by capture_frame's own
        backpressure. Stops early if paused/stopped mid-drain (checked
        between frames, since asyncio is cooperative -- pause()/stop()
        can only run between our awaits)."""
        drain_start = time.monotonic()
        start_sample = self._submitted_samples
        while (
            self._state == PlaybackState.PLAYING
            and self._submitted_samples < len(self._audio)
        ):
            frame = self._audio[
                self._submitted_samples : self._submitted_samples + self._frame_samples
            ]
            await self._capture_frame(frame)
            self._submitted_samples += len(frame)
        drained_samples = self._submitted_samples - start_sample
        drained_s = drained_samples / self._sample_rate
        wall_s = time.monotonic() - drain_start
        if drained_samples > 0:
            # TEMPORARY diagnostic (INFO so it's visible without changing
            # log level) -- investigating a live-observed discrepancy
            # between expected and actual pause/resume position.
            logger.info(
                "pump: drained %.2fs of audio in %.2fs wall-clock (ratio=%.2f)",
                drained_s,
                wall_s,
                wall_s / drained_s if drained_s else 0.0,
            )
