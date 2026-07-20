"""Voice agent worker: Silero VAD, Smart Turn, gated STT, streaming LLM,
sentence-buffered TTS, and barge-in.

Joins a LiveKit room dispatched to us, subscribes to the first remote audio
track, and runs every incoming frame through the ingress format contract
(16 kHz mono float32) to feed Silero VAD and Smart Turn. TurnGate consults
both to decide when an utterance is ready to transcribe; a confirmed
transcript goes to the LLM, whose streamed reply is flushed sentence-by-
sentence to TTS and published back into the room as the agent's spoken
reply. If a new confirmed utterance arrives while a reply is still being
generated or spoken, that reply is cooperatively interrupted and its
audio queue cleared -- see
docs/superpowers/specs/2026-07-19-barge-in-design.md (including its
"Revision" section: interruption is a checked flag, not
asyncio.Task.cancel(), which corrupts LiveKit's AudioSource if it lands
mid-capture_frame()).
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents.vad import VADEventType

from agent.audio import to_16k_mono_f32
from agent.llm import LLMBackend, create_llm_backend
from agent.sentence_buffer import SentenceBuffer
from agent.smart_turn import SmartTurnObserver, create_smart_turn_scorer
from agent.stt import STTBackend, create_stt_backend
from agent.tts import TTSBackend, create_tts_backend
from agent.turn_gate import Continue, ForceFire, GateResult, TurnGate, create_turn_gate
from agent.vad import create_vad

PlayAudio = Callable[[np.ndarray], Awaitable[None]]
ClearAudio = Callable[[], None]


@dataclass
class HeardWord:
    """One word actually pushed to play_audio, with the absolute
    time.monotonic() timestamp its playback is modeled to end at -- see
    docs/superpowers/specs/2026-07-20-barge-in_heard_text.md."""

    text: str
    end: float


def heard_text(timeline: list[HeardWord], interrupted_at: float | None) -> str:
    """Words whose modeled playback ended before the barge-in cut time,
    joined back into text. Returns "" if nothing was heard yet (empty
    timeline, or the interrupt landed before any word finished, or no
    interrupt happened at all)."""
    if interrupted_at is None:
        return ""
    return "".join(w.text for w in timeline if w.end <= interrupted_at).strip()


@dataclass
class ActiveReply:
    """Tracks the currently in-flight _dispatch_gate_result task (if any)
    and whether it's been asked to stop -- see barge-in design spec.
    Interruption is cooperative: the task checks `interrupted` at natural
    checkpoints rather than being cancelled, since asyncio.Task.cancel()
    corrupts LiveKit's AudioSource if it lands mid-capture_frame().

    heard_timeline/playback_cursor/interrupted_at exist to reconstruct how
    much of an interrupted reply the user actually heard -- see
    docs/superpowers/specs/2026-07-20-barge-in_heard_text.md."""

    task: asyncio.Task | None = None
    interrupted: bool = False
    heard_timeline: list[HeardWord] = field(default_factory=list)
    playback_cursor: float = 0.0
    interrupted_at: float | None = None


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent-worker")
logger.setLevel(logging.INFO)

# File handler so subprocess logs (dev-mode PROCESS executor) are visible
# alongside the parent's. Guard against duplicate registration on re-import.
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    _fh = logging.FileHandler("/tmp/voice-agent-worker.log")
    _fh.setFormatter(
        logging.Formatter("%(asctime)s %(process)d %(levelname)s %(message)s")
    )
    logger.addHandler(_fh)


async def entrypoint(ctx: agents.JobContext) -> None:
    tts_backend = create_tts_backend()

    # Outbound track carries the synthesised TTS reply, at Kokoro's native rate.
    source = rtc.AudioSource(tts_backend.sample_rate, num_channels=1)
    out_track = rtc.LocalAudioTrack.create_audio_track("agent-reply", source)

    async def play_audio(audio: np.ndarray) -> None:
        if len(audio) == 0:
            return
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        await source.capture_frame(
            rtc.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=tts_backend.sample_rate,
                num_channels=1,
                samples_per_channel=len(pcm),
            )
        )

    def clear_audio() -> None:
        source.clear_queue()

    # Set up track queue *before* connect so no early track is missed.
    remote_audio: asyncio.Queue[tuple[rtc.Track, rtc.RemoteParticipant]] = (
        asyncio.Queue()
    )

    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        _pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            remote_audio.put_nowait((track, participant))

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    await ctx.room.local_participant.publish_track(out_track)
    logger.info("joined room=%s, published agent-reply track", ctx.room.name)

    remote_track, remote_p = await remote_audio.get()
    logger.info("listening to remote audio from %s", remote_p.identity)

    # Register the leave handler now — filtered by identity so stale disconnect
    # events (from a previous browser session in the same room) can't fire it.
    user_left = asyncio.Event()

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(p: rtc.RemoteParticipant) -> None:
        if p.identity == remote_p.identity:
            logger.info("participant %s left, ending session", p.identity)
            user_left.set()

    turn_observer = SmartTurnObserver(create_smart_turn_scorer())
    turn_observer.start()
    turn_gate = create_turn_gate()
    stt_backend = create_stt_backend()
    llm_backend = create_llm_backend()
    # Session-scoped, NOT a singleton like the backends above -- a fresh
    # list/lock/reply-tracker per room job, or state would leak across rooms.
    history: list[dict[str, str]] = []
    history_lock = asyncio.Lock()
    active_reply = ActiveReply()

    vad_stream = create_vad().stream()
    vad_task = asyncio.create_task(
        _consume_vad_events(
            vad_stream,
            turn_observer,
            turn_gate,
            stt_backend,
            llm_backend,
            tts_backend,
            history,
            history_lock,
            active_reply,
            play_audio,
            clear_audio,
        )
    )
    ingest_task = asyncio.create_task(
        _ingest(
            remote_track,
            vad_stream,
            turn_observer,
            turn_gate,
            stt_backend,
            llm_backend,
            tts_backend,
            history,
            history_lock,
            active_reply,
            play_audio,
            clear_audio,
        )
    )
    left_task = asyncio.create_task(user_left.wait())

    try:
        _, pending = await asyncio.wait(
            [ingest_task, left_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
    finally:
        turn_observer.stop()
        await vad_stream.aclose()
        vad_task.cancel()
        logger.info("entrypoint done")


async def _ingest(
    remote: rtc.RemoteAudioTrack,
    vad_stream,
    turn_observer: SmartTurnObserver,
    turn_gate: TurnGate,
    stt_backend: STTBackend,
    llm_backend: LLMBackend,
    tts_backend: TTSBackend,
    history: list[dict[str, str]],
    history_lock: asyncio.Lock,
    active_reply: ActiveReply,
    play_audio: PlayAudio,
    clear_audio: ClearAudio,
) -> None:
    """Read frames, normalise, and mirror into the VAD stream, the Smart
    Turn ring buffer, and the turn gate's utterance buffer."""
    stream = rtc.AudioStream(remote, sample_rate=48_000, num_channels=1)
    frame_count, last_log = 0, time.monotonic()

    async for event in stream:
        f = event.frame
        samples = np.frombuffer(f.data, dtype=np.int16)
        mono_16k = to_16k_mono_f32(samples, f.sample_rate, f.num_channels)
        turn_observer.push(mono_16k)

        if turn_gate.push(mono_16k):
            # Max utterance duration crossed — force-fire without waiting
            # for a Silero END_OF_SPEECH a long monologue might never produce.
            result = turn_gate.evaluate(turn_observer.latest_probability)
            asyncio.create_task(
                _dispatch_gate_result(
                    result,
                    stt_backend,
                    llm_backend,
                    tts_backend,
                    history,
                    history_lock,
                    active_reply,
                    play_audio,
                    clear_audio,
                )
            )

        # Float32 [-1, +1] → int16 so Silero gets its expected PCM frame.
        pcm = np.clip(mono_16k * 32767.0, -32768, 32767).astype(np.int16)
        vad_stream.push_frame(
            rtc.AudioFrame(
                data=pcm.tobytes(),
                sample_rate=16_000,
                num_channels=1,
                samples_per_channel=len(pcm),
            )
        )

        frame_count += 1
        now = time.monotonic()
        if now - last_log >= 2.0:
            logger.info("ingested %d frames in %.1fs", frame_count, now - last_log)
            frame_count, last_log = 0, now


async def _consume_vad_events(
    vad_stream,
    turn_observer: SmartTurnObserver,
    turn_gate: TurnGate,
    stt_backend: STTBackend,
    llm_backend: LLMBackend,
    tts_backend: TTSBackend,
    history: list[dict[str, str]],
    history_lock: asyncio.Lock,
    active_reply: ActiveReply,
    play_audio: PlayAudio,
    clear_audio: ClearAudio,
) -> None:
    """Log Silero VAD speech-boundary events, consulting the Smart Turn
    completion probability at END_OF_SPEECH to decide — via TurnGate —
    whether the turn is actually over. Also log the first VAD inference so
    we can confirm the task is alive and Silero is producing outputs."""
    logger.info("VAD event task started")
    first_inference_logged = False
    async for event in vad_stream:
        if event.type == VADEventType.START_OF_SPEECH:
            logger.info("SPEECH_START (prob=%.2f)", event.probability)
            turn_gate.begin()
        elif event.type == VADEventType.END_OF_SPEECH:
            prob = turn_observer.latest_probability
            logger.info(
                "SPEECH_END (duration=%.2fs) smart_turn_prob=%.2f",
                event.speech_duration,
                prob,
            )
            result = turn_gate.evaluate(prob)
            asyncio.create_task(
                _dispatch_gate_result(
                    result,
                    stt_backend,
                    llm_backend,
                    tts_backend,
                    history,
                    history_lock,
                    active_reply,
                    play_audio,
                    clear_audio,
                )
            )
        elif event.type == VADEventType.INFERENCE_DONE and not first_inference_logged:
            logger.info("VAD first inference: prob=%.2f", event.probability)
            first_inference_logged = True


async def _synthesize_and_play(
    tts_backend: TTSBackend,
    text: str,
    play_audio: PlayAudio,
    active_reply: ActiveReply,
) -> None:
    """Synthesize one sentence-buffer-flushed segment and play it. A failed
    segment is logged and skipped rather than aborting the turn -- the LLM
    reply still lands in history even if TTS drops a sentence.

    Synthesis runs in a thread executor and can't be cancelled mid-call, so
    a barge-in can flip active_reply.interrupted while this segment is
    already synthesizing. Re-checked here, right before playback, so that
    a now-stale segment is discarded instead of played -- checking only
    before/after the whole synthesize+play unit (as callers also do) still
    lets one full segment play after every barge-in.

    Records each word's modeled absolute end-time into
    active_reply.heard_timeline and advances playback_cursor -- see
    docs/superpowers/specs/2026-07-20-barge-in_heard_text.md. Segments
    queue behind each other (play_audio doesn't block for real-time
    playback), so a segment's start is modeled as max(now, cursor), not
    `now` directly."""
    try:
        result = await asyncio.to_thread(tts_backend.synthesize, text)
    except Exception:
        logger.exception("TTS synthesis failed for segment: %r", text)
        return
    if active_reply.interrupted:
        return
    logger.info(
        "TTS segment (%.2fs audio): %r",
        len(result.audio) / tts_backend.sample_rate,
        text,
    )
    seg_start = max(time.monotonic(), active_reply.playback_cursor)
    for word in result.words:
        active_reply.heard_timeline.append(HeardWord(word.text, seg_start + word.end))
    active_reply.playback_cursor = seg_start + len(result.audio) / tts_backend.sample_rate
    await play_audio(result.audio)


def _check_interrupted(active_reply: ActiveReply) -> bool:
    """Checked at each natural pause point in _dispatch_gate_result's LLM
    loop; logs once and reports whether this reply should stop now."""
    if active_reply.interrupted:
        logger.info("barge-in: reply interrupted mid-stream")
        return True
    return False


def _append_heard(active_reply: ActiveReply, history: list[dict[str, str]]) -> None:
    """Called wherever _dispatch_gate_result returns early on interrupt.
    Appends only what the user actually heard (per heard_text), not the
    full generated reply -- and appends nothing if nothing was heard yet
    (e.g. interrupted before the first segment finished synthesizing)."""
    text = heard_text(active_reply.heard_timeline, active_reply.interrupted_at)
    if text:
        logger.info("barge-in: heard before interrupt: %r", text)
        history.append({"role": "assistant", "content": text})


async def _dispatch_gate_result(
    result: GateResult,
    stt_backend: STTBackend,
    llm_backend: LLMBackend,
    tts_backend: TTSBackend,
    history: list[dict[str, str]],
    history_lock: asyncio.Lock,
    active_reply: ActiveReply,
    play_audio: PlayAudio,
    clear_audio: ClearAudio,
) -> None:
    """Handle a TurnGate decision: log-and-return on Continue, transcribe
    and log the transcript on Fire/ForceFire, then stream an LLM reply,
    flushing it sentence-by-sentence to TTS as it arrives.

    Barge-in: if a reply is still in flight (LLM streaming, TTS synthesis,
    or mid-playback) when a new confirmed Fire/ForceFire arrives, that
    prior task is cooperatively interrupted -- not cancelled -- and the
    audio queue cleared before this turn does anything else -- see
    docs/superpowers/specs/2026-07-19-barge-in-design.md's "Revision"
    section: asyncio.Task.cancel() corrupts LiveKit's AudioSource if it
    lands mid-capture_frame() (confirmed via live reproduction), so
    interruption is a checked flag instead. The interrupted task's user
    turn stays in history (it was really said); its assistant reply never
    gets appended since it returns as soon as it notices the flag, so the
    next request just sees two consecutive user turns, which chat-tuned
    models tolerate.

    history_lock is now a defensive invariant rather than the primary
    serialization mechanism: interrupting the predecessor and awaiting
    its exit guarantees the lock is free by the time this turn tries to
    acquire it.
    """
    if isinstance(result, Continue):
        logger.info("turn incomplete, continuing to listen")
        return

    current = asyncio.current_task()
    if active_reply.task is not None and not active_reply.task.done():
        active_reply.interrupted = True
        active_reply.interrupted_at = time.monotonic()
        clear_audio()
        logger.info("barge-in: interrupting in-flight reply")
        await active_reply.task
    active_reply.interrupted = False
    active_reply.task = current
    active_reply.heard_timeline = []
    active_reply.playback_cursor = 0.0
    active_reply.interrupted_at = None

    forced = isinstance(result, ForceFire)
    if forced:
        logger.warning("max utterance duration exceeded, forcing STT")
    try:
        text = await asyncio.to_thread(stt_backend.transcribe, result.audio)
    except Exception:
        logger.exception("STT transcription failed")
        return
    logger.info("TRANSCRIPT (%s): %r", "forced" if forced else "confirmed", text)

    async with history_lock:
        history.append({"role": "user", "content": text})
        sentence_buffer = SentenceBuffer()
        try:
            start = time.monotonic()
            first_chunk = True
            chunks: list[str] = []
            async for chunk in llm_backend.stream_chat(history):
                if _check_interrupted(active_reply):
                    _append_heard(active_reply, history)
                    return
                if first_chunk:
                    logger.info("LLM TTFT: %.3fs", time.monotonic() - start)
                    first_chunk = False
                chunks.append(chunk)
                for sentence in sentence_buffer.push(chunk):
                    await _synthesize_and_play(
                        tts_backend, sentence, play_audio, active_reply
                    )
                    if _check_interrupted(active_reply):
                        _append_heard(active_reply, history)
                        return
        except Exception:
            # The user turn stays in history unanswered; the next turn's
            # request will just include two consecutive user messages,
            # which chat-tuned models tolerate.
            logger.exception("LLM streaming failed")
            return
        remainder = sentence_buffer.flush()
        if remainder:
            await _synthesize_and_play(
                tts_backend, remainder, play_audio, active_reply
            )
            if _check_interrupted(active_reply):
                _append_heard(active_reply, history)
                return
        response = "".join(chunks)
        history.append({"role": "assistant", "content": response})
        logger.info("LLM RESPONSE: %r", response)


def _prewarm(proc: agents.JobProcess) -> None:
    """Runs in each subprocess before entrypoint so Silero, Smart Turn, STT,
    LLM, and TTS backends are already loaded/constructed.

    TTS also gets one dummy synthesis call here: MLX lazily compiles
    Kokoro's graph on the first real call (~2-3s cold vs. ~0.1-0.2s once
    warm, measured), and prewarm exists precisely so that cost lands here
    rather than mid-reply in a live conversation.
    """
    create_vad()
    create_smart_turn_scorer()
    create_stt_backend()
    create_llm_backend()
    create_tts_backend().synthesize("Ready.")
    logger.info(
        "Silero VAD + Smart Turn + STT + LLM + TTS preloaded (pid=%d)",
        proc.pid if hasattr(proc, "pid") else 0,
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=_prewarm)
    )
