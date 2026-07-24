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
import json
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents.vad import VADEventType

from agent.audio import TARGET_SR, to_16k_mono_f32
from agent.llm import LLMBackend, create_llm_backend
from agent.metrics import DEFAULT_METRICS_LOG_PATH, TurnMetrics, append_turn_metrics
from agent.playback import PlaybackPump, PlaybackState
from agent.sentence_buffer import SentenceBuffer
from agent.smart_turn import SmartTurnObserver, create_smart_turn_scorer
from agent.stt import WHISPER_MODEL, STTBackend, create_stt_backend, is_repetition_loop
from agent.tts import KOKORO_REPO, TTSBackend, create_tts_backend
from agent.turn_gate import Continue, ForceFire, GateResult, TurnGate, create_turn_gate
from agent.vad import create_vad

METRICS_LOG_PATH = os.environ.get("METRICS_LOG_PATH", DEFAULT_METRICS_LOG_PATH)

# Spoken when STT output is a detected repetition loop (see
# agent/stt.py's is_repetition_loop) -- design.md's "Error recovery"
# section already specifies this exact fallback for stage failures; this
# is the first path that actually speaks it rather than silently
# returning. The garbage transcript never reaches the LLM or history.
FALLBACK_REPLY = "Sorry, I didn't catch that."

# Bump whenever SYSTEM_PROMPT's text changes and add an entry to
# benchmarks/prompts.md -- this travels into every turn's metrics record
# (see TurnMetrics.prompt_version) so benchmark results stay traceable to
# the exact prompt text that produced them, not just the STT/LLM/TTS
# combination_id.
SYSTEM_PROMPT_VERSION = "v1-concise-en"

# Without this, llama3.2:3b defaults to long replies and, on at least one
# live turn, drifted into Hindi -- transcripts confirmed the LLM itself
# produced the Hindi text, and Kokoro (English-only TTS) rendered it as
# garbage audio. See benchmarks/experiments.md.
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are spoken aloud, so "
    "keep them short and conversational -- one to three sentences unless "
    "the user explicitly asks for more detail, a list, or a recipe/steps. "
    "Always reply in English, even if a question references another "
    "language or asks about earlier turns in the conversation."
)


def _new_history() -> list[dict[str, str]]:
    """A fresh per-room conversation history, seeded with SYSTEM_PROMPT."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


@dataclass
class HeardWord:
    """One word actually submitted for playback, with the absolute
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
    docs/superpowers/specs/2026-07-20-barge-in_heard_text.md.

    speech_started_at/escalation_handle exist so a barge-in doesn't have
    to wait for Smart Turn to confirm the interrupting utterance is a
    complete turn -- see _arm_escalation/_escalate_barge_in. paused_at is
    set for the same window: non-None while the reply is provisionally
    (reversibly) paused, pending escalation or a resume -- see
    PlaybackPump.pause/resume in agent/playback.py."""

    task: asyncio.Task | None = None
    interrupted: bool = False
    heard_timeline: list[HeardWord] = field(default_factory=list)
    playback_cursor: float = 0.0
    interrupted_at: float | None = None
    speech_started_at: float | None = None
    escalation_handle: asyncio.TimerHandle | None = None
    paused_at: float | None = None
    room: str = ""
    turn_count: int = 0


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

# This module runs in both the parent worker process and every job
# subprocess livekit-agents spawns, so each gets its own FileHandler above.
# livekit-agents *also* forwards every job subprocess log record to the
# parent over IPC (see ipc/log_queue.py) for console visibility -- and that
# forwarded record still carries the subprocess's original pid/timestamp, so
# without this it silently re-enters this same logger in the parent and
# gets written to the file a second time (confirmed live: every job-process
# log line appeared twice, byte-identical down to the pid). propagate=False
# only stops that redundant round-trip for *this* logger -- other loggers
# (e.g. livekit's own) still propagate normally for console output.
logger.propagate = False


async def _shutdown_active_reply(active_reply: ActiveReply, pump: PlaybackPump) -> None:
    """Called when a session is ending. An in-flight reply's LLM/TTS task
    isn't cancelled by anything else -- confirmed live, it kept
    generating and submitting TTS segments for several more seconds
    after the participant left, audible to no one. Interrupt it the same
    cooperative way as any other barge-in (asyncio.Task.cancel() corrupts
    LiveKit's AudioSource if it lands mid-capture_frame() -- see barge-in
    design spec) rather than cancelling it directly."""
    if active_reply.task is None or active_reply.task.done():
        return
    active_reply.interrupted = True
    pump.stop()
    try:
        await active_reply.task
    except Exception:
        logger.exception("in-flight reply task raised during shutdown")


async def entrypoint(ctx: agents.JobContext) -> None:
    tts_backend = create_tts_backend()

    # Outbound track carries the synthesised TTS reply, at Kokoro's native
    # rate. queue_size_ms is cut from the 1000ms default to 200ms: with the
    # default, PlaybackPump's real-time-paced drain loop can race up to a
    # full second ahead of actual playback before AudioSource.capture_frame
    # starts blocking for backpressure (confirmed live: pause()'s reported
    # position was word-count-plausible for "1s ahead" but not for real
    # time elapsed, undermining both the pause/resume rewind point and
    # _synthesize_and_play's heard_timeline wall-clock model, which both
    # assume submission timing tracks real playback closely).
    source = rtc.AudioSource(tts_backend.sample_rate, num_channels=1, queue_size_ms=200)
    out_track = rtc.LocalAudioTrack.create_audio_track("agent-reply", source)

    async def _capture_frame(audio: np.ndarray) -> None:
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

    # Submits TTS audio to `source` in small real-time-paced frames instead
    # of one call per segment, so a barge-in can pause mid-segment and
    # resume from a rewound word boundary instead of only ever hard-killing
    # playback -- see agent/playback.py.
    pump = PlaybackPump(_capture_frame, source.clear_queue, tts_backend.sample_rate)

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
    room_name = ctx.room.name
    logger.info("joined room=%s, published agent-reply track", room_name)

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
    history: list[dict[str, str]] = _new_history()
    history_lock = asyncio.Lock()
    active_reply = ActiveReply(room=room_name)

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
            pump,
            ctx.room.local_participant,
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
            pump,
            ctx.room.local_participant,
        )
    )
    left_task = asyncio.create_task(user_left.wait())

    # Backends preload before entrypoint runs (see _prewarm), but VAD/
    # ingest only start consuming audio frames from here -- anything
    # spoken before this line won't be picked up. Logged as an explicit
    # signal (not just implied by "listening to remote audio from ..."
    # above, which fires once the track is subscribed but before these
    # tasks exist) since a user speaking during that gap loses their
    # first few words with no other indication why.
    logger.info("ready — you can start talking now")

    try:
        _, pending = await asyncio.wait(
            [ingest_task, left_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
    finally:
        await _shutdown_active_reply(active_reply, pump)
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
    pump: PlaybackPump,
    local_participant: rtc.LocalParticipant,
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
            # Always a ForceFire (never Continue), so the "listening"
            # indicator always clears here -- see _consume_vad_events'
            # docstring on why turn_gate.is_open (not a local flag) is the
            # single source of truth shared across this and that function.
            prob = turn_observer.latest_probability
            result = turn_gate.evaluate(prob)
            asyncio.create_task(_publish_turn_state(local_participant, "idle"))
            asyncio.create_task(
                _dispatch_gate_result(
                    result,
                    stt_backend,
                    llm_backend,
                    tts_backend,
                    history,
                    history_lock,
                    active_reply,
                    pump,
                    t0=time.monotonic(),
                    smart_turn_prob=prob,
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


async def _publish_turn_state(local_participant: rtc.LocalParticipant, state: str) -> None:
    """Tells the client's visual indicator whether a turn is open -- see
    design.md's Known Issues: "No feedback during a long pause" (a real
    hesitation and a stuck turn were indistinguishable to the user, who
    repeated themselves live). Callers should wrap this in
    asyncio.create_task rather than awaiting inline, so a slow/stalled
    data-channel publish can never delay turn dispatch."""
    await local_participant.publish_data(
        json.dumps({"state": state}), reliable=True, topic="turn_state"
    )


def _speech_duration_since(started_at: float | None, now: float) -> float:
    """Wall-clock seconds since START_OF_SPEECH, used for the SPEECH_END log
    line instead of the VAD event's own speech_duration field -- confirmed
    that field is hardcoded to 0.0 on every END_OF_SPEECH event by the
    installed livekit-plugins-silero (it zeroes pub_speech_duration right
    before constructing the event), which made a real incident (a turn
    stuck fragmented across a long gap) much harder to diagnose from logs
    than it should have been. See benchmarks/experiments.md."""
    return now - started_at if started_at is not None else 0.0


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
    pump: PlaybackPump,
    local_participant: rtc.LocalParticipant,
) -> None:
    """Log Silero VAD speech-boundary events, consulting the Smart Turn
    completion probability at END_OF_SPEECH to decide — via TurnGate —
    whether the turn is actually over. Also log the first VAD inference so
    we can confirm the task is alive and Silero is producing outputs.

    SPEECH_START additionally arms a barge-in escalation timer and
    provisionally pauses playback (see _arm_escalation) so an interrupting
    utterance doesn't have to wait for Smart Turn to confirm it's a
    complete turn before the agent goes quiet; SPEECH_END disarms it --
    resuming (with a rewind) if speech turned out to be brief, since Smart
    Turn's own Fire/Continue evaluation below can now run normally.

    Also publishes a "listening"/"idle" turn-state to the client over the
    data channel, driven by turn_gate.is_open (the single source of truth
    -- _ingest's own sample-count ForceFire path can also resolve the same
    turn, so this can't be a local flag duplicated in both places): "on"
    the moment a turn opens, "off" once it resolves via Fire/ForceFire.
    This deliberately spans the silent gaps between speech bursts within
    one open turn, since that's exactly where a real pause and a stuck
    turn looked identical to the user (see design.md's Known Issues)."""
    logger.info("VAD event task started")
    first_inference_logged = False
    speech_started_at: float | None = None
    async for event in vad_stream:
        if event.type == VADEventType.START_OF_SPEECH:
            speech_started_at = time.monotonic()
            logger.info("SPEECH_START (prob=%.2f)", event.probability)
            was_open = turn_gate.is_open
            turn_gate.begin()
            if not was_open:
                asyncio.create_task(_publish_turn_state(local_participant, "listening"))
            _arm_escalation(active_reply, pump)
        elif event.type == VADEventType.END_OF_SPEECH:
            _disarm_escalation(active_reply, pump)
            prob = turn_observer.latest_probability
            t0 = time.monotonic()
            logger.info(
                "SPEECH_END (duration=%.2fs) smart_turn_prob=%.2f",
                _speech_duration_since(speech_started_at, t0),
                prob,
            )
            result = turn_gate.evaluate(prob)
            if not isinstance(result, Continue):
                asyncio.create_task(_publish_turn_state(local_participant, "idle"))
            asyncio.create_task(
                _dispatch_gate_result(
                    result,
                    stt_backend,
                    llm_backend,
                    tts_backend,
                    history,
                    history_lock,
                    active_reply,
                    pump,
                    t0=t0,
                    smart_turn_prob=prob,
                )
            )
        elif event.type == VADEventType.INFERENCE_DONE and not first_inference_logged:
            logger.info("VAD first inference: prob=%.2f", event.probability)
            first_inference_logged = True


SYNTH_PAUSE_POLL_S = 0.02


async def _wait_while_paused(pump: PlaybackPump, active_reply: ActiveReply) -> None:
    """Blocks until the pump stops being provisionally paused or the
    reply is interrupted, whichever comes first."""
    while pump.state == PlaybackState.PAUSED and not active_reply.interrupted:
        await asyncio.sleep(SYNTH_PAUSE_POLL_S)


async def _synthesize_and_play(
    tts_backend: TTSBackend,
    text: str,
    pump: PlaybackPump,
    active_reply: ActiveReply,
    metrics: TurnMetrics | None = None,
) -> None:
    """Synthesize one sentence-buffer-flushed segment and submit it to the
    playback pump. A failed segment is logged and skipped rather than
    aborting the turn -- the LLM reply still lands in history even if TTS
    drops a sentence.

    Synthesis runs in a thread executor and can't be cancelled mid-call, so
    a barge-in can flip active_reply.interrupted while this segment is
    already synthesizing. Re-checked here, right before submission, so
    that a now-stale segment is discarded instead of played -- checking
    only before/after the whole synthesize+submit unit (as callers also
    do) still lets one full segment play after every barge-in.

    Also records each word's modeled absolute end-time into
    active_reply.heard_timeline and advances playback_cursor -- see
    docs/superpowers/specs/2026-07-20-barge-in_heard_text.md. This is a
    separate, wall-clock-estimated model from the pump's own
    submitted-sample-count clock (used for pause/resume rewind); both are
    built from the same result.words in the same order, so they stay
    index-aligned -- see _disarm_escalation's heard_timeline truncation.

    While the pump is provisionally paused (see _arm_escalation), waits
    here before calling the expensive tts_backend.synthesize() at all --
    confirmed live, without this a sentence generated during the pause
    window got fully synthesized (real Kokoro/MLX compute) and buffered
    with zero chance of being heard if the pause escalated into a real
    interrupt seconds later, thrown away by pump.stop(). Synthesis can't
    be cancelled or checked mid-call, so a pause can also land *while*
    it's already in flight (confirmed live: a segment logged 65ms after
    "pausing playback provisionally") -- waited for again after
    synthesis completes, before logging/submitting, to close that race.

    metrics is optional so existing callers/tests that don't instrument a
    turn keep working unchanged. When given, stamps t4 (first sentence
    handed to TTS) and t5 (first audio actually submitted) only once per
    turn -- a turn spans multiple segments/calls, and only the first
    matters for TTFA."""
    if metrics is not None and metrics.t4 is None:
        metrics.t4 = time.monotonic()
    await _wait_while_paused(pump, active_reply)
    if active_reply.interrupted:
        return
    try:
        result = await asyncio.to_thread(tts_backend.synthesize, text)
    except Exception:
        logger.exception("TTS synthesis failed for segment: %r", text)
        return
    if active_reply.interrupted:
        return
    await _wait_while_paused(pump, active_reply)
    if active_reply.interrupted:
        return
    logger.info(
        "TTS segment (%.2fs audio): %r",
        len(result.audio) / tts_backend.sample_rate,
        text,
    )
    seg_start = max(time.monotonic(), active_reply.playback_cursor)
    for i, word in enumerate(result.words):
        word_text = word.text
        if (
            i == 0
            and active_reply.heard_timeline
            and not active_reply.heard_timeline[-1].text.endswith((" ", "\n"))
        ):
            # Kokoro gives each word its own trailing whitespace, but
            # nothing separates one segment's last word from the next
            # segment's first -- without this, back-to-back segments
            # read "Hello!Hello!" once joined in heard_text().
            word_text = " " + word_text
        active_reply.heard_timeline.append(HeardWord(word_text, seg_start + word.end))
    active_reply.playback_cursor = seg_start + len(result.audio) / tts_backend.sample_rate
    if metrics is not None and metrics.t5 is None:
        metrics.t5 = time.monotonic()
    await pump.submit(result.audio, result.words)


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


# Barge-in previously only fired on a confirmed Fire/ForceFire
# (smart_turn_prob >= TurnGate's threshold) -- a short interjection Smart
# Turn scored low fell into the Continue branch below, which returns
# immediately without ever touching active_reply. Confirmed live: a real
# SPEECH_END with smart_turn_prob=0.02 landed while a reply was actively
# playing and didn't interrupt it. PROVISIONAL_MUTE_S decouples barge-in
# responsiveness from Smart Turn's turn-completion question: VAD SPEECH_
# START immediately pauses playback (reversible) and arms a wall-clock
# timer; if speech is sustained past this long the pause escalates into a
# real interrupt, independent of waiting for SPEECH_END -- see
# docs/superpowers/specs/2026-07-20-barge-in_heard_text.md for why
# interrupted_at is stamped at speech start, not escalation time.
#
# Must exceed agent/vad.py's MIN_SILENCE_DURATION (0.3s): SPEECH_END can't
# arrive before (sound duration + MIN_SILENCE_DURATION), so a shorter
# value than that makes escalation always win the race regardless of how
# brief the speech was -- confirmed live: at 0.25s, "escalating" fired on
# every single real interruption and "resuming" (the false-positive path)
# never fired once. Muting itself is unaffected by this value -- pump.
# pause() always fires immediately on SPEECH_START -- only how long we
# wait before committing to kill the LLM generation is delayed.
PROVISIONAL_MUTE_S = 0.7


def _escalate_barge_in(active_reply: ActiveReply, pump: PlaybackPump) -> None:
    """Timer callback: speech has been sustained past PROVISIONAL_MUTE_S,
    so commit to interrupting the in-flight reply -- hard-stopping the
    pump (there's nothing to resume to) -- rather than wait for Smart
    Turn to eventually confirm the interrupting utterance is a complete
    turn. A no-op if the reply already finished or was already
    interrupted by the time the timer fires."""
    if active_reply.task is None or active_reply.task.done():
        return
    if active_reply.interrupted:
        return
    active_reply.interrupted = True
    active_reply.interrupted_at = active_reply.speech_started_at
    pump.stop()
    active_reply.escalation_handle = None
    active_reply.paused_at = None
    logger.info("barge-in: escalating after sustained speech")


def _arm_escalation(
    active_reply: ActiveReply, pump: PlaybackPump, delay_s: float = PROVISIONAL_MUTE_S
) -> None:
    """Called on VAD SPEECH_START. If a reply is actively in flight and
    not already interrupted, immediately (reversibly) pauses it and
    starts the escalation timer instead of waiting for Smart Turn to
    eventually decide the interrupting utterance is a complete turn."""
    if active_reply.task is None or active_reply.task.done():
        return
    if active_reply.interrupted:
        return
    active_reply.speech_started_at = time.monotonic()
    active_reply.paused_at = pump.pause()
    logger.info("barge-in: pausing playback provisionally (speech detected)")
    loop = asyncio.get_running_loop()
    active_reply.escalation_handle = loop.call_later(
        delay_s, _escalate_barge_in, active_reply, pump
    )


def _disarm_escalation(active_reply: ActiveReply, pump: PlaybackPump) -> None:
    """Called on VAD SPEECH_END. Cancels a pending escalation timer -- the
    speech that armed it turned out to be short enough that Smart Turn's
    own Fire/Continue evaluation can handle it normally. If the reply was
    provisionally paused (and not since escalated -- a non-None handle
    here means the timer hasn't fired), resumes it with a rewind and
    truncates heard_timeline to match, since it's built index-aligned
    with the pump's own word list (see _synthesize_and_play)."""
    if active_reply.escalation_handle is None:
        return
    active_reply.escalation_handle.cancel()
    active_reply.escalation_handle = None
    if active_reply.paused_at is not None:
        keep_count = pump.resume()
        logger.info(
            "barge-in: resuming playback (false alarm, rewound to word %d)", keep_count
        )
        active_reply.heard_timeline = active_reply.heard_timeline[:keep_count]
        active_reply.paused_at = None


async def _dispatch_gate_result(
    result: GateResult,
    stt_backend: STTBackend,
    llm_backend: LLMBackend,
    tts_backend: TTSBackend,
    history: list[dict[str, str]],
    history_lock: asyncio.Lock,
    active_reply: ActiveReply,
    pump: PlaybackPump,
    t0: float | None = None,
    smart_turn_prob: float | None = None,
) -> None:
    """Handle a TurnGate decision: log-and-return on Continue, transcribe
    and log the transcript on Fire/ForceFire, then stream an LLM reply,
    flushing it sentence-by-sentence to TTS as it arrives.

    Barge-in: if a reply is still in flight (LLM streaming, TTS synthesis,
    or mid-playback) when a new confirmed Fire/ForceFire arrives, that
    prior task is cooperatively interrupted -- not cancelled -- and the
    pump hard-stopped before this turn does anything else -- see
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

    t0/smart_turn_prob default to None so existing callers/tests that
    don't instrument a turn keep working unchanged -- see agent/metrics.py.
    A metrics record is only written when t0 is given (real dispatch sites
    always give one); the finally block emits it on every exit path
    (success, STT/LLM failure, or barge-in), with un-reached stages left
    null, so a JSONL line always exists for every Fire/ForceFire turn.
    """
    if isinstance(result, Continue):
        logger.info("turn incomplete, continuing to listen")
        return

    current = asyncio.current_task()
    if active_reply.task is not None and not active_reply.task.done():
        if not active_reply.interrupted:
            # Not already interrupted (e.g. by _escalate_barge_in, whose
            # interrupted_at is stamped at the moment speech actually
            # started -- more accurate than "now"). Don't clobber that
            # with a later timestamp, which would credit heard_text with
            # words spoken after the user had already started talking.
            active_reply.interrupted = True
            active_reply.interrupted_at = time.monotonic()
            pump.stop()
            logger.info("barge-in: interrupting in-flight reply")
        await active_reply.task
    active_reply.interrupted = False
    active_reply.task = current
    active_reply.heard_timeline = []
    active_reply.playback_cursor = 0.0
    active_reply.interrupted_at = None
    active_reply.speech_started_at = None
    active_reply.paused_at = None
    _disarm_escalation(active_reply, pump)
    pump.reset_for_new_reply()

    active_reply.turn_count += 1
    forced = isinstance(result, ForceFire)
    metrics = TurnMetrics(
        turn_id=f"{active_reply.room}-{active_reply.turn_count}",
        room=active_reply.room,
        combination_id=f"{WHISPER_MODEL}|{llm_backend.model}|{KOKORO_REPO}",
        t0=t0,
        forced=forced,
        smart_turn_prob=smart_turn_prob,
        prompt_version=SYSTEM_PROMPT_VERSION,
    )
    try:
        if forced:
            logger.warning("max utterance duration exceeded, forcing STT")
        try:
            metrics.t1 = time.monotonic()
            text = await asyncio.to_thread(stt_backend.transcribe, result.audio)
        except Exception:
            logger.exception("STT transcription failed")
            return
        metrics.t2 = time.monotonic()
        logger.info("TRANSCRIPT (%s): %r", "forced" if forced else "confirmed", text)

        if is_repetition_loop(text):
            metrics.stt_repetition_detected = True
            logger.warning(
                "STT repetition-loop detected (turn_id=%s): %r -- skipping LLM, "
                "speaking fallback reply",
                metrics.turn_id,
                text,
            )
            await _synthesize_and_play(tts_backend, FALLBACK_REPLY, pump, active_reply, metrics)
            return

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
                        metrics.t3 = time.monotonic()
                        logger.info("LLM TTFT: %.3fs", metrics.t3 - start)
                        first_chunk = False
                    chunks.append(chunk)
                    for sentence in sentence_buffer.push(chunk):
                        await _synthesize_and_play(
                            tts_backend, sentence, pump, active_reply, metrics
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
                    tts_backend, remainder, pump, active_reply, metrics
                )
                if _check_interrupted(active_reply):
                    _append_heard(active_reply, history)
                    return
        response = "".join(chunks)
        history.append({"role": "assistant", "content": response})
        logger.info("LLM RESPONSE: %r", response)
    finally:
        metrics.interrupted = active_reply.interrupted
        if t0 is not None:
            append_turn_metrics(METRICS_LOG_PATH, metrics)


async def _warm_llm(llm_backend: LLMBackend) -> None:
    """One throwaway round-trip so Ollama loads the model into memory here
    instead of on the first live turn -- see _prewarm's docstring."""
    async for _ in llm_backend.stream_chat([{"role": "user", "content": "Hi"}]):
        pass


def _prewarm(proc: agents.JobProcess) -> None:
    """Runs in each subprocess before entrypoint so Silero, Smart Turn, STT,
    LLM, and TTS backends are already loaded/constructed.

    STT and TTS each get one dummy inference call here, and the LLM one
    dummy round-trip: mlx-whisper doesn't load/compile its weights until
    the first real transcribe() call, Ollama doesn't load its model into
    memory until the first real /api/chat, and MLX lazily compiles
    Kokoro's graph on its first real call (~2-3s cold vs. ~0.1-0.2s once
    warm, measured). Without this, the first real utterance of a session
    pays all of that cost at once -- confirmed live as the direct cause of
    a too-short first utterance (see turn_gate.py's MIN_UTTERANCE_S) also
    landing on a cold STT call and hallucinating (benchmarks/experiments.md).
    Measured: _prewarm now takes ~6.8s total (was ~0s for STT/LLM before
    this fix); a real STT/LLM call immediately after is back to normal
    steady-state latency (~1.3s / ~0.2s) instead of paying a multi-second
    model-load spike on the first live turn.

    The LLM warmup is wrapped in try/except: unlike STT/TTS (purely local,
    already-cached weights), it depends on Ollama actually running, which
    README.md documents as best-effort -- the worker must still start
    (and just pay the cold-load cost on the first real turn) if it isn't.
    """
    create_vad()
    create_smart_turn_scorer()
    create_stt_backend().transcribe(np.zeros(TARGET_SR, dtype=np.float32))
    llm_backend = create_llm_backend()
    try:
        asyncio.run(_warm_llm(llm_backend))
    except Exception:
        logger.warning("LLM warmup failed (is Ollama running?) -- continuing")
    create_tts_backend().synthesize("Ready.")
    logger.info(
        "Silero VAD + Smart Turn + STT + LLM + TTS preloaded (pid=%d)",
        proc.pid if hasattr(proc, "pid") else 0,
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=_prewarm,
            # Default is 10.0s. _prewarm now does a real STT transcribe, an
            # LLM round-trip, and Kokoro's graph compile, back to back --
            # confirmed live: livekit-agents killed the process with
            # "initialization timed out" before prewarm could finish,
            # silently (no worker ever registers, nothing ever logs).
            # See benchmarks/experiments.md.
            initialize_process_timeout=60.0,
        )
    )
