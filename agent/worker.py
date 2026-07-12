"""Echo-loop agent worker with Silero VAD observation.

Joins a LiveKit room dispatched to us, subscribes to the first remote audio
track, runs every incoming frame through the ingress format contract
(16 kHz mono float32), and republishes the same audio back into the room.
Silero VAD runs alongside and logs speech start/end events on the same
normalised buffer that the echo consumes.
"""

import asyncio
import logging
import time

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents.vad import VADEventType

from agent.audio import TARGET_SR, to_16k_mono_f32
from agent.vad import create_vad

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echo-worker")
logger.setLevel(logging.INFO)

# File handler so subprocess logs (dev-mode PROCESS executor) are visible
# alongside the parent's. Guard against duplicate registration on re-import.
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    _fh = logging.FileHandler("/tmp/echo-worker.log")
    _fh.setFormatter(
        logging.Formatter("%(asctime)s %(process)d %(levelname)s %(message)s")
    )
    logger.addHandler(_fh)


async def entrypoint(ctx: agents.JobContext) -> None:
    # Outbound track at the pipeline's canonical rate.
    source = rtc.AudioSource(TARGET_SR, num_channels=1)
    out_track = rtc.LocalAudioTrack.create_audio_track("echo-out", source)

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
    logger.info("joined room=%s, published echo track", ctx.room.name)

    remote_track, remote_p = await remote_audio.get()
    logger.info("echoing remote audio from %s", remote_p.identity)

    # Register the leave handler now — filtered by identity so stale disconnect
    # events (from a previous browser session in the same room) can't fire it.
    user_left = asyncio.Event()

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(p: rtc.RemoteParticipant) -> None:
        if p.identity == remote_p.identity:
            logger.info("participant %s left, ending session", p.identity)
            user_left.set()

    vad_stream = create_vad().stream()
    vad_task = asyncio.create_task(_consume_vad_events(vad_stream))
    echo_task = asyncio.create_task(_echo(remote_track, source, vad_stream))
    left_task = asyncio.create_task(user_left.wait())

    try:
        _, pending = await asyncio.wait(
            [echo_task, left_task], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
    finally:
        await vad_stream.aclose()
        vad_task.cancel()
        logger.info("entrypoint done")


async def _echo(
    remote: rtc.RemoteAudioTrack,
    source: rtc.AudioSource,
    vad_stream,
) -> None:
    """Read frames, normalise, republish, and mirror into the VAD stream."""
    stream = rtc.AudioStream(remote, sample_rate=48_000, num_channels=1)
    frame_count, last_log = 0, time.monotonic()

    async for event in stream:
        f = event.frame
        samples = np.frombuffer(f.data, dtype=np.int16)
        mono_16k = to_16k_mono_f32(samples, f.sample_rate, f.num_channels)

        # Float32 [-1, +1] → int16 for the outbound wire format.
        out = np.clip(mono_16k * 32767.0, -32768, 32767).astype(np.int16)
        out_frame = rtc.AudioFrame(
            data=out.tobytes(),
            sample_rate=TARGET_SR,
            num_channels=1,
            samples_per_channel=len(out),
        )
        await source.capture_frame(out_frame)
        vad_stream.push_frame(out_frame)

        frame_count += 1
        now = time.monotonic()
        if now - last_log >= 2.0:
            logger.info("echoed %d frames in %.1fs", frame_count, now - last_log)
            frame_count, last_log = 0, now


async def _consume_vad_events(vad_stream) -> None:
    """Log Silero VAD speech-boundary events. Also log the first inference so
    we can confirm the task is alive and Silero is producing outputs."""
    logger.info("VAD event task started")
    first_inference_logged = False
    async for event in vad_stream:
        if event.type == VADEventType.START_OF_SPEECH:
            logger.info("SPEECH_START (prob=%.2f)", event.probability)
        elif event.type == VADEventType.END_OF_SPEECH:
            logger.info("SPEECH_END (duration=%.2fs)", event.speech_duration)
        elif event.type == VADEventType.INFERENCE_DONE and not first_inference_logged:
            logger.info("VAD first inference: prob=%.2f", event.probability)
            first_inference_logged = True


def _prewarm(proc: agents.JobProcess) -> None:
    """Runs in each subprocess before entrypoint so Silero is already loaded."""
    create_vad()
    logger.info("Silero VAD preloaded (pid=%d)", proc.pid if hasattr(proc, "pid") else 0)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=_prewarm)
    )
