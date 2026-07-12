"""Echo-loop agent worker.

Joins a LiveKit room dispatched to us, subscribes to the first remote audio
track, runs every incoming frame through the ingress format contract
(16 kHz mono float32), and republishes the same audio back into the room.

This is the transport-trunk verification for the pipeline — no VAD, STT, LLM,
or TTS yet. Downstream stages will plug in on the normalised buffer.
"""

import asyncio
import logging
import time

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc

from agent.audio import TARGET_SR, to_16k_mono_f32

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("echo-worker")


async def entrypoint(ctx: agents.JobContext) -> None:
    # Outbound track at the pipeline's canonical rate.
    source = rtc.AudioSource(TARGET_SR, num_channels=1)
    out_track = rtc.LocalAudioTrack.create_audio_track("echo-out", source)

    # Set up the subscription queue *before* connect so no early track is missed.
    remote_audio: asyncio.Queue[rtc.RemoteAudioTrack] = asyncio.Queue()

    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        _pub: rtc.RemoteTrackPublication,
        _participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            remote_audio.put_nowait(track)

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    await ctx.room.local_participant.publish_track(out_track)
    logger.info("joined room=%s, published echo track", ctx.room.name)

    remote = await remote_audio.get()
    logger.info("echoing remote audio track")
    await _echo(remote, source)


async def _echo(remote: rtc.RemoteAudioTrack, source: rtc.AudioSource) -> None:
    """Read frames, normalise via the ingress contract, republish."""
    stream = rtc.AudioStream(remote, sample_rate=48_000, num_channels=1)
    frame_count, last_log = 0, time.monotonic()

    async for event in stream:
        f = event.frame
        samples = np.frombuffer(f.data, dtype=np.int16)
        mono_16k = to_16k_mono_f32(samples, f.sample_rate, f.num_channels)

        # Float32 [-1, +1] → int16 for the outbound wire format.
        out = np.clip(mono_16k * 32767.0, -32768, 32767).astype(np.int16)
        await source.capture_frame(
            rtc.AudioFrame(
                data=out.tobytes(),
                sample_rate=TARGET_SR,
                num_channels=1,
                samples_per_channel=len(out),
            )
        )

        frame_count += 1
        now = time.monotonic()
        if now - last_log >= 2.0:
            logger.info("echoed %d frames in %.1fs", frame_count, now - last_log)
            frame_count, last_log = 0, now


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
