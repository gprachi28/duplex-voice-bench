"""Per-turn latency metrics -- see design.md's "Observability" section.

TurnMetrics holds raw time.monotonic() timestamps for each pipeline stage
boundary (set by the caller in worker.py as the turn progresses) and turns
them into named deltas on demand. append_turn_metrics writes one JSON line
per turn to a shared file -- the FastAPI sidecar runs in a separate OS
process with no other access to this data, so the file is the IPC.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

# Shared with server/main.py (via METRICS_LOG_PATH) so both processes agree
# on where the file-based IPC lives without duplicating the literal path.
DEFAULT_METRICS_LOG_PATH = "/tmp/voice-agent-metrics.jsonl"


@dataclass
class TurnMetrics:
    turn_id: str
    room: str
    combination_id: str
    t0: float | None = None  # end of speech (Silero SPEECH_END)
    t1: float | None = None  # Smart Turn confirmed / gate fired
    t2: float | None = None  # STT transcription complete
    t3: float | None = None  # LLM first token
    t4: float | None = None  # first sentence flushed to TTS
    t5: float | None = None  # first TTS audio submitted to the pump
    forced: bool = False
    interrupted: bool = False
    smart_turn_prob: float | None = None

    def to_record(self) -> dict:
        return {
            "ts": time.time(),
            "turn_id": self.turn_id,
            "room": self.room,
            "combination_id": self.combination_id,
            "end_of_turn_s": _delta(self.t0, self.t1),
            "transcription_s": _delta(self.t1, self.t2),
            "llm_ttft_s": _delta(self.t2, self.t3),
            "sentence_buffer_s": _delta(self.t3, self.t4),
            "tts_first_chunk_s": _delta(self.t4, self.t5),
            "ttfa_s": _delta(self.t0, self.t5),
            "forced": self.forced,
            "interrupted": self.interrupted,
            "smart_turn_prob": self.smart_turn_prob,
        }


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def append_turn_metrics(path: str, metrics: TurnMetrics) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(metrics.to_record()) + "\n")
