"""Reads the JSONL file agent/metrics.py writes to (the worker's job
subprocess and this FastAPI sidecar are separate OS processes with no
other shared state -- see design.md's "Observability" section) and
aggregates it into Prometheus summary exposition text.

"Aggregate" is an in-memory sliding window of the most recent `window`
turns, computed with exact nearest-rank percentiles -- no persistence
across sidecar restarts, no cross-instance aggregation. That's the right
size for a single-instance, one-active-room-at-a-time project.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict, deque

STAGES = [
    "end_of_turn_s",
    "transcription_s",
    "llm_ttft_s",
    "sentence_buffer_s",
    "tts_first_chunk_s",
    "ttfa_s",
]

# stage name -> Prometheus label value (design.md's per-stage metric names)
_STAGE_LABELS = {
    "end_of_turn_s": "end_of_turn",
    "transcription_s": "transcription",
    "llm_ttft_s": "llm_ttft",
    "sentence_buffer_s": "sentence_buffer",
    "tts_first_chunk_s": "tts_first_chunk",
    "ttfa_s": "ttfa",
}

DEFAULT_WINDOW = 500


class MetricsAggregator:
    def __init__(self, path: str, window: int = DEFAULT_WINDOW) -> None:
        self._path = path
        self._offset = 0
        self._stage_values: dict[str, deque[float]] = {
            stage: deque(maxlen=window) for stage in STAGES
        }
        self._combination_counts: dict[str, int] = defaultdict(int)
        self._turn_count = 0

    def refresh(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path) as f:
            f.seek(self._offset)
            new_lines = f.readlines()
            self._offset = f.tell()
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            self._turn_count += 1
            self._combination_counts[record.get("combination_id", "unknown")] += 1
            for stage in STAGES:
                value = record.get(stage)
                if value is not None:
                    self._stage_values[stage].append(value)

    def turn_count(self) -> int:
        return self._turn_count

    def percentile(self, stage: str, q: float) -> float | None:
        values = sorted(self._stage_values[stage])
        if not values:
            return None
        rank = max(1, math.ceil(q * len(values)))
        return values[rank - 1]

    def render_prometheus(self) -> str:
        lines = [
            "# HELP voice_agent_stage_latency_seconds Per-stage voice pipeline "
            "latency (sliding window, most recent turns).",
            "# TYPE voice_agent_stage_latency_seconds summary",
        ]
        for stage in STAGES:
            label = _STAGE_LABELS[stage]
            for q in (0.5, 0.95):
                value = self.percentile(stage, q)
                if value is not None:
                    lines.append(
                        f'voice_agent_stage_latency_seconds{{stage="{label}",'
                        f'quantile="{q}"}} {value}'
                    )
        lines.append("# HELP voice_agent_turns_total Turns recorded since the "
                      "metrics file started.")
        lines.append("# TYPE voice_agent_turns_total counter")
        for combination_id, count in self._combination_counts.items():
            lines.append(
                f'voice_agent_turns_total{{combination_id="{combination_id}"}} {count}'
            )
        return "\n".join(lines) + "\n"
