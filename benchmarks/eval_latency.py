"""Offline latency analysis over benchmark run JSONL snapshots.

Reads the per-turn records agent/metrics.py writes (one JSON object per
line), groups them by combination_id, and computes p50/p95/p99 per stage --
see design.md: "Do not rely on averages ... compute p50, p95, and p99
percentiles for every single delta." Unlike server/metrics.py's live sliding
window (bounded, single-combination-agnostic), this reads the full history
of whatever run files are on disk and breaks results out per combination.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

STAGES = [
    "end_of_turn_s",
    "transcription_s",
    "llm_ttft_s",
    "sentence_buffer_s",
    "tts_first_chunk_s",
    "ttfa_s",
]


def load_runs(runs_dir: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def group_by_combination(records: list[dict]) -> dict[str, list[dict]]:
    """Groups by (combination_id, prompt_version) -- a prompt change is as
    much a change in what's being measured as a model swap is, so pooling
    two prompt versions under one combination_id would blur the very
    difference this is meant to surface. Older records with no
    prompt_version key are grouped under the bare combination_id."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        combo = record.get("combination_id", "unknown")
        prompt_version = record.get("prompt_version")
        key = f"{combo} [prompt={prompt_version}]" if prompt_version else combo
        grouped[key].append(record)
    return grouped


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = max(1, math.ceil(q * len(values)))
    return values[rank - 1]


def summarize(records: list[dict]) -> dict:
    summary: dict = {"n": len(records)}
    for stage in STAGES:
        values = [r[stage] for r in records if r.get(stage) is not None]
        summary[stage] = {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }
    return summary


def analyze(runs_dir: str) -> dict[str, dict]:
    grouped = group_by_combination(load_runs(runs_dir))
    return {combo: summarize(records) for combo, records in grouped.items()}


if __name__ == "__main__":
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/runs"
    result = analyze(runs_dir)
    out_path = "benchmarks/results/latency_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)
