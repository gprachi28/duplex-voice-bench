"""Unit tests for the /metrics aggregator -- reads the JSONL file agent/metrics.py
writes to (a separate process's output; the sidecar has no other access to it),
computes p50/p95 over a bounded in-memory window, and renders Prometheus
text exposition format. See design.md's "Observability" section.
"""

import json

from server.metrics import MetricsAggregator, STAGES


def _write_records(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(**overrides):
    base = {
        "turn_id": "t",
        "room": "r",
        "combination_id": "combo",
        "end_of_turn_s": 0.01,
        "transcription_s": 0.1,
        "llm_ttft_s": 0.2,
        "sentence_buffer_s": 0.05,
        "tts_first_chunk_s": 0.1,
        "ttfa_s": 0.46,
        "forced": False,
        "interrupted": False,
        "smart_turn_prob": 0.9,
    }
    base.update(overrides)
    return base


def test_refresh_reads_records_written_before_construction(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_records(path, [_record(ttfa_s=0.5), _record(ttfa_s=0.7)])

    agg = MetricsAggregator(str(path))
    agg.refresh()

    assert agg.percentile("ttfa_s", 0.5) in (0.5, 0.7)
    assert agg.turn_count() == 2


def test_refresh_only_reads_newly_appended_lines_on_subsequent_calls(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_records(path, [_record(ttfa_s=0.5)])

    agg = MetricsAggregator(str(path))
    agg.refresh()
    assert agg.turn_count() == 1

    _write_records(path, [_record(ttfa_s=0.9)])
    agg.refresh()
    assert agg.turn_count() == 2


def test_refresh_is_safe_when_file_does_not_exist_yet(tmp_path):
    path = tmp_path / "does-not-exist.jsonl"
    agg = MetricsAggregator(str(path))
    agg.refresh()  # must not raise
    assert agg.turn_count() == 0
    assert agg.percentile("ttfa_s", 0.5) is None


def test_null_stage_deltas_are_excluded_from_percentiles(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_records(
        path,
        [_record(transcription_s=None), _record(transcription_s=0.2)],
    )
    agg = MetricsAggregator(str(path))
    agg.refresh()
    assert agg.percentile("transcription_s", 0.5) == 0.2


def test_percentile_p50_and_p95_over_a_known_distribution(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_records(path, [_record(ttfa_s=float(i)) for i in range(1, 21)])  # 1..20

    agg = MetricsAggregator(str(path))
    agg.refresh()

    assert agg.percentile("ttfa_s", 0.5) == 10.0
    assert agg.percentile("ttfa_s", 0.95) == 19.0


def test_window_bounds_percentile_memory_to_most_recent_n_turns(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_records(path, [_record(ttfa_s=float(i)) for i in range(10)])  # 0..9

    agg = MetricsAggregator(str(path), window=3)
    agg.refresh()

    assert agg.turn_count() == 10  # all-time total, unbounded by the window
    assert agg.percentile("ttfa_s", 0.5) == 8.0  # only [7, 8, 9] retained


def test_render_prometheus_includes_help_type_and_stage_labels(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _write_records(path, [_record(combination_id="combo-a")])

    agg = MetricsAggregator(str(path))
    agg.refresh()
    text = agg.render_prometheus()

    assert "# TYPE voice_agent_stage_latency_seconds summary" in text
    assert 'voice_agent_stage_latency_seconds{stage="ttfa",quantile="0.5"}' in text
    assert 'voice_agent_turns_total{combination_id="combo-a"} 1' in text


def test_all_stages_are_covered():
    assert STAGES == [
        "end_of_turn_s",
        "transcription_s",
        "llm_ttft_s",
        "sentence_buffer_s",
        "tts_first_chunk_s",
        "ttfa_s",
    ]
