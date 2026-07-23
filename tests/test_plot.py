"""Unit tests for benchmark plotting -- see benchmarks/README.md.

Only the pure data-shaping helper is unit-tested in detail; matplotlib
rendering itself is covered by a smoke test that a PNG file is actually
produced (mirrors the level of testing agent/tts.py's synthesis output gets:
shape/non-emptiness, not pixel content).
"""

from benchmarks.plot import (
    plot_stage_breakdown,
    plot_ttfa_distribution,
    plot_ttfa_trend,
    stage_breakdown_series,
)


def _summary(**overrides):
    base = {
        "end_of_turn_s": {"p50": 0.01, "p95": 0.02, "p99": 0.03},
        "transcription_s": {"p50": 0.1, "p95": 0.2, "p99": 0.3},
        "llm_ttft_s": {"p50": 0.2, "p95": 0.4, "p99": 0.5},
        "sentence_buffer_s": {"p50": 0.05, "p95": 0.08, "p99": 0.1},
        "tts_first_chunk_s": {"p50": 0.1, "p95": 0.15, "p99": 0.2},
        "ttfa_s": {"p50": 0.46, "p95": 0.85, "p99": 1.1},
    }
    base.update(overrides)
    return base


def test_stage_breakdown_series_orders_stages_and_excludes_the_total():
    series = stage_breakdown_series(_summary(), "p50")

    labels = [label for label, _ in series]
    assert labels == [
        "End of turn",
        "Transcription",
        "LLM TTFT",
        "Sentence buffer",
        "TTS first chunk",
    ]
    assert "TTFA" not in labels


def test_stage_breakdown_series_reads_the_requested_quantile():
    series = stage_breakdown_series(_summary(), "p95")

    values = dict(series)
    assert values["Transcription"] == 0.2


def test_stage_breakdown_series_skips_stages_with_no_data():
    summary = _summary()
    summary["transcription_s"]["p50"] = None

    series = stage_breakdown_series(summary, "p50")

    assert "Transcription" not in dict(series)


def test_plot_stage_breakdown_writes_a_png(tmp_path):
    out_path = tmp_path / "stage_breakdown.png"

    plot_stage_breakdown({"combo-a": _summary()}, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_ttfa_distribution_writes_a_png(tmp_path):
    out_path = tmp_path / "ttfa_distribution.png"
    grouped = {"combo-a": [{"ttfa_s": float(i)} for i in range(1, 11)]}

    plot_ttfa_distribution(grouped, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_ttfa_trend_writes_a_png_when_two_tags_exist(tmp_path):
    out_path = tmp_path / "ttfa_trend.png"
    latency_summary = {
        "combo-a [baseline]": _summary(ttfa_s={"p50": 2.02, "p95": 3.42, "p99": 5.0}),
        "combo-a [fix1]": _summary(ttfa_s={"p50": 1.39, "p95": 2.30, "p99": 3.0}),
    }
    ordered_tags = {"combo-a": ["baseline", "fix1"]}

    plot_ttfa_trend(latency_summary, ordered_tags, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_ttfa_trend_skips_when_only_one_tag_exists(tmp_path, capsys):
    out_path = tmp_path / "ttfa_trend.png"
    latency_summary = {"combo-a [baseline]": _summary()}
    ordered_tags = {"combo-a": ["baseline"]}

    plot_ttfa_trend(latency_summary, ordered_tags, str(out_path))

    assert not out_path.exists()
    assert "skipped" in capsys.readouterr().err


def test_plot_ttfa_trend_skips_when_multiple_combinations_qualify(tmp_path, capsys):
    out_path = tmp_path / "ttfa_trend.png"
    latency_summary = {
        "combo-a [baseline]": _summary(),
        "combo-a [fix1]": _summary(),
        "combo-b [baseline]": _summary(),
        "combo-b [fix1]": _summary(),
    }
    ordered_tags = {"combo-a": ["baseline", "fix1"], "combo-b": ["baseline", "fix1"]}

    plot_ttfa_trend(latency_summary, ordered_tags, str(out_path))

    assert not out_path.exists()
    assert "skipped" in capsys.readouterr().err
