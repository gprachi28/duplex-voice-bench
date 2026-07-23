"""Unit tests for the offline benchmark latency analyzer -- reads run JSONL
snapshots under benchmarks/results/runs/, groups by combination_id, and
computes p50/p95/p99 per stage. See benchmarks/README.md.
"""

import json

import pytest

from benchmarks.eval_latency import (
    STAGES,
    combo_base_key,
    combo_tag_key,
    group_by_combination_and_tag,
    load_runs,
    load_runs_with_tags,
    parse_change_tag,
    percentile,
    summarize,
)


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(**overrides):
    base = {
        "turn_id": "t",
        "combination_id": "combo-a",
        "prompt_version": "v1-concise-en",
        "end_of_turn_s": 0.01,
        "transcription_s": 0.1,
        "llm_ttft_s": 0.2,
        "sentence_buffer_s": 0.05,
        "tts_first_chunk_s": 0.1,
        "ttfa_s": 0.46,
    }
    base.update(overrides)
    return base


def test_load_runs_reads_and_concatenates_every_jsonl_file_in_a_directory(tmp_path):
    _write_jsonl(tmp_path / "run1.jsonl", [_record(turn_id="a")])
    _write_jsonl(tmp_path / "run2.jsonl", [_record(turn_id="b"), _record(turn_id="c")])

    records = load_runs(str(tmp_path))

    assert {r["turn_id"] for r in records} == {"a", "b", "c"}


def test_load_runs_is_safe_on_an_empty_directory(tmp_path):
    assert load_runs(str(tmp_path)) == []


def test_group_by_combination_and_tag_splits_records_by_combination_id():
    tagged_records = [
        ("t1", _record(combination_id="combo-a")),
        ("t1", _record(combination_id="combo-b")),
        ("t1", _record(combination_id="combo-a")),
    ]

    grouped = group_by_combination_and_tag(tagged_records)

    assert len(grouped) == 2
    sizes = sorted(len(v) for v in grouped.values())
    assert sizes == [1, 2]


def test_group_by_combination_and_tag_splits_the_same_combination_id_by_prompt_version():
    # Same STT/LLM/TTS combination, different SYSTEM_PROMPT -- pooling these
    # would blur a prompt change's effect into the combination's numbers.
    tagged_records = [
        ("t1", _record(combination_id="combo-a", prompt_version="v1-concise-en")),
        ("t1", _record(combination_id="combo-a", prompt_version="v2-shorter")),
        ("t1", _record(combination_id="combo-a", prompt_version="v1-concise-en")),
    ]

    grouped = group_by_combination_and_tag(tagged_records)

    assert len(grouped) == 2
    assert sum(len(v) for v in grouped.values()) == 3


def test_group_by_combination_and_tag_tolerates_records_with_no_prompt_version():
    # Older run files (recorded before prompt_version existed) shouldn't
    # crash the analyzer.
    record = _record(combination_id="combo-a")
    del record["prompt_version"]

    grouped = group_by_combination_and_tag([("t1", record)])

    assert grouped["combo-a [t1]"] == [record]


def test_group_by_combination_and_tag_splits_the_same_combination_by_change_tag():
    # The exact scenario this feature fixes: two runs sharing
    # combination_id and prompt_version but captured against different
    # code states must not be pooled.
    tagged_records = [
        ("baseline", _record(combination_id="combo-a")),
        ("stt-lang-temp", _record(combination_id="combo-a")),
        ("baseline", _record(combination_id="combo-a")),
    ]

    grouped = group_by_combination_and_tag(tagged_records)

    assert len(grouped) == 2
    sizes = sorted(len(v) for v in grouped.values())
    assert sizes == [1, 2]


def test_percentile_p50_and_p95_over_a_known_distribution():
    values = [float(i) for i in range(1, 21)]  # 1..20

    assert percentile(values, 0.50) == 10.0
    assert percentile(values, 0.95) == 19.0


def test_percentile_returns_none_for_empty_values():
    assert percentile([], 0.50) is None


def test_summarize_reports_turn_count_and_per_stage_percentiles():
    records = [_record(ttfa_s=float(i)) for i in range(1, 21)]  # 1..20

    summary = summarize(records)

    assert summary["n"] == 20
    assert summary["ttfa_s"]["p50"] == 10.0
    assert summary["ttfa_s"]["p95"] == 19.0
    assert summary["ttfa_s"]["p99"] == 20.0


def test_summarize_excludes_null_stage_values_from_percentiles():
    records = [_record(transcription_s=None), _record(transcription_s=0.2)]

    summary = summarize(records)

    assert summary["transcription_s"]["p50"] == 0.2


def test_summarize_covers_every_stage():
    summary = summarize([_record()])

    assert set(STAGES) <= summary.keys()


def test_parse_change_tag_returns_segment_after_double_underscore():
    path = "benchmarks/results/runs/20260722_223140_local-lv3-ollama3b-kokoro__stt-lang-temp.jsonl"

    assert parse_change_tag(path) == "stt-lang-temp"


def test_parse_change_tag_works_on_a_bare_filename_without_directory():
    assert parse_change_tag("20260722_100000_combo__baseline.jsonl") == "baseline"


def test_parse_change_tag_raises_when_no_double_underscore():
    path = "benchmarks/results/runs/20260722_223140_local-lv3-ollama3b-kokoro.jsonl"

    with pytest.raises(ValueError, match="20260722_223140_local-lv3-ollama3b-kokoro.jsonl"):
        parse_change_tag(path)


def test_combo_base_key_includes_prompt_version_when_present():
    assert combo_base_key("combo-a", "v1-concise-en") == "combo-a [prompt=v1-concise-en]"


def test_combo_base_key_omits_prompt_bracket_when_none():
    assert combo_base_key("combo-a", None) == "combo-a"


def test_combo_tag_key_appends_bracketed_tag():
    assert combo_tag_key("combo-a", "baseline") == "combo-a [baseline]"


def test_load_runs_with_tags_pairs_each_record_with_its_file_tag(tmp_path):
    _write_jsonl(tmp_path / "20260722_100000_combo__baseline.jsonl", [_record(turn_id="a")])
    _write_jsonl(tmp_path / "20260722_110000_combo__fix1.jsonl", [_record(turn_id="b")])

    tagged = load_runs_with_tags(str(tmp_path))

    tags_by_turn = {record["turn_id"]: tag for tag, record in tagged}
    assert tags_by_turn == {"a": "baseline", "b": "fix1"}


def test_load_runs_with_tags_is_safe_on_an_empty_directory(tmp_path):
    assert load_runs_with_tags(str(tmp_path)) == []


def test_load_runs_with_tags_raises_on_untagged_file_and_names_it(tmp_path):
    _write_jsonl(tmp_path / "20260722_100000_combo__baseline.jsonl", [_record(turn_id="a")])
    _write_jsonl(tmp_path / "20260722_110000_combo.jsonl", [_record(turn_id="b")])

    with pytest.raises(ValueError, match="20260722_110000_combo.jsonl"):
        load_runs_with_tags(str(tmp_path))


def test_load_runs_with_tags_names_every_untagged_file_when_several_are_missing_tags(tmp_path):
    _write_jsonl(tmp_path / "20260722_100000_combo.jsonl", [_record(turn_id="a")])
    _write_jsonl(tmp_path / "20260722_110000_combo.jsonl", [_record(turn_id="b")])

    with pytest.raises(ValueError) as excinfo:
        load_runs_with_tags(str(tmp_path))
    assert "20260722_100000_combo.jsonl" in str(excinfo.value)
    assert "20260722_110000_combo.jsonl" in str(excinfo.value)
