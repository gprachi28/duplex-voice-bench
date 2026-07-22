"""Per-turn latency metrics -- see design.md's "Observability" section
("structured JSON log with per-stage latencies and the combination ID").

TurnMetrics holds raw monotonic timestamps for each pipeline stage boundary
and turns them into named deltas; append_turn_metrics writes one JSON line
per turn so the FastAPI sidecar (a separate process) can read and aggregate
them without any other IPC.
"""

import json

from agent.metrics import TurnMetrics, append_turn_metrics


def test_to_record_computes_stage_deltas_from_raw_timestamps():
    m = TurnMetrics(
        turn_id="room-1",
        room="room",
        combination_id="mlx-large-v3.ollama-3.2-3b.kokoro",
        t0=0.0,
        t1=0.01,
        t2=0.20,
        t3=0.35,
        t4=0.40,
        t5=0.50,
    )
    record = m.to_record()
    assert record["end_of_turn_s"] == 0.01
    assert round(record["transcription_s"], 10) == 0.19
    assert round(record["llm_ttft_s"], 10) == 0.15
    assert round(record["sentence_buffer_s"], 10) == 0.05
    assert round(record["tts_first_chunk_s"], 10) == 0.10
    assert round(record["ttfa_s"], 10) == 0.50


def test_to_record_includes_identifying_fields_and_flags():
    m = TurnMetrics(
        turn_id="room-1-turn-3",
        room="room-1",
        combination_id="combo",
        forced=True,
        interrupted=True,
        smart_turn_prob=0.95,
    )
    record = m.to_record()
    assert record["turn_id"] == "room-1-turn-3"
    assert record["room"] == "room-1"
    assert record["combination_id"] == "combo"
    assert record["forced"] is True
    assert record["interrupted"] is True
    assert record["smart_turn_prob"] == 0.95


def test_to_record_includes_prompt_version():
    m = TurnMetrics(
        turn_id="t", room="r", combination_id="c", prompt_version="v1-concise-en"
    )
    assert m.to_record()["prompt_version"] == "v1-concise-en"


def test_to_record_prompt_version_defaults_to_none():
    m = TurnMetrics(turn_id="t", room="r", combination_id="c")
    assert m.to_record()["prompt_version"] is None


def test_to_record_includes_stt_repetition_detected():
    m = TurnMetrics(
        turn_id="t", room="r", combination_id="c", stt_repetition_detected=True
    )
    assert m.to_record()["stt_repetition_detected"] is True


def test_to_record_stt_repetition_detected_defaults_to_false():
    m = TurnMetrics(turn_id="t", room="r", combination_id="c")
    assert m.to_record()["stt_repetition_detected"] is False


def test_to_record_is_null_for_deltas_missing_a_timestamp():
    # An interrupted turn never reaches later stages -- those timestamps
    # stay None, so the deltas that need them must be None too, not a
    # crash and not a misleading 0.0.
    m = TurnMetrics(turn_id="t", room="r", combination_id="c", t0=0.0, t1=0.02)
    record = m.to_record()
    assert record["end_of_turn_s"] == 0.02
    assert record["transcription_s"] is None
    assert record["llm_ttft_s"] is None
    assert record["sentence_buffer_s"] is None
    assert record["tts_first_chunk_s"] is None
    assert record["ttfa_s"] is None


def test_append_turn_metrics_writes_one_json_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    m = TurnMetrics(turn_id="t1", room="r", combination_id="c", t0=0.0, t5=0.5)

    append_turn_metrics(str(path), m)

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["turn_id"] == "t1"
    assert record["ttfa_s"] == 0.5


def test_append_turn_metrics_appends_without_truncating(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_turn_metrics(str(path), TurnMetrics(turn_id="t1", room="r", combination_id="c"))
    append_turn_metrics(str(path), TurnMetrics(turn_id="t2", room="r", combination_id="c"))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["turn_id"] == "t1"
    assert json.loads(lines[1])["turn_id"] == "t2"
