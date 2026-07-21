"""GET /metrics on the FastAPI sidecar -- see server/metrics.py for the
aggregator this endpoint serves."""

import json

from fastapi.testclient import TestClient

import server.main as main_module
from server.metrics import MetricsAggregator


def test_metrics_endpoint_serves_prometheus_text_from_the_aggregator(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
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
        )
        + "\n"
    )
    main_module._metrics_aggregator = MetricsAggregator(str(path))

    client = TestClient(main_module.app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "voice_agent_stage_latency_seconds" in response.text
    assert 'voice_agent_turns_total{combination_id="combo"} 1' in response.text
