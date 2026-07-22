"""Unit tests for _publish_turn_state -- the client-side visual indicator
fix for design.md's Known Issues: 'No feedback during a long pause'. The
data-channel publish itself is faked here; the client/index.html rendering
is verified live (see benchmarks/experiments.md).
"""

import asyncio
import json

from agent.worker import _publish_turn_state


class FakeLocalParticipant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, str]] = []

    async def publish_data(self, payload, *, reliable=True, topic="") -> None:
        self.calls.append((payload, reliable, topic))


def test_publish_turn_state_sends_json_payload_on_the_turn_state_topic():
    participant = FakeLocalParticipant()

    asyncio.run(_publish_turn_state(participant, "listening"))

    assert len(participant.calls) == 1
    payload, reliable, topic = participant.calls[0]
    assert json.loads(payload) == {"state": "listening"}
    assert reliable is True
    assert topic == "turn_state"


def test_publish_turn_state_sends_the_given_state_verbatim():
    participant = FakeLocalParticipant()

    asyncio.run(_publish_turn_state(participant, "idle"))

    payload, _, _ = participant.calls[0]
    assert json.loads(payload)["state"] == "idle"
