"""T2-01: SSE event contract tests (risk R-03 test hierarchy gap).

The streaming contract is: agent_state events first, then token events,
terminated by exactly one done event. Callers depend on this ordering for
the UI progress bar and completion detection.
"""

from __future__ import annotations

import asyncio
import json

from app.api.routers.workflow_utils import create_streaming_response, format_sse_event


def _collect(gen):
    return asyncio.run(_collect_async(gen))


async def _collect_async(gen):
    events = []
    async for chunk in gen:
        events.append(chunk)
    return events


def _parse_sse(events: list[str]) -> list[dict]:
    parsed = []
    for event in events:
        lines = event.strip().split("\n")
        data = None
        event_type = None
        for line in lines:
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        parsed.append({"event": event_type, "data": json.loads(data) if data else None})
    return parsed


def test_sse_emits_agent_state_tokens_then_done():
    def fake_graph(safe_input: str) -> str:
        return "Line one\nLine two"

    events = _collect(
        create_streaming_response(
            fake_graph,
            "input",
            [("analyst", "Analyzing"), ("reviewer", "Reviewing")],
        )
    )
    parsed = _parse_sse(events)

    # Two agent_state events first
    assert [e["event"] for e in parsed[:2]] == ["agent_state", "agent_state"]
    assert parsed[0]["data"]["agent"] == "analyst"
    assert parsed[0]["data"]["status"] == "running"
    assert parsed[1]["data"]["agent"] == "reviewer"

    # Then token events carrying the report lines
    token_events = [e for e in parsed if e["event"] == "token"]
    assert len(token_events) == 2
    assert token_events[0]["data"]["text"] == "Line one\n"

    # Exactly one done event at the end
    assert parsed[-1]["event"] == "done"
    assert parsed[-1]["data"]["status"] == "complete"
    assert sum(1 for e in parsed if e["event"] == "done") == 1


def test_sse_emits_done_even_when_report_is_empty():
    def fake_graph(safe_input: str) -> str:
        return ""

    events = _collect(create_streaming_response(fake_graph, "input", []))
    parsed = _parse_sse(events)

    # Contract: exactly one done event terminates the stream; an empty
    # report still yields a single (empty) token event before it.
    assert parsed[-1]["event"] == "done"
    assert [e["event"] for e in parsed] == ["token", "done"]


def test_sse_agent_state_events_are_flushed_before_graph_execution():
    """UI must see agent_state events even if the graph is slow."""

    async def slow_graph(safe_input: str) -> str:
        await asyncio.sleep(0.05)
        return "Result"

    order: list[str] = []

    async def observe():
        async for chunk in create_streaming_response(slow_graph, "input", [("analyst", "Start")]):
            event_type = None
            for line in chunk.strip().split("\n"):
                if line.startswith("event:"):
                    event_type = line[6:].strip()
            order.append(event_type)

    asyncio.run(observe())
    assert order[0] == "agent_state"
    assert order[-1] == "done"


def test_format_sse_event_is_properly_framed():
    event = format_sse_event("token", {"text": "hello"})
    assert event.startswith("event: token")
    assert '"text": "hello"' in event
