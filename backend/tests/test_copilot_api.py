from fastapi import HTTPException

from app.schemas.copilot import CopilotChatRequest, CopilotRuntimePayload
from app.services.copilot.intent_classifier import IntentDecision
from app.services.copilot.tool_router import ToolRouteResult


def test_copilot_api_missing_credentials_returns_safe_error(monkeypatch):
    from app.api.routers import copilot

    class _Settings:
        DEEPSEEK_API_KEY = ""

    monkeypatch.setattr(copilot, "get_settings", lambda: _Settings())

    import asyncio

    try:
        asyncio.run(copilot.chat_stream(CopilotChatRequest(message="Hello")))
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "not configured" in str(exc.detail).lower()
    else:
        raise AssertionError("Expected HTTPException for missing credentials")


def test_copilot_stream_emits_required_events(monkeypatch):
    from app.api.routers import copilot

    monkeypatch.setattr(copilot, "build_case_context", lambda request: {"case_context": request.case_context.model_dump()})
    monkeypatch.setattr(
        copilot,
        "classify_intent",
        lambda _message: IntentDecision(intent="regulatory_qa", engine="rag", reason="test"),
    )
    monkeypatch.setattr(
        copilot,
        "route_tools",
        lambda *_args, **_kwargs: ToolRouteResult(payload=CopilotRuntimePayload(evidence_chunks=[]), tool_name="rag"),
    )
    monkeypatch.setattr(
        copilot,
        "write_bilingual_response",
        lambda *_args, **_kwargs: (
            "## 绻侀珨涓枃\n測試\n\n## English\nTest",
            {"unsupported_claim_rate": 0.0},
        ),
    )

    async def _collect():
        chunks = []
        async for chunk in copilot._stream_chat(CopilotChatRequest(message="What is CDD?")):
            chunks.append(chunk)
        return chunks

    import asyncio

    chunks = asyncio.run(_collect())
    joined = "".join(chunks)
    assert "event: intent" in joined
    assert "event: tool_call" in joined
    assert "event: token" in joined
    assert "event: citation_audit" in joined
    assert "event: done" in joined
