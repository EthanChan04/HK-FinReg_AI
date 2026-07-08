"""Compliance Copilot streaming chat API."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.schemas.copilot import CopilotChatRequest
from app.services.copilot.context_builder import build_case_context
from app.services.copilot.intent_classifier import classify_intent
from app.services.copilot.model import build_copilot_llm
from app.services.copilot.response_writer import write_bilingual_response
from app.services.copilot.tool_router import route_tools
from app.services.utils import pii_scrubber

router = APIRouter(prefix="/copilot", tags=["Compliance Copilot"])


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, size: int = 120):
    for index in range(0, len(text), size):
        yield text[index : index + size]


async def _stream_chat(request: CopilotChatRequest):
    conversation_id = request.conversation_id or f"copilot-{uuid4().hex[:10]}"

    # PII 脱敏：在进入 LLM 处理前统一脱敏用户输入
    safe_message = pii_scrubber(request.message)

    try:
        compact_context = build_case_context(request)
        decision = classify_intent(safe_message)

        yield _sse("intent", {"intent": decision.intent, "engine": decision.engine, "reason": decision.reason})

        routed = await asyncio.to_thread(route_tools, decision, request, compact_context)
        yield _sse("tool_call", {"tool": routed.tool_name, "status": "running"})

        payload = routed.payload.model_dump()

        if payload.get("evidence_chunks"):
            evidence_data = []
            for item in payload["evidence_chunks"]:
                if hasattr(item, "model_dump"):
                    evidence_data.append(item.model_dump())
                else:
                    evidence_data.append(item)
            yield _sse("evidence", {"evidence_chunks": evidence_data})

        if payload.get("graph_paths"):
            yield _sse("graph", {"graph_paths": payload["graph_paths"]})

        response_text, citation_audit = await asyncio.to_thread(
            write_bilingual_response,
            request.message,
            decision,
            compact_context,
            routed.payload,
        )

        yield _sse("tool_call", {"tool": routed.tool_name, "status": "done"})

        for chunk in _chunk_text(response_text):
            yield _sse("token", {"text": chunk})

        yield _sse(
            "citation_audit",
            {"unsupported_claim_rate": citation_audit.get("unsupported_claim_rate", 0.0)},
        )

        yield _sse(
            "done",
            {
                "conversation_id": conversation_id,
                "intent": decision.intent,
                "engine": decision.engine,
            },
        )
    except Exception as exc:
        yield _sse("tool_call", {"tool": "mimo", "status": "error", "message": str(exc)})
        yield _sse("done", {"conversation_id": conversation_id, "intent": "smalltalk_or_help", "engine": "error"})


@router.post("/chat/stream")
async def chat_stream(request: CopilotChatRequest):
    settings = get_settings()
    if not (settings.COPILOT_API_KEY or settings.ZHIPU_API_KEY):
        raise HTTPException(
            status_code=503,
            detail="Compliance Copilot model credentials are not configured.",
        )

    # Proactive validation to return a safe API error before streaming.
    try:
        build_copilot_llm()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StreamingResponse(
        _stream_chat(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
