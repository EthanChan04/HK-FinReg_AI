"""Case-aware context compaction for Compliance Copilot."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.schemas.copilot import CopilotCaseContext, CopilotChatRequest

_SECRET_KEYS = {"api_key", "apikey", "token", "secret", "password", "authorization"}


def _trim_text(value: str | None, max_chars: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    head = max(0, max_chars - 24)
    return text[:head].rstrip() + "\n...[truncated]"


def _sanitize_dict(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        key_norm = str(key).lower()
        if any(secret_key in key_norm for secret_key in _SECRET_KEYS):
            continue
        if isinstance(value, dict):
            cleaned[key] = _sanitize_dict(value)
        elif isinstance(value, list):
            compact_list = []
            for item in value:
                if isinstance(item, dict):
                    compact_list.append(_sanitize_dict(item))
                else:
                    compact_list.append(item)
            cleaned[key] = compact_list
        else:
            cleaned[key] = value
    return cleaned


def _compact_evidence(case_context: CopilotCaseContext, max_chars: int) -> list[dict[str, Any]]:
    budget = max_chars // 2
    used = 0
    compact: list[dict[str, Any]] = []
    for item in case_context.evidence_chunks:
        if not isinstance(item, dict):
            continue
        normalized = _sanitize_dict(item)
        snippet = _trim_text(str(normalized.get("text", "")), 360)
        entry = {
            "evidence_id": normalized.get("evidence_id") or normalized.get("id"),
            "regulator": normalized.get("regulator"),
            "title": normalized.get("title"),
            "page": normalized.get("page"),
            "section_title": normalized.get("section_title"),
            "text_snippet": snippet,
        }
        entry_size = sum(len(str(v)) for v in entry.values() if v is not None)
        if used + entry_size > budget and compact:
            break
        compact.append(entry)
        used += entry_size
    return compact


def _compact_graph_paths(case_context: CopilotCaseContext, limit: int = 8) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    for raw in case_context.graph_paths[:limit]:
        if not isinstance(raw, dict):
            continue
        cleaned = _sanitize_dict(raw)
        paths.append(
            {
                "path": cleaned.get("path", [])[:8],
                "matched_node": cleaned.get("matched_node"),
                "matched_topics": (cleaned.get("matched_topics") or [])[:5],
                "matched_obligations": (cleaned.get("matched_obligations") or [])[:5],
            }
        )
    return paths


def build_case_context(request: CopilotChatRequest, settings: Settings | None = None) -> dict[str, Any]:
    """Build compact, secret-safe copilot context from incoming request."""

    cfg = settings or get_settings()
    max_chars = cfg.COPILOT_MAX_CONTEXT_CHARS

    history = request.history[-cfg.COPILOT_MAX_HISTORY_MESSAGES :]
    compact_history = [
        {
            "role": message.role,
            "content": _trim_text(message.content, 800),
        }
        for message in history
    ]

    case_context = request.case_context
    compact = {
        "workspace_id": case_context.workspace_id,
        "workflow_id": case_context.workflow_id,
        "workflow_name": case_context.workflow_name,
        "workflow_run_id": case_context.workflow_run_id,
        "input_text": _trim_text(case_context.input_text, max_chars // 4),
        "report_text": _trim_text(case_context.report_text, max_chars // 3),
        "evidence_chunks": _compact_evidence(case_context, max_chars=max_chars),
        "graph_paths": _compact_graph_paths(case_context),
        "research_plan": _sanitize_dict(case_context.research_plan or {}),
        "confidence_data": _sanitize_dict(case_context.confidence_data),
        "current_gate": case_context.current_gate,
        "gate_message": _trim_text(case_context.gate_message, 300),
    }

    return {
        "message": _trim_text(request.message, 2000),
        "preferred_language": request.preferred_language,
        "history": compact_history,
        "case_context": _sanitize_dict(compact),
    }
