"""Response guardrails for Compliance Copilot."""

from __future__ import annotations

import re
from typing import Any

from app.schemas.evidence import EvidenceChunk
from app.services.retrieval.citation_verifier import verify_citations

ZH_HEADER = "## 绻侀珨涓枃"
EN_HEADER = "## English"

_APPROVAL_PATTERNS = (
    "approved",
    "approve",
    "reject",
    "final approval",
    "最終批准",
    "最终批准",
    "批准",
    "拒絕",
    "拒绝",
)

_LEGAL_ADVICE_PATTERNS = (
    "legal advice",
    "法律意見",
    "法律意见",
)


def _split_sections(text: str) -> tuple[str, str]:
    zh = ""
    en = ""
    if ZH_HEADER in text and EN_HEADER in text:
        _, tail = text.split(ZH_HEADER, 1)
        zh_part, en_part = tail.split(EN_HEADER, 1)
        zh = zh_part.strip()
        en = en_part.strip()
        return zh, en
    return zh, en


def ensure_bilingual_sections(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        normalized = "Evidence is insufficient to produce a complete answer."

    if ZH_HEADER in normalized and EN_HEADER in normalized:
        return normalized

    if EN_HEADER in normalized and ZH_HEADER not in normalized:
        return f"{ZH_HEADER}\n資料不足，需要更多證據支持。\n\n{normalized}"

    if ZH_HEADER in normalized and EN_HEADER not in normalized:
        return f"{normalized}\n\n{EN_HEADER}\nEvidence is insufficient; more support is required."

    return (
        f"{ZH_HEADER}\n"
        "資料不足，需要更多證據支持，以下為初步分析。\n\n"
        f"{EN_HEADER}\n"
        f"{normalized}"
    )


def _inject_disclaimer(text: str) -> str:
    zh, en = _split_sections(text)
    if not zh and not en:
        return text

    if "僅供" not in zh and "不構成最終合規批准" not in zh:
        zh += "\n\n- 本回覆僅供內部合規分析，不構成最終合規批准或法律意見。"

    if "does not constitute" not in en.lower():
        en += "\n\n- This response is for internal compliance analysis and does not constitute final approval or legal advice."

    return f"{ZH_HEADER}\n{zh.strip()}\n\n{EN_HEADER}\n{en.strip()}"


def _remove_prohibited_statements(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(pattern in lowered for pattern in _APPROVAL_PATTERNS):
            if "do not" in lowered or "not provide" in lowered:
                lines.append(line)
            else:
                continue
            
        elif any(pattern in lowered for pattern in _LEGAL_ADVICE_PATTERNS):
            if "not" in lowered:
                lines.append(line)
            else:
                continue
        else:
            lines.append(line)

    cleaned = "\n".join(lines).strip()
    return cleaned or text


def _normalize_evidence_chunks(evidence_chunks: list[Any]) -> list[EvidenceChunk]:
    normalized: list[EvidenceChunk] = []
    for index, item in enumerate(evidence_chunks, start=1):
        if isinstance(item, EvidenceChunk):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            payload = dict(item)
            payload.setdefault("evidence_id", payload.get("id") or f"source_{index}")
            payload.setdefault("text", payload.get("text_snippet") or "")
            normalized.append(EvidenceChunk(**payload))
    return normalized


def apply_guardrails(
    text: str,
    evidence_chunks: list[Any],
    confidence_data: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Apply response safety and enforce bilingual output contract."""

    guarded = ensure_bilingual_sections(text)
    guarded = _remove_prohibited_statements(guarded)
    guarded = ensure_bilingual_sections(guarded)
    guarded = _inject_disclaimer(guarded)

    confidence = confidence_data or {}
    low_confidence = False
    retrieval_score = confidence.get("retrieval")
    if isinstance(retrieval_score, (int, float)) and retrieval_score < 0.5:
        low_confidence = True
    if confidence.get("cross_validation_passed") is False:
        low_confidence = True

    if low_confidence:
        zh, en = _split_sections(guarded)
        zh += "\n\n- 目前置信度偏低，建議轉交 Human Review 進一步覆核。"
        en += "\n\n- Confidence is currently low; route this case to Human Review for additional validation."
        guarded = f"{ZH_HEADER}\n{zh.strip()}\n\n{EN_HEADER}\n{en.strip()}"

    audit = verify_citations(guarded, _normalize_evidence_chunks(evidence_chunks)).model_dump()
    return guarded, audit
