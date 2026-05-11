"""Rule-based retrieval routing for HK financial regulatory queries."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


class QueryProfile(BaseModel):
    """Classifier output consumed by the retrieval router."""

    retrieval_mode: Literal["rag", "kag", "deep_research"] = "rag"
    filters: dict[str, list[str]] = Field(default_factory=dict)
    confidence: float = 0.5
    reasons: list[str] = Field(default_factory=list)


def _add_filter(filters: dict[str, list[str]], key: str, values: list[str]) -> None:
    existing = filters.setdefault(key, [])
    for value in values:
        if value not in existing:
            existing.append(value)


def classify_query(query: str) -> QueryProfile:
    """Classify a query into RAG, KAG, or DeepResearch using conservative rules."""

    text = (query or "").lower()
    filters: dict[str, list[str]] = {}
    reasons: list[str] = []
    mode: Literal["rag", "kag", "deep_research"] = "rag"
    confidence = 0.55

    if re.search(r"\b(svf|stored value|儲值|储值)\b", text):
        _add_filter(filters, "module_tags", ["svf"])
        reasons.append("svf")
    if re.search(r"\b(aml|cft|cdd|kyc|money laundering|洗錢|洗钱)\b", text):
        _add_filter(filters, "topics", ["AML", "CFT", "CDD", "KYC"])
        reasons.append("aml")
    if re.search(r"\b(ai|artificial intelligence|genai|生成式|人工智能|投顧|投顾)\b", text):
        _add_filter(filters, "topics", ["AI", "GenAI", "ai_governance"])
        mode = "kag"
        confidence = 0.72
        reasons.append("ai")
    if re.search(r"\b(privacy|personal data|私隱|私隐|個人資料|个人资料)\b", text):
        _add_filter(filters, "topics", ["privacy", "personal_data"])
        _add_filter(filters, "regulator", ["PCPD"])
        mode = "kag"
        confidence = 0.74
        reasons.append("privacy")
    if re.search(r"\b(regulators?|multi[-\s]?regulator|監管機構|监管机构|obligations?)\b", text) and "ai" in reasons:
        mode = "kag"
        confidence = max(confidence, 0.78)
        reasons.append("relationship_query")
    if re.search(r"(分析|報告|报告|checklist|檢查清單|检查清单|上线前|上線前|合規風險|合规风险|research)", text):
        mode = "deep_research"
        confidence = 0.82
        reasons.append("research")

    return QueryProfile(retrieval_mode=mode, filters=filters, confidence=confidence, reasons=reasons)
