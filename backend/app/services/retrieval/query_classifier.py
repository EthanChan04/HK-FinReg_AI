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


def _has_any(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


# CJK characters count as \w in Python regex, so \b boundaries never fire
# between Chinese characters, and Latin terms embedded in CJK sentences
# (e.g. "GenAI客戶") have no boundary either. Latin terms therefore use an
# ASCII-only boundary (a term like "ai" must not match inside "training"),
# while CJK terms use plain substring matching (Chinese has no spaces).
_CJK_RANGE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_BOUNDED_TEMPLATE = r"(?<![a-zA-Z0-9])(?:{})(?![a-zA-Z0-9])"


def _has_term(text: str, *terms: str) -> bool:
    lowered = text.lower()
    for term in terms:
        if _CJK_RANGE.search(term):
            if term in lowered:
                return True
        elif re.search(_ASCII_BOUNDED_TEMPLATE.format(re.escape(term)), lowered):
            return True
    return False


def _is_ai_wealth_product_launch(text: str, reasons: list[str]) -> bool:
    has_ai = "ai" in reasons or _has_term(
        text, "ai", "artificial intelligence", "genai", "人工智能", "生成式", "投顧", "投顾"
    )
    has_wealth = _has_term(
        text,
        "wealth",
        "investment",
        "investor",
        "advisory",
        "advisor",
        "suitability",
        "portfolio",
        "投資",
        "投顾",
        "投顧",
        "理財",
        "理财",
    )
    has_launch_or_product = _has_term(
        text,
        "product",
        "launch",
        "launches",
        "launching",
        "pre-launch",
        "pre_launch",
        "prelaunch",
        "pre launch",
        "onboarding",
        "推出",
        "上线前",
        "上線前",
        "產品",
        "产品",
    )
    return has_ai and has_wealth and has_launch_or_product


def classify_query(query: str) -> QueryProfile:
    """Classify a query into RAG, KAG, or DeepResearch using conservative rules."""

    text = (query or "").lower()
    filters: dict[str, list[str]] = {}
    reasons: list[str] = []
    mode: Literal["rag", "kag", "deep_research"] = "rag"
    confidence = 0.55

    if _has_term(text, "svf", "stored value", "儲值", "储值"):
        _add_filter(filters, "module_tags", ["svf"])
        reasons.append("svf")
    if _has_term(text, "aml", "cft", "cdd", "kyc", "money laundering", "洗錢", "洗钱", "打擊洗錢", "打击洗钱"):
        _add_filter(filters, "topics", ["AML", "CFT", "CDD", "KYC"])
        reasons.append("aml")
    if _has_term(text, "ai", "artificial intelligence", "genai", "生成式", "人工智能", "投顧", "投顾"):
        _add_filter(filters, "topics", ["AI", "GenAI", "ai_governance"])
        mode = "kag"
        confidence = 0.72
        reasons.append("ai")
    if _has_term(text, "privacy", "personal data", "私隱", "私隐", "個人資料", "个人资料"):
        _add_filter(filters, "topics", ["privacy", "personal_data"])
        _add_filter(filters, "regulator", ["PCPD"])
        mode = "kag"
        confidence = 0.74
        reasons.append("privacy")
    if _has_term(text, "regulator", "regulators", "multi-regulator", "multi regulator", "multiregulator", "監管機構", "监管机构", "obligation", "obligations") and "ai" in reasons:
        mode = "kag"
        confidence = max(confidence, 0.78)
        reasons.append("relationship_query")
    if _is_ai_wealth_product_launch(text, reasons):
        _add_filter(filters, "regulator", ["HKMA", "SFC", "PCPD"])
        _add_filter(
            filters,
            "topics",
            [
                "wealth_management",
                "consumer_protection",
                "suitability",
                "personal_data",
            ],
        )
        mode = "kag"
        confidence = max(confidence, 0.79)
        reasons.append("ai_wealth_product_launch")
    if _has_term(text, "分析", "報告", "报告", "checklist", "檢查清單", "检查清单", "上线前", "上線前", "合規風險", "合规风险", "research", "researching", "deepresearch"):
        mode = "deep_research"
        confidence = 0.82
        reasons.append("research")

    return QueryProfile(retrieval_mode=mode, filters=filters, confidence=confidence, reasons=reasons)
