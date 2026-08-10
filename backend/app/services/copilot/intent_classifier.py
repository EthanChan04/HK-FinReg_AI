"""Deterministic intent classifier for Compliance Copilot."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.copilot import CopilotIntent


@dataclass(frozen=True)
class IntentDecision:
    intent: CopilotIntent
    engine: str
    reason: str


_DEEP_RESEARCH_KEYWORDS = (
    "memo",
    "compare",
    "impact",
    "launch",
    "ai governance",
    "policy change",
    "监管备忘录",
    "監管備忘錄",
    "政策變更",
    "政策变更",
    "产品发布",
    "產品發布",
)

_OBLIGATION_MAPPING_KEYWORDS = (
    "why",
    "obligation",
    "risk path",
    "control",
    "regulator applies",
    "obligation mapping",
    "義務",
    "义务",
    "風險路徑",
    "风险路径",
    "控制",
    "監管機構",
    "监管机构",
)

_CASE_EXPLANATION_KEYWORDS = (
    "current report",
    "this case",
    "evidence insufficient",
    "low confidence",
    "當前報告",
    "当前报告",
    "此案例",
    "這個案例",
    "證據不足",
    "证据不足",
    "低置信度",
)

_WORKFLOW_KEYWORDS = (
    "which module",
    "which workflow",
    "where should i go",
    "what workflow",
    "哪個模組",
    "哪个模块",
    "哪個工作流",
    "哪个工作流",
)

_REVIEW_KEYWORDS = (
    "review",
    "approve",
    "reject",
    "pending",
    "human review",
    "審核",
    "审核",
    "批准",
    "拒絕",
    "拒绝",
    "待审核",
)

_SMALLTALK_KEYWORDS = (
    "hello",
    "hi",
    "thanks",
    "help",
    "你是谁",
    "你是誰",
    "你好",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        if keyword in text:
            return keyword
    return None


def classify_intent(message: str) -> IntentDecision:
    """Classify message into explicit Copilot routes using deterministic rules."""

    normalized = (message or "").strip().lower()

    matched = _contains_any(normalized, _DEEP_RESEARCH_KEYWORDS)
    if matched:
        return IntentDecision("deep_research", "deep_research", f"matched keyword: {matched}")

    matched = _contains_any(normalized, _OBLIGATION_MAPPING_KEYWORDS)
    if matched:
        return IntentDecision("obligation_mapping", "rag_kag", f"matched keyword: {matched}")

    matched = _contains_any(normalized, _CASE_EXPLANATION_KEYWORDS)
    if matched:
        return IntentDecision("case_explanation", "case_context_rag", f"matched keyword: {matched}")

    matched = _contains_any(normalized, _WORKFLOW_KEYWORDS)
    if matched:
        return IntentDecision("workflow_recommendation", "workflow_router", f"matched keyword: {matched}")

    matched = _contains_any(normalized, _REVIEW_KEYWORDS)
    if matched:
        return IntentDecision("human_review_help", "human_review", f"matched keyword: {matched}")

    matched = _contains_any(normalized, _SMALLTALK_KEYWORDS)
    if matched and len(normalized.split()) <= 12:
        return IntentDecision("smalltalk_or_help", "deepseek", f"matched keyword: {matched}")

    return IntentDecision("regulatory_qa", "rag", "default regulatory Q&A fallback")
