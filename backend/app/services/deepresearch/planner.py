"""DeepResearch planner with deterministic fallback."""

from __future__ import annotations

from app.core.config import get_settings
from app.schemas.deepresearch import ResearchPlan, ResearchSubQuestion


def fallback_research_plan(query: str) -> ResearchPlan:
    """Create a stable rule-based plan when LLM planning is unavailable."""

    settings = get_settings()
    sub_questions = [
        ResearchSubQuestion(
            id="SQ1",
            question=f"What HKMA requirements are relevant to: {query}",
            retrieval_mode="rag",
            required_topics=["HKMA"],
        ),
        ResearchSubQuestion(
            id="SQ2",
            question=f"What SFC or conduct obligations may be relevant to: {query}",
            retrieval_mode="kag",
            required_topics=["SFC", "conduct"],
        ),
        ResearchSubQuestion(
            id="SQ3",
            question=f"What AML/CFT and customer due diligence obligations are relevant to: {query}",
            retrieval_mode="rag",
            required_topics=["AML", "CDD"],
        ),
        ResearchSubQuestion(
            id="SQ4",
            question=f"What AI governance, privacy, and consumer protection risks are relevant to: {query}",
            retrieval_mode="kag",
            required_topics=["AI", "privacy", "consumer_protection"],
        ),
    ]
    return ResearchPlan(
        research_goal=query,
        sub_questions=sub_questions[: settings.DEEP_RESEARCH_MAX_SUBQUESTIONS],
        expected_output_sections=[
            "Executive Summary",
            "Regulatory Scope",
            "Key Obligations",
            "Risk Analysis",
            "Compliance Checklist",
            "Information Gaps",
        ],
    )


def build_research_plan(query: str, llm=None) -> ResearchPlan:
    """Build a research plan, falling back deterministically on any LLM error."""

    if llm is None:
        return fallback_research_plan(query)
    try:
        # The online planner can be expanded later; fallback keeps the workflow safe.
        return fallback_research_plan(query)
    except Exception:
        return fallback_research_plan(query)
