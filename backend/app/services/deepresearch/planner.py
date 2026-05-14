"""DeepResearch planner with task-type templates and deterministic fallback."""

from __future__ import annotations

from app.core.config import get_settings
from app.schemas.deepresearch import ResearchPlan, ResearchRequest, ResearchSubQuestion


def _sections_for_output(output_format: str) -> list[str]:
    if output_format == "checklist":
        return ["Checklist", "Evidence Table", "Information Gaps"]
    if output_format == "memo":
        return ["Subject", "Background", "Comparative Analysis", "Recommendations", "Evidence Appendix"]
    if output_format == "matrix":
        return ["Regulator-Obligation Matrix", "Controls", "Evidence", "Open Issues"]
    return [
        "Executive Summary",
        "Regulatory Scope",
        "Key Obligations",
        "Risk Analysis",
        "Compliance Checklist",
        "Information Gaps",
    ]


def fallback_research_plan(query: str, request: ResearchRequest | None = None) -> ResearchPlan:
    """Create a stable rule-based plan when LLM planning is unavailable."""

    request = request or ResearchRequest(query=query)
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
        expected_output_sections=_sections_for_output(request.output_format),
    )


def _build_ai_governance_plan(query: str, request: ResearchRequest) -> ResearchPlan:
    sub_questions = [
        ResearchSubQuestion(id="SQ1", question=f"What AI policy principles apply to: {query}", retrieval_mode="kag"),
        ResearchSubQuestion(id="SQ2", question=f"What HKMA obligations apply to: {query}", retrieval_mode="rag"),
        ResearchSubQuestion(id="SQ3", question=f"What PCPD obligations are triggered by: {query}", retrieval_mode="kag"),
        ResearchSubQuestion(id="SQ4", question=f"What SFC or conduct obligations may apply to: {query}", retrieval_mode="kag"),
        ResearchSubQuestion(
            id="SQ5",
            question=f"What model risk, human oversight, cybersecurity and third-party risks exist for: {query}",
            retrieval_mode="kag",
        ),
        ResearchSubQuestion(
            id="SQ6",
            question=f"What controls and audit evidence should be prepared for: {query}",
            retrieval_mode="rag",
        ),
    ]
    return ResearchPlan(
        research_goal=query,
        sub_questions=sub_questions,
        expected_output_sections=_sections_for_output(request.output_format),
    )


def _build_product_launch_plan(query: str, request: ResearchRequest) -> ResearchPlan:
    sub_questions = [
        ResearchSubQuestion(
            id="SQ1",
            question=f"What product category and regulated activity does this use case fall into: {query}",
            retrieval_mode="kag",
        ),
        ResearchSubQuestion(
            id="SQ2",
            question=f"Which regulators and regulatory documents apply to: {query}",
            retrieval_mode="kag",
        ),
        ResearchSubQuestion(
            id="SQ3",
            question=f"What licensing or authorization issues arise for: {query}",
            retrieval_mode="rag",
        ),
        ResearchSubQuestion(
            id="SQ4",
            question=f"What AML/CFT, KYC/CDD and fraud monitoring obligations arise for: {query}",
            retrieval_mode="rag",
        ),
        ResearchSubQuestion(
            id="SQ5",
            question=f"What data privacy and AI governance obligations arise for: {query}",
            retrieval_mode="kag",
        ),
        ResearchSubQuestion(
            id="SQ6",
            question=f"What operational, outsourcing, cybersecurity and record-keeping controls are required for: {query}",
            retrieval_mode="rag",
        ),
        ResearchSubQuestion(
            id="SQ7",
            question=f"What unresolved evidence gaps require human legal or compliance review for: {query}",
            retrieval_mode="kag",
        ),
    ]
    return ResearchPlan(
        research_goal=query,
        sub_questions=sub_questions,
        expected_output_sections=_sections_for_output(request.output_format),
    )


def _build_cross_regulator_plan(query: str, request: ResearchRequest) -> ResearchPlan:
    sub_questions = [
        ResearchSubQuestion(
            id="SQ1",
            question=f"Which regulators have jurisdiction over: {query}",
            retrieval_mode="kag",
        ),
        ResearchSubQuestion(
            id="SQ2",
            question=f"What are common compliance themes across regulators for: {query}",
            retrieval_mode="kag",
        ),
        ResearchSubQuestion(
            id="SQ3",
            question=f"What obligations differ by regulator for: {query}",
            retrieval_mode="kag",
        ),
        ResearchSubQuestion(
            id="SQ4",
            question=f"What evidence-backed controls should harmonize these requirements for: {query}",
            retrieval_mode="rag",
        ),
    ]
    return ResearchPlan(
        research_goal=query,
        sub_questions=sub_questions,
        expected_output_sections=_sections_for_output(request.output_format),
    )


def build_research_plan(query: str, request: ResearchRequest | None = None, llm=None) -> ResearchPlan:
    """Build a research plan, falling back deterministically on any LLM error."""

    request = request or ResearchRequest(query=query)
    if llm is None:
        if request.task_type == "ai_governance_review":
            return _build_ai_governance_plan(query, request)
        if request.task_type in {"product_launch_review", "regulatory_change_impact"}:
            return _build_product_launch_plan(query, request)
        if request.task_type in {"cross_regulator_analysis", "regulatory_memo", "checklist_generation"}:
            return _build_cross_regulator_plan(query, request)
        return fallback_research_plan(query, request)
    try:
        return fallback_research_plan(query, request)
    except Exception:
        return fallback_research_plan(query, request)
