"""Evaluate evidence coverage for DeepResearch sub-questions."""

from __future__ import annotations

from enum import Enum

from app.schemas.deepresearch import EvidenceGap, ResearchPlan


class EvidenceGapType(str, Enum):
    INSUFFICIENT_COUNT = "insufficient_count"
    MISSING_REGULATOR = "missing_regulator"
    LOW_SOURCE_QUALITY = "low_source_quality"
    LOW_RECENCY = "low_recency"


def _coverage_score(
    evidence_count: int,
    min_count: int,
    regulator_hits: int,
    topic_hits: int,
) -> float:
    sub_question_coverage = min(1.0, evidence_count / max(min_count, 1))
    regulator_coverage = 1.0 if regulator_hits > 0 else 0.0
    topic_coverage = 1.0 if topic_hits > 0 else 0.0
    source_quality = 1.0 if evidence_count >= min_count else 0.5
    citation_support = 1.0 if evidence_count > 0 else 0.0
    recency_score = 0.8
    return round(
        0.25 * sub_question_coverage
        + 0.20 * regulator_coverage
        + 0.20 * topic_coverage
        + 0.15 * source_quality
        + 0.10 * citation_support
        + 0.10 * recency_score,
        3,
    )


def evaluate_evidence_coverage(
    plan: ResearchPlan,
    evidence_by_subquestion: dict[str, list[dict]],
) -> list[EvidenceGap]:
    """Return coverage-based evidence gaps for weak sub-questions."""

    gaps: list[EvidenceGap] = []
    for sub_question in plan.sub_questions:
        evidence = evidence_by_subquestion.get(sub_question.id, [])
        merged_text = " ".join(str(item.get("text", "")).lower() for item in evidence)
        regulator_hits = sum(
            1 for topic in sub_question.required_topics if topic.lower() in merged_text
        )
        topic_hits = regulator_hits
        coverage_score = _coverage_score(
            evidence_count=len(evidence),
            min_count=sub_question.evidence_min_count,
            regulator_hits=regulator_hits,
            topic_hits=topic_hits,
        )
        if coverage_score >= 0.65:
            continue
        gap_type = (
            EvidenceGapType.INSUFFICIENT_COUNT.value
            if len(evidence) < sub_question.evidence_min_count
            else EvidenceGapType.MISSING_REGULATOR.value
        )
        gaps.append(
            EvidenceGap(
                sub_question_id=sub_question.id,
                reason=(
                    f"Coverage score {coverage_score} below threshold for {sub_question.id}; "
                    f"gap_type={gap_type}."
                ),
                suggested_followup_query=sub_question.question,
            )
        )
    return gaps

