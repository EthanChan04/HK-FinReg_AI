"""Evaluate evidence coverage for DeepResearch sub-questions."""

from app.schemas.deepresearch import EvidenceGap, ResearchPlan


def evaluate_evidence_coverage(
    plan: ResearchPlan,
    evidence_by_subquestion: dict[str, list[dict]],
) -> list[EvidenceGap]:
    """Return evidence gaps for sub-questions below their minimum evidence count."""

    gaps: list[EvidenceGap] = []
    for sub_question in plan.sub_questions:
        evidence = evidence_by_subquestion.get(sub_question.id, [])
        if len(evidence) < sub_question.evidence_min_count:
            gaps.append(
                EvidenceGap(
                    sub_question_id=sub_question.id,
                    reason=(
                        f"Only {len(evidence)} evidence chunks found; "
                        f"minimum {sub_question.evidence_min_count} required."
                    ),
                    suggested_followup_query=sub_question.question,
                )
            )
    return gaps
