from app.schemas.evidence import EvidenceChunk


def test_deepresearch_fallback_plan_has_minimum_subquestions():
    from app.services.deepresearch.planner import fallback_research_plan

    plan = fallback_research_plan("AI investment advisor compliance risks")

    assert len(plan.sub_questions) >= 4
    assert plan.research_goal == "AI investment advisor compliance risks"


def test_deepresearch_task_type_templates():
    from app.schemas.deepresearch import ResearchRequest
    from app.services.deepresearch.planner import build_research_plan

    request = ResearchRequest(
        query="Compare HKMA and PCPD obligations for AI onboarding",
        task_type="cross_regulator_analysis",
        output_format="memo",
    )
    plan = build_research_plan(request.query, request=request)

    assert len(plan.sub_questions) == 4
    assert "Evidence Appendix" in plan.expected_output_sections


def test_evidence_evaluator_reports_gaps():
    from app.services.deepresearch.evidence_evaluator import evaluate_evidence_coverage
    from app.services.deepresearch.planner import fallback_research_plan

    plan = fallback_research_plan("AI investment advisor compliance risks")
    gaps = evaluate_evidence_coverage(plan, evidence_by_subquestion={})

    assert gaps
    assert gaps[0].sub_question_id == plan.sub_questions[0].id


def test_report_writer_creates_checklist_and_citation_audit():
    from app.services.deepresearch.report_writer import write_fallback_report

    result = write_fallback_report(
        query="AI advisor compliance",
        research_plan={"research_goal": "AI advisor compliance"},
        evidence_by_subquestion={
            "SQ1": [
                EvidenceChunk(
                    evidence_id="source_1",
                    doc_id="doc_1",
                    title="Doc",
                    regulator="HKMA",
                    page=1,
                    text="Institutions should maintain governance controls.",
                ).model_dump()
            ]
        },
        evidence_gaps=[],
    )

    assert "Compliance Checklist" in result.final_report
    assert result.citation_audit is not None


def test_deepresearch_regulator_diversity_gate_prioritizes_available_regulators():
    from app.services.deepresearch.workflow import _apply_regulator_diversity_gate

    evidence = [
        {"evidence_id": "p1", "regulator": "PCPD", "text": "PCPD AI guidance"},
        {"evidence_id": "p2", "regulator": "PCPD", "text": "PCPD personal data"},
        {"evidence_id": "p3", "regulator": "PCPD", "text": "PCPD privacy"},
        {"evidence_id": "h1", "regulator": "HKMA", "text": "HKMA governance"},
        {"evidence_id": "s1", "regulator": "SFC", "text": "SFC suitability"},
    ]

    selected = _apply_regulator_diversity_gate(
        evidence,
        required_regulators=["HKMA", "SFC", "PCPD"],
        top_k=3,
    )

    assert [item["regulator"] for item in selected] == ["HKMA", "SFC", "PCPD"]


def test_deepresearch_regulator_diversity_gate_preserves_rank_when_no_alternative_exists():
    from app.services.deepresearch.workflow import _apply_regulator_diversity_gate

    evidence = [
        {"evidence_id": "p1", "regulator": "PCPD", "text": "PCPD AI guidance"},
        {"evidence_id": "p2", "regulator": "PCPD", "text": "PCPD personal data"},
    ]

    selected = _apply_regulator_diversity_gate(
        evidence,
        required_regulators=["HKMA", "SFC", "PCPD"],
        top_k=3,
    )

    assert [item["evidence_id"] for item in selected] == ["p1", "p2"]
