def test_evaluation_metric_errors_are_reported(monkeypatch):
    from app.services.evaluation import run_eval

    monkeypatch.setattr(
        run_eval,
        "load_benchmark_questions",
        lambda: [
            {
                "id": "ERR_001",
                "question": "What are SVF CDD requirements?",
                "expected_retrieval_mode": "rag",
                "expected_topics": [],
                "expected_regulators": [],
            }
        ],
    )
    monkeypatch.setattr(run_eval, "classify_query", lambda query: type(
        "Profile",
        (),
        {"retrieval_mode": "rag", "filters": {}},
    )())

    def raise_metric_error(item, errors=None):
        raise RuntimeError("retriever boom")

    monkeypatch.setattr(run_eval, "_compute_evidence_count", raise_metric_error)
    monkeypatch.setattr(run_eval, "_compute_graph_path_count", lambda item: 0)
    monkeypatch.setattr(run_eval, "_compute_citation_audit", lambda item: (0.0, 0.0))
    monkeypatch.setattr(run_eval, "_compute_deepresearch_gap_count", lambda item: 0)

    summary = run_eval.run_eval()

    assert summary["metric_errors"]
    assert summary["metric_errors"][0]["question_id"] == "ERR_001"
    assert summary["metric_errors"][0]["metric"] == "evidence_count"
    assert "retriever boom" in summary["metric_errors"][0]["error"]


def test_evaluation_splits_classifier_and_evidence_regulator_coverage(monkeypatch):
    from langchain_core.documents import Document

    from app.services.evaluation import run_eval

    monkeypatch.setattr(
        run_eval,
        "load_benchmark_questions",
        lambda: [
            {
                "id": "KAG_AI_ADVISOR_001",
                "question": "AI wealth advisory product launch",
                "expected_retrieval_mode": "kag",
                "expected_strategy_id": "ai_governance_kag",
                "expected_topics": [],
                "expected_regulators": ["HKMA", "SFC", "PCPD"],
                "expected_expansion_terms": [],
            }
        ],
    )
    monkeypatch.setattr(
        run_eval,
        "_retrieve_eval_documents",
        lambda question, top_k=10: [
            Document(page_content="HKMA governance", metadata={"regulator": "HKMA"}),
            Document(page_content="SFC suitability", metadata={"regulator": "SFC"}),
            Document(page_content="PCPD personal data", metadata={"regulator": "PCPD"}),
        ],
    )
    monkeypatch.setattr(run_eval, "_compute_graph_path_count", lambda item: 0)
    monkeypatch.setattr(run_eval, "_compute_citation_audit", lambda item: (1.0, 0.0))
    monkeypatch.setattr(run_eval, "_compute_deepresearch_gap_count", lambda item: 0)

    summary = run_eval.run_eval()
    row = summary["rows"][0]

    assert row["classifier_regulator_coverage"] == 1.0
    assert row["evidence_regulator_coverage"] == 1.0
    assert summary["avg_classifier_regulator_coverage"] == 1.0
    assert summary["avg_evidence_regulator_coverage"] == 1.0
    assert summary["avg_regulator_coverage"] == summary["avg_classifier_regulator_coverage"]


def test_deepresearch_gap_eval_uses_regulator_diversity_gate(monkeypatch):
    from langchain_core.documents import Document

    from app.schemas.deepresearch import ResearchPlan, ResearchSubQuestion
    from app.services.evaluation import run_eval

    plan = ResearchPlan(
        research_goal="AI advisor launch",
        sub_questions=[
            ResearchSubQuestion(
                id="SQ1",
                question="AI advisor launch regulators",
                retrieval_mode="kag",
                required_topics=["HKMA", "SFC", "PCPD"],
                evidence_min_count=3,
            )
        ],
    )

    monkeypatch.setattr(run_eval, "build_research_plan", lambda question: plan, raising=False)
    monkeypatch.setattr(
        run_eval,
        "_retrieve_eval_documents",
        lambda question, top_k=10: [
            Document(page_content="PCPD AI", metadata={"regulator": "PCPD"}),
            Document(page_content="PCPD privacy", metadata={"regulator": "PCPD"}),
            Document(page_content="HKMA governance", metadata={"regulator": "HKMA"}),
            Document(page_content="SFC suitability", metadata={"regulator": "SFC"}),
        ],
    )

    assert run_eval._compute_deepresearch_gap_count(
        {
            "id": "DR_AI_ADVISOR_001",
            "question": "AI advisor launch",
            "expected_retrieval_mode": "deep_research",
        }
    ) == 0
