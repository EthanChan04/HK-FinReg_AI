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
