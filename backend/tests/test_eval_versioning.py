"""T1-03: Evaluation versioning tests.

Every run_eval summary must carry provenance metadata so results are
auditable: which benchmark snapshot, which corpus state, which eval code
version, and when the run happened.
"""

from __future__ import annotations

from app.services.evaluation import run_eval
from app.services.evaluation.run_eval import (
    _benchmark_fingerprint,
    _corpus_fingerprint,
    _eval_version,
    _build_provenance,
)


def test_eval_version_is_stable_and_descriptive():
    version = _eval_version()
    assert isinstance(version, str)
    assert len(version) > 4
    # Stable across calls within the same process.
    assert version == _eval_version()


def test_benchmark_fingerprint_changes_with_questions():
    questions_a = [{"id": "Q1", "question": "What is CDD?"}]
    questions_b = [{"id": "Q1", "question": "What is KYC?"}]
    fingerprint_a = _benchmark_fingerprint(questions_a)
    fingerprint_b = _benchmark_fingerprint(questions_b)
    assert fingerprint_a != fingerprint_b
    assert len(fingerprint_a) == 64  # sha256 hex


def test_corpus_fingerprint_detects_missing_manifest():
    fingerprint = _corpus_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64


def test_provenance_includes_required_fields():
    provenance = _build_provenance(
        questions=[{"id": "Q1", "question": "x"}],
    )
    assert "eval_version" in provenance
    assert "evaluated_at" in provenance
    assert "benchmark_fingerprint" in provenance
    assert "corpus_fingerprint" in provenance
    assert "question_count" in provenance
    assert provenance["question_count"] == 1
    # evaluated_at must be ISO-8601-ish
    assert "T" in provenance["evaluated_at"]


def test_run_eval_summary_contains_provenance(monkeypatch):
    from app.services.evaluation import run_eval as module

    monkeypatch.setattr(
        module,
        "load_benchmark_questions",
        lambda: [
            {
                "id": "PROV_001",
                "question": "CDD requirements?",
                "expected_retrieval_mode": "rag",
                "expected_topics": [],
                "expected_regulators": [],
            }
        ],
    )
    monkeypatch.setattr(module, "classify_query", lambda query: type(
        "Profile", (), {"retrieval_mode": "rag", "filters": {}}
    )())
    monkeypatch.setattr(module, "_compute_evidence_count", lambda item: 0)
    monkeypatch.setattr(module, "_compute_graph_path_count", lambda item: 0)
    monkeypatch.setattr(module, "_compute_citation_audit", lambda item: (0.0, 0.0))
    monkeypatch.setattr(module, "_compute_deepresearch_gap_count", lambda item: 0)
    monkeypatch.setattr(module, "_compute_evidence_regulator_coverage", lambda item: 0.0)
    monkeypatch.setattr(module, "_evaluate_claim_metrics", lambda item: {
        "claim_recall": 0.0,
        "context_precision": 0.0,
        "faithfulness": None,
        "hallucination_rate": None,
        "noise_sensitivity": 0.0,
        "context_utilization": 0.0,
        "claim_diagnostics": [],
    })

    summary = run_eval.run_eval()

    assert "provenance" in summary
    assert summary["provenance"]["question_count"] == 1
    assert summary["provenance"]["benchmark_fingerprint"]
