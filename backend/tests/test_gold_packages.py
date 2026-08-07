"""Auditable gold-package structure for the 108-case benchmark."""

from __future__ import annotations

import pytest


def _question(case_id: str = "Q1") -> dict:
    return {
        "id": case_id,
        "question": "What must an institution do?",
        "expected_claims": ["The institution must perform customer due diligence."],
        "expected_regulators": ["HKMA"],
        "language": "en",
        "task_type": "obligation_extraction",
    }


def test_build_gold_packages_preserves_decision_and_pending_review_state():
    from app.services.evaluation.gold_packages import build_gold_packages

    document = build_gold_packages([_question()])

    package = document["packages"][0]
    assert package["case_id"] == "Q1"
    assert package["decision"] == [
        "The institution must perform customer due diligence."
    ]
    assert package["witness_trace"][0]["claim"] == package["decision"][0]
    assert package["witness_trace"][0]["evidence_ids"] == []
    assert package["clause_set"] == []
    assert package["review"] == {
        "status": "pending",
        "reviewer": None,
        "reviewed_at": None,
        "revision_history": [],
    }


def test_validate_gold_packages_rejects_approval_without_human_identity():
    from app.services.evaluation.gold_packages import (
        build_gold_packages,
        validate_gold_packages,
    )

    document = build_gold_packages([_question()])
    document["packages"][0]["review"]["status"] = "approved"

    with pytest.raises(ValueError, match="reviewer and reviewed_at"):
        validate_gold_packages(document, [_question()])


def test_validate_gold_packages_requires_one_package_per_benchmark_case():
    from app.services.evaluation.gold_packages import (
        build_gold_packages,
        validate_gold_packages,
    )

    questions = [_question("Q1"), _question("Q2")]
    document = build_gold_packages(questions)
    document["packages"].pop()

    with pytest.raises(ValueError, match="case IDs do not match"):
        validate_gold_packages(document, questions)
