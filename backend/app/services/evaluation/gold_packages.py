"""Build and validate ScenarioBench-style gold decision packages."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.evaluation.benchmark_loader import load_benchmark_questions

GOLD_PACKAGE_SCHEMA_VERSION = 1


def default_gold_package_path() -> Path:
    backend_root = Path(__file__).resolve().parents[3]
    return (
        backend_root
        / "data"
        / "evaluation"
        / "gold_packages"
        / "benchmark-gold-packages.json"
    )


def build_gold_packages(questions: list[dict]) -> dict:
    """Create deterministic pending-review packages from benchmark claims."""

    packages = []
    for item in questions:
        decision = [str(claim) for claim in item.get("expected_claims", [])]
        packages.append(
            {
                "case_id": str(item["id"]),
                "decision": decision,
                "witness_trace": [
                    {
                        "claim": claim,
                        "evidence_ids": [],
                        "clause_ids": [],
                        "reasoning": "Pending human evidence-to-clause review.",
                    }
                    for claim in decision
                ],
                "clause_set": [],
                "expected_regulators": list(item.get("expected_regulators", [])),
                "language": item.get("language"),
                "task_type": item.get("task_type"),
                "review": {
                    "status": "pending",
                    "reviewer": None,
                    "reviewed_at": None,
                    "revision_history": [],
                },
            }
        )
    return {
        "schema_version": GOLD_PACKAGE_SCHEMA_VERSION,
        "benchmark_case_count": len(questions),
        "packages": packages,
    }


def validate_gold_packages(document: dict, questions: list[dict]) -> None:
    """Fail closed on missing cases, traces, or fabricated review approval."""

    if document.get("schema_version") != GOLD_PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported gold package schema version")
    packages = document.get("packages")
    if not isinstance(packages, list):
        raise ValueError("gold packages must be a list")

    expected_ids = {str(item["id"]) for item in questions}
    actual_ids = {str(item.get("case_id", "")) for item in packages}
    if expected_ids != actual_ids or len(packages) != len(expected_ids):
        raise ValueError("gold package case IDs do not match benchmark case IDs")

    for package in packages:
        decision = package.get("decision")
        trace = package.get("witness_trace")
        clause_set = package.get("clause_set")
        review = package.get("review")
        if not isinstance(decision, list) or not decision:
            raise ValueError(f"{package.get('case_id')} has no gold decision")
        if not isinstance(trace, list) or [item.get("claim") for item in trace] != decision:
            raise ValueError(f"{package.get('case_id')} has an invalid witness trace")
        if not isinstance(clause_set, list):
            raise ValueError(f"{package.get('case_id')} has no clause set")
        if not isinstance(review, dict) or review.get("status") not in {
            "pending",
            "approved",
            "rejected",
        }:
            raise ValueError(f"{package.get('case_id')} has an invalid review status")
        if review["status"] in {"approved", "rejected"} and not (
            review.get("reviewer") and review.get("reviewed_at")
        ):
            raise ValueError(
                f"{package.get('case_id')} approved/rejected review requires reviewer and reviewed_at"
            )


def write_gold_packages(path: str | Path, questions: list[dict]) -> dict:
    document = build_gold_packages(questions)
    validate_gold_packages(document, questions)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return document


def load_gold_packages(path: str | Path | None = None) -> dict:
    package_path = Path(path) if path is not None else default_gold_package_path()
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gold package document must be a JSON object")
    return payload


def main() -> None:
    questions = load_benchmark_questions()
    output_path = default_gold_package_path()
    document = write_gold_packages(output_path, questions)
    print(f"Built {len(document['packages'])} pending-review gold packages")


if __name__ == "__main__":
    main()
