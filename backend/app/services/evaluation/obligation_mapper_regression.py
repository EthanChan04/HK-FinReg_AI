"""Golden-case regression helpers for Obligation Mapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "applicable_regulators",
    "risk_types",
    "obligations",
    "evidence_chunks",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _coverage(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 1.0
    return round(len(expected.intersection(actual)) / len(expected), 3)


def evaluate_regression(
    expected_path: str | Path,
    actual_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute core regression metrics for obligation mapping outputs."""

    expected_rows = _read_jsonl(expected_path)
    expected_by_id = {row["case_id"]: row for row in expected_rows}
    actual_by_id = {row["case_id"]: row for row in actual_rows}

    per_case: list[dict[str, Any]] = []
    regulator_scores: list[float] = []
    obligation_scores: list[float] = []
    evidence_support_hits = 0
    structured_valid_count = 0

    for case_id, expected in expected_by_id.items():
        actual = actual_by_id.get(case_id, {})
        structured_valid = REQUIRED_KEYS.issubset(actual.keys())
        structured_valid_count += 1 if structured_valid else 0

        expected_regs = set(expected.get("applicable_regulators", []))
        actual_regs = set(actual.get("applicable_regulators", []))
        reg_cov = _coverage(expected_regs, actual_regs)
        regulator_scores.append(reg_cov)

        expected_ob = set(expected.get("obligations", []))
        actual_ob = set(actual.get("obligations", []))
        ob_cov = _coverage(expected_ob, actual_ob)
        obligation_scores.append(ob_cov)

        has_evidence = bool(actual.get("evidence_chunks"))
        evidence_support_hits += 1 if has_evidence else 0

        per_case.append(
            {
                "case_id": case_id,
                "structured_valid": structured_valid,
                "regulator_coverage": reg_cov,
                "obligation_coverage": ob_cov,
                "expected": expected,
                "actual": actual,
            }
        )

    total = len(expected_rows) or 1
    metrics = {
        "regulator_coverage": round(sum(regulator_scores) / total, 3),
        "obligation_coverage": round(sum(obligation_scores) / total, 3),
        "evidence_support_rate": round(evidence_support_hits / total, 3),
        "structured_output_validity": round(structured_valid_count / total, 3),
    }
    return {"metrics": metrics, "per_case": per_case}

