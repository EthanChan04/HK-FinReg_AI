from pathlib import Path


def test_obligation_mapper_regression_metrics_shape():
    from app.services.evaluation.obligation_mapper_regression import evaluate_regression

    expected_path = (
        Path(__file__).parent / "regression" / "obligation_mapper" / "golden_expected.jsonl"
    )

    actual_rows = [
        {
            "case_id": "L1-001",
            "applicable_regulators": ["HKMA"],
            "risk_types": ["AML/CFT"],
            "obligations": ["CDD"],
            "evidence_chunks": ["source_1"],
        }
    ]
    result = evaluate_regression(expected_path=expected_path, actual_rows=actual_rows)

    assert "metrics" in result
    assert "per_case" in result
    assert set(result["metrics"].keys()) == {
        "regulator_coverage",
        "obligation_coverage",
        "evidence_support_rate",
        "structured_output_validity",
    }
    assert len(result["per_case"]) == 20

