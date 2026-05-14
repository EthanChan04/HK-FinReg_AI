def test_obligation_mapper_release_gate_thresholds_pass():
    from app.services.evaluation.run_obligation_mapper_regression import run

    report = run()
    assert report["failures"] == []
    assert report["metrics"]["regulator_coverage"] >= 0.9
    assert report["metrics"]["obligation_coverage"] >= 0.85
    assert report["metrics"]["evidence_support_rate"] >= 0.9
    assert report["metrics"]["structured_output_validity"] >= 1.0

