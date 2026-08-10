import json

import pytest

from app.services.evaluation import live_demo_gate
from app.services.evaluation.live_demo_eval import LIVE_DEMO_CASE_IDS


def _document():
    return {
        "schema_version": 1,
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "prompt_version": "demo-grounded-v1",
        "cases": [
            {
                "case_id": case_id,
                "response": f"Grounded response for {case_id} [1].",
                "evidence_ids": ["source-1#page=1"],
                "latency_ms": 10,
                "usage": {},
                "error": None,
            }
            for case_id in LIVE_DEMO_CASE_IDS
        ],
    }


def _summary(faithfulness=0.5, unsupported=0.1):
    return {
        "metric_errors": [],
        "rows": [
            {
                "id": case_id,
                "faithfulness": faithfulness,
                "unsupported_claim_rate": unsupported,
            }
            for case_id in LIVE_DEMO_CASE_IDS
        ],
    }


def test_live_demo_gate_passes_only_complete_measured_deepseek_run():
    report = live_demo_gate.evaluate_live_demo_gate(_document(), _summary())

    assert report["passed"] is True
    assert report["actual"]["response_count"] == 12
    assert report["actual"]["faithfulness_measured_rows"] == 12
    assert report["actual"]["avg_faithfulness"] == 0.5
    assert report["actual"]["avg_unsupported_claim_rate"] == 0.1


@pytest.mark.parametrize(
    ("mutation", "failure_metric"),
    [
        (lambda document, summary: document.update(model="deepseek-chat"), "runtime_model"),
        (lambda document, summary: document["cases"][0].update(response=""), "response_count"),
        (lambda document, summary: document["cases"][0].update(error="timeout"), "live_errors"),
        (lambda document, summary: summary["rows"][0].update(faithfulness=None), "faithfulness_measured_rows"),
        (
            lambda document, summary: summary.update(
                rows=[{**row, "faithfulness": 0.0} for row in summary["rows"]]
            ),
            "avg_faithfulness",
        ),
        (lambda document, summary: summary["rows"][0].update(unsupported_claim_rate=0.2), "avg_unsupported_claim_rate"),
    ],
)
def test_live_demo_gate_fails_closed(mutation, failure_metric):
    document = _document()
    summary = _summary()
    mutation(document, summary)

    report = live_demo_gate.evaluate_live_demo_gate(document, summary)

    assert report["passed"] is False
    assert failure_metric in {failure["metric"] for failure in report["failures"]}


def test_run_gate_rejects_missing_key_before_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        live_demo_gate,
        "deepseek_runtime_status",
        lambda: {"configured": False, "provider": "deepseek", "model": "deepseek-v4-flash", "reason": "missing"},
    )
    called = False

    def should_not_capture(output_dir):
        nonlocal called
        called = True

    monkeypatch.setattr(live_demo_gate, "capture_live_responses", should_not_capture)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        live_demo_gate.run_live_demo_gate(tmp_path)
    assert called is False


def test_run_gate_writes_report_before_raising_on_quality_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        live_demo_gate,
        "deepseek_runtime_status",
        lambda: {"configured": True, "provider": "deepseek", "model": "deepseek-v4-flash", "reason": "configured"},
    )
    monkeypatch.setattr(live_demo_gate, "capture_live_responses", lambda output_dir: _document())
    monkeypatch.setattr(live_demo_gate, "run_eval", lambda response_provider=None: _summary(faithfulness=0.0))

    with pytest.raises(RuntimeError, match="live DeepSeek demo gate failed"):
        live_demo_gate.run_live_demo_gate(tmp_path)

    report_path = tmp_path / "deepseek-v4-flash-live-gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
