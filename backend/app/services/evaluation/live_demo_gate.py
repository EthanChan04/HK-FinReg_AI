"""Blocking acceptance gate for the real DeepSeek V4 Flash demo run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.evaluation.live_demo_eval import (
    LIVE_DEMO_CASE_IDS,
    MODEL,
    PROVIDER,
    capture_live_responses,
    load_live_response_provider,
)
from app.services.evaluation.run_eval import run_eval
from app.services.llm.deepseek import deepseek_runtime_status


LIVE_GATE_THRESHOLDS = {
    "response_count": 12,
    "faithfulness_measured_rows": 12,
    "avg_faithfulness": 0.45,
    "avg_unsupported_claim_rate": 0.10,
}
GATE_REPORT_FILENAME = "deepseek-v4-flash-live-gate.json"


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def evaluate_live_demo_gate(document: dict, summary: dict) -> dict:
    """Evaluate the explicit runtime, capture completeness, and selected metrics."""

    raw_cases = document.get("cases", []) if isinstance(document, dict) else []
    cases = raw_cases if isinstance(raw_cases, list) else []
    cases_by_id = {
        str(case.get("case_id")): case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    expected_ids = set(LIVE_DEMO_CASE_IDS)
    valid_response_ids = {
        case_id
        for case_id, case in cases_by_id.items()
        if case_id in expected_ids
        and isinstance(case.get("response"), str)
        and bool(case["response"].strip())
    }
    live_errors = [
        {"case_id": case_id, "error": case.get("error")}
        for case_id, case in cases_by_id.items()
        if case_id in expected_ids and case.get("error")
    ]

    rows = summary.get("rows", []) if isinstance(summary, dict) else []
    selected_rows = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("id")) in expected_ids
    }
    faithfulness_values = [
        float(row["faithfulness"])
        for row in selected_rows.values()
        if row.get("faithfulness") is not None
    ]
    unsupported_values = [
        float(row["unsupported_claim_rate"])
        for row in selected_rows.values()
        if row.get("unsupported_claim_rate") is not None
    ]
    selected_metric_errors = [
        error
        for error in summary.get("metric_errors", [])
        if isinstance(error, dict) and str(error.get("question_id")) in expected_ids
    ]
    actual = {
        "provider": document.get("provider") if isinstance(document, dict) else None,
        "model": document.get("model") if isinstance(document, dict) else None,
        "response_count": len(valid_response_ids),
        "faithfulness_measured_rows": len(faithfulness_values),
        "avg_faithfulness": _average(faithfulness_values),
        "avg_unsupported_claim_rate": _average(unsupported_values),
        "live_error_count": len(live_errors),
        "selected_metric_error_count": len(selected_metric_errors),
    }
    failures = []

    def fail(metric: str, value, operator: str, threshold) -> None:
        failures.append(
            {"metric": metric, "actual": value, "operator": operator, "threshold": threshold}
        )

    if actual["provider"] != PROVIDER:
        fail("runtime_provider", actual["provider"], "==", PROVIDER)
    if actual["model"] != MODEL:
        fail("runtime_model", actual["model"], "==", MODEL)
    if set(cases_by_id) != expected_ids or len(cases) != len(LIVE_DEMO_CASE_IDS):
        fail("case_set", sorted(cases_by_id), "==", list(LIVE_DEMO_CASE_IDS))
    if actual["response_count"] != LIVE_GATE_THRESHOLDS["response_count"]:
        fail("response_count", actual["response_count"], "==", 12)
    if live_errors:
        fail("live_errors", live_errors, "==", [])
    if set(selected_rows) != expected_ids:
        fail("evaluated_case_set", sorted(selected_rows), "==", list(LIVE_DEMO_CASE_IDS))
    if actual["faithfulness_measured_rows"] != 12:
        fail("faithfulness_measured_rows", actual["faithfulness_measured_rows"], "==", 12)
    if actual["avg_faithfulness"] is None or actual["avg_faithfulness"] < 0.45:
        fail("avg_faithfulness", actual["avg_faithfulness"], ">=", 0.45)
    if (
        len(unsupported_values) != 12
        or actual["avg_unsupported_claim_rate"] is None
        or actual["avg_unsupported_claim_rate"] > 0.10
    ):
        fail(
            "avg_unsupported_claim_rate",
            actual["avg_unsupported_claim_rate"],
            "<=",
            0.10,
        )
    if selected_metric_errors:
        fail("selected_metric_errors", selected_metric_errors, "==", [])

    return {
        "schema_version": 1,
        "gate": "deepseek-v4-flash-live-demo",
        "passed": not failures,
        "thresholds": LIVE_GATE_THRESHOLDS,
        "actual": actual,
        "failures": failures,
        "selected_case_ids": list(LIVE_DEMO_CASE_IDS),
    }


def run_live_demo_gate(output_dir: str | Path) -> dict:
    """Capture, evaluate, write a gate report, and fail closed on any violation."""

    runtime = deepseek_runtime_status()
    if not runtime.get("configured"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for the live DeepSeek demo gate")
    if runtime.get("provider") != PROVIDER or runtime.get("model") != MODEL:
        raise RuntimeError(
            f"live demo requires explicit {PROVIDER}/{MODEL}, got "
            f"{runtime.get('provider')}/{runtime.get('model')}"
        )

    document = capture_live_responses(output_dir)
    provider = load_live_response_provider(document)
    summary = run_eval(response_provider=provider)
    report = evaluate_live_demo_gate(document, summary)
    destination = Path(output_dir) / GATE_REPORT_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise RuntimeError(f"live DeepSeek demo gate failed: {report['failures']}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/live"),
    )
    args = parser.parse_args(argv)
    report = run_live_demo_gate(args.output_dir)
    print(
        "Live DeepSeek demo gate passed: "
        f"responses={report['actual']['response_count']}, "
        f"faithfulness={report['actual']['avg_faithfulness']}, "
        f"unsupported={report['actual']['avg_unsupported_claim_rate']}"
    )


if __name__ == "__main__":
    main()
