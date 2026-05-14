"""Run Obligation Mapper golden regression and enforce release thresholds."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.evidence import EvidenceChunk
from app.schemas.kag import ProductProfile
from app.services.evaluation.obligation_mapper_regression import evaluate_regression
from app.services.kag.graph_retriever import GraphRetriever
from app.services.kag.graph_store import NetworkXGraphStore
from app.services.kag.obligation_mapper import ObligationMapper


THRESHOLDS = {
    "regulator_coverage": 0.90,
    "obligation_coverage": 0.85,
    "evidence_support_rate": 0.90,
    "structured_output_validity": 1.00,
}


class _SyntheticRetrievalService:
    def retrieve(self, query: str, retrieval_mode: str = "kag", top_k: int = 8) -> list[EvidenceChunk]:
        del retrieval_mode, top_k
        return [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="synthetic_doc",
                page=1,
                text=f"Synthetic evidence for regression: {query}",
            )
        ]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run() -> dict:
    base = Path(__file__).resolve().parents[3] / "tests" / "regression" / "obligation_mapper"
    cases_path = base / "golden_cases.jsonl"
    expected_path = base / "golden_expected.jsonl"
    cases = _read_jsonl(cases_path)

    graph_store = NetworkXGraphStore(Path(__file__).resolve().parents[3] / "data" / "graph" / "regression_graph.json")
    graph_retriever = GraphRetriever(graph_store)
    mapper = ObligationMapper()
    retrieval = _SyntheticRetrievalService()

    actual_rows = []
    for case in cases:
        profile = ProductProfile(**(case.get("product_profile") or {}))
        result = mapper.map_obligations(case["query"], profile, graph_retriever, retrieval)
        actual_rows.append(
            {
                "case_id": case["case_id"],
                "applicable_regulators": result.applicable_regulators,
                "risk_types": result.risks,
                "obligations": [item.obligation for item in result.obligations],
                "evidence_chunks": [item for item in ("source_1",) if result.obligations],
            }
        )

    report = evaluate_regression(expected_path, actual_rows)
    metrics = report["metrics"]
    failures = []
    for key, threshold in THRESHOLDS.items():
        value = float(metrics.get(key, 0.0))
        if value < threshold:
            failures.append({"metric": key, "value": value, "threshold": threshold})

    output = {
        "metrics": metrics,
        "thresholds": THRESHOLDS,
        "failures": failures,
        "total_cases": len(cases),
    }

    out_path = base / "latest_regression_report.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> int:
    output = run()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if output["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

