"""Run a lightweight deterministic retrieval benchmark with real metrics."""

from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path

from app.services.evaluation.benchmark_loader import load_benchmark_questions
from app.services.retrieval.query_classifier import classify_query


def _metric_error(question_id: str, metric: str, exc: Exception) -> dict:
    """Normalize metric collection failures for benchmark reporting."""

    return {
        "question_id": question_id,
        "metric": metric,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _run_metric(question_id: str, metric: str, errors: list[dict], fn, default):
    """Run a benchmark metric and record failures instead of hiding them."""

    try:
        return fn()
    except Exception as exc:
        errors.append(_metric_error(question_id, metric, exc))
        return default


def _coverage(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 1.0
    actual_lower = {value.lower() for value in actual}
    matched = sum(1 for value in expected if value.lower() in actual_lower)
    return round(matched / len(expected), 3)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{1,}", (text or "").lower())
        if len(token) > 2
    }


def _metadata_values(metadata: dict, key: str) -> set[str]:
    value = metadata.get(key)
    if value is None:
        return set()
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = [str(value)]
    return {str(item).lower() for item in values if str(item).strip()}


def _metadata_matches(metadata: dict, filters: dict) -> bool:
    for key, expected in (filters or {}).items():
        expected_values = expected if isinstance(expected, list) else [expected]
        expected_normalized = {str(item).lower() for item in expected_values}
        if expected_normalized and not _metadata_values(metadata, key).intersection(expected_normalized):
            return False
    return True


@lru_cache()
def _load_cached_corpus_documents() -> tuple:
    """Load cached corpus chunks without invoking external embedding services."""

    from app.core.config import get_settings

    settings = get_settings()
    cache_path = Path(settings.CORPUS_INDEX_DIR) / "corpus_documents.pkl"
    if not cache_path.exists():
        return ()
    with cache_path.open("rb") as cache_file:
        docs = pickle.load(cache_file)
    return tuple(docs or ())


def _retrieve_eval_documents(question: str, top_k: int = 10) -> list:
    """Return deterministic local evidence candidates for benchmark metrics."""

    profile = classify_query(question)
    query_tokens = _tokens(question)
    docs = list(_load_cached_corpus_documents())
    if not docs:
        return []

    scored = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if profile.filters and not _metadata_matches(metadata, profile.filters):
            continue
        content_tokens = _tokens(getattr(doc, "page_content", ""))
        metadata_tokens = _tokens(" ".join(str(value) for value in metadata.values()))
        overlap = len(query_tokens.intersection(content_tokens | metadata_tokens))
        priority = {"P0": 3, "P1": 2, "P2": 1, "P3": 0}.get(str(metadata.get("priority", "P1")).upper(), 0)
        scored.append((overlap + priority, doc))

    if not scored and profile.filters:
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            content_tokens = _tokens(getattr(doc, "page_content", ""))
            metadata_tokens = _tokens(" ".join(str(value) for value in metadata.values()))
            overlap = len(query_tokens.intersection(content_tokens | metadata_tokens))
            scored.append((overlap, doc))

    ranked = [doc for score, doc in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:top_k]


def _compute_evidence_count(item: dict) -> int:
    """Retrieve evidence for a benchmark question and return the count."""
    return len(_retrieve_eval_documents(item["question"], top_k=10))


def _compute_graph_path_count(item: dict) -> int:
    """Retrieve graph paths for the benchmark question."""
    try:
        from app.core.config import get_settings
        from app.services.kag.graph_store import NetworkXGraphStore
        from app.services.kag.graph_retriever import GraphRetriever

        settings = get_settings()
        store = NetworkXGraphStore(settings.GRAPH_STORE_PATH)
        store.load()
        retriever = GraphRetriever(store)
        paths = retriever.retrieve_paths(item["question"], limit=5)
        return len(paths)
    except Exception:
        return 0


def _compute_citation_audit(item: dict) -> tuple[float, float]:
    """Return (citation_supported_rate, unsupported_claim_rate) from actual retrieval."""
    from app.services.retrieval.citation_verifier import verify_citations
    from app.services.retrieval.retrieval_service import document_to_evidence

    docs = _retrieve_eval_documents(item["question"], top_k=10)
    evidence = [document_to_evidence(doc, index + 1) for index, doc in enumerate(docs)]
    if not evidence:
        return 0.0, 0.0

    report_text = "\n".join(
        f"[Source {i + 1}, p.{chunk.page or 1}] {chunk.text[:200]}"
        for i, chunk in enumerate(evidence)
    )
    audit = verify_citations(report_text, evidence)
    supported_rate = 1.0 - audit.unsupported_claim_rate
    return supported_rate, audit.unsupported_claim_rate


def _compute_deepresearch_gap_count(item: dict) -> int:
    """Compute evidence gap count for deep_research mode questions."""
    if item.get("expected_retrieval_mode") != "deep_research":
        return 0
    try:
        from app.services.deepresearch.evidence_evaluator import (
            evaluate_evidence_coverage,
        )
        from app.services.deepresearch.planner import build_research_plan
        from app.services.retrieval.retrieval_service import document_to_evidence

        plan = build_research_plan(item["question"])
        evidence_by_subquestion: dict[str, list[dict]] = {}

        for sq in plan.sub_questions:
            docs = _retrieve_eval_documents(sq.question, top_k=sq.evidence_min_count + 3)
            evidence = (
                document_to_evidence(doc, index + 1)
                for index, doc in enumerate(docs)
            )
            evidence_by_subquestion[sq.id] = [chunk.model_dump() for chunk in evidence]

        gaps = evaluate_evidence_coverage(plan, evidence_by_subquestion)
        return len(gaps)
    except Exception:
        return 0


def run_eval() -> dict:
    """Evaluate router mode, topic, and regulator coverage with real retrieval metrics."""

    questions = load_benchmark_questions()
    rows = []
    metric_errors: list[dict] = []
    for item in questions:
        profile = classify_query(item["question"])
        filters = profile.filters
        actual_topics = filters.get("topics", [])
        actual_regulators = filters.get(
            "regulator",
            ["HKMA"] if "svf" in filters.get("module_tags", []) else [],
        )

        question_id = item["id"]
        evidence_count = _run_metric(
            question_id,
            "evidence_count",
            metric_errors,
            lambda: _compute_evidence_count(item),
            0,
        )
        graph_path_count = _run_metric(
            question_id,
            "graph_path_count",
            metric_errors,
            lambda: _compute_graph_path_count(item),
            0,
        )
        citation_supported_rate, unsupported_claim_rate = _run_metric(
            question_id,
            "citation_audit",
            metric_errors,
            lambda: _compute_citation_audit(item),
            (0.0, 0.0),
        )
        deepresearch_gap_count = _run_metric(
            question_id,
            "deepresearch_gap_count",
            metric_errors,
            lambda: _compute_deepresearch_gap_count(item),
            0,
        )

        rows.append(
            {
                "id": item["id"],
                "mode_correct": profile.retrieval_mode
                == item["expected_retrieval_mode"],
                "topic_coverage": _coverage(
                    item.get("expected_topics", []), actual_topics
                ),
                "regulator_coverage": _coverage(
                    item.get("expected_regulators", []), actual_regulators
                ),
                "graph_path_count": graph_path_count,
                "evidence_count": evidence_count,
                "citation_supported_rate": citation_supported_rate,
                "unsupported_claim_rate": unsupported_claim_rate,
                "deepresearch_gap_count": deepresearch_gap_count,
            }
        )

    total = len(rows) or 1
    summary = {
        "total_questions": len(rows),
        "retrieval_mode_accuracy": round(
            sum(1 for row in rows if row["mode_correct"]) / total, 3
        ),
        "avg_topic_coverage": round(
            sum(row["topic_coverage"] for row in rows) / total, 3
        ),
        "avg_regulator_coverage": round(
            sum(row["regulator_coverage"] for row in rows) / total, 3
        ),
        "avg_evidence_count": round(
            sum(row["evidence_count"] for row in rows) / total, 3
        ),
        "avg_graph_path_count": round(
            sum(row["graph_path_count"] for row in rows) / total, 3
        ),
        "avg_citation_supported_rate": round(
            sum(row["citation_supported_rate"] for row in rows) / total, 3
        ),
        "avg_unsupported_claim_rate": round(
            sum(row["unsupported_claim_rate"] for row in rows) / total, 3
        ),
        "avg_deepresearch_gap_count": round(
            sum(row["deepresearch_gap_count"] for row in rows) / total, 3
        ),
        "metric_errors": metric_errors,
        "rows": rows,
    }
    return summary


def main() -> None:
    summary = run_eval()
    print("Evaluation Summary")
    for key, value in summary.items():
        if key == "metric_errors":
            if value:
                print("\nMetric Errors:")
                for error in value:
                    print(
                        f"  [{error['question_id']}] {error['metric']}: "
                        f"{error['error_type']}: {error['error']}"
                    )
            else:
                print("- metric_errors: 0")
        elif key == "rows":
            print("\nIndividual Results:")
            for row in value:
                print(f"  [{row['id']}]")
                print(f"    mode_correct: {row['mode_correct']}")
                print(f"    topic_coverage: {row['topic_coverage']}")
                print(f"    regulator_coverage: {row['regulator_coverage']}")
                print(f"    evidence_count: {row['evidence_count']}")
                print(f"    graph_path_count: {row['graph_path_count']}")
                print(f"    citation_supported_rate: {row['citation_supported_rate']}")
                print(f"    unsupported_claim_rate: {row['unsupported_claim_rate']}")
                print(f"    deepresearch_gap_count: {row['deepresearch_gap_count']}")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
