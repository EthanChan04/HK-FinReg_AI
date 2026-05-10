"""Run a lightweight deterministic retrieval benchmark with real metrics."""

from __future__ import annotations


from app.services.evaluation.benchmark_loader import load_benchmark_questions
from app.services.retrieval.query_classifier import classify_query


def _coverage(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 1.0
    actual_lower = {value.lower() for value in actual}
    matched = sum(1 for value in expected if value.lower() in actual_lower)
    return round(matched / len(expected), 3)


def _compute_evidence_count(item: dict) -> int:
    """Retrieve evidence for a benchmark question and return the count."""
    try:
        from app.services.agents.builder import build_reranked_retriever
        from app.services.retrieval.retrieval_service import RetrievalService

        base_retriever = build_reranked_retriever()
        if base_retriever is None:
            return 0
        retrieval_service = RetrievalService(retriever=base_retriever)
        profile = classify_query(item["question"])
        evidence = retrieval_service.retrieve(
            query=item["question"],
            filters=profile.filters,
            retrieval_mode=profile.retrieval_mode,
            top_k=10,
        )
        return len(evidence)
    except Exception:
        return 0


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
    try:
        from app.services.agents.builder import build_reranked_retriever
        from app.services.retrieval.citation_verifier import verify_citations
        from app.services.retrieval.retrieval_service import RetrievalService

        base_retriever = build_reranked_retriever()
        if base_retriever is None:
            return 0.0, 0.0
        retrieval_service = RetrievalService(retriever=base_retriever)
        profile = classify_query(item["question"])
        evidence = retrieval_service.retrieve(
            query=item["question"],
            filters=profile.filters,
            retrieval_mode=profile.retrieval_mode,
            top_k=10,
        )
        if not evidence:
            return 0.0, 0.0

        report_text = "\n".join(
            f"[Source {i + 1}, p.{chunk.page or 1}] {chunk.text[:200]}"
            for i, chunk in enumerate(evidence)
        )
        audit = verify_citations(report_text, evidence)
        supported_rate = 1.0 - audit.unsupported_claim_rate
        return supported_rate, audit.unsupported_claim_rate
    except Exception:
        return 0.0, 0.0


def _compute_deepresearch_gap_count(item: dict) -> int:
    """Compute evidence gap count for deep_research mode questions."""
    if item.get("expected_retrieval_mode") != "deep_research":
        return 0
    try:
        from app.services.agents.builder import build_reranked_retriever
        from app.services.deepresearch.evidence_evaluator import (
            evaluate_evidence_coverage,
        )
        from app.services.deepresearch.planner import build_research_plan
        from app.services.retrieval.retrieval_service import RetrievalService

        plan = build_research_plan(item["question"])

        base_retriever = build_reranked_retriever()
        if base_retriever is None:
            return len(plan.sub_questions)

        retrieval_service = RetrievalService(retriever=base_retriever)
        evidence_by_subquestion: dict[str, list[dict]] = {}

        for sq in plan.sub_questions:
            sq_profile = classify_query(sq.question)
            top_k = sq.evidence_min_count + 3
            evidence = retrieval_service.retrieve(
                query=sq.question,
                filters=sq_profile.filters,
                retrieval_mode=sq.retrieval_mode,
                top_k=top_k,
            )
            evidence_by_subquestion[sq.id] = (
                [chunk.model_dump() for chunk in evidence] if evidence else []
            )

        gaps = evaluate_evidence_coverage(plan, evidence_by_subquestion)
        return len(gaps)
    except Exception:
        return 0


def run_eval() -> dict:
    """Evaluate router mode, topic, and regulator coverage with real retrieval metrics."""

    questions = load_benchmark_questions()
    rows = []
    for item in questions:
        profile = classify_query(item["question"])
        filters = profile.filters
        actual_topics = filters.get("topics", [])
        actual_regulators = filters.get(
            "regulator",
            ["HKMA"] if "svf" in filters.get("module_tags", []) else [],
        )

        evidence_count = _compute_evidence_count(item)
        graph_path_count = _compute_graph_path_count(item)
        citation_supported_rate, unsupported_claim_rate = _compute_citation_audit(
            item
        )
        deepresearch_gap_count = _compute_deepresearch_gap_count(item)

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
        "rows": rows,
    }
    return summary


def main() -> None:
    summary = run_eval()
    print("Evaluation Summary")
    for key, value in summary.items():
        if key == "rows":
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
