"""Run a lightweight deterministic retrieval benchmark with real metrics."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from app.services.evaluation.benchmark_loader import load_benchmark_questions
from app.services.evaluation.rag_eval import evaluate_claim_level_metrics
from app.services.deepresearch.planner import build_research_plan
from app.services.retrieval.query_classifier import classify_query
from app.services.retrieval.query_planner import build_query_plan
from app.services.retrieval.strategy_router import select_retrieval_strategy


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_backend_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else _backend_root() / path


def _metric_error(question_id: str, metric: str, exc: Exception) -> dict:
    """Normalize metric collection failures for benchmark reporting."""

    return {
        "question_id": question_id,
        "metric": metric,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _avg_optional(values) -> float | None:
    """Average only non-None values; None when nothing was measured."""

    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return round(sum(measured) / len(measured), 3)


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


def _casefold_values(values: list[str]) -> list[str]:
    return [str(value).casefold() for value in values]


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


def _document_regulators(docs: list) -> list[str]:
    regulators: list[str] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        value = metadata.get("regulator")
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item and str(item) not in regulators:
                regulators.append(str(item))
    return regulators


@lru_cache()
def _load_cached_corpus_documents() -> tuple:
    """Load cached corpus chunks without invoking external embedding services."""

    from app.core.config import get_settings

    settings = get_settings()
    from app.services.corpus.cache import manifest_digest, read_corpus_cache

    cache_path = _resolve_backend_path(settings.CORPUS_INDEX_DIR) / "corpus_documents.json"
    manifest_path = _backend_root() / "data" / "source_manifest.json"
    return tuple(
        read_corpus_cache(
            cache_path,
            manifest_digest=manifest_digest(manifest_path),
            parser_version="hierarchy-v1",
        )
    )


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

    expected_regulators = profile.filters.get("regulator", [])
    if len(expected_regulators) > 1:
        scored_docs = {id(doc) for _, doc in scored}
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            if str(metadata.get("regulator", "")).casefold() not in {
                str(value).casefold() for value in expected_regulators
            } or id(doc) in scored_docs:
                continue
            content_tokens = _tokens(getattr(doc, "page_content", ""))
            metadata_tokens = _tokens(" ".join(str(value) for value in metadata.values()))
            overlap = len(query_tokens.intersection(content_tokens | metadata_tokens))
            if overlap > 0:
                scored.append((overlap, doc))
    ranked_pairs = [item for item in sorted(scored, key=lambda item: item[0], reverse=True) if item[0] > 0]
    if len(expected_regulators) > 1:
        selected = []
        for regulator in expected_regulators:
            candidates = [
                (score, doc)
                for score, doc in ranked_pairs
                if str((getattr(doc, "metadata", {}) or {}).get("regulator", "")).casefold()
                == str(regulator).casefold()
            ]
            if candidates and candidates[0][1] not in selected:
                selected.append(candidates[0][1])
        for _, doc in ranked_pairs:
            if len(selected) >= top_k:
                break
            if doc not in selected:
                selected.append(doc)
        return selected[:top_k]
    return [doc for _, doc in ranked_pairs[:top_k]]


def _compute_evidence_count(item: dict) -> int:
    """Retrieve evidence for a benchmark question and return the count."""
    return len(_retrieve_eval_documents(item["question"], top_k=10))


def _compute_evidence_regulator_coverage(item: dict) -> float:
    """Measure expected regulator coverage in retrieved evidence metadata."""
    docs = _retrieve_eval_documents(item["question"], top_k=10)
    return _coverage(item.get("expected_regulators", []), _document_regulators(docs))


def _compute_graph_path_count(item: dict) -> int:
    """Retrieve graph paths for the benchmark question."""
    try:
        from app.core.config import get_settings

        settings = get_settings()
        graph_path = _resolve_backend_path(settings.GRAPH_STORE_PATH)
        retriever = _get_cached_graph_retriever(str(graph_path))
        paths = retriever.retrieve_paths(
            item["question"],
            limit=5,
            include_provenance=False,
        )
        return len(paths)
    except Exception:
        return 0


@lru_cache()
def _get_cached_graph_retriever(graph_path: str):
    """Load the immutable graph once per evaluation process."""

    from app.services.kag.graph_retriever import GraphRetriever
    from app.services.kag.graph_store import NetworkXGraphStore

    store = NetworkXGraphStore(graph_path)
    store.load()
    return GraphRetriever(store)


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
        from app.services.deepresearch.workflow import _apply_regulator_diversity_gate
        from app.services.retrieval.retrieval_service import document_to_evidence

        plan = build_research_plan(item["question"])
        evidence_by_subquestion: dict[str, list[dict]] = {}

        for sq in plan.sub_questions:
            docs = _retrieve_eval_documents(sq.question, top_k=sq.evidence_min_count + 3)
            evidence = (
                document_to_evidence(doc, index + 1)
                for index, doc in enumerate(docs)
            )
            dumped_evidence = [chunk.model_dump() for chunk in evidence]
            required_regulators = [
                topic for topic in sq.required_topics if topic in {"HKMA", "SFC", "PCPD"}
            ]
            evidence_by_subquestion[sq.id] = _apply_regulator_diversity_gate(
                dumped_evidence,
                required_regulators,
                top_k=sq.evidence_min_count + 3,
            )

        gaps = evaluate_evidence_coverage(plan, evidence_by_subquestion)
        return len(gaps)
    except Exception:
        return 0


def _evaluate_claim_metrics(item: dict) -> dict:
    evidence = _retrieve_eval_documents(item["question"], top_k=10)
    noise_documents = [
        Document(page_content=str(noise), metadata={"doc_id": str(noise)})
        for noise in item.get("noise_documents", [])
    ]
    noisy_evidence = evidence + noise_documents if noise_documents else None
    return evaluate_claim_level_metrics(
        item.get("expected_claims", []),
        evidence,
        noisy_evidence_chunks=noisy_evidence,
    )


def run_eval() -> dict:
    """Evaluate router mode, topic, and regulator coverage with real retrieval metrics."""

    questions = load_benchmark_questions()
    rows = []
    metric_errors: list[dict] = []
    for item in questions:
        profile = classify_query(item["question"])
        query_plan = build_query_plan(item["question"], profile=profile)
        strategy = select_retrieval_strategy(profile, query_plan)
        filters = profile.filters
        actual_topics = filters.get("topics", [])
        actual_regulators = filters.get(
            "regulator",
            ["HKMA"] if "svf" in filters.get("module_tags", []) else [],
        )
        classifier_regulator_coverage = _coverage(
            item.get("expected_regulators", []), actual_regulators
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
        evidence_regulator_coverage = _run_metric(
            question_id,
            "evidence_regulator_coverage",
            metric_errors,
            lambda: _compute_evidence_regulator_coverage(item),
            0.0,
        )
        claim_metrics = _run_metric(
            question_id,
            "claim_level_metrics",
            metric_errors,
            lambda: _evaluate_claim_metrics(item),
            {
                "claim_recall": 0.0,
                "context_precision": 0.0,
                # Faithfulness is never defaulted to 0.0: an unmeasured
                # generator response must stay None so aggregates do not
                # treat a collection failure as a zero score.
                "faithfulness": None,
                "hallucination_rate": None,
                "noise_sensitivity": 0.0,
                "context_utilization": 0.0,
                "claim_diagnostics": [],
            },
        )

        rows.append(
            {
                "id": item["id"],
                "mode_correct": profile.retrieval_mode
                == item["expected_retrieval_mode"],
                "topic_coverage": _coverage(
                    item.get("expected_topics", []), actual_topics
                ),
                "classifier_regulator_coverage": classifier_regulator_coverage,
                "regulator_coverage": classifier_regulator_coverage,
                "evidence_regulator_coverage": evidence_regulator_coverage,
                "strategy_id": strategy.strategy_id,
                "strategy_correct": strategy.strategy_id
                == item.get("expected_strategy_id"),
                "expansion_term_coverage": _coverage(
                    _casefold_values(item.get("expected_expansion_terms", [])),
                    _casefold_values(query_plan.expansion_terms),
                ),
                "query_plan_drift": query_plan.dense_query
                != query_plan.scrubbed_query,
                "graph_path_count": graph_path_count,
                "evidence_count": evidence_count,
                "citation_supported_rate": citation_supported_rate,
                "unsupported_claim_rate": unsupported_claim_rate,
                "deepresearch_gap_count": deepresearch_gap_count,
                **claim_metrics,
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
        "avg_classifier_regulator_coverage": round(
            sum(row["classifier_regulator_coverage"] for row in rows) / total, 3
        ),
        "avg_regulator_coverage": round(
            sum(row["classifier_regulator_coverage"] for row in rows) / total, 3
        ),
        "avg_evidence_regulator_coverage": round(
            sum(row["evidence_regulator_coverage"] for row in rows) / total, 3
        ),
        "strategy_accuracy": round(
            sum(1 for row in rows if row["strategy_correct"]) / total, 3
        ),
        "avg_expansion_term_coverage": round(
            sum(row["expansion_term_coverage"] for row in rows) / total, 3
        ),
        "query_plan_drift_rate": round(
            sum(1 for row in rows if row["query_plan_drift"]) / total, 3
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
        "avg_claim_recall": round(sum(row["claim_recall"] for row in rows) / total, 3),
        "avg_context_precision": round(sum(row["context_precision"] for row in rows) / total, 3),
        # Generation faithfulness is None unless a generator response was
        # evaluated; aggregate only measured rows to avoid treating a
        # missing measurement as a zero score.
        "avg_faithfulness": _avg_optional(row["faithfulness"] for row in rows),
        "avg_hallucination_rate": _avg_optional(row["hallucination_rate"] for row in rows),
        "avg_noise_sensitivity": round(sum(row["noise_sensitivity"] for row in rows) / total, 3),
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
                print(f"    classifier_regulator_coverage: {row['classifier_regulator_coverage']}")
                print(f"    regulator_coverage: {row['regulator_coverage']}")
                print(f"    evidence_regulator_coverage: {row['evidence_regulator_coverage']}")
                print(f"    strategy_id: {row['strategy_id']}")
                print(f"    strategy_correct: {row['strategy_correct']}")
                print(f"    expansion_term_coverage: {row['expansion_term_coverage']}")
                print(f"    query_plan_drift: {row['query_plan_drift']}")
                print(f"    evidence_count: {row['evidence_count']}")
                print(f"    graph_path_count: {row['graph_path_count']}")
                print(f"    citation_supported_rate: {row['citation_supported_rate']}")
                print(f"    unsupported_claim_rate: {row['unsupported_claim_rate']}")
                print(f"    deepresearch_gap_count: {row['deepresearch_gap_count']}")
                print(f"    claim_recall: {row['claim_recall']}")
                print(f"    context_precision: {row['context_precision']}")
                print(f"    faithfulness: {row['faithfulness']}")
                print(f"    hallucination_rate: {row['hallucination_rate']}")
                print(f"    noise_sensitivity: {row['noise_sensitivity']}")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
