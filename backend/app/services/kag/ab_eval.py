"""Small deterministic A/B harness for vector-like and dual-graph retrieval."""

from __future__ import annotations

import re

from app.services.kag.graph_retriever import GraphRetriever


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").casefold()))


def _recall(doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    expected = {doc.casefold() for doc in expected_doc_ids}
    if not expected:
        return 1.0
    return round(len({doc.casefold() for doc in doc_ids} & expected) / len(expected), 3)


def compare_graph_retrieval(
    retriever: GraphRetriever,
    query: str,
    *,
    expected_doc_ids: list[str],
    limit: int = 10,
) -> dict:
    """Compare title/metadata lexical retrieval with bounded dual-graph retrieval."""

    query_tokens = _tokens(query)
    baseline = []
    for node_id, attrs in retriever.store.graph.nodes(data=True):
        if attrs.get("node_type") not in {"RegulatoryDocument", "Document"}:
            continue
        searchable = " ".join(
            [str(attrs.get("title", "")), str(attrs.get("doc_id", node_id))]
            + [str(value) for key in ("topics", "module_tags") for value in attrs.get(key, [])]
        )
        if query_tokens & _tokens(searchable):
            baseline.append(node_id)
    baseline = baseline[:limit]

    dual_paths = retriever.retrieve_paths(query, limit=limit)
    dual_graph = []
    for path in dual_paths:
        for doc_id in path["matched_doc_ids"]:
            if doc_id not in dual_graph:
                dual_graph.append(doc_id)

    baseline_recall = _recall(baseline, expected_doc_ids)
    dual_graph_recall = _recall(dual_graph, expected_doc_ids)
    return {
        "baseline": baseline,
        "dual_graph": dual_graph,
        "baseline_recall": baseline_recall,
        "dual_graph_recall": dual_graph_recall,
        "recall_delta": round(dual_graph_recall - baseline_recall, 3),
    }
