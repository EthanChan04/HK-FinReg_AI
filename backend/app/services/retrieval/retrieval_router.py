"""Route queries across RAG, KAG, and DeepResearch retrieval paths."""

from __future__ import annotations

from app.schemas.evidence import EvidenceBundle
from app.services.retrieval.query_classifier import classify_query


def route_query(query: str):
    """Return the query profile without executing retrieval."""

    return classify_query(query)


def route_and_retrieve(
    query: str,
    retrieval_service,
    graph_retriever=None,
    allow_deep_research: bool = False,
    top_k: int = 5,
) -> EvidenceBundle:
    """Classify and retrieve a normalized evidence bundle."""

    profile = classify_query(query)
    mode = profile.retrieval_mode
    warnings: list[str] = []
    graph_paths: list[dict] = []

    if mode == "deep_research" and not allow_deep_research:
        warnings.append("DeepResearch requested but disabled for this path; falling back to KAG/RAG retrieval.")
        mode = "kag" if graph_retriever is not None else "rag"

    if mode == "kag" and graph_retriever is not None:
        graph_paths = graph_retriever.retrieve_paths(query, filters=profile.filters, limit=top_k)

    evidence = retrieval_service.retrieve(
        query=query,
        filters=profile.filters,
        retrieval_mode=mode,
        top_k=top_k,
    )

    return EvidenceBundle(
        query=query,
        retrieval_mode=mode,
        evidence_chunks=evidence,
        graph_paths=graph_paths,
        warnings=warnings,
    )
