"""Route queries across RAG, KAG, and DeepResearch retrieval paths."""

from __future__ import annotations

from app.schemas.evidence import EvidenceBundle
from app.core.config import get_settings
from app.services.retrieval.query_classifier import classify_query
from app.services.retrieval.query_planner import build_query_plan
from app.services.retrieval.strategy_memory import StrategyExperienceStore
from app.services.retrieval.strategy_router import query_traits, select_retrieval_strategy
from app.services.retrieval.term_statistics import load_term_statistics


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
    settings = get_settings()
    query_plan = None
    strategy = None

    if getattr(settings, "SIRA_QUERY_PLANNER_ENABLED", False):
        stats = load_term_statistics(getattr(settings, "SIRA_TERM_STATS_PATH", "")) if getattr(settings, "SIRA_TERM_STATS_PATH", "") else None
        query_plan = build_query_plan(query, profile=profile, term_statistics=stats)

    if getattr(settings, "EXPERIENCE_RAG_ENABLED", False):
        experiences = []
        memory_path = getattr(settings, "EXPERIENCE_RAG_MEMORY_PATH", "")
        if memory_path:
            store = StrategyExperienceStore(memory_path, max_records=getattr(settings, "EXPERIENCE_RAG_MAX_RECORDS", 1000))
            traits_plan = query_plan or build_query_plan(query, profile=profile)
            experiences = store.find_similar(query_traits(profile, traits_plan), limit=5)
            strategy = select_retrieval_strategy(profile, traits_plan, experiences=experiences)
        else:
            strategy = select_retrieval_strategy(profile, query_plan or build_query_plan(query, profile=profile))

    if strategy is not None:
        mode = strategy.retrieval_mode
        top_k = strategy.top_k

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
        query_plan=query_plan,
        strategy=strategy,
    )

    for chunk in evidence:
        if query_plan is not None:
            chunk.metadata["query_plan"] = query_plan.model_dump()
        if strategy is not None:
            chunk.metadata["retrieval_strategy"] = strategy.model_dump()

    return EvidenceBundle(
        query=query,
        retrieval_mode=mode,
        evidence_chunks=evidence,
        graph_paths=graph_paths,
        warnings=warnings,
        query_plan=query_plan.model_dump() if query_plan is not None else None,
        retrieval_strategy=strategy.model_dump() if strategy is not None else None,
    )
