"""Experience-RAG retrieval strategy selection."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.services.retrieval.query_classifier import QueryProfile
from app.services.retrieval.query_planner import QueryPlan
from app.services.retrieval.strategy_memory import StrategyExperience


class RetrievalStrategy(BaseModel):
    strategy_id: str
    retrieval_mode: str
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    top_k: int = 6
    use_rerank: bool = True
    use_graph: bool = False
    allow_deep_research: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    memory_hit: bool = False


def query_traits(profile: QueryProfile, plan: QueryPlan) -> list[str]:
    traits = {profile.retrieval_mode}
    traits.update(getattr(profile, "reasons", []) or [])
    for values in (getattr(profile, "filters", {}) or {}).values():
        traits.update(str(value).lower() for value in values)
    text = plan.scrubbed_query.lower()
    if re.search(r"\b(cdd|kyc|aml|cft)\b", text):
        traits.update({"aml", "cdd"})
    if re.search(r"\b(ai|genai|governance)\b", text):
        traits.update({"ai_governance"})
    if re.search(r"\b(section|paragraph|chapter|clause)\b", text):
        traits.add("clause_lookup")
    return sorted(trait for trait in traits if trait)


def _from_experience(experience: StrategyExperience) -> RetrievalStrategy:
    return RetrievalStrategy(
        strategy_id=experience.strategy_id,
        retrieval_mode=experience.retrieval_mode,
        bm25_weight=experience.bm25_weight,
        dense_weight=experience.dense_weight,
        top_k=experience.top_k,
        use_rerank=True,
        use_graph=experience.retrieval_mode == "kag",
        allow_deep_research=experience.retrieval_mode == "deep_research",
        reason_codes=["memory_preferred"],
        memory_hit=True,
    )


def select_retrieval_strategy(
    profile: QueryProfile,
    plan: QueryPlan,
    *,
    experiences: list[StrategyExperience] | None = None,
) -> RetrievalStrategy:
    """Select a deterministic retrieval recipe, optionally using prior experience."""

    profile_reasons = getattr(profile, "reasons", []) or []
    candidates = [
        experience
        for experience in (experiences or [])
        if experience.quality_score >= 0.75 and set(experience.query_traits).intersection(query_traits(profile, plan))
    ]
    if candidates:
        candidates.sort(key=lambda item: item.quality_score, reverse=True)
        return _from_experience(candidates[0])

    traits = set(query_traits(profile, plan))
    if "ai_governance" in traits or profile.retrieval_mode == "kag":
        return RetrievalStrategy(
            strategy_id="ai_governance_kag",
            retrieval_mode="kag",
            bm25_weight=0.35,
            dense_weight=0.65,
            top_k=6,
            use_graph=True,
            reason_codes=["ai_governance", *profile_reasons],
        )

    if profile.retrieval_mode == "deep_research":
        return RetrievalStrategy(
            strategy_id="cross_regulator_deepresearch",
            retrieval_mode="deep_research",
            bm25_weight=0.4,
            dense_weight=0.6,
            top_k=8,
            allow_deep_research=True,
            reason_codes=["deep_research", *profile_reasons],
        )

    if "clause_lookup" in traits:
        return RetrievalStrategy(
            strategy_id="clause_lookup_sparse_heavy",
            retrieval_mode="rag",
            bm25_weight=0.75,
            dense_weight=0.25,
            top_k=8,
            reason_codes=["clause_lookup", *profile_reasons],
        )

    if {"aml", "cdd"}.intersection(traits):
        return RetrievalStrategy(
            strategy_id="aml_kyc_balanced_rerank",
            retrieval_mode="rag",
            bm25_weight=0.45,
            dense_weight=0.55,
            top_k=6,
            reason_codes=["aml_kyc", *profile_reasons],
        )

    return RetrievalStrategy(
        strategy_id="default_hybrid",
        retrieval_mode=profile.retrieval_mode,
        bm25_weight=0.4,
        dense_weight=0.6,
        top_k=6,
        reason_codes=["default", *profile_reasons],
    )
