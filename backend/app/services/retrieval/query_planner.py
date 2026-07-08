"""SIRA-style deterministic query planning for regulatory retrieval."""

from __future__ import annotations

import hashlib
from typing import Iterable

from pydantic import BaseModel, Field

from app.services.retrieval.query_classifier import QueryProfile
from app.services.retrieval.term_statistics import TermStatistics
from app.services.utils import pii_scrubber


class QueryPlan(BaseModel):
    raw_query: str
    scrubbed_query: str
    bm25_query: str
    dense_query: str
    expansion_terms: list[str] = Field(default_factory=list)
    rejected_terms: list[str] = Field(default_factory=list)
    filters: dict[str, list[str]] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    query_plan_id: str


_ALIASES: dict[str, list[str]] = {
    "svf": ["SVF", "stored value facility", "HKMA"],
    "cdd": ["CDD", "customer due diligence"],
    "kyc": ["KYC", "know your customer"],
    "aml": ["AML", "anti-money laundering"],
    "cft": ["CFT", "counter-financing of terrorism"],
    "ai": ["AI", "GenAI", "governance"],
    "genai": ["GenAI", "generative AI", "governance"],
    "wealth": ["wealth management", "suitability"],
    "advisory": ["wealth management", "suitability"],
    "advisor": ["wealth management", "suitability"],
    "investment": ["wealth management", "suitability"],
    "product": ["product launch"],
    "launch": ["product launch"],
    "launches": ["product launch"],
    "launching": ["product launch"],
    "wealth_management": ["wealth management"],
    "consumer_protection": ["consumer protection"],
    "suitability": ["suitability"],
    "personal_data": ["personal data", "PCPD"],
    "privacy": ["personal data", "PCPD"],
    "hkma": ["HKMA"],
    "sfc": ["SFC"],
    "pcpd": ["PCPD"],
    "requirement": ["required"],
    "requirements": ["required"],
}

_PROTECTED_TERMS = {
    "ai",
    "aml",
    "cdd",
    "cft",
    "consumer protection",
    "genai",
    "governance",
    "hkma",
    "kyc",
    "personal data",
    "pcpd",
    "sfc",
    "suitability",
    "svf",
    "wealth management",
    "stored value facility",
    "product launch",
}


def _query_plan_id(scrubbed_query: str, expansion_terms: Iterable[str]) -> str:
    payload = scrubbed_query + "|" + "|".join(sorted(expansion_terms))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _metadata_terms(filters: dict[str, list[str]]) -> list[str]:
    terms: list[str] = []
    for values in (filters or {}).values():
        terms.extend(str(value) for value in values)
    return terms


def _candidate_terms(query: str, profile: QueryProfile) -> list[tuple[str, str]]:
    text = (query or "").lower()
    candidates: list[tuple[str, str]] = []

    for trigger, terms in _ALIASES.items():
        if trigger in text:
            candidates.extend((term, f"alias:{trigger}") for term in terms)

    for topic in profile.filters.get("topics", []):
        topic_l = topic.lower()
        if topic_l in _ALIASES:
            candidates.extend((term, f"filter_topic:{topic}") for term in _ALIASES[topic_l])

    for module_tag in profile.filters.get("module_tags", []):
        tag_l = module_tag.lower()
        if tag_l in _ALIASES:
            candidates.extend((term, f"filter_module:{module_tag}") for term in _ALIASES[tag_l])

    for regulator in profile.filters.get("regulator", []):
        regulator_l = regulator.lower()
        if regulator_l in _ALIASES:
            candidates.extend((term, f"filter_regulator:{regulator}") for term in _ALIASES[regulator_l])

    return candidates


def build_query_plan(
    query: str,
    *,
    profile: QueryProfile,
    term_statistics: TermStatistics | None = None,
) -> QueryPlan:
    """Build a deterministic, auditable retrieval query plan."""

    stats = term_statistics or TermStatistics()
    scrubbed_query = pii_scrubber((query or "").strip())
    metadata_terms = _metadata_terms(profile.filters)
    expansions: list[str] = []
    rejected: list[str] = []
    reasons: list[str] = []

    for term, reason in _candidate_terms(scrubbed_query, profile):
        protected = term.lower() in _PROTECTED_TERMS
        if stats.is_allowed(term, query=scrubbed_query, metadata_terms=metadata_terms, protected=protected):
            if term not in expansions:
                expansions.append(term)
                reasons.append(reason)
        elif term not in rejected:
            rejected.append(term)

    bm25_parts = [scrubbed_query, *expansions]
    return QueryPlan(
        raw_query=query,
        scrubbed_query=scrubbed_query,
        bm25_query=" ".join(part for part in bm25_parts if part).strip(),
        dense_query=scrubbed_query,
        expansion_terms=expansions,
        rejected_terms=rejected,
        filters=profile.filters,
        reasons=reasons,
        query_plan_id=_query_plan_id(scrubbed_query, expansions),
    )
