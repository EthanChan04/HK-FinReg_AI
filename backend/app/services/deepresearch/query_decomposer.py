"""Sub-question decomposition for deep research queries."""

from __future__ import annotations


def decompose_query(
    query: str,
    max_sub_questions: int = 8,
) -> list[dict]:
    """Split a regulatory research query into focused sub-questions.

    Uses a deterministic rule-based approach (matching the pattern from
    ``planner.fallback_research_plan``) so it works without an LLM.
    Each returned dict contains ``id``, ``question``, ``retrieval_mode``,
    and ``required_topics``.
    """
    if not query or not query.strip():
        return []

    sub_questions = [
        {
            "id": "SQ1",
            "question": f"What HKMA requirements apply to: {query}",
            "retrieval_mode": "rag",
            "required_topics": ["HKMA"],
        },
        {
            "id": "SQ2",
            "question": f"What SFC or conduct obligations may be relevant to: {query}",
            "retrieval_mode": "kag",
            "required_topics": ["SFC", "conduct"],
        },
        {
            "id": "SQ3",
            "question": f"What AML/CFT and customer due diligence obligations apply to: {query}",
            "retrieval_mode": "rag",
            "required_topics": ["AML", "CDD"],
        },
        {
            "id": "SQ4",
            "question": f"What AI governance, privacy, and consumer protection risks exist for: {query}",
            "retrieval_mode": "kag",
            "required_topics": ["AI", "privacy", "consumer_protection"],
        },
        {
            "id": "SQ5",
            "question": f"What penalties or enforcement actions have been taken related to: {query}",
            "retrieval_mode": "rag",
            "required_topics": ["enforcement", "penalty"],
        },
        {
            "id": "SQ6",
            "question": f"What cross-border or jurisdictional considerations apply to: {query}",
            "retrieval_mode": "kag",
            "required_topics": ["cross_border", "jurisdiction"],
        },
    ]

    # Respect caller's limit (but always return at least 1 question)
    return sub_questions[:max(max_sub_questions, 1)]
