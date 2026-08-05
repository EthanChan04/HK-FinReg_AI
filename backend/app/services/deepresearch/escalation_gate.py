"""T3-01: PEA-CAE escalation gate (NR-01, arXiv 2607.24791).

Progressive Evidence Acquisition with Cost-Aware Escalation, adapted as a
single-scenario research prototype: start with cheap high-precision
retrieval; escalate to full-text reading only when the expected evidence
gain justifies the added cost. This module is a standalone experiment --
the existing DeepResearch workflow is NOT modified.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Cost model: full-text reading is priced in "retrieval-call equivalents".
# Reading ~1k chars costs roughly one extra retrieval call (LLM context
# pricing), so a 20k-char regulation reads like ~20 retrievals.
CHARS_PER_RETRIEVAL = 1000.0


def evidence_gain_score(
    coverage: float,
    gap_ratio: float,
    *,
    min_coverage: float = 0.5,
    max_gap_ratio: float = 0.6,
) -> float:
    """Expected evidence gain from escalating to full-text reading.

    Gain is high when current coverage is below target AND the unresolved
    gap is substantial. Once coverage reaches ``min_coverage`` the gain is
    zero: further escalation yields diminishing returns and must not be
    justified by a residual gap alone.

    Returns a score in [0, 1].
    """
    if coverage >= min_coverage:
        return 0.0
    coverage_deficit = max(0.0, min_coverage - coverage) / min_coverage
    gap_term = min(1.0, gap_ratio / max_gap_ratio) if max_gap_ratio > 0 else 0.0
    return round(min(1.0, coverage_deficit * 0.6 + gap_term * 0.4), 3)


def escalation_cost(
    full_text_chars: int,
    *,
    retrieval_cost: float = 1.0,
    chars_per_retrieval: float = CHARS_PER_RETRIEVAL,
) -> float:
    """Cost of reading the full text, in retrieval-call equivalents."""
    return round(retrieval_cost + full_text_chars / chars_per_retrieval, 3)


def should_escalate(
    *,
    coverage: float,
    gap_ratio: float,
    full_text_chars: int,
    retrieval_cost: float = 1.0,
    min_coverage: float = 0.5,
    max_gap_ratio: float = 0.6,
    cost_threshold: float = 6.0,
    gain_threshold: float = 0.35,
) -> tuple[bool, dict]:
    """Decide whether to escalate to full-text reading.

    Escalation happens only when BOTH:
      - expected gain >= gain_threshold (quality side), AND
      - full-text cost <= cost_threshold (cost side).

    Returns ``(decision, diagnostics)`` where diagnostics explain the
    scoring for A/B reporting.
    """
    gain = evidence_gain_score(
        coverage, gap_ratio, min_coverage=min_coverage, max_gap_ratio=max_gap_ratio
    )
    cost = escalation_cost(full_text_chars, retrieval_cost=retrieval_cost)
    decision = gain >= gain_threshold and cost <= cost_threshold
    diagnostics = {
        "gain": gain,
        "cost": cost,
        "coverage": coverage,
        "gap_ratio": gap_ratio,
        "full_text_chars": full_text_chars,
        "gain_threshold": gain_threshold,
        "cost_threshold": cost_threshold,
        "reason": (
            "escalate: gain and cost within thresholds"
            if decision
            else (
                "hold: gain below threshold"
                if gain < gain_threshold
                else "hold: full-text cost above threshold"
            )
        ),
    }
    logger.debug("PEA-CAE escalation decision: %s", diagnostics)
    return decision, diagnostics
