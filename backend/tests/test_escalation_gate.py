"""T3-01: PEA-CAE escalation gate tests (NR-01).

Behavioral contract:
  - escalate when gain >= gain_threshold AND cost <= cost_threshold
  - hold when coverage is already sufficient (diminishing returns)
  - hold when the full text is too expensive
  - hold when the unresolved gap is trivial
  - diagnostics always explain the decision for A/B reporting
"""

from __future__ import annotations

from app.services.deepresearch.escalation_gate import (
    escalation_cost,
    evidence_gain_score,
    should_escalate,
)


class TestEvidenceGainScore:
    def test_full_coverage_yields_zero_gain(self):
        assert evidence_gain_score(coverage=1.0, gap_ratio=0.0) == 0.0

    def test_low_coverage_with_large_gap_yields_high_gain(self):
        score = evidence_gain_score(coverage=0.2, gap_ratio=0.8)
        assert score >= 0.7

    def test_coverage_above_min_yields_zero_gain(self):
        score = evidence_gain_score(coverage=0.9, gap_ratio=0.5)
        assert score == 0.0

    def test_gain_is_bounded_to_one(self):
        assert evidence_gain_score(coverage=0.0, gap_ratio=1.0) <= 1.0


class TestEscalationCost:
    def test_cost_scales_with_full_text_size(self):
        small = escalation_cost(1000)
        large = escalation_cost(100_000)
        assert large > small

    def test_cost_never_below_retrieval_baseline(self):
        assert escalation_cost(0) >= 1.0


class TestShouldEscalate:
    def test_escalates_when_gain_high_and_cost_low(self):
        decision, diag = should_escalate(
            coverage=0.3, gap_ratio=0.8, full_text_chars=2000
        )
        assert decision is True
        assert diag["reason"].startswith("escalate")

    def test_holds_when_coverage_already_sufficient(self):
        decision, diag = should_escalate(
            coverage=0.9, gap_ratio=0.5, full_text_chars=2000
        )
        assert decision is False
        assert diag["reason"] == "hold: gain below threshold"

    def test_holds_when_full_text_too_expensive(self):
        decision, diag = should_escalate(
            coverage=0.1,
            gap_ratio=0.9,
            full_text_chars=500_000,
            cost_threshold=100.0,
        )
        assert decision is False
        assert diag["reason"] == "hold: full-text cost above threshold"

    def test_holds_when_gap_trivial(self):
        decision, diag = should_escalate(
            coverage=0.4, gap_ratio=0.05, full_text_chars=2000
        )
        assert decision is False
        assert diag["reason"] == "hold: gain below threshold"

    def test_diagnostics_are_ab_report_friendly(self):
        _, diag = should_escalate(
            coverage=0.3, gap_ratio=0.8, full_text_chars=2000
        )
        for key in ("gain", "cost", "coverage", "gap_ratio", "reason"):
            assert key in diag

    def test_cost_threshold_blocks_escalation(self):
        decision, _ = should_escalate(
            coverage=0.1,
            gap_ratio=0.9,
            full_text_chars=2000,
            cost_threshold=0.5,  # impossible: cost >= retrieval baseline
        )
        assert decision is False
