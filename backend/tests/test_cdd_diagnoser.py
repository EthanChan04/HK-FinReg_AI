"""T3-02: CDD conflict diagnosis tests (NR-02).

Behavioral contract:
  - conflict detected when prior knowledge supports a claim but context does not
  - no conflict when context and prior agree
  - detection/false-positive rates computed against ground truth
  - report always explains each diagnosis (diagnostic tool, not a gate)
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.services.retrieval.cdd_diagnoser import diagnose_conflicts


def _ctx(text: str) -> Document:
    return Document(page_content=text, metadata={"doc_id": "ctx"})


def _prior(text: str) -> Document:
    return Document(page_content=text, metadata={"doc_id": "prior"})


class TestDiagnoseConflicts:
    def test_detects_conflict_when_prior_supports_but_context_does_not(self):
        context = [_ctx("The current guideline requires a licence from the HKMA.")]
        prior = [_prior("The outdated guideline required a licence from the SFC.")]
        claims = ["The licence must be obtained from the SFC."]

        report = diagnose_conflicts(claims, context, prior)

        assert report.diagnoses[0].conflict is True
        assert report.diagnoses[0].context_supported is False
        assert report.diagnoses[0].prior_supported is True

    def test_no_conflict_when_context_and_prior_agree(self):
        context = [_ctx("CDD must be performed before onboarding.")]
        prior = [_prior("CDD must be performed before onboarding.")]
        claims = ["CDD must be performed before onboarding."]

        report = diagnose_conflicts(claims, context, prior)

        assert report.diagnoses[0].conflict is False
        assert report.diagnoses[0].context_supported is True
        assert report.diagnoses[0].prior_supported is True

    def test_no_conflict_when_neither_supports(self):
        context = [_ctx("Transaction monitoring requirements.")]
        prior = [_prior("Capital adequacy requirements.")]
        claims = ["The firm must file annual returns."]

        report = diagnose_conflicts(claims, context, prior)

        assert report.diagnoses[0].conflict is False

    def test_detection_and_false_positive_rates(self):
        context = [_ctx("HKMA requires a licence.")]
        prior = [_prior("SFC requires a licence.")]
        claims = [
            "A licence from the SFC is required.",  # true conflict
            "A licence from the HKMA is required.",  # no conflict
        ]

        report = diagnose_conflicts(
            claims,
            context,
            prior,
            conflicting_claims=["A licence from the SFC is required."],
        )

        assert report.conflict_detection_rate == 1.0
        assert report.false_positive_rate == 0.0
        assert report.total_claims == 2

    def test_detection_rate_misses_undetected_conflict(self):
        context = [_ctx("HKMA requires a licence.")]
        prior = [_prior("SFC requires a licence.")]
        claims = ["A licence from the SFC is required."]

        # No ground-truth claim provided -> rates stay 0, diagnosis still works
        report = diagnose_conflicts(claims, context, prior)
        assert report.diagnoses[0].conflict is True
        assert report.conflict_detection_rate == 0.0

    def test_report_summary_counts_conflicts(self):
        context = [_ctx("HKMA requires a licence.")]
        prior = [_prior("SFC requires a licence.")]
        claims = ["A licence from the SFC is required."]

        report = diagnose_conflicts(claims, context, prior)

        assert "1/1 claims conflict" in report.summary

    def test_empty_claims_produce_empty_report(self):
        report = diagnose_conflicts([], [_ctx("anything")], [_prior("anything")])
        assert report.total_claims == 0
        assert report.diagnoses == []
