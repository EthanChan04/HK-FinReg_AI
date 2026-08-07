"""T1-01: Independent generation faithfulness metric.

Requirement (from system-evaluation-report-2026-08-04.md):
  faithfulness must NOT equal claim_recall. It measures whether each
  claim in the GENERATOR's actual response is supported by the retrieved
  context, not whether expected claims from the benchmark are retrieved.
"""

from __future__ import annotations

import json

from langchain_core.documents import Document

from app.services.evaluation.rag_eval import (
    evaluate_claim_level_metrics,
    evaluate_generation_faithfulness,
    split_response_claims,
)


def _evidence(text: str) -> list[Document]:
    return [Document(page_content=text, metadata={"doc_id": "doc-1"})]


class TestSplitResponseClaims:
    def test_splits_english_sentences(self):
        response = "The SVF must obtain a licence. It must also conduct CDD."
        claims = split_response_claims(response)
        assert claims == ["The SVF must obtain a licence.", "It must also conduct CDD."]

    def test_splits_traditional_chinese_sentences(self):
        response = "持牌人須進行客戶盡職審查。亦須保存交易記錄。"
        claims = split_response_claims(response)
        assert claims == ["持牌人須進行客戶盡職審查。", "亦須保存交易記錄。"]

    def test_skips_short_noise_fragments(self):
        response = "Yes. The licensee must maintain adequate capital. OK."
        claims = split_response_claims(response)
        assert claims == ["The licensee must maintain adequate capital."]


class TestGenerationFaithfulness:
    def test_faithfulness_measures_generated_response_claims(self):
        evidence = _evidence(
            "An SVF licensee must obtain a licence from the HKMA and "
            "perform customer due diligence under the AMLO."
        )
        response = (
            "The SVF must obtain a licence from the HKMA. "
            "It must submit quarterly returns to the SFC. "  # NOT supported
            "It must perform customer due diligence."
        )
        result = evaluate_generation_faithfulness(response, evidence)
        assert result["faithfulness"] == round(2 / 3, 3)
        assert result["hallucination_rate"] == round(1 / 3, 3)
        # Per-claim traceability
        assert len(result["per_claim"]) == 3
        supported = [item["claim"] for item in result["per_claim"] if item["supported"]]
        assert supported == [
            "The SVF must obtain a licence from the HKMA.",
            "It must perform customer due diligence.",
        ]

    def test_faithfulness_differs_from_claim_recall(self):
        evidence = _evidence(
            "An SVF licensee must obtain a licence from the HKMA and "
            "perform customer due diligence under the AMLO."
        )
        # Benchmark expects claims that ARE retrieved (high claim_recall)
        benchmark_claims = ["An SVF licensee must obtain a licence."]
        # Generator response contains an unsupported claim (lower faithfulness)
        response = (
            "The SVF must obtain a licence from the HKMA. "
            "It must report to the PCPD quarterly."  # unsupported
        )
        metrics = evaluate_claim_level_metrics(
            benchmark_claims,
            evidence,
            generated_response=response,
        )
        assert metrics["claim_recall"] == 1.0
        assert metrics["faithfulness"] == 0.5
        assert metrics["faithfulness"] != metrics["claim_recall"]

    def test_faithfulness_is_none_without_generated_response(self):
        """Without a real generator response, faithfulness must not be
        silently reported as claim_recall."""
        evidence = _evidence("An SVF licensee must obtain a licence from the HKMA.")
        metrics = evaluate_claim_level_metrics(
            ["An SVF licensee must obtain a licence."],
            evidence,
        )
        assert metrics["faithfulness"] is None
        assert metrics["hallucination_rate"] is None
        assert metrics["faithfulness_measured"] is False
        # claim_recall stays available as a retrieval-quality metric
        assert metrics["claim_recall"] == 1.0

    def test_empty_response_has_no_claims(self):
        result = evaluate_generation_faithfulness("", _evidence("Any evidence text."))
        assert result["faithfulness"] is None
        assert result["per_claim"] == []

    def test_no_evidence_means_nothing_supported(self):
        response = "The SVF must obtain a licence from the HKMA."
        result = evaluate_generation_faithfulness(response, [])
        assert result["faithfulness"] == 0.0
        assert result["per_claim"][0]["supported"] is False


def test_eval_claim_metrics_uses_actual_response_provider(monkeypatch):
    from app.services.evaluation import run_eval

    evidence = [
        Document(
            page_content="Authorized institutions must perform customer due diligence.",
            metadata={"doc_id": "hkma-cdd"},
        )
    ]
    item = {
        "id": "GEN_001",
        "question": "What must authorized institutions perform?",
        "expected_claims": ["CDD evidence should be retrievable."],
    }
    observed = {}

    def response_provider(case, retrieved):
        observed["case_id"] = case["id"]
        observed["evidence"] = retrieved
        return "Authorized institutions must perform customer due diligence."

    monkeypatch.setattr(run_eval, "_retrieve_eval_documents", lambda question, top_k: evidence)

    metrics = run_eval._evaluate_claim_metrics(item, response_provider=response_provider)

    assert observed == {"case_id": "GEN_001", "evidence": evidence}
    assert metrics["faithfulness"] == 1.0
    assert metrics["hallucination_rate"] == 0.0


def test_captured_response_provider_loads_actual_outputs_by_case_id(tmp_path):
    from app.services.evaluation.run_eval import load_captured_response_provider

    path = tmp_path / "captured-responses.json"
    path.write_text(
        json.dumps(
            {
                "GEN_001": "Authorized institutions must perform CDD.",
                "GEN_002": "Customers may request human review.",
            }
        ),
        encoding="utf-8",
    )

    provider = load_captured_response_provider(path)

    assert provider({"id": "GEN_001"}, []) == "Authorized institutions must perform CDD."
    assert provider({"id": "MISSING"}, []) is None
