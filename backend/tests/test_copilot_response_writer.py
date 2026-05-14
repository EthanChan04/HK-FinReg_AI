from app.services.copilot.intent_classifier import IntentDecision
from app.services.copilot.response_writer import write_bilingual_response


def test_response_writer_enforces_bilingual_sections(monkeypatch):
    class _FakeLLM:
        def invoke(self, _messages):
            class R:
                content = "This is English only and should be repaired."

            return R()

    monkeypatch.setattr("app.services.copilot.response_writer.build_copilot_llm", lambda: _FakeLLM())

    text, audit = write_bilingual_response(
        message="Explain HKMA eKYC",
        intent=IntentDecision(intent="regulatory_qa", engine="rag", reason="default"),
        compact_context={"case_context": {}},
        runtime_payload={"evidence_chunks": []},
    )

    assert "## 绻侀珨涓枃" in text
    assert "## English" in text
    assert isinstance(audit.get("unsupported_claim_rate"), float)


def test_response_writer_guardrails_remove_approval_language(monkeypatch):
    class _FakeLLM:
        def invoke(self, _messages):
            class R:
                content = (
                    "## 绻侀珨涓枃\n"
                    "批准該客戶。\n\n"
                    "## English\n"
                    "We approve this customer immediately."
                )

            return R()

    monkeypatch.setattr("app.services.copilot.response_writer.build_copilot_llm", lambda: _FakeLLM())

    text, _audit = write_bilingual_response(
        message="Can we approve this?",
        intent=IntentDecision(intent="human_review_help", engine="human_review", reason="review"),
        compact_context={"case_context": {"confidence_data": {"retrieval": 0.2}}},
        runtime_payload={"evidence_chunks": []},
    )

    assert "approve this customer" not in text.lower()
    assert "Human Review" in text
