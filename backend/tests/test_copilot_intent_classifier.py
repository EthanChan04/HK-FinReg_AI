from pydantic import ValidationError


def test_copilot_chat_request_defaults_preferred_language():
    from app.schemas.copilot import CopilotChatRequest

    payload = CopilotChatRequest(message="Explain HKMA CDD expectations")
    assert payload.preferred_language == "zh-HK+en"


def test_copilot_chat_request_rejects_overlong_message():
    from app.schemas.copilot import CopilotChatRequest

    too_long = "x" * 8001
    try:
        CopilotChatRequest(message=too_long)
    except ValidationError as exc:
        assert "at most 8000" in str(exc)
    else:
        raise AssertionError("Expected ValidationError for oversized message")


def test_intent_classifier_routes_all_required_intents():
    from app.services.copilot.intent_classifier import classify_intent

    assert classify_intent("please compare HKMA SFC PCPD policy impact").intent == "deep_research"
    assert classify_intent("why does this obligation apply").intent == "obligation_mapping"
    assert classify_intent("this case has low confidence evidence insufficient").intent == "case_explanation"
    assert classify_intent("which workflow should i use").intent == "workflow_recommendation"
    assert classify_intent("can reviewer approve this pending queue item").intent == "human_review_help"
    smalltalk = classify_intent("hello, help")
    assert smalltalk.intent == "smalltalk_or_help"
    assert smalltalk.engine == "deepseek"
    assert classify_intent("What does HKMA require for eKYC?").intent == "regulatory_qa"
