from app.core.config import Settings
from app.schemas.copilot import CopilotCaseContext, CopilotChatRequest, CopilotMessage
from app.services.copilot.context_builder import build_case_context


def test_context_builder_limits_history_and_trims_fields():
    cfg = Settings(
        COPILOT_MAX_HISTORY_MESSAGES=2,
        COPILOT_MAX_CONTEXT_CHARS=120,
    )

    request = CopilotChatRequest(
        message="Explain this case",
        history=[
            CopilotMessage(role="user", content="m1"),
            CopilotMessage(role="assistant", content="m2"),
            CopilotMessage(role="user", content="m3"),
        ],
        case_context=CopilotCaseContext(
            report_text="R" * 500,
            input_text="I" * 400,
            evidence_chunks=[
                {
                    "evidence_id": "source_1",
                    "title": "Doc",
                    "regulator": "HKMA",
                    "page": 2,
                    "section_title": "CDD",
                    "text": "T" * 1000,
                }
            ],
        ),
    )

    built = build_case_context(request, settings=cfg)

    assert len(built["history"]) == 2
    assert built["history"][0]["content"] == "m2"
    assert "...[truncated]" in built["case_context"]["report_text"]
    assert built["case_context"]["evidence_chunks"][0]["evidence_id"] == "source_1"
    assert len(built["case_context"]["evidence_chunks"][0]["text_snippet"]) <= 360 + len("\n...[truncated]")


def test_context_builder_excludes_secret_like_keys():
    request = CopilotChatRequest(
        message="explain",
        case_context=CopilotCaseContext(
            confidence_data={
                "score": 0.8,
                "api_key": "should-not-leak",
                "nested": {"token": "secret-token", "keep": "ok"},
            }
        ),
    )

    built = build_case_context(request)
    confidence_data = built["case_context"]["confidence_data"]

    assert "api_key" not in confidence_data
    assert "token" not in confidence_data.get("nested", {})
    assert confidence_data.get("nested", {}).get("keep") == "ok"
