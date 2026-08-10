import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.services.evaluation import live_demo_eval


EXPECTED_CASE_IDS = (
    "RAG_SVF_AML_001",
    "KAG_AI_ADVISOR_001",
    "DR_AI_ADVISOR_001",
    "EXP_051",
    "EXP_062",
    "EXP_071",
    "EXP_077",
    "EXP_082",
    "EXP_088",
    "EXP_089",
    "EXP_090",
    "EXP_095",
)


def _benchmark_items():
    return [{"id": case_id, "question": f"Question for {case_id}?"} for case_id in EXPECTED_CASE_IDS]


def _evidence():
    return [
        Document(
            page_content="A licensee must maintain documented controls.",
            metadata={"doc_id": "hkma-controls", "page": 4, "regulator": "HKMA"},
        ),
        Document(
            page_content="Personal data must be processed fairly.",
            metadata={"source_id": "pcpd-data", "page_number": 9, "regulator": "PCPD"},
        ),
    ]


def test_live_demo_case_selection_is_fixed_and_stratified():
    assert live_demo_eval.LIVE_DEMO_CASE_IDS == EXPECTED_CASE_IDS
    assert len(set(live_demo_eval.LIVE_DEMO_CASE_IDS)) == 12


def test_grounded_prompt_numbers_evidence_and_requires_insufficiency_disclosure():
    messages = live_demo_eval.build_grounded_prompt(
        {"id": "RAG_SVF_AML_001", "question": "What applies?"},
        _evidence(),
    )

    rendered = "\n".join(str(message.content) for message in messages)
    assert "[1]" in rendered and "[2]" in rendered
    assert "hkma-controls" in rendered and "pcpd-data" in rendered
    assert "insufficient" in rendered.lower()
    assert "What applies?" in rendered


class _FakeLLM:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def invoke(self, messages):
        del messages
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _HTTPError(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def test_capture_retries_transient_errors_and_writes_loadable_artifact(tmp_path, monkeypatch):
    items = _benchmark_items()
    monkeypatch.setattr(live_demo_eval, "load_benchmark_questions", lambda: items)
    monkeypatch.setattr(live_demo_eval, "retrieve_eval_documents", lambda question, top_k=10: _evidence())
    monkeypatch.setattr(live_demo_eval.time, "sleep", lambda seconds: None)
    llm = _FakeLLM(
        [
            _HTTPError(429, "rate limited"),
            _HTTPError(503, "temporary"),
            AIMessage(
                content="Grounded answer [1].",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            ),
        ]
        + [AIMessage(content=f"Answer {case_id} [1].") for case_id in EXPECTED_CASE_IDS[1:]]
    )

    document = live_demo_eval.capture_live_responses(tmp_path, llm=llm)

    assert llm.calls == 14
    assert document["provider"] == "deepseek"
    assert document["model"] == "deepseek-v4-flash"
    assert [case["case_id"] for case in document["cases"]] == list(EXPECTED_CASE_IDS)
    assert document["cases"][0]["usage"]["total_tokens"] == 15
    assert all(case["latency_ms"] >= 0 for case in document["cases"])
    artifact = tmp_path / "deepseek-v4-flash-live-responses.json"
    assert json.loads(artifact.read_text(encoding="utf-8")) == document
    provider = live_demo_eval.load_live_response_provider(document)
    assert provider(items[0], _evidence()) == "Grounded answer [1]."


def test_capture_does_not_retry_auth_error_and_redacts_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(live_demo_eval, "load_benchmark_questions", lambda: _benchmark_items())
    monkeypatch.setattr(live_demo_eval, "retrieve_eval_documents", lambda question, top_k=10: _evidence())
    secret = "sk-sensitive-demo-key"
    llm = _FakeLLM([_HTTPError(401, f"unauthorized bearer {secret}")])

    document = live_demo_eval.capture_live_responses(tmp_path, llm=llm)

    assert llm.calls == 1
    first = document["cases"][0]
    assert first["response"] == ""
    assert "401" in first["error"]
    assert secret not in json.dumps(document)
    assert all(case["error"] == "not attempted after fatal authentication failure" for case in document["cases"][1:])


def test_capture_records_empty_model_output_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(live_demo_eval, "load_benchmark_questions", lambda: _benchmark_items())
    monkeypatch.setattr(live_demo_eval, "retrieve_eval_documents", lambda question, top_k=10: _evidence())
    llm = _FakeLLM([AIMessage(content="   ")])

    document = live_demo_eval.capture_live_responses(tmp_path, llm=llm)

    assert llm.calls == 1
    assert document["cases"][0]["error"] == "empty model response"


def test_live_response_provider_rejects_wrong_runtime_and_duplicate_cases():
    base = {
        "schema_version": 1,
        "provider": "openai-compatible",
        "model": "deepseek-v4-flash",
        "prompt_version": "demo-grounded-v1",
        "cases": [],
    }
    with pytest.raises(ValueError, match="provider"):
        live_demo_eval.load_live_response_provider(base)

    base["provider"] = "deepseek"
    base["cases"] = [
        {"case_id": "A", "response": "one"},
        {"case_id": "A", "response": "two"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        live_demo_eval.load_live_response_provider(base)
