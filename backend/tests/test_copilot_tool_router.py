from types import SimpleNamespace

from app.schemas.copilot import CopilotCaseContext, CopilotChatRequest
from app.services.copilot.intent_classifier import IntentDecision
from app.services.copilot.tool_router import route_tools


class _FakeRetriever:
    def invoke(self, query: str):
        from langchain_core.documents import Document

        return [
            Document(
                page_content=f"Evidence for {query}",
                metadata={
                    "doc_id": "doc_1",
                    "title": "HKMA Guideline",
                    "regulator": "HKMA",
                    "page": 3,
                    "section_title": "CDD",
                    "rerank_score": 0.9,
                },
            )
        ]


def test_tool_router_regulatory_qa_uses_retrieval(monkeypatch):
    monkeypatch.setattr("app.services.copilot.tool_router.build_reranked_retriever", lambda: _FakeRetriever())

    request = CopilotChatRequest(message="What is CDD?")
    decision = IntentDecision(intent="regulatory_qa", engine="rag", reason="default")

    result = route_tools(decision, request, {"case_context": {}})

    assert result.tool_name == "rag"
    assert result.payload.evidence_chunks
    assert result.payload.evidence_chunks[0].regulator == "HKMA"


def test_tool_router_regulatory_qa_uses_strategy_aware_router(monkeypatch):
    from app.schemas.evidence import EvidenceBundle, EvidenceChunk

    def fake_route_and_retrieve(query, retrieval_service, **kwargs):
        return EvidenceBundle(
            query=query,
            retrieval_mode="rag",
            evidence_chunks=[
                EvidenceChunk(
                    evidence_id="source_1",
                    doc_id="doc_1",
                    title="Doc",
                    regulator="HKMA",
                    page=1,
                    text="strategy-aware evidence",
                    retrieval_method="hybrid",
                )
            ],
            retrieval_strategy={"strategy_id": "aml_kyc_balanced_rerank"},
        )

    monkeypatch.setattr("app.services.copilot.tool_router.build_reranked_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr("app.services.copilot.tool_router.route_and_retrieve", fake_route_and_retrieve)

    request = CopilotChatRequest(message="What are SVF CDD requirements?")
    decision = IntentDecision(intent="regulatory_qa", engine="rag", reason="default")

    result = route_tools(decision, request, {"case_context": {}})

    assert result.tool_name == "rag"
    assert result.payload.evidence_chunks[0].metadata["retrieval_strategy"]["strategy_id"] == "aml_kyc_balanced_rerank"


def test_tool_router_workflow_recommendation_returns_metadata():
    request = CopilotChatRequest(
        message="Which workflow should I use for product launch?",
        case_context=CopilotCaseContext(workflow_id="x", workflow_name="Current Workflow"),
    )
    decision = IntentDecision(intent="workflow_recommendation", engine="workflow_router", reason="keyword")

    result = route_tools(decision, request, {"case_context": request.case_context.model_dump()})

    assert result.tool_name == "workflow_router"
    assert result.payload.workflow_recommendation is not None
    assert result.payload.workflow_recommendation["workflow_id"] == "product-launch-review"


def test_tool_router_human_review_uses_gate_context():
    request = CopilotChatRequest(
        message="why pending review",
        case_context=CopilotCaseContext(
            current_gate="low_confidence_gate",
            gate_message="Need review",
            confidence_data={"retrieval": 0.3},
        ),
    )
    decision = IntentDecision(intent="human_review_help", engine="human_review", reason="keyword")

    result = route_tools(decision, request, {"case_context": request.case_context.model_dump()})

    assert result.tool_name == "human_review"
    assert result.payload.review_guidance is not None
    assert result.payload.review_guidance["current_gate"] == "low_confidence_gate"
