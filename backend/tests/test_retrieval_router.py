from app.schemas.evidence import EvidenceChunk


class StubRetrievalService:
    def retrieve(self, query, filters=None, retrieval_mode="rag", top_k=5, **kwargs):
        return [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="doc_1",
                title="Doc",
                regulator="HKMA",
                page=1,
                text=f"Evidence for {query}",
                retrieval_method="hybrid",
            )
        ]


class StubGraphRetriever:
    def retrieve_paths(self, query, filters=None, limit=5):
        return [
            {
                "path": ["HKMA", "Doc", "AI"],
                "matched_doc_ids": ["doc_1"],
                "matched_topics": ["AI"],
            }
        ]


def test_retrieval_router_returns_rag_bundle():
    from app.services.retrieval.retrieval_router import route_and_retrieve

    bundle = route_and_retrieve(
        "What are SVF CDD requirements?",
        retrieval_service=StubRetrievalService(),
    )

    assert bundle.retrieval_mode == "rag"
    assert len(bundle.evidence_chunks) == 1
    assert bundle.graph_paths == []


def test_retrieval_router_returns_kag_bundle_with_graph_paths():
    from app.services.retrieval.retrieval_router import route_and_retrieve

    bundle = route_and_retrieve(
        "Which regulators apply to an AI wealth advisory product?",
        retrieval_service=StubRetrievalService(),
        graph_retriever=StubGraphRetriever(),
    )

    assert bundle.retrieval_mode == "kag"
    assert len(bundle.evidence_chunks) == 1
    assert bundle.graph_paths[0]["path"] == ["HKMA", "Doc", "AI"]


def test_retrieval_router_does_not_execute_deepresearch_for_svf_path():
    from app.services.retrieval.retrieval_router import route_and_retrieve

    bundle = route_and_retrieve(
        "請分析 AI 投資顧問合規風險並生成 checklist",
        retrieval_service=StubRetrievalService(),
        allow_deep_research=False,
    )

    assert bundle.retrieval_mode in {"rag", "kag"}
    assert bundle.evidence_chunks
    assert any("DeepResearch" in warning for warning in bundle.warnings)


class RecordingRetrievalService:
    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, filters=None, retrieval_mode="rag", top_k=5, **kwargs):
        self.kwargs = kwargs
        return [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="doc_1",
                title="Doc",
                regulator="HKMA",
                page=1,
                text="SVF CDD evidence",
                retrieval_method="hybrid",
            )
        ]


def test_retrieval_router_attaches_sira_strategy_metadata_when_enabled(monkeypatch):
    from app.services.retrieval import retrieval_router
    from app.services.retrieval.retrieval_router import route_and_retrieve

    class Settings:
        SIRA_QUERY_PLANNER_ENABLED = True
        EXPERIENCE_RAG_ENABLED = True
        EXPERIENCE_RAG_MEMORY_PATH = "unused.jsonl"
        EXPERIENCE_RAG_MAX_RECORDS = 100

    monkeypatch.setattr(retrieval_router, "get_settings", lambda: Settings())
    service = RecordingRetrievalService()

    bundle = route_and_retrieve("What are SVF CDD requirements?", retrieval_service=service)

    assert service.kwargs["query_plan"].expansion_terms
    assert service.kwargs["strategy"].strategy_id == "aml_kyc_balanced_rerank"
    assert bundle.query_plan["expansion_terms"]
    assert bundle.retrieval_strategy["strategy_id"] == "aml_kyc_balanced_rerank"
    assert bundle.evidence_chunks[0].metadata["query_plan"]["expansion_terms"]
    assert bundle.evidence_chunks[0].metadata["retrieval_strategy"]["strategy_id"] == "aml_kyc_balanced_rerank"
