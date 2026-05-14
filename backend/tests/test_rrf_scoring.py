from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever


def test_rrf_fusion_writes_rrf_score_metadata():
    from app.services.agents.builder import reciprocal_rank_fusion

    bm25 = [
        Document(page_content="doc a", metadata={"doc_id": "a"}),
        Document(page_content="doc b", metadata={"doc_id": "b"}),
    ]
    dense = [
        Document(page_content="doc b", metadata={"doc_id": "b"}),
        Document(page_content="doc c", metadata={"doc_id": "c"}),
    ]
    fused = reciprocal_rank_fusion([bm25, dense], [0.4, 0.6])

    assert fused
    assert "rrf_score" in fused[0].metadata
    assert isinstance(fused[0].metadata["rrf_score"], float)


class FailingDenseRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        raise RuntimeError("dense retriever unavailable")


def test_hybrid_retriever_falls_back_to_bm25_when_dense_fails():
    from app.services.agents.builder import HybridRetriever

    bm25 = BM25Retriever.from_documents(
        [
            Document(page_content="KYC CDD control obligations", metadata={"doc_id": "a"}),
            Document(page_content="PEP enhanced due diligence", metadata={"doc_id": "b"}),
        ],
        k=2,
    )

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=FailingDenseRetriever(),
        bm25_weight=0.4,
        dense_weight=0.6,
    )

    docs = hybrid.invoke("KYC PEP due diligence")
    assert len(docs) > 0
    assert all("rrf_score" in (doc.metadata or {}) for doc in docs)
