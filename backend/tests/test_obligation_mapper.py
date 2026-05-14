from app.schemas.evidence import EvidenceChunk
from app.schemas.kag import ProductProfile
from app.services.kag.graph_builder import build_graph_from_sources
from app.services.kag.graph_retriever import GraphRetriever
from app.services.kag.obligation_mapper import ObligationMapper
from app.schemas.corpus import SourceDocument


class _FakeRetrievalService:
    def retrieve(self, query: str, retrieval_mode: str = "kag", top_k: int = 8):
        del query, retrieval_mode, top_k
        return [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="doc_1",
                page=1,
                text="dummy evidence",
            )
        ]


class _EmptyRiskGraphRetriever:
    def retrieve_paths(self, query: str, limit: int = 8):
        del query, limit
        return [
            {
                "path": ["HKMA", "Doc A"],
                "matched_doc_ids": ["doc_1"],
                "matched_topics": ["SVF"],
                "matched_obligations": ["CDD"],
                "matched_risks": [],
            }
        ]


def test_obligation_mapper_returns_structured_output(tmp_path):
    docs = [
        SourceDocument(
            doc_id="doc_1",
            title="AI Circular",
            regulator="HKMA",
            doc_type="Circular",
            topics=["AI"],
            risk_tags=["model_risk"],
            file_path="ai.pdf",
        )
    ]
    store = build_graph_from_sources(docs, [], graph_path=tmp_path / "graph.json")
    retriever = GraphRetriever(store)
    mapper = ObligationMapper()

    result = mapper.map_obligations(
        query="AI governance obligations for SVF",
        product_profile=ProductProfile(product_type="SVF", ai_used=True),
        graph_retriever=retriever,
        retrieval_service=_FakeRetrievalService(),
    )

    assert result.applicable_regulators
    assert result.obligations
    assert result.risks
    assert result.obligations[0].evidence_ids
    assert "SVF" in result.applicable_products


def test_obligation_mapper_handles_empty_matched_risks():
    mapper = ObligationMapper()
    result = mapper.map_obligations(
        query="SVF onboarding checks",
        product_profile=ProductProfile(product_type="SVF"),
        graph_retriever=_EmptyRiskGraphRetriever(),
        retrieval_service=_FakeRetrievalService(),
    )

    assert result.obligations
    assert result.obligations[0].risk == "Operational Risk"
