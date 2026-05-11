from langchain_core.documents import Document


def test_evidence_chunk_instantiation():
    from app.schemas.evidence import EvidenceChunk

    chunk = EvidenceChunk(
        evidence_id="source_1",
        chunk_id="chunk_1",
        doc_id="hkma_svf_amlcft_guideline_2023",
        title="Guideline on AML/CFT for SVF Licensees",
        regulator="HKMA",
        doc_type="Guideline",
        page=12,
        section_title="Customer Due Diligence",
        hierarchy_path="Chapter 4 > Section 4.1",
        text="Customer due diligence requirements apply to SVF licensees.",
        retrieval_method="hybrid",
        score=0.87,
    )

    assert chunk.jurisdiction == "Hong Kong"
    assert chunk.metadata == {}
    assert chunk.doc_id == "hkma_svf_amlcft_guideline_2023"


def test_document_to_evidence_uses_metadata():
    from app.services.retrieval.retrieval_service import document_to_evidence

    doc = Document(
        page_content="CDD requirements text",
        metadata={
            "chunk_id": "chunk_1",
            "doc_id": "doc_1",
            "title": "Title",
            "regulator": "HKMA",
            "doc_type": "Guideline",
            "page": 3,
            "section_title": "CDD",
            "hierarchy_path": "Chapter 4 > CDD",
            "source_url": "https://example.com/doc.pdf",
            "rerank_score": 0.91,
        },
    )

    evidence = document_to_evidence(doc, 1, retrieval_method="rerank")

    assert evidence.evidence_id == "source_1"
    assert evidence.chunk_id == "chunk_1"
    assert evidence.doc_id == "doc_1"
    assert evidence.page == 3
    assert evidence.score == 0.91
    assert evidence.text == "CDD requirements text"


def test_render_evidence_context_contains_source_metadata():
    from app.schemas.evidence import EvidenceChunk
    from app.services.retrieval.evidence_renderer import render_evidence_context

    context = render_evidence_context(
        [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="doc_1",
                title="Doc Title",
                regulator="HKMA",
                page=7,
                section_title="Section 2.1",
                hierarchy_path="Chapter 2 > Section 2.1",
                text="Evidence text",
                score=0.8,
            )
        ]
    )

    assert "Source 1" in context
    assert "doc_id=doc_1" in context
    assert "regulator=HKMA" in context
    assert "p.7" in context
    assert "Section 2.1" in context
