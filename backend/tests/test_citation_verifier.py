from app.schemas.evidence import EvidenceChunk


def test_citation_verifier_supports_existing_source_and_page():
    from app.services.retrieval.citation_verifier import verify_citations

    audit = verify_citations(
        "CDD is required [Source 1, p.3].",
        [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="doc_1",
                page=3,
                text="CDD is required.",
            )
        ],
    )

    assert len(audit.supported_citations) == 1
    assert audit.unsupported_citations == []
    assert audit.unsupported_claim_rate == 0.0


def test_citation_verifier_flags_missing_source():
    from app.services.retrieval.citation_verifier import verify_citations

    audit = verify_citations(
        "CDD is required [Source 2, p.3].",
        [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="doc_1",
                page=3,
                text="CDD is required.",
            )
        ],
    )

    assert audit.supported_citations == []
    assert len(audit.unsupported_citations) == 1
    assert audit.unsupported_citations[0].reason == "source_not_found"


def test_citation_verifier_flags_page_mismatch():
    from app.services.retrieval.citation_verifier import verify_citations

    audit = verify_citations(
        "CDD is required [Source 1, p.9].",
        [
            EvidenceChunk(
                evidence_id="source_1",
                doc_id="doc_1",
                page=3,
                text="CDD is required.",
            )
        ],
    )

    assert len(audit.unsupported_citations) == 1
    assert audit.unsupported_citations[0].reason == "page_mismatch"
