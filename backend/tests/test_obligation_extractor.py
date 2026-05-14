from app.schemas.evidence import EvidenceChunk


def test_obligation_extractor_detects_obligation_and_risk():
    from app.services.kag.obligation_extractor import extract_obligations_from_chunks

    chunks = [
        EvidenceChunk(
            evidence_id="source_1",
            doc_id="doc_1",
            page=3,
            text="Institutions must perform customer due diligence and keep records.",
        )
    ]

    extracted = extract_obligations_from_chunks(chunks)
    assert extracted
    assert extracted[0].name in {"CDD", "Record Keeping"}
    assert extracted[0].risk_type == "AML/CFT"

