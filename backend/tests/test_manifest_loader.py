import json
from pathlib import Path

import pytest
from pydantic import ValidationError


def test_source_document_defaults_and_validation():
    from app.schemas.corpus import SourceDocument

    doc = SourceDocument(
        doc_id="hkma_svf_test",
        title="Test Document",
        regulator="HKMA",
        doc_type="Guideline",
        file_path="hkma_svf/test.pdf",
    )

    assert doc.jurisdiction == "Hong Kong"
    assert doc.status == "active"
    assert doc.priority == "P1"
    assert doc.language == "en"
    assert doc.topics == []
    assert doc.risk_tags == []
    assert doc.module_tags == []

    with pytest.raises(ValidationError):
        SourceDocument(
            doc_id="bad_priority",
            title="Bad",
            regulator="HKMA",
            doc_type="Guideline",
            file_path="bad.pdf",
            priority="P9",
        )

    with pytest.raises(ValidationError):
        SourceDocument(
            doc_id="bad_status",
            title="Bad",
            regulator="HKMA",
            doc_type="Guideline",
            file_path="bad.pdf",
            status="current",
        )


def test_manifest_loader_reads_existing_files(tmp_path, monkeypatch):
    from app.services.corpus.manifest_loader import load_source_manifest

    reg_dir = tmp_path / "regulations"
    reg_dir.mkdir()
    source_pdf = reg_dir / "hkma_svf" / "test.pdf"
    source_pdf.parent.mkdir()
    source_pdf.write_bytes(b"%PDF-1.4 test")

    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "doc_id": "hkma_svf_test",
                    "title": "Test Document",
                    "regulator": "HKMA",
                    "doc_type": "Guideline",
                    "topics": ["AML", "CDD"],
                    "module_tags": ["svf", "aml"],
                    "file_path": "hkma_svf/test.pdf",
                    "priority": "P0",
                }
            ]
        ),
        encoding="utf-8",
    )

    docs = load_source_manifest(manifest_path=manifest, reg_doc_dir=reg_dir)

    assert len(docs) == 1
    assert docs[0].doc_id == "hkma_svf_test"
    assert docs[0].resolved_path == source_pdf
    assert docs[0].topics == ["AML", "CDD"]
    assert docs[0].module_tags == ["svf", "aml"]


def test_manifest_loader_missing_manifest_returns_empty(tmp_path):
    from app.services.corpus.manifest_loader import load_source_manifest

    docs = load_source_manifest(
        manifest_path=tmp_path / "missing.json",
        reg_doc_dir=tmp_path / "regulations",
    )

    assert docs == []


def test_manifest_loader_warns_for_missing_source_file(tmp_path, caplog):
    from app.services.corpus.manifest_loader import load_source_manifest

    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "doc_id": "missing_doc",
                    "title": "Missing Document",
                    "regulator": "HKMA",
                    "doc_type": "Guideline",
                    "file_path": "hkma_svf/missing.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )

    docs = load_source_manifest(
        manifest_path=manifest,
        reg_doc_dir=tmp_path / "regulations",
    )

    assert docs == []
    assert "Source file not found" in caplog.text
