from pathlib import Path
import json

from langchain_core.documents import Document

from app.schemas.corpus import SourceDocument
from app.services.corpus import corpus_ingestor


def _source(path: Path, *, required: bool = True) -> SourceDocument:
    return SourceDocument(
        doc_id=path.stem,
        title="Test regulatory source",
        regulator="HKMA",
        doc_type="Guideline",
        file_path=path.name,
        source_url="https://www.hkma.gov.hk/test.pdf",
        required_for_demo=required,
        resolved_path=path,
    )


def test_required_pdf_failure_is_reported(monkeypatch, tmp_path):
    source = _source(tmp_path / "broken.pdf")
    monkeypatch.setattr(corpus_ingestor, "load_source_manifest", lambda **_: [source])

    def broken_pdf(_source):
        raise ValueError("invalid PDF catalog")

    monkeypatch.setattr(corpus_ingestor, "_load_source_pages", broken_pdf)

    result = corpus_ingestor.ingest_corpus_documents()

    assert result.documents == []
    assert result.loaded_source_ids == []
    assert [failure.doc_id for failure in result.required_failures] == [source.doc_id]
    assert result.required_failures[0].error_type == "ValueError"


def test_zero_page_required_pdf_is_reported_as_empty_document(monkeypatch, tmp_path):
    source = _source(tmp_path / "empty.pdf")
    monkeypatch.setattr(corpus_ingestor, "load_source_manifest", lambda **_: [source])
    monkeypatch.setattr(corpus_ingestor, "_load_source_pages", lambda _source: [])

    result = corpus_ingestor.ingest_corpus_documents()

    assert result.required_failures[0].error_type == "EmptyDocument"
    assert result.required_failures[0].required is True


def test_successful_source_is_counted_and_compatibility_wrapper_returns_documents(monkeypatch, tmp_path):
    source = _source(tmp_path / "valid.pdf")
    page = Document(page_content="Customer due diligence controls apply.", metadata={"page": 1})
    monkeypatch.setattr(corpus_ingestor, "load_source_manifest", lambda **_: [source])
    monkeypatch.setattr(corpus_ingestor, "_load_source_pages", lambda _source: [page])
    monkeypatch.setattr(
        corpus_ingestor,
        "_parse_source_pages",
        lambda pages, _source: pages,
    )

    result = corpus_ingestor.ingest_corpus_documents()

    assert result.loaded_source_ids == [source.doc_id]
    assert result.failures == []
    assert result.documents[0].metadata["doc_id"] == source.doc_id

    monkeypatch.setattr(corpus_ingestor, "ingest_corpus_documents", lambda **_: result)
    assert corpus_ingestor.load_corpus_documents() == result.documents


def test_missing_required_manifest_file_is_reported(tmp_path):
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "doc_id": "missing_required_source",
                    "title": "Missing required source",
                    "regulator": "HKMA",
                    "doc_type": "Guideline",
                    "file_path": "missing.pdf",
                    "required_for_demo": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = corpus_ingestor.ingest_corpus_documents(
        manifest_path=manifest,
        reg_doc_dir=tmp_path / "regulations",
    )

    assert [failure.doc_id for failure in result.required_failures] == ["missing_required_source"]
    assert result.required_failures[0].error_type == "FileNotFoundError"
