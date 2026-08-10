"""Safety tests for the versioned JSON corpus cache (T2-04, risk R-06).

Every unsafe or stale cache input must be rejected with an empty list so the
caller rebuilds the corpus instead of trusting corrupted data.
"""

import json
from types import SimpleNamespace

import pytest

from langchain_core.documents import Document

from app.services.corpus.cache import read_corpus_cache, write_corpus_cache

MANIFEST_DIGEST = "abc123"
PARSER_VERSION = "hierarchy-v1"


def _write_valid_cache(path):
    write_corpus_cache(
        path,
        [Document(page_content="CDD obligation", metadata={"regulator": "HKMA"})],
        manifest_digest=MANIFEST_DIGEST,
        parser_version=PARSER_VERSION,
    )


def test_tampered_schema_version_returns_empty_list(tmp_path):
    cache_path = tmp_path / "corpus_documents.json"
    _write_valid_cache(cache_path)

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        read_corpus_cache(
            cache_path,
            manifest_digest=MANIFEST_DIGEST,
            parser_version=PARSER_VERSION,
        )
        == []
    )


def test_manifest_digest_mismatch_returns_empty_list(tmp_path):
    cache_path = tmp_path / "corpus_documents.json"
    _write_valid_cache(cache_path)

    # Cache was written for a different source manifest -> must be rejected.
    assert (
        read_corpus_cache(
            cache_path,
            manifest_digest="different-digest",
            parser_version=PARSER_VERSION,
        )
        == []
    )


def test_corrupted_json_returns_empty_list(tmp_path):
    cache_path = tmp_path / "corpus_documents.json"
    cache_path.write_text("{not valid json!!", encoding="utf-8")

    assert (
        read_corpus_cache(
            cache_path,
            manifest_digest=MANIFEST_DIGEST,
            parser_version=PARSER_VERSION,
        )
        == []
    )


def test_cache_build_fails_before_write_when_required_source_failed(monkeypatch, tmp_path):
    from app.schemas.corpus import CorpusIngestionFailure, CorpusIngestionResult
    from app.services.corpus import build_cache

    result = CorpusIngestionResult(
        documents=[Document(page_content="partial", metadata={})],
        loaded_source_ids=["healthy_source"],
        failures=[
            CorpusIngestionFailure(
                doc_id="required_broken_source",
                path="broken.pdf",
                required=True,
                error_type="PdfReadError",
                message="Cannot find Root object",
            )
        ],
    )
    monkeypatch.setattr(build_cache, "get_settings", lambda: SimpleNamespace(CORPUS_INDEX_DIR=str(tmp_path)))
    monkeypatch.setattr(build_cache, "ingest_corpus_documents", lambda: result)

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("a partial corpus cache must never be written")

    monkeypatch.setattr(build_cache, "write_corpus_cache", unexpected_write)

    with pytest.raises(RuntimeError, match="required_broken_source"):
        build_cache.main()
