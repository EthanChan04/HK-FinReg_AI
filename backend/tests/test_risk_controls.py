import asyncio
import json
from pathlib import Path

import pytest
from fastapi import Request


def _request(path: str = "/api/v1/test", headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": raw_headers,
        "client": ("127.0.0.1", 1234),
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    return Request(scope)


def test_release_manifest_has_complete_official_metadata_and_sfc_coverage():
    manifest_path = Path(__file__).parents[1] / "data" / "source_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    regulators = {item["regulator"] for item in raw}
    assert {"HKMA", "SFC", "PCPD"}.issubset(regulators)
    assert len(raw) >= 20
    assert all(item.get("source_url", "").startswith("https://") for item in raw)
    assert all(item.get("status") in {"active", "superseded", "archived"} for item in raw)
    assert all(item.get("effective_date") or item.get("metadata_note") for item in raw)


def test_manifest_loader_rejects_non_official_source_url(tmp_path):
    from app.services.corpus.manifest_loader import validate_source_metadata

    item = {
        "doc_id": "bad-source",
        "title": "Bad source",
        "regulator": "HKMA",
        "doc_type": "Guideline",
        "file_path": "bad.pdf",
        "source_url": "https://example.com/bad.pdf",
        "effective_date": "2025-01-01",
        "status": "active",
    }

    with pytest.raises(ValueError, match="official regulator domain"):
        validate_source_metadata(item)


def test_rate_limit_identity_is_stable_and_not_shared_between_users():
    from app.core.rate_limit import rate_limit_identity

    first = rate_limit_identity(_request(headers={"authorization": "Bearer tenant-a"}))
    second = rate_limit_identity(_request(headers={"authorization": "Bearer tenant-b"}))
    assert first != second
    assert first.startswith("credential:")


def test_rate_limit_store_is_shared_across_middleware_instances():
    from app.core.rate_limit import InMemoryRateLimitStore

    async def exercise():
        store = InMemoryRateLimitStore()
        assert await store.allow("credential:a", 1, 10, 10)
        assert not await store.allow("credential:a", 1, 1, 1)
        assert await store.allow("credential:b", 1, 1, 1)

    asyncio.run(exercise())


def test_health_readiness_reports_dependency_failure():
    from app.core.health import readiness_report

    report = readiness_report(
        {
            "llm_service": False,
            "corpus_index": True,
            "graph_store": True,
        }
    )
    assert report["status"] == "degraded"
    assert report["engines"]["llm_service"] == "unavailable"


def test_fastapi_health_endpoints_expose_live_and_ready_states():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    live = client.get("/api/v1/health/live")
    ready = client.get("/api/v1/health/ready")

    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code in {200, 503}
    assert ready.json()["status"] in {"ready", "degraded"}


def test_json_corpus_cache_roundtrip_and_schema_validation(tmp_path):
    from langchain_core.documents import Document

    from app.services.corpus.cache import read_corpus_cache, write_corpus_cache

    cache_path = tmp_path / "corpus_documents.json"
    docs = [Document(page_content="CDD obligation", metadata={"regulator": "HKMA"})]
    write_corpus_cache(cache_path, docs, manifest_digest="abc", parser_version="v1")
    loaded = read_corpus_cache(cache_path, manifest_digest="abc", parser_version="v1")
    assert loaded[0].page_content == docs[0].page_content
    assert loaded[0].metadata == docs[0].metadata

    cache_path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    assert read_corpus_cache(cache_path, manifest_digest="abc", parser_version="v1") == []


def test_benchmark_contains_claim_level_metrics_and_at_least_fifty_cases():
    from app.services.evaluation.benchmark_loader import load_benchmark_questions
    from app.services.evaluation.rag_eval import evaluate_claim_level_metrics

    questions = load_benchmark_questions()
    assert len(questions) >= 50
    assert all(item.get("expected_claims") for item in questions)

    metrics = evaluate_claim_level_metrics(
        ["The firm must assess suitability before making a recommendation."],
        [{"page_content": "The firm must assess suitability before making a recommendation."}],
    )
    assert metrics["claim_recall"] == 1.0
    # Without a generator response, faithfulness must not be silently
    # reported as claim_recall (system-evaluation-report-2026-08-04, sec 3.2).
    assert metrics["faithfulness"] is None
    assert metrics["hallucination_rate"] is None
    assert metrics["faithfulness_measured"] is False


def test_eval_retrieval_preserves_expected_regulator_diversity(monkeypatch):
    from langchain_core.documents import Document

    from app.services.evaluation import run_eval

    docs = [
        Document(page_content="AI governance", metadata={"regulator": "HKMA", "topics": "AI"}),
        Document(page_content="AI governance", metadata={"regulator": "SFC", "topics": "AI suitability"}),
        Document(page_content="AI governance", metadata={"regulator": "PCPD", "topics": "AI personal_data"}),
    ]
    monkeypatch.setattr(run_eval, "_load_cached_corpus_documents", lambda: tuple(docs))

    retrieved = run_eval._retrieve_eval_documents(
        "Which HKMA SFC and PCPD obligations apply to an AI wealth advisory product?",
        top_k=3,
    )
    assert {doc.metadata["regulator"] for doc in retrieved} == {"HKMA", "SFC", "PCPD"}
