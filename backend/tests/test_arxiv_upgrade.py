import json
import os

import pytest
from langchain_core.documents import Document


PHASE4_EXPERIMENT = pytest.mark.skipif(
    os.environ.get("ENABLE_PHASE4_EXPERIMENTS") != "1",
    reason="Phase 4 architecture experiments remain behind the decision gate",
)


def test_eval_cache_resolves_backend_relative_paths_from_repo_root(tmp_path, monkeypatch):
    from app.services.evaluation import run_eval
    from app.services.corpus.cache import manifest_digest, write_corpus_cache
    from app.core import config

    cache_dir = tmp_path / "indexes"
    manifest_path = run_eval._backend_root() / "data" / "source_manifest.json"
    write_corpus_cache(
        cache_dir / "corpus_documents.json",
        [Document(page_content="SFC suitability", metadata={"regulator": "SFC"})],
        manifest_digest=manifest_digest(manifest_path),
        parser_version="hierarchy-v1",
    )

    class Settings:
        CORPUS_INDEX_DIR = str(cache_dir)

    monkeypatch.setattr(config, "get_settings", lambda: Settings())

    run_eval._load_cached_corpus_documents.cache_clear()
    documents = run_eval._load_cached_corpus_documents()

    assert documents
    assert any("regulator" in document.metadata for document in documents)


def test_build_cache_targets_a_json_file_not_the_index_directory(monkeypatch):
    from app.schemas.corpus import CorpusIngestionResult
    from app.services.corpus import build_cache

    captured = {}

    class Settings:
        CORPUS_INDEX_DIR = "data/indexes"

    monkeypatch.setattr(build_cache, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        build_cache,
        "ingest_corpus_documents",
        lambda: CorpusIngestionResult(
            documents=[Document(page_content="ok")],
            loaded_source_ids=["test_source"],
            failures=[],
        ),
    )
    monkeypatch.setattr(build_cache, "manifest_digest", lambda path: "digest")

    def capture_write(path, documents, **kwargs):
        captured["path"] = path

    monkeypatch.setattr(build_cache, "write_corpus_cache", capture_write)
    build_cache.main()

    assert captured["path"].name == "corpus_documents.json"


def test_quality_gate_reports_retriever_and_generator_regressions():
    from app.services.evaluation.release_gate import evaluate_quality_gate

    result = evaluate_quality_gate(
        {
            "avg_claim_recall": 0.89,
            "avg_context_precision": 0.74,
            "avg_faithfulness": 0.94,
            "avg_unsupported_claim_rate": 0.06,
        }
    )

    assert result["passed"] is False
    assert {failure["metric"] for failure in result["failures"]} == {
        "claim_recall",
        "context_precision",
        "faithfulness",
        "unsupported_claim_rate",
    }


def test_quality_gate_does_not_fail_unmeasured_faithfulness():
    """Unmeasured (None) generation faithfulness is reported, not failed."""

    from app.services.evaluation.release_gate import evaluate_quality_gate

    result = evaluate_quality_gate(
        {
            "avg_claim_recall": 0.90,
            "avg_context_precision": 0.75,
            "avg_faithfulness": None,
            "avg_unsupported_claim_rate": 0.05,
        }
    )

    assert result["passed"] is True
    assert result["actual"]["faithfulness"] is None


@PHASE4_EXPERIMENT
def test_spo_extraction_is_normalized_and_traceable():
    from app.schemas.corpus import SourceDocument
    from app.schemas.evidence import EvidenceChunk
    from app.services.kag.triples import extract_regulatory_triples

    documents = [
        SourceDocument(
            doc_id="hkma_cdd",
            title="CDD Guideline",
            regulator="HKMA",
            doc_type="Guideline",
            file_path="hkma_cdd.pdf",
            source_url="https://www.hkma.gov.hk/example",
        )
    ]
    evidence = [
        EvidenceChunk(
            evidence_id="ev-1",
            chunk_id="clause-4-1",
            doc_id="hkma_cdd",
            page=12,
            hierarchy_path="Chapter 4 > Section 4.1 > Clause 4.1.2",
            text="Authorized institutions shall perform customer due diligence before establishing a business relationship.",
        )
    ]

    triples = extract_regulatory_triples(evidence, documents)

    assert len(triples) == 1
    triple = triples[0]
    assert triple.predicate == "must_perform"
    assert triple.source.clause_id == "clause-4-1"
    assert triple.source.page == 12
    assert triple.source.official_url == "https://www.hkma.gov.hk/example"
    assert triple.extraction.extractor_version
    assert triple.triple_id == extract_regulatory_triples(evidence, documents)[0].triple_id


@PHASE4_EXPERIMENT
def test_dual_graph_builds_structure_and_semantic_provenance_paths(tmp_path):
    from app.schemas.corpus import SourceDocument
    from app.schemas.evidence import EvidenceChunk
    from app.services.kag.graph_builder import build_graph_from_sources
    from app.services.kag.graph_retriever import GraphRetriever

    documents = [
        SourceDocument(
            doc_id="hkma_cdd",
            title="CDD Guideline",
            regulator="HKMA",
            doc_type="Guideline",
            file_path="hkma_cdd.pdf",
            source_url="https://www.hkma.gov.hk/example",
        )
    ]
    evidence = [
        EvidenceChunk(
            evidence_id="ev-1",
            chunk_id="clause-4-1",
            doc_id="hkma_cdd",
            page=12,
            hierarchy_path="Chapter 4 > Section 4.1 > Clause 4.1.2",
            section_title="Customer Due Diligence",
            text="Authorized institutions shall perform customer due diligence before establishing a business relationship.",
        )
    ]

    store = build_graph_from_sources(documents, evidence, graph_path=tmp_path / "graph.json")

    assert any(attrs.get("node_type") == "Chapter" for _, attrs in store.graph.nodes(data=True))
    assert any(attrs.get("node_type") == "Section" for _, attrs in store.graph.nodes(data=True))
    assert any(attrs.get("node_type") == "Clause" for _, attrs in store.graph.nodes(data=True))
    assert any(attrs.get("node_type") == "RegulatoryTriple" for _, attrs in store.graph.nodes(data=True))

    paths = GraphRetriever(store).retrieve_paths("who must perform customer due diligence", limit=5)

    assert paths
    assert paths[0]["matched_doc_ids"] == ["hkma_cdd"]
    assert "SUPPORTED_BY" in paths[0]["relation_chain"]
    assert paths[0]["source_refs"][0]["page"] == 12


@PHASE4_EXPERIMENT
def test_dual_graph_ab_comparison_exposes_recall_delta(tmp_path):
    from app.schemas.corpus import SourceDocument
    from app.schemas.evidence import EvidenceChunk
    from app.services.kag.ab_eval import compare_graph_retrieval
    from app.services.kag.graph_builder import build_graph_from_sources
    from app.services.kag.graph_retriever import GraphRetriever

    documents = [
        SourceDocument(
            doc_id="hkma_cdd",
            title="CDD Guideline",
            regulator="HKMA",
            doc_type="Guideline",
            file_path="hkma_cdd.pdf",
            source_url="https://www.hkma.gov.hk/example",
        )
    ]
    evidence = [
        EvidenceChunk(
            evidence_id="ev-1",
            chunk_id="clause-4-1",
            doc_id="hkma_cdd",
            hierarchy_path="Chapter 4 > Section 4.1 > Clause 4.1.2",
            text="Authorized institutions shall perform customer due diligence.",
        )
    ]
    retriever = GraphRetriever(
        build_graph_from_sources(documents, evidence, graph_path=tmp_path / "graph.json")
    )

    result = compare_graph_retrieval(
        retriever,
        "what must be performed before a relationship?",
        expected_doc_ids=["hkma_cdd"],
    )

    assert set(result) == {"baseline", "dual_graph", "baseline_recall", "dual_graph_recall", "recall_delta"}
    assert result["dual_graph_recall"] == 1.0
    assert result["recall_delta"] >= 0.0


@PHASE4_EXPERIMENT
def test_graph_retriever_bounds_multi_hop_candidate_expansion():
    from app.services.kag.graph_retriever import GraphRetriever
    from app.services.kag.graph_store import NetworkXGraphStore

    store = NetworkXGraphStore("unused.json")
    for index in range(200):
        store.add_node(f"doc-{index}", node_type="RegulatoryDocument", title=f"Common requirement {index}")
        store.add_node(f"chunk-{index}", node_type="EvidenceChunk", title="Common requirement")
        store.add_edge(f"doc-{index}", f"chunk-{index}", relation="CONTAINS")

    retriever = GraphRetriever(store)
    calls = 0
    original = retriever._candidate_docs_for_node

    def count_calls(node_id, max_hops=4):
        nonlocal calls
        calls += 1
        return original(node_id, max_hops=max_hops)

    retriever._candidate_docs_for_node = count_calls
    retriever.retrieve_paths("what are the common requirements", limit=5)

    assert calls <= 15


def test_eval_reuses_read_only_graph_retriever(monkeypatch, tmp_path):
    from app.services.evaluation import run_eval
    from app.services.kag.graph_store import NetworkXGraphStore

    graph_path = tmp_path / "graph.json"
    NetworkXGraphStore(graph_path).save()

    class Settings:
        GRAPH_STORE_PATH = str(graph_path)

    monkeypatch.setattr(run_eval, "get_settings", lambda: Settings(), raising=False)
    run_eval._get_cached_graph_retriever.cache_clear()

    first = run_eval._get_cached_graph_retriever(str(graph_path))
    second = run_eval._get_cached_graph_retriever(str(graph_path))

    assert first is second


def test_eval_graph_count_uses_lightweight_retrieval(monkeypatch, tmp_path):
    from app.services.evaluation import run_eval

    class Settings:
        GRAPH_STORE_PATH = str(tmp_path / "graph.json")

    class FakeRetriever:
        called = False

        def retrieve_paths(self, query, limit):
            self.called = True
            return [{"path": ["doc"]}]

    fake = FakeRetriever()
    monkeypatch.setattr(run_eval, "get_settings", lambda: Settings(), raising=False)
    monkeypatch.setattr(run_eval, "_get_cached_graph_retriever", lambda path: fake)

    assert run_eval._compute_graph_path_count({"question": "test"}) == 1
    assert fake.called is True


def test_eval_graph_count_uses_the_released_graph_retriever_contract(monkeypatch, tmp_path):
    from app.core import config
    from app.services.evaluation import run_eval
    from app.services.kag.graph_store import NetworkXGraphStore

    graph_path = tmp_path / "graph.json"
    store = NetworkXGraphStore(graph_path)
    store.add_node("doc", node_type="RegulatoryDocument", title="SFC suitability requirement")
    store.save()

    class Settings:
        GRAPH_STORE_PATH = str(graph_path)

    monkeypatch.setattr(config, "get_settings", lambda: Settings())
    run_eval._get_cached_graph_retriever.cache_clear()

    assert run_eval._compute_graph_path_count({"question": "SFC suitability"}) == 1
