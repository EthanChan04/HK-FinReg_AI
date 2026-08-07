from app.schemas.corpus import SourceDocument
from app.schemas.evidence import EvidenceChunk


def test_graph_store_round_trip(tmp_path):
    from app.services.kag.graph_store import NetworkXGraphStore

    path = tmp_path / "graph.json"
    store = NetworkXGraphStore(path)
    store.add_node("doc_1", node_type="Document", title="Doc")
    store.add_node("HKMA", node_type="Regulator", title="HKMA")
    store.add_edge("doc_1", "HKMA", relation="issued_by")
    store.save()

    loaded = NetworkXGraphStore(path)
    loaded.load()

    assert loaded.graph.has_node("doc_1")
    assert loaded.graph.has_edge("doc_1", "HKMA")
    assert loaded.graph.edges["doc_1", "HKMA"]["relation"] == "issued_by"


def test_graph_builder_creates_document_regulator_and_topic_edges(tmp_path):
    from app.services.kag.graph_builder import build_graph_from_sources

    docs = [
        SourceDocument(
            doc_id="doc_1",
            title="Doc",
            regulator="HKMA",
            doc_type="Guideline",
            topics=["AML", "CDD"],
            module_tags=["svf"],
            file_path="doc.pdf",
        )
    ]
    evidence = [
        EvidenceChunk(
            evidence_id="source_1",
            chunk_id="chunk_1",
            doc_id="doc_1",
            title="Doc",
            regulator="HKMA",
            section_title="Customer Due Diligence",
            text="SVF licensees must perform customer due diligence.",
        )
    ]

    store = build_graph_from_sources(docs, evidence, graph_path=tmp_path / "graph.json")

    assert store.graph.has_edge("doc_1", "regulator:HKMA")
    assert store.graph.has_edge("doc_1", "topic:AML")
    assert store.graph.has_edge("doc_1", "chunk:chunk_1")
    assert store.graph.edges["doc_1", "regulator:HKMA"]["relation"] == "ISSUED_BY"


def test_graph_retriever_returns_matching_paths(tmp_path):
    from app.services.kag.graph_builder import build_graph_from_sources
    from app.services.kag.graph_retriever import GraphRetriever

    docs = [
        SourceDocument(
            doc_id="doc_ai",
            title="AI Circular",
            regulator="HKMA",
            doc_type="Circular",
            topics=["AI", "consumer_protection"],
            module_tags=["ai_regtech"],
            file_path="ai.pdf",
        )
    ]
    store = build_graph_from_sources(docs, [], graph_path=tmp_path / "graph.json")
    retriever = GraphRetriever(store)

    paths = retriever.retrieve_paths("AI consumer protection obligations")

    assert paths
    assert paths[0]["matched_doc_ids"] == ["doc_ai"]
    assert "confidence" in paths[0]


def test_graph_retriever_returns_regulator_document_topic_path(tmp_path):
    from app.services.kag.graph_builder import build_graph_from_sources
    from app.services.kag.graph_retriever import GraphRetriever

    docs = [
        SourceDocument(
            doc_id="doc_ai",
            title="AI Circular",
            regulator="HKMA",
            doc_type="Circular",
            topics=["AI"],
            module_tags=["ai_regtech"],
            file_path="ai.pdf",
        )
    ]
    store = build_graph_from_sources(docs, [], graph_path=tmp_path / "graph.json")
    retriever = GraphRetriever(store)

    paths = retriever.retrieve_paths("Which regulators apply to AI?")

    assert ["HKMA", "AI Circular", "AI"] in [item["path"] for item in paths]


def test_graph_retriever_respects_metadata_filters(tmp_path):
    from app.services.kag.graph_builder import build_graph_from_sources
    from app.services.kag.graph_retriever import GraphRetriever

    docs = [
        SourceDocument(
            doc_id="doc_hkma",
            title="HKMA Circular",
            regulator="HKMA",
            doc_type="Circular",
            topics=["AI"],
            module_tags=["ai_regtech"],
            file_path="hkma.pdf",
        ),
        SourceDocument(
            doc_id="doc_sfc",
            title="SFC Circular",
            regulator="SFC",
            doc_type="Circular",
            topics=["AI"],
            module_tags=["ai_regtech"],
            file_path="sfc.pdf",
        ),
    ]
    store = build_graph_from_sources(docs, [], graph_path=tmp_path / "graph.json")
    retriever = GraphRetriever(store)

    paths = retriever.retrieve_paths("AI obligations", filters={"regulator": "SFC"})

    assert paths
    assert all(path["matched_doc_ids"] == ["doc_sfc"] for path in paths)


def test_graph_builder_activates_reference_and_supersedes_relations(tmp_path):
    from app.services.kag.graph_builder import build_graph_from_sources

    docs = [
        SourceDocument(
            doc_id="doc_current",
            title="Current",
            regulator="SFC",
            doc_type="Guideline",
            file_path="current.pdf",
            references=["doc_reference"],
            supersedes=["doc_old"],
        ),
        SourceDocument(
            doc_id="doc_old",
            title="Old",
            regulator="SFC",
            doc_type="Guideline",
            file_path="old.pdf",
            status="superseded",
        ),
        SourceDocument(
            doc_id="doc_reference",
            title="Reference",
            regulator="SFC",
            doc_type="Code",
            file_path="reference.pdf",
        ),
    ]

    store = build_graph_from_sources(docs, [], graph_path=tmp_path / "graph.json")

    assert store.graph.edges["doc_current", "doc_reference"]["relation"] == "REFERENCES"
    assert store.graph.edges["doc_current", "doc_old"]["relation"] == "SUPERSEDES"
