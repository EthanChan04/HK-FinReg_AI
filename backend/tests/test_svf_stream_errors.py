import asyncio
import time
from types import SimpleNamespace

from langchain_core.documents import Document


def _collect_stream(generator):
    async def _collect():
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)
        return chunks

    return asyncio.run(_collect())


def test_stream_svf_reports_graph_initialization_errors(monkeypatch):
    from app.api.routers import svf

    def fail_build_graph(*args, **kwargs):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr(svf, "_build_svf_graph", fail_build_graph)

    chunks = _collect_stream(svf._stream_svf("test svf input"))

    joined = "".join(chunks)
    assert "event: error" in joined
    assert "embedding service unavailable" in joined
    assert "event: done" in joined


def test_stream_svf_sends_keepalive_during_chat_model_stream(monkeypatch):
    from app.api.routers import svf

    class FakeGraph:
        async def astream_events(self, *args, **kwargs):
            for _ in range(20):
                yield {"event": "on_chat_model_stream"}
            yield {
                "event": "on_chain_end",
                "name": "reviewer",
                "data": {"output": {"final_report": "ok"}},
            }

    monkeypatch.setattr(svf, "_build_svf_graph", lambda *args, **kwargs: FakeGraph())

    chunks = _collect_stream(svf._stream_svf("test svf input"))

    assert ": keepalive" in "".join(chunks)


def test_stream_svf_sends_keepalive_while_building_graph(monkeypatch):
    from app.api.routers import svf

    class FakeGraph:
        async def astream_events(self, *args, **kwargs):
            yield {
                "event": "on_chain_end",
                "name": "reviewer",
                "data": {"output": {"final_report": "ok"}},
            }

    def slow_build_graph(*args, **kwargs):
        time.sleep(0.05)
        return FakeGraph()

    monkeypatch.setattr(svf, "_build_svf_graph", slow_build_graph)
    monkeypatch.setattr(svf, "GRAPH_BUILD_KEEPALIVE_INTERVAL_SECONDS", 0.01)

    chunks = _collect_stream(svf._stream_svf("test svf input"))

    assert "".join(chunks).count(": keepalive") >= 2


def test_stream_svf_sends_keepalive_while_waiting_for_graph_event(monkeypatch):
    from app.api.routers import svf

    class FakeGraph:
        async def astream_events(self, *args, **kwargs):
            await asyncio.sleep(0.05)
            yield {
                "event": "on_chain_end",
                "name": "reviewer",
                "data": {"output": {"final_report": "ok"}},
            }

    monkeypatch.setattr(svf, "_build_svf_graph", lambda *args, **kwargs: FakeGraph())
    monkeypatch.setattr(svf, "GRAPH_EVENT_KEEPALIVE_INTERVAL_SECONDS", 0.01)

    chunks = _collect_stream(svf._stream_svf("test svf input"))

    assert "".join(chunks).count(": keepalive") >= 2


def test_blocking_svf_graph_has_checkpoint_imports_available():
    from app.api.routers import svf

    assert callable(svf.get_checkpoint_manager)
    assert hasattr(svf.CheckpointManager, "generate_workflow_run_id")


def test_stream_svf_emits_retrieval_payloads_for_frontend_panels(monkeypatch):
    from app.api.routers import svf

    class FakeGraph:
        async def astream_events(self, *args, **kwargs):
            yield {
                "event": "on_chain_end",
                "name": "retriever",
                "data": {
                    "output": {
                        "evidence_chunks": [
                            {
                                "evidence_id": "source_1",
                                "doc_id": "doc_1",
                                "title": "Doc",
                                "regulator": "HKMA",
                                "text": "Evidence text",
                            }
                        ],
                        "graph_paths": [
                            {
                                "path": ["HKMA", "Doc", "AI"],
                                "matched_node": "topic:AI",
                                "matched_topics": ["AI"],
                            }
                        ],
                    }
                },
            }
            yield {
                "event": "on_chain_end",
                "name": "reviewer",
                "data": {"output": {"final_report": "ok"}},
            }

    monkeypatch.setattr(svf, "_build_svf_graph", lambda *args, **kwargs: FakeGraph())

    chunks = _collect_stream(svf._stream_svf("Which regulators apply to AI?"))
    joined = "".join(chunks)

    assert "event: evidence_chunks" in joined
    assert "source_1" in joined
    assert "event: graph_paths" in joined
    assert "topic:AI" in joined


def test_blocking_svf_graph_uses_draft_report_in_reviewer(monkeypatch):
    from app.api.routers import svf

    report = """# SVF Compliance Report

報告日期: 2026-05-10

## 申請人概覽
This applicant is an SVF licensee seeking AML and CDD review support.

## 法規事實摘要
- SVF licensees must maintain customer due diligence controls. [Source 1, p.1]

## 合規差距分析
The available facts are sufficient for a focused control gap review.

## 合規建議
Maintain documented controls and review suspicious transaction monitoring.

## 風險評級
中風險

## 資訊不足聲明
No material information gap was identified for this test.

```json
{"overall_confidence": 0.8, "low_confidence_areas": [], "high_confidence_areas": ["AML"]}
```
"""

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content='{"entity_type":"SVF","topic":"CDD"}')
            return SimpleNamespace(content=report)

    class FakeRetriever:
        def invoke(self, query):
            return [
                Document(
                    page_content="SVF licensees must maintain customer due diligence controls.",
                    metadata={
                        "page": 1,
                        "rerank_score": 0.9,
                        "doc_id": "doc_1",
                        "title": "SVF AML Guideline",
                        "regulator": "HKMA",
                    },
                )
            ]

    monkeypatch.setattr(svf, "build_deepseek_llm", lambda _profile: FakeLLM())
    monkeypatch.setattr(svf, "build_reranked_retriever", lambda: FakeRetriever())

    final_report = svf._run_svf_graph("SVF CDD obligations")

    assert "SVF Compliance Report" in final_report
