from langchain_core.documents import Document


def test_reranker_skips_calls_during_cooldown(monkeypatch):
    from app.services.agents import reranker

    docs = [Document(page_content="a"), Document(page_content="b")]
    monkeypatch.setattr(reranker, "_COHERE_RATE_LIMITED_UNTIL", 9_999_999_999.0)

    called = {"value": False}

    class FakeClient:
        def rerank(self, **kwargs):
            called["value"] = True
            raise AssertionError("should not be called during cooldown")

    monkeypatch.setattr(reranker, "_get_cohere_client", lambda: FakeClient())
    out = reranker.rerank_documents("test", docs, top_k=1)

    assert len(out) == 1
    assert called["value"] is False


def test_reranker_sets_cooldown_on_rate_limit(monkeypatch):
    from app.services.agents import reranker

    docs = [Document(page_content="a"), Document(page_content="b")]
    monkeypatch.setattr(reranker, "_COHERE_RATE_LIMITED_UNTIL", 0.0)

    class FakeRateLimitError(Exception):
        status_code = 429

    class FakeClient:
        def rerank(self, **kwargs):
            raise FakeRateLimitError("status_code: 429")

    monkeypatch.setattr(reranker, "_get_cohere_client", lambda: FakeClient())
    out = reranker.rerank_documents("test", docs, top_k=2)

    assert len(out) == 2
    assert reranker._COHERE_RATE_LIMITED_UNTIL > 0
