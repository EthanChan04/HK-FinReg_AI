import types


def test_local_hash_embeddings_are_deterministic():
    from app.services.agents.builder import LocalHashEmbeddings

    emb = LocalHashEmbeddings(dimensions=64)
    v1 = emb.embed_query("KYC risk assessment for cross-border onboarding")
    v2 = emb.embed_query("KYC risk assessment for cross-border onboarding")

    assert len(v1) == 64
    assert v1 == v2
    assert any(abs(x) > 0 for x in v1)


def test_build_embeddings_client_falls_back_to_local_when_probe_fails(monkeypatch):
    from app.services.agents import builder

    settings = types.SimpleNamespace(
        EMBEDDING_PROVIDER="openai_compatible",
        EMBEDDING_MODEL="embedding-3",
        EMBEDDING_BASE_URL="https://invalid.example/v1",
        EMBEDDING_API_KEY="fake-key",
    )

    class FailingEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_query(self, text):
            raise RuntimeError("probe failed")

    monkeypatch.setattr(builder, "get_settings", lambda: settings)
    monkeypatch.setattr(builder, "OpenAIEmbeddings", FailingEmbeddings)

    emb = builder.build_embeddings_client()
    assert isinstance(emb, builder.LocalHashEmbeddings)


def test_empty_embedding_provider_uses_local_hash_without_remote_probe(monkeypatch):
    from app.services.agents import builder

    settings = types.SimpleNamespace(
        EMBEDDING_PROVIDER="",
        EMBEDDING_MODEL="",
        EMBEDDING_BASE_URL="",
        EMBEDDING_API_KEY="",
        EMBEDDING_DIMENSIONS=96,
    )

    class UnexpectedRemoteClient:
        def __init__(self, **_kwargs):
            raise AssertionError("the default embedding path must stay local")

    monkeypatch.setattr(builder, "get_settings", lambda: settings)
    monkeypatch.setattr(builder, "OpenAIEmbeddings", UnexpectedRemoteClient)

    emb = builder.build_embeddings_client()

    assert isinstance(emb, builder.LocalHashEmbeddings)
    assert emb.dimensions == 96
