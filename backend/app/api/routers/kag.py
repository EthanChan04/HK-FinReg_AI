"""KAG API routes."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.kag import ObligationMapRequest, ObligationMapResponse
from app.services.agents.builder import build_reranked_retriever
from app.services.kag.graph_retriever import GraphRetriever
from app.services.kag.graph_store import NetworkXGraphStore
from app.services.kag.obligation_mapper import ObligationMapper
from app.services.retrieval.retrieval_service import RetrievalService

router = APIRouter(prefix="/kag", tags=["KAG"])


def _build_graph_retriever() -> GraphRetriever:
    settings = get_settings()
    store = NetworkXGraphStore(settings.GRAPH_STORE_PATH)
    store.load()
    return GraphRetriever(store)


@router.post("/obligation-map", response_model=ObligationMapResponse)
def map_obligations(request: ObligationMapRequest) -> ObligationMapResponse:
    graph_retriever = _build_graph_retriever()
    retrieval = RetrievalService(retriever=build_reranked_retriever())
    mapper = ObligationMapper()
    return mapper.map_obligations(
        query=request.query,
        product_profile=request.product_profile,
        graph_retriever=graph_retriever,
        retrieval_service=retrieval,
    )


@router.post("/graph/search")
def search_graph(request: ObligationMapRequest):
    graph_retriever = _build_graph_retriever()
    return {"paths": graph_retriever.retrieve_paths(request.query, limit=8)}

