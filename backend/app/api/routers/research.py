"""DeepResearch API routes."""

from fastapi import APIRouter

from app.schemas.deepresearch import ResearchRequest
from app.services.deepresearch.workflow import build_deepresearch_graph

router = APIRouter(prefix="/research", tags=["DeepResearch"])


@router.post("/analyze")
def analyze_research(request: ResearchRequest):
    """Run the bounded DeepResearch workflow."""

    graph = build_deepresearch_graph()
    return graph.invoke(
        {
            "original_query": request.query,
            "iteration": 0,
        }
    )
