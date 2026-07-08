"""DeepResearch API routes."""

from fastapi import APIRouter

from app.schemas.deepresearch import ResearchRequest
from app.services.deepresearch.workflow import build_deepresearch_graph
from app.services.utils import pii_scrubber

router = APIRouter(prefix="/research", tags=["DeepResearch"])


@router.post("/analyze")
def analyze_research(request: ResearchRequest):
    """Run the bounded DeepResearch workflow."""

    # PII 脱敏
    safe_query = pii_scrubber(request.query)

    graph = build_deepresearch_graph()
    return graph.invoke(
        {
            "original_query": safe_query,
            "request": request.model_dump(),
            "iteration": 0,
        }
    )
