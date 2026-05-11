"""Schemas for DeepResearch-style multi-step compliance research."""

from typing import Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str
    module: str | None = None
    max_iterations: int = 3
    language: str = "zh-HK"


class ResearchSubQuestion(BaseModel):
    id: str
    question: str
    retrieval_mode: Literal["rag", "kag"] = "rag"
    required_topics: list[str] = Field(default_factory=list)
    evidence_min_count: int = 2


class ResearchPlan(BaseModel):
    research_goal: str
    sub_questions: list[ResearchSubQuestion]
    expected_output_sections: list[str] = Field(default_factory=list)


class EvidenceGap(BaseModel):
    sub_question_id: str
    reason: str
    suggested_followup_query: str


class DeepResearchResult(BaseModel):
    research_plan: dict
    evidence_by_subquestion: dict[str, list[dict]]
    evidence_gaps: list[dict] = Field(default_factory=list)
    final_report: str
    citation_audit: dict
