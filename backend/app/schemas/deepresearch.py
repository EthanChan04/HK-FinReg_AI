"""Schemas for DeepResearch-style multi-step compliance research."""

from typing import Literal

from pydantic import BaseModel, Field


class ProductProfile(BaseModel):
    product_type: str | None = None
    business_activity: list[str] = Field(default_factory=list)
    target_customers: list[str] = Field(default_factory=list)
    data_used: list[str] = Field(default_factory=list)
    ai_used: bool = False
    cross_border: bool = False
    regulated_entities: list[str] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    query: str
    module: str | None = None
    task_type: Literal[
        "routine_review",
        "product_launch_review",
        "ai_governance_review",
        "cross_regulator_analysis",
        "regulatory_memo",
        "checklist_generation",
        "regulatory_change_impact",
    ] = "routine_review"
    product_profile: ProductProfile | None = None
    forced_regulators: list[str] = Field(default_factory=list)
    output_format: Literal["report", "checklist", "memo", "matrix"] = "report"
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
