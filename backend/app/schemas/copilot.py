"""Schemas for Compliance Copilot chat streaming."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.evidence import EvidenceChunk

CopilotIntent = Literal[
    "regulatory_qa",
    "case_explanation",
    "obligation_mapping",
    "workflow_recommendation",
    "deep_research",
    "human_review_help",
    "smalltalk_or_help",
]

CopilotToolName = Literal[
    "rag",
    "kag",
    "deepresearch",
    "human_review",
    "workflow_router",
    "deepseek",
]


class CopilotMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class CopilotCaseContext(BaseModel):
    workspace_id: str | None = None
    workflow_id: str | None = None
    workflow_name: str | None = None
    input_text: str | None = None
    report_text: str | None = None
    evidence_chunks: list[dict] = Field(default_factory=list)
    graph_paths: list[dict] = Field(default_factory=list)
    research_plan: dict | None = None
    confidence_data: dict = Field(default_factory=dict)
    workflow_run_id: str | None = None
    current_gate: str | None = None
    gate_message: str | None = None


class CopilotChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    conversation_id: str | None = None
    history: list[CopilotMessage] = Field(default_factory=list)
    case_context: CopilotCaseContext = Field(default_factory=CopilotCaseContext)
    preferred_language: Literal["zh-HK+en"] = "zh-HK+en"


class CopilotIntentEvent(BaseModel):
    intent: CopilotIntent
    engine: str
    reason: str


class CopilotToolEvent(BaseModel):
    tool: CopilotToolName
    status: Literal["running", "done", "error"]
    message: str | None = None


class CopilotCitationAuditEvent(BaseModel):
    unsupported_claim_rate: float = 0.0


class CopilotDoneEvent(BaseModel):
    conversation_id: str
    intent: CopilotIntent
    engine: str


class CopilotRuntimePayload(BaseModel):
    """Intermediate payload returned by tool router for response writing."""

    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    graph_paths: list[dict] = Field(default_factory=list)
    research_plan: dict | None = None
    workflow_recommendation: dict | None = None
    review_guidance: dict | None = None
    notes: list[str] = Field(default_factory=list)
