"""
Pydantic 请求体与响应体定义 (Schemas)
用于 FastAPI 路由做严格的数据入参验证与出参序列化。
"""
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ==========================================
# Phase 1: HITL Gate 类型
# ==========================================

class GateType(str, Enum):
    """三类标准中断 gate"""
    LOW_CONFIDENCE = "low_confidence_gate"
    MISSING_EVIDENCE = "missing_evidence_gate"
    MANUAL_APPROVAL = "manual_approval_gate"


# ==========================================
# 通用请求体
# ==========================================
class ComplianceRequest(BaseModel):
    """所有合规审查接口的统一请求体"""
    application_data: str = Field(
        ..., min_length=10,
        description="用户提交的业务申请文本",
    )
    business_context: Optional[str] = Field(None, description="可选的附加业务上下文")
    stream_agents_state: bool = Field(False, description="是否通过 SSE 播报 Agent 节点状态")

    @field_validator("application_data")
    @classmethod
    def validate_max_length(cls, v: str) -> str:
        """从配置中读取最大长度限制，防止超大文本导致 LLM API 成本失控"""
        from app.core.config import get_settings
        settings = get_settings()
        max_len = getattr(settings, "MAX_INPUT_LENGTH", 50000)
        if len(v) > max_len:
            raise ValueError(f"application_data exceeds maximum length of {max_len} characters")
        return v


# ==========================================
# 通用响应体
# ==========================================
class ComplianceMetrics(BaseModel):
    """性能指标"""
    processing_time: float
    total_agents: int = 5


class ComplianceResponse(BaseModel):
    """阻塞式接口统一响应体"""
    status: str = "success"
    scrubbed_input: str = ""
    final_report: str = ""
    metrics: ComplianceMetrics


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "2.0.0"
    engines: dict = {}


class ErrorResponse(BaseModel):
    """错误响应"""
    status: str = "error"
    detail: str


class SourceCitation(BaseModel):
    """A single source citation extracted from the report."""

    source_number: int = Field(..., ge=1, description="Source index in retrieved context.")
    page: int = Field(..., ge=0, description="Source page number.")
    content: str = Field(..., min_length=10, description="Quoted or cited claim content.")


class RegulatoryFact(BaseModel):
    """A grounded regulatory fact that must carry at least one citation."""

    statement: str = Field(..., min_length=10)
    citations: List[SourceCitation] = Field(..., min_length=1)

    @field_validator("statement")
    @classmethod
    def must_not_contain_recommendation_language(cls, value: str) -> str:
        forbidden_patterns = ("建議", "建议", "should", "recommend", "suggest")
        lowered = value.lower()
        for pattern in forbidden_patterns:
            if pattern.lower() in lowered:
                raise ValueError(f"Regulatory facts must not contain recommendation language: '{pattern}'")
        return value


class AnalyzerOutput(BaseModel):
    """Structured Analyzer output enforced after markdown parsing."""

    report_title: str = Field(..., min_length=5)
    report_date: str = Field(..., min_length=8)
    applicant_overview: str = Field(..., min_length=20)
    regulatory_facts: List[RegulatoryFact] = Field(..., min_length=1)
    gap_analysis: str = Field(..., min_length=20)
    recommendations: str = Field(..., min_length=20)
    risk_rating: Literal["低風險", "中風險", "高風險"]
    insufficiency_disclaimer: str = Field(..., min_length=10)

    @field_validator("regulatory_facts")
    @classmethod
    def all_facts_must_have_citations(cls, facts: List[RegulatoryFact]) -> List[RegulatoryFact]:
        for fact in facts:
            if not fact.citations:
                raise ValueError(f"Regulatory fact is missing citations: {fact.statement[:50]}...")
        return facts


class ConfidenceScore(BaseModel):
    """三维置信度模型 (P2 M1)

    三维指标：
      1. retrieval_confidence — 检索置信度（Rerank Top-1 / Top-5 gap）
      2. reasoning_confidence — 推理置信度（Analyzer 自评）
      3. cross_validation_passed — Reviewer 交叉验证是否通过（偏差检测）
    """

    retrieval_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="检索置信度：Cohere Rerank Top-1 分数 (0.0-1.0)",
    )
    top5_gap: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Rerank Top-1 与 Top-5 均值的分数差距（越大表示 Top-1 独特性越高）",
    )
    reasoning_confidence: float = Field(
        0.5, ge=0.0, le=1.0,
        description="推理置信度：Analyzer 自评的综合置信度 (0.0-1.0)",
    )
    low_confidence_areas: List[str] = Field(
        default_factory=list,
        description="低置信度领域列表（如 ['CDD/KYC', '制裁筛查']）",
    )
    high_confidence_areas: List[str] = Field(
        default_factory=list,
        description="高置信度领域列表",
    )
    cross_validation_passed: bool = Field(
        True,
        description="Reviewer 交叉验证是否通过（retrieval vs reasoning 偏差 <= 阈值）",
    )
    cross_validation_deviation: float = Field(
        0.0, ge=0.0, le=1.0,
        description="retrieval_confidence 与 reasoning_confidence 的绝对偏差",
    )

    @property
    def overall_confidence(self) -> float:
        """综合置信度 = 加权平均（retrieval 40% + reasoning 60%，交叉验证失败时 -0.2 惩罚）"""
        base = self.retrieval_confidence * 0.4 + self.reasoning_confidence * 0.6
        if not self.cross_validation_passed:
            base = max(0.0, base - 0.2)
        return round(base, 3)

    @property
    def warning_level(self) -> str:
        """置信度警告级别: low / medium / high"""
        o = self.overall_confidence
        if o < 0.4:
            return "high"
        if o < 0.6:
            return "medium"
        return "low"


class ReviewerVerdict(BaseModel):
    """Structured reviewer verdict."""

    decision: Literal["APPROVED", "REJECTED"]
    failed_checks: List[int] = Field(default_factory=list, description="Checklist numbers that failed.")
    reason: Optional[str] = None
    required_fix: Optional[str] = None
    rejection_type: Optional[Literal["insufficient_info", "quality_issue"]] = Field(
        None,
        description=(
            "insufficient_info: the retrieved context lacks information needed for the report. "
            "quality_issue: the report has structural/logical/formatting problems."
        ),
    )
    reviewer_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Reviewer 独立评估的置信度分数 (0.0-1.0)，用于交叉验证",
    )

    @model_validator(mode="after")
    def rejected_must_have_reason(self) -> "ReviewerVerdict":
        if self.decision == "REJECTED" and not self.reason:
            raise ValueError("REJECTED verdict must include a reason")
        if self.decision == "REJECTED" and not self.required_fix:
            raise ValueError("REJECTED verdict must include a required_fix")
        if self.decision == "REJECTED" and not self.rejection_type:
            raise ValueError("REJECTED verdict must include a rejection_type")
        return self


# ==========================================
# Phase 1: HITL Review Queue Schemas
# ==========================================

class ReviewQueueItemResponse(BaseModel):
    """审查队列条目响应"""
    workflow_run_id: str
    gate_type: str = Field(..., description="触发暂停的 gate 类型")
    checkpoint_created_at: float = Field(..., description="Checkpoint 创建时间戳")
    evidence_snapshot: dict = Field(default_factory=dict, description="证据快照")
    latest_draft_report: str = Field("", description="最新草稿报告")
    confidence_data: dict = Field(default_factory=dict, description="置信度数据")
    original_input: str = Field("", description="原始输入")
    human_review_status: str = Field("pending", description="审查状态: pending/approved/rejected")
    human_review_notes: str = Field("", description="人工批注")
    reviewed_at: Optional[float] = Field(None, description="审查时间戳")
    reviewed_by: Optional[str] = Field(None, description="审查人")


class ReviewResumeRequest(BaseModel):
    """恢复执行请求"""
    notes: str = Field("", description="人工批注")
    reviewed_by: str = Field("", description="审查人标识")
    additional_context: Optional[str] = Field(None, description="补充的外部事实或上下文")


class ReviewRejectRequest(BaseModel):
    """驳回请求"""
    notes: str = Field(..., min_length=1, description="驳回原因")
    reviewed_by: str = Field("", description="审查人标识")
