"""
Pydantic 请求体与响应体定义 (Schemas)
用于 FastAPI 路由做严格的数据入参验证与出参序列化。
"""
from pydantic import BaseModel, Field
from typing import Optional


# ==========================================
# 通用请求体
# ==========================================
class ComplianceRequest(BaseModel):
    """所有合规审查接口的统一请求体"""
    application_data: str = Field(..., min_length=10, description="用户提交的业务申请文本")
    business_context: Optional[str] = Field(None, description="可选的附加业务上下文")
    stream_agents_state: bool = Field(False, description="是否通过 SSE 播报 Agent 节点状态")


# ==========================================
# 通用响应体
# ==========================================
class ComplianceMetrics(BaseModel):
    """性能指标"""
    processing_time: float
    total_agents: int = 4


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
