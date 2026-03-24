"""
FastAPI 应用主入口 (Main Entrypoint)
挂载所有业务路由、配置 CORS、启动 LangSmith 追踪、暴露健康检查端点。
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.core.config import get_settings
from app.core.monitoring import get_tracker, setup_langsmith
from app.schemas.requests import HealthResponse

from app.api.routers import svf, bank_account, cross_border, sme_lending

settings = get_settings()

# --- 启动 LangSmith 追踪 ---
setup_langsmith()

app = FastAPI(
    title=settings.APP_TITLE,
    version="2.0.0",
    description="HK-FinReg AI Backend — Multi-Agent Compliance Engine powered by LangGraph",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS 配置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 挂载业务路由 ---
app.include_router(svf.router, prefix="/api/v1")
app.include_router(bank_account.router, prefix="/api/v1")
app.include_router(cross_border.router, prefix="/api/v1")
app.include_router(sme_lending.router, prefix="/api/v1")


# --- 测试客户端页面 ---
@app.get("/test", tags=["System"])
async def serve_test_client():
    """返回 HTML 测试客户端"""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_client.html")
    return FileResponse(html_path, media_type="text/html")


# --- 健康检查 ---
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    tracker = get_tracker()
    langsmith_status = "enabled" if os.environ.get("LANGCHAIN_TRACING_V2") == "true" else "disabled"
    return HealthResponse(
        status="ok",
        version="2.0.0",
        engines={
            "zhipu_glm": "configured" if settings.ZHIPU_API_KEY else "missing",
            "longcat_thinking": "configured" if settings.LONGCAT_API_KEY else "missing",
            "langsmith_tracing": langsmith_status,
            "langsmith_project": settings.LANGSMITH_PROJECT,
            "total_queries": tracker.session_stats["total_queries"]
        }
    )


@app.get("/api/v1/metrics", tags=["System"])
async def get_metrics():
    """返回当前会话性能统计"""
    tracker = get_tracker()
    return tracker.get_session_summary()
