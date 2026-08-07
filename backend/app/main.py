"""
FastAPI 应用主入口 (Main Entrypoint)
挂载所有业务路由、配置 CORS、启动 LangSmith 追踪、暴露健康检查端点。
安全策略：可选的 API Key 认证 + 生产环境关闭 Swagger 文档。
"""
import sys

# Windows GBK 控制台兼容：强制 stdout/stderr 使用 UTF-8 编码，避免 emoji 字符输出崩溃
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import traceback

from app.core.config import get_settings
from app.core.monitoring import get_tracker, setup_langsmith
from app.core.security import verify_api_key
from app.core.rate_limit import RateLimitMiddleware
from app.core.health import readiness_report
from app.core.startup_checks import run_startup_checks
from app.schemas.requests import HealthResponse

from app.api.routers import svf, bank_account, cross_border, sme_lending, review_queue, research, kag, copilot

settings = get_settings()

# --- 启动 LangSmith 追踪 ---
setup_langsmith()

# --- Swagger 文档仅在 DEBUG 模式下开放 ---
app = FastAPI(
    title=settings.APP_TITLE,
    version="2.0.0",
    description="HK-FinReg AI Backend — Multi-Agent Compliance Engine powered by LangGraph",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# --- 启动安全检查 ---
startup_warnings = run_startup_checks(settings)

# --- CORS 配置 (已移除 * 通配符) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# --- 速率限制中间件 ---
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.RATE_LIMIT_RPM,
    requests_per_hour=settings.RATE_LIMIT_RPH,
    storage_url=settings.RATE_LIMIT_STORAGE_URL,
)
app.state.settings = settings

# --- 挂载业务路由 (受 API Key 保护) ---
app.include_router(svf.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(bank_account.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(cross_border.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(sme_lending.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(review_queue.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(research.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(kag.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(copilot.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])


# --- 测试客户端页面 (仅 DEBUG 模式) ---
if settings.DEBUG:
    @app.get("/test", tags=["System"])
    async def serve_test_client():
        """返回 HTML 测试客户端 (仅开发环境)"""
        html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_client.html")
        return FileResponse(html_path, media_type="text/html")


# --- 健康检查 (公开端点，不需要认证，但隐藏敏感配置状态) ---
def _dependency_checks() -> dict[str, bool]:
    """Perform bounded local readiness checks without calling external services."""

    graph_path = os.path.join(os.path.dirname(__file__), "..", "data", "graph", "regulatory_graph.json")
    corpus_cache_path = os.path.join(
        settings.CORPUS_INDEX_DIR, "corpus_documents.json"
    )
    llm_configured = bool(settings.COPILOT_API_KEY or settings.ZHIPU_API_KEY or settings.LONGCAT_API_KEY)
    return {
        "llm_service": llm_configured,
        "corpus_index": os.path.isfile(corpus_cache_path),
        "graph_store": os.path.exists(settings.GRAPH_STORE_PATH) or os.path.exists(graph_path),
    }


@app.get("/api/v1/health/live", response_model=HealthResponse, tags=["System"])
async def health_live():
    tracker = get_tracker()
    langsmith_status = "enabled" if os.environ.get("LANGCHAIN_TRACING_V2") == "true" else "disabled"
    return HealthResponse(
        status="ok",
        version="2.0.0",
        engines={
            "process": "alive",
            "langsmith_tracing": langsmith_status,
            "total_queries": tracker.session_stats["total_queries"]
        }
    )


@app.get("/api/v1/health/ready", response_model=HealthResponse, tags=["System"])
async def health_ready():
    report = readiness_report(_dependency_checks())
    status_code = 200 if report["status"] == "ready" else 503
    return JSONResponse(
        status_code=status_code,
        content=HealthResponse(
            status=report["status"], version="2.0.0", engines=report["engines"]
        ).model_dump(),
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return await health_live()


@app.get("/api/v1/metrics", tags=["System"], dependencies=[Depends(verify_api_key)])
async def get_metrics():
    """返回当前会话性能统计（需认证）"""
    tracker = get_tracker()
    return tracker.get_session_summary()


# --- 全局异常处理 (防止内部错误信息泄露) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理异常，避免堆栈信息泄露到客户端"""
    # 在服务端记录完整错误
    traceback.print_exc()
    # 返回通用错误响应，不暴露内部细节
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error. Please try again later."},
    )
