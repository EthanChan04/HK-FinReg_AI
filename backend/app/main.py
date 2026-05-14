"""
FastAPI 应用主入口 (Main Entrypoint)
挂载所有业务路由、配置 CORS、启动 LangSmith 追踪、暴露健康检查端点。
安全策略：可选的 API Key 认证 + 生产环境关闭 Swagger 文档。
"""
import sys
import io

# Windows GBK 控制台兼容：强制 stdout/stderr 使用 UTF-8 编码，避免 emoji 字符输出崩溃
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
import traceback

from app.core.config import get_settings
from app.core.monitoring import get_tracker, setup_langsmith
from app.core.security import verify_api_key
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

# --- CORS 配置 (已移除 * 通配符) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

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
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    tracker = get_tracker()
    langsmith_status = "enabled" if os.environ.get("LANGCHAIN_TRACING_V2") == "true" else "disabled"
    return HealthResponse(
        status="ok",
        version="2.0.0",
        engines={
            "llm_service": "available",
            "langsmith_tracing": langsmith_status,
            "total_queries": tracker.session_stats["total_queries"]
        }
    )


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
