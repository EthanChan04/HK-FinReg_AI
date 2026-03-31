"""
安全认证模块 (Security)

提供可配置的 API Key 认证机制。
通过 .env 中的 API_KEY_ENABLED 和 API_KEY 控制是否启用。
在 FastAPI 路由中使用 Depends(verify_api_key) 进行保护。
"""
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from app.core.config import get_settings

# HTTP Bearer scheme (在 Swagger 文档中自动显示锁图标)
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)
):
    """
    API Key 校验依赖项。

    行为：
      - 当 API_KEY_ENABLED=False (默认) 时：跳过验证，所有请求放行
      - 当 API_KEY_ENABLED=True 时：要求请求头携带 Authorization: Bearer <API_KEY>

    用法：
      @router.post("/endpoint", dependencies=[Depends(verify_api_key)])
      或在路由器级别:
      router = APIRouter(dependencies=[Depends(verify_api_key)])
    """
    settings = get_settings()

    # 未启用 API Key 认证 — 开发模式直接放行
    if not settings.API_KEY_ENABLED:
        return

    # 启用了认证但未配置 API_KEY — 服务端配置错误
    if not settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: API_KEY_ENABLED is True but API_KEY is not set"
        )

    # 检查请求是否携带了有效的 Bearer Token
    if credentials is None or credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Use header: Authorization: Bearer <your_api_key>",
            headers={"WWW-Authenticate": "Bearer"}
        )
