"""
CORS 与安全配置 (Security)
"""
from fastapi import Request, HTTPException, status


async def verify_api_key(request: Request):
    """
    可选的 API Key 校验中间件。
    生产环境下可启用 Authorization: Bearer <key> 校验。
    开发阶段默认跳过。
    """
    # 开发阶段直接 pass
    pass
