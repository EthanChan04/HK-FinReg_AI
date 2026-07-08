"""
API 速率限制中间件 (Rate Limiting)

基于滑动窗口算法的内存限流器。
防止 API 滥用和 LLM 调用成本失控。

生产环境建议替换为 Redis-backed 方案以支持多实例部署。
"""
import time
from collections import defaultdict

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """按客户端 IP 的滑动窗口限流中间件。

    配置项（通过 Settings 注入）：
        requests_per_minute: 每分钟最大请求数
        requests_per_hour: 每小时最大请求数

    跳过限流的端点：
        /api/v1/health — 健康检查不计入限流
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 500,
    ):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._windows: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # 健康检查端点不限流
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # 清理过期记录（保留最近 1 小时）
        self._windows[client_ip] = [
            t for t in self._windows[client_ip] if now - t < 3600
        ]

        window = self._windows[client_ip]

        # 检查每分钟限制
        recent_minute = [t for t in window if now - t < 60]
        if len(recent_minute) >= self.rpm:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: too many requests per minute. Please retry later.",
            )

        # 检查每小时限制
        if len(window) >= self.rph:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: too many requests per hour. Please retry later.",
            )

        # 记录本次请求
        window.append(now)

        return await call_next(request)
