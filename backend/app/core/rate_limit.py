"""Identity-aware rate limiting with an optional Redis-backed store."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def rate_limit_identity(request: Request) -> str:
    """Prefer authenticated identities, falling back to a client address."""

    authorization = request.headers.get("authorization")
    if authorization:
        return f"credential:{_stable_digest(authorization)}"
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"credential:{_stable_digest(api_key)}"
    tenant = request.headers.get("x-tenant-id")
    if tenant:
        return f"tenant:{_stable_digest(tenant)}"
    user = request.headers.get("x-user-id")
    if user:
        return f"user:{_stable_digest(user)}"

    settings = getattr(request.app.state, "settings", None)
    if settings and getattr(settings, "TRUSTED_PROXY_HEADERS", False):
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


class InMemoryRateLimitStore:
    """Process-local fallback store used for development and single replicas."""

    def __init__(self):
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    async def allow(self, key: str, now: float, rpm: int, rph: int) -> bool:
        with self._lock:
            window = [timestamp for timestamp in self._windows[key] if now - timestamp < 3600]
            self._windows[key] = window
            if sum(now - timestamp < 60 for timestamp in window) >= rpm:
                return False
            if len(window) >= rph:
                return False
            window.append(now)
            return True


class RedisRateLimitStore:
    """Atomic sliding-window store shared by all application replicas."""

    _SCRIPT = """
    local now = tonumber(ARGV[1])
    local rpm = tonumber(ARGV[2])
    local rph = tonumber(ARGV[3])
    redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - 3600)
    local minute = redis.call('ZCOUNT', KEYS[1], now - 60, '+inf')
    local hour = redis.call('ZCARD', KEYS[1])
    if minute >= rpm or hour >= rph then
      return 0
    end
    local sequence = redis.call('INCR', KEYS[1] .. ':sequence')
    redis.call('ZADD', KEYS[1], now, tostring(now) .. '-' .. tostring(sequence))
    redis.call('EXPIRE', KEYS[1], 3700)
    redis.call('EXPIRE', KEYS[1] .. ':sequence', 3700)
    return 1
    """

    def __init__(self, redis_url: str):
        from redis import asyncio as redis

        # redis.from_url() is lazy: it does not connect. Connection failures
        # surface on the first eval() call, so allow() must degrade at call time.
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._fallback = InMemoryRateLimitStore()
        self._degraded = False

    async def allow(self, key: str, now: float, rpm: int, rph: int) -> bool:
        if self._degraded:
            return await self._fallback.allow(key, now, rpm, rph)
        try:
            result = await self._client.eval(self._SCRIPT, 1, f"rate-limit:{key}", now, rpm, rph)
            return bool(result)
        except Exception as exc:
            # Redis unreachable: fall back to the process-local store and warn once.
            self._degraded = True
            logger.warning(
                "Redis rate limit store unavailable (%s); degraded to in-memory store "
                "for the rest of this process. Counts are no longer shared across replicas.",
                exc,
            )
            return await self._fallback.allow(key, now, rpm, rph)


def build_rate_limit_store(storage_url: str | None):
    if not storage_url:
        return InMemoryRateLimitStore()
    try:
        return RedisRateLimitStore(storage_url)
    except Exception as exc:
        logger.warning("Redis rate limit unavailable; using local fallback: %s", exc)
        return InMemoryRateLimitStore()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-identity minute/hour limits, shared through Redis when configured."""

    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 500, storage_url: str | None = None):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.store = build_rate_limit_store(storage_url)

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/v1/health"):
            return await call_next(request)

        allowed = await self.store.allow(rate_limit_identity(request), time.time(), self.rpm, self.rph)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
