"""Fallback behaviour of the Redis-backed rate limit store (T2-03, risk R-05).

redis.asyncio connects lazily, so a down Redis instance only fails on the
first eval() call. The store must degrade to the in-memory implementation
instead of letting the exception escape into the request path (which would
turn a rate limit outage into HTTP 500s).
"""

import asyncio

from app.core.rate_limit import RedisRateLimitStore


class _FailingClient:
    """Stand-in for redis.asyncio client whose eval() always fails."""

    async def eval(self, *args, **kwargs):
        raise ConnectionError("redis is down")


def test_redis_store_degrades_to_in_memory_on_connection_failure():
    store = RedisRateLimitStore("redis://127.0.0.1:6379")
    store._client = _FailingClient()

    async def exercise():
        # First call hits Redis, fails, and degrades permanently.
        assert await store.allow("credential:a", 1.0, 10, 10) is True
        assert store._degraded is True
        # Degraded store still enforces limits via the in-memory fallback.
        assert await store.allow("credential:a", 1.0, 1, 1) is False
        assert await store.allow("credential:b", 1.0, 1, 1) is True

    asyncio.run(exercise())
