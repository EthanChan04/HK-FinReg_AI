"""T2-01: FastAPI HTTP integration tests.

Covers authentication, CORS, rate limiting, error responses and health
endpoint semantics at the HTTP layer (risk R-03: test hierarchy gap).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """TestClient with API key auth enabled and a fixed test key."""

    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "API_KEY_ENABLED", True)
    monkeypatch.setattr(settings, "API_KEY", "test-secret-key")

    from app.main import app

    return TestClient(app)


def test_protected_route_rejects_missing_api_key(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code in {401, 403}


def test_protected_route_rejects_wrong_api_key(client):
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code in {401, 403}


def test_protected_route_accepts_valid_api_key(client):
    response = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code == 200


def test_health_endpoints_are_public_without_api_key(client):
    live = client.get("/api/v1/health/live")
    ready = client.get("/api/v1/health/ready")
    assert live.status_code == 200
    assert ready.status_code in {200, 503}


def test_cors_headers_present_on_health_response(client):
    response = client.get(
        "/api/v1/health/live",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") is not None


def test_unknown_route_returns_404_json(client):
    response = client.get("/api/v1/definitely-not-a-route")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_invalid_body_returns_422_with_detail(client):
    response = client.post(
        "/api/v1/copilot/chat/stream",
        headers={"Authorization": "Bearer test-secret-key"},
        json={"not_the_expected_field": "value"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()


def _set_rate_limit(app, rpm: int, rph: int) -> None:
    """Lower the rate limit budget by rebuilding the middleware stack.

    RateLimitMiddleware copies constructor kwargs into instance attributes
    at build time, so the running middleware must be re-instantiated.
    """
    from starlette.middleware import Middleware

    from app.core.rate_limit import RateLimitMiddleware

    rebuilt = []
    for entry in app.user_middleware:
        if entry.cls.__name__ == "RateLimitMiddleware":
            rebuilt.append(
                Middleware(
                    RateLimitMiddleware,
                    requests_per_minute=rpm,
                    requests_per_hour=rph,
                    storage_url=entry.kwargs.get("storage_url", ""),
                )
            )
        else:
            rebuilt.append(entry)
    app.user_middleware = rebuilt
    app.middleware_stack = app.build_middleware_stack()


def test_health_requests_are_exempt_from_rate_limiting(client):
    """Health checks must never be throttled (orchestration depends on it)."""

    from app.main import app

    _set_rate_limit(app, 1, 1)

    for _ in range(5):
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200


def test_rate_limit_returns_429_when_exceeded(client):
    """Business routes are throttled when the per-minute budget is spent."""

    from app.main import app

    _set_rate_limit(app, 2, 2)

    # Burn the budget with valid requests.
    for _ in range(2):
        response = client.get(
            "/api/v1/metrics",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert response.status_code == 200

    # Next request within the same window is throttled.
    throttled = client.get(
        "/api/v1/metrics",
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert throttled.status_code == 429
