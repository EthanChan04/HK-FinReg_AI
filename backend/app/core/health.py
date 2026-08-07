"""Liveness and dependency readiness reporting."""

from __future__ import annotations


def readiness_report(checks: dict[str, bool]) -> dict:
    """Convert dependency checks into a stable readiness response."""

    engines = {
        name: "available" if available else "unavailable"
        for name, available in checks.items()
    }
    ready = all(checks.values())
    return {"status": "ready" if ready else "degraded", "engines": engines}
