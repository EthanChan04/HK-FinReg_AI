"""Central DeepSeek V4 Flash runtime factory."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI

from app.core.config import get_settings


DeepSeekProfile = Literal["interactive", "reasoning", "evaluation"]
_VALID_PROFILES = frozenset({"interactive", "reasoning", "evaluation"})


@lru_cache(maxsize=3)
def build_deepseek_llm(profile: DeepSeekProfile) -> ChatOpenAI:
    """Build one cached DeepSeek client for an approved runtime profile."""

    if profile not in _VALID_PROFILES:
        raise ValueError(f"Unknown DeepSeek profile: {profile}")

    settings = get_settings()
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required for real LLM requests")

    thinking = (
        settings.DEEPSEEK_REASONING_THINKING
        if profile == "reasoning"
        else settings.DEEPSEEK_INTERACTIVE_THINKING
    )
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        temperature=0,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.DEEPSEEK_TIMEOUT_SECONDS,
        max_retries=0,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )


def deepseek_runtime_status() -> dict[str, str | bool]:
    """Return a non-secret summary suitable for health and gate reports."""

    settings = get_settings()
    configured = bool(settings.DEEPSEEK_API_KEY)
    return {
        "configured": configured,
        "provider": "deepseek",
        "model": settings.DEEPSEEK_MODEL,
        "reason": "configured" if configured else "DEEPSEEK_API_KEY is missing",
    }
