"""Model builder for Compliance Copilot."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def _normalize_model_name(model_name: str) -> str:
    if not model_name:
        return model_name
    if model_name.lower() == "mimo-v2.5":
        return "mimo-v2.5"
    return model_name


@lru_cache()
def build_copilot_llm() -> ChatOpenAI:
    settings = get_settings()
    api_key = settings.COPILOT_API_KEY or settings.ZHIPU_API_KEY
    if not api_key:
        raise RuntimeError("COPILOT_API_KEY or ZHIPU_API_KEY must be configured for Compliance Copilot.")
    model_name = _normalize_model_name(settings.COPILOT_MODEL)
    return ChatOpenAI(
        model_name=model_name,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=settings.COPILOT_BASE_URL,
        timeout=settings.COPILOT_TIMEOUT_SECONDS,
    )
