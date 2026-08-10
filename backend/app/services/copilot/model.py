"""Model builder for Compliance Copilot."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.services.llm.deepseek import build_deepseek_llm


def build_copilot_llm() -> ChatOpenAI:
    """Build the interactive DeepSeek profile for Compliance Copilot."""
    return build_deepseek_llm("interactive")
