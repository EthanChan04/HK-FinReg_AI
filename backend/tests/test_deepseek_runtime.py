from types import SimpleNamespace

import pytest

from app.services.llm import deepseek


class CapturingChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _settings(api_key: str = "test-deepseek-key") -> SimpleNamespace:
    return SimpleNamespace(
        DEEPSEEK_API_KEY=api_key,
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        DEEPSEEK_MODEL="deepseek-v4-flash",
        DEEPSEEK_TIMEOUT_SECONDS=60,
        DEEPSEEK_INTERACTIVE_THINKING=False,
        DEEPSEEK_REASONING_THINKING=True,
    )


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    deepseek.build_deepseek_llm.cache_clear()
    yield
    deepseek.build_deepseek_llm.cache_clear()


@pytest.mark.parametrize("profile", ["interactive", "evaluation"])
def test_non_reasoning_profiles_use_v4_flash_without_thinking(monkeypatch, profile):
    monkeypatch.setattr(deepseek, "get_settings", lambda: _settings())
    monkeypatch.setattr(deepseek, "ChatOpenAI", CapturingChatOpenAI)

    client = deepseek.build_deepseek_llm(profile)

    assert client.kwargs["model"] == "deepseek-v4-flash"
    assert client.kwargs["base_url"] == "https://api.deepseek.com"
    assert client.kwargs["timeout"] == 60
    assert client.kwargs["max_retries"] == 0
    assert client.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_reasoning_profile_enables_thinking(monkeypatch):
    monkeypatch.setattr(deepseek, "get_settings", lambda: _settings())
    monkeypatch.setattr(deepseek, "ChatOpenAI", CapturingChatOpenAI)

    client = deepseek.build_deepseek_llm("reasoning")

    assert client.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_missing_key_fails_closed(monkeypatch):
    monkeypatch.setattr(deepseek, "get_settings", lambda: _settings(api_key=""))

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is required"):
        deepseek.build_deepseek_llm("interactive")


def test_runtime_status_is_explicit_and_never_returns_the_api_key(monkeypatch):
    monkeypatch.setattr(deepseek, "get_settings", lambda: _settings())

    status = deepseek.deepseek_runtime_status()

    assert status == {
        "configured": True,
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "reason": "configured",
    }
    assert "test-deepseek-key" not in repr(status)


def test_unknown_profile_is_rejected_before_client_creation(monkeypatch):
    monkeypatch.setattr(deepseek, "get_settings", lambda: _settings())
    monkeypatch.setattr(deepseek, "ChatOpenAI", CapturingChatOpenAI)

    with pytest.raises(ValueError, match="Unknown DeepSeek profile"):
        deepseek.build_deepseek_llm("batch")
