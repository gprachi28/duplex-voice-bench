"""create_llm_backend()'s LLM_BACKEND-driven selection.

No network calls -- OllamaBackend/GPT4oBackend construction only builds an
HTTP client (see agent/llm.py docstrings), so these test selection logic
only, resetting the module-level singleton between cases.
"""

import agent.llm as llm_module
from agent.llm import OLLAMA_MODEL, GPT4oBackend, OllamaBackend, create_llm_backend

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton():
    llm_module._backend_instance = None
    yield
    llm_module._backend_instance = None


def test_defaults_to_ollama_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    backend = create_llm_backend()
    assert isinstance(backend, OllamaBackend)
    assert backend.model == OLLAMA_MODEL


def test_selects_gpt4o_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gpt4o")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    backend = create_llm_backend()
    assert isinstance(backend, GPT4oBackend)
    assert backend.model == "gpt-4o"


def test_selects_gpt4o_mini_via_openai_model_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gpt4o")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    backend = create_llm_backend()
    assert isinstance(backend, GPT4oBackend)
    assert backend.model == "gpt-4o-mini"


def test_gpt4o_without_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gpt4o")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        create_llm_backend()


def test_unknown_backend_name_raises(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "not-a-real-backend")
    with pytest.raises(ValueError):
        create_llm_backend()
