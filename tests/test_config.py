"""Tests for reading the configuration from the environment."""

from __future__ import annotations

import pytest

from paperless_ollama_ocr.config import DEFAULT_ENDPOINT
from paperless_ollama_ocr.config import DEFAULT_MODEL
from paperless_ollama_ocr.config import DEFAULT_SCORE
from paperless_ollama_ocr.config import OllamaConfig

ALL_VARS = [
    "PAPERLESS_OLLAMA_OCR_ENDPOINT",
    "PAPERLESS_OLLAMA_OCR_MODEL",
    "PAPERLESS_OLLAMA_OCR_API_KEY",
    "PAPERLESS_OLLAMA_OCR_SCORE",
    "PAPERLESS_OLLAMA_OCR_PREPROCESS",
    "PAPERLESS_OLLAMA_OCR_FALLBACK",
    "PAPERLESS_OLLAMA_OCR_FORCE",
    "PAPERLESS_OLLAMA_OCR_DPI",
    "PAPERLESS_OLLAMA_OCR_MAX_SIDE",
    "PAPERLESS_OLLAMA_OCR_TIMEOUT",
    "PAPERLESS_OLLAMA_OCR_NUM_CTX",
    "PAPERLESS_OLLAMA_OCR_PARALLEL",
    "PAPERLESS_OLLAMA_OCR_STRUCTURED",
    "PAPERLESS_OLLAMA_OCR_KEEP_ALIVE",
    "PAPERLESS_OLLAMA_OCR_PROMPT",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)


class TestValidity:
    def test_valid_without_any_api_key(self):
        # The whole point: a self-hosted Ollama needs no credentials.
        assert OllamaConfig.from_env().is_valid() is True

    def test_invalid_without_endpoint(self, monkeypatch):
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_ENDPOINT", "   ")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_MODEL", "")
        # Empty values fall back to the defaults, so the config stays valid;
        # declining is the job of score(), not of the config.
        assert OllamaConfig.from_env().endpoint == DEFAULT_ENDPOINT


class TestDefaults:
    def test_endpoint_and_model_have_defaults(self):
        config = OllamaConfig.from_env()
        assert config.endpoint == DEFAULT_ENDPOINT
        assert config.model == DEFAULT_MODEL

    def test_plugin_defaults(self):
        config = OllamaConfig.from_env()
        assert config.score == DEFAULT_SCORE
        assert config.fallback is True
        assert config.preprocess is True
        assert config.force is False
        assert config.structured is False
        assert config.bbox_space == "auto"
        assert config.dpi == 200
        assert config.parallel == 1
        assert config.api_key is None
        assert config.prompt is None


class TestOverrides:
    def test_values_are_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_ENDPOINT", "http://gpu.lan:11434/")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_MODEL", "deepseek-ocr")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_API_KEY", "proxy-token")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_SCORE", "40")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_PREPROCESS", "false")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_FALLBACK", "no")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_FORCE", "yes")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_DPI", "300")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_TIMEOUT", "120")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_NUM_CTX", "4096")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_PARALLEL", "4")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_STRUCTURED", "off")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_KEEP_ALIVE", "30m")
        config = OllamaConfig.from_env()
        assert config.endpoint == "http://gpu.lan:11434/"
        assert config.model == "deepseek-ocr"
        assert config.api_key == "proxy-token"
        assert config.score == 40
        assert config.preprocess is False
        assert config.fallback is False
        assert config.force is True
        assert config.dpi == 300
        assert config.timeout == 120
        assert config.num_ctx == 4096
        assert config.parallel == 4
        assert config.structured is False
        assert config.keep_alive == "30m"

    def test_prompt_override(self, monkeypatch):
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_PROMPT", "Read this invoice.")
        assert OllamaConfig.from_env().prompt == "Read this invoice."

    def test_whitespace_key_is_no_key(self, monkeypatch):
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_API_KEY", "   ")
        assert OllamaConfig.from_env().api_key is None

    def test_bad_numbers_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_DPI", "lots")
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_SCORE", "high")
        config = OllamaConfig.from_env()
        assert config.dpi == 200
        assert config.score == DEFAULT_SCORE

    def test_parallel_is_at_least_one(self, monkeypatch):
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_PARALLEL", "0")
        assert OllamaConfig.from_env().parallel == 1
