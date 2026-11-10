"""Tests for the Ollama HTTP client against a fake server."""

from __future__ import annotations

import base64

import pytest

from paperless_ollama_ocr.client import OllamaClient
from paperless_ollama_ocr.client import OllamaError
from paperless_ollama_ocr.client import _base_url
from paperless_ollama_ocr.client import endpoint_available
from paperless_ollama_ocr.config import OllamaConfig


def make_config(**overrides) -> OllamaConfig:
    base = OllamaConfig.from_env()
    return OllamaConfig(**{**base.__dict__, **overrides})


class TestBaseUrl:
    def test_trailing_slash_is_removed(self):
        assert _base_url("http://host:11434/") == "http://host:11434"

    def test_appended_api_is_removed(self):
        assert _base_url("http://host:11434/api/") == "http://host:11434"

    def test_plain_url_is_untouched(self):
        assert _base_url("http://host:11434") == "http://host:11434"


class TestProbe:
    def test_reachable(self, fake_ollama):
        assert endpoint_available(fake_ollama.endpoint) is True

    def test_unreachable(self):
        assert endpoint_available("http://127.0.0.1:1", use_cache=False) is False

    def test_result_is_cached(self, fake_ollama):
        before = len(fake_ollama.api_key_seen)
        assert endpoint_available(fake_ollama.endpoint) is True
        assert endpoint_available(fake_ollama.endpoint) is True
        assert len(fake_ollama.api_key_seen) - before == 1

    def test_cache_can_be_bypassed(self, fake_ollama):
        before = len(fake_ollama.api_key_seen)
        endpoint_available(fake_ollama.endpoint)
        endpoint_available(fake_ollama.endpoint, use_cache=False)
        assert len(fake_ollama.api_key_seen) - before == 2

    def test_api_key_is_sent_as_bearer(self, fake_ollama):
        endpoint_available(fake_ollama.endpoint, "secret", use_cache=False)
        assert fake_ollama.api_key_seen[-1] == "Bearer secret"


class TestModels:
    def test_list_models(self, fake_ollama):
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))
        assert client.list_models() == ["qwen2.5vl:3b", "granite3.2-vision:2b"]

    def test_has_model_exact(self, fake_ollama):
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))
        assert client.has_model("qwen2.5vl:3b") is True

    def test_has_model_by_base_name(self, fake_ollama):
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))
        assert client.has_model("qwen2.5vl") is True

    def test_missing_model(self, fake_ollama):
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))
        assert client.has_model("llama3.2-vision:11b") is False

    def test_unreachable_lists_nothing(self):
        client = OllamaClient(make_config(endpoint="http://127.0.0.1:1"))
        assert client.list_models() == []
        assert client.has_model() is False


class TestTranscribe:
    def test_request_shape(self, fake_ollama):
        fake_ollama.replies = "some text"
        client = OllamaClient(
            make_config(endpoint=fake_ollama.endpoint, model="qwen2.5vl:3b")
        )
        assert client.transcribe(b"\xff\xd8image", "prompt here") == "some text"

        payload = fake_ollama.requests[-1]
        assert payload["model"] == "qwen2.5vl:3b"
        assert payload["stream"] is False
        assert payload["options"]["num_ctx"] == 8192
        assert payload["keep_alive"] == "5m"
        assert "format" not in payload
        message = payload["messages"][0]
        assert message["content"] == "prompt here"
        assert base64.b64decode(message["images"][0]) == b"\xff\xd8image"

    def test_schema_is_forwarded(self, fake_ollama):
        fake_ollama.replies = '{"lines":[]}'
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))
        schema = {"type": "object", "properties": {}}
        client.transcribe(b"img", "prompt", schema=schema)
        assert fake_ollama.requests[-1]["format"] == schema

    def test_http_error_becomes_ollama_error(self, fake_ollama):
        fake_ollama.replies = 500
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))
        with pytest.raises(OllamaError, match="HTTP 500"):
            client.transcribe(b"img", "prompt")

    def test_model_missing_error_is_raised(self, fake_ollama, monkeypatch):
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))

        class Response:
            status_code = 200

            def json(self):
                return {"error": "model not found"}

        monkeypatch.setattr("httpx.post", lambda *a, **kw: Response())
        with pytest.raises(OllamaError, match="model not found"):
            client.transcribe(b"img", "prompt")

    def test_unreachable_raises(self):
        client = OllamaClient(make_config(endpoint="http://127.0.0.1:1"))
        with pytest.raises(OllamaError, match="failed"):
            client.transcribe(b"img", "prompt")

    def test_unexpected_body_raises(self, fake_ollama, monkeypatch):
        client = OllamaClient(make_config(endpoint=fake_ollama.endpoint))

        class Response:
            status_code = 200

            def json(self):
                return {"message": {"content": None}}

        monkeypatch.setattr("httpx.post", lambda *a, **kw: Response())
        with pytest.raises(OllamaError, match="unexpected"):
            client.transcribe(b"img", "prompt")
