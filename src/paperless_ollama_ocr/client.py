"""Minimal client for a self-hosted Ollama instance.

Only two endpoints are used: ``GET /api/tags`` (reachability and model
discovery) and ``POST /api/chat`` (transcription). No SDK is required, so
nothing has to be installed inside the container beyond what the official
paperless-ngx image already ships.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paperless_ollama_ocr.config import OllamaConfig

logger = logging.getLogger("paperless.parsing.ollama")

#: How long a reachability answer is reused. ``score()`` runs for every
#: document, so the probe must not turn into one HTTP call per file.
PROBE_TTL: float = 60.0

#: Connect timeout for the reachability probe, in seconds.
PROBE_TIMEOUT: float = 4.0

_probe_cache: dict[str, tuple[float, bool]] = {}


class OllamaError(RuntimeError):
    """Raised when the Ollama instance answers with an error or not at all."""


def clear_probe_cache() -> None:
    """Forget cached reachability results (used by tests and the health check)."""
    _probe_cache.clear()


def _base_url(endpoint: str) -> str:
    """Normalise the endpoint: strip trailing slashes and a trailing /api."""
    url = (endpoint or "").strip().rstrip("/")
    if url.endswith("/api"):
        url = url[: -len("/api")]
    return url


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def endpoint_available(
    endpoint: str,
    api_key: str | None = None,
    *,
    use_cache: bool = True,
) -> bool:
    """True when the Ollama endpoint answers on ``GET /api/tags``.

    Cached for :data:`PROBE_TTL` seconds. A short timeout: an unreachable
    endpoint must not stall document processing, and returning False simply
    means Paperless falls back to Tesseract for that run.
    """
    import httpx

    key = f"{_base_url(endpoint)}|{bool(api_key)}"
    now = time.monotonic()
    if use_cache:
        cached = _probe_cache.get(key)
        if cached is not None and now - cached[0] < PROBE_TTL:
            return cached[1]

    try:
        response = httpx.get(
            f"{_base_url(endpoint)}/api/tags",
            headers=_headers(api_key),
            timeout=PROBE_TIMEOUT,
        )
        available = response.status_code == 200
    except Exception as e:  # noqa: BLE001 - any transport error means "not there"
        logger.debug("Ollama probe failed for %s: %s", endpoint, e)
        available = False

    _probe_cache[key] = (now, available)
    return available


class OllamaClient:
    """Sends single page images to a vision model and returns its text."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._base = _base_url(config.endpoint)

    # -- discovery ------------------------------------------------------

    def list_models(self) -> list[str]:
        """Names of the models the instance has pulled; empty list on error."""
        import httpx

        try:
            response = httpx.get(
                f"{self._base}/api/tags",
                headers=_headers(self._config.api_key),
                timeout=PROBE_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not list Ollama models: %s", e)
            return []
        return [
            str(entry.get("name") or entry.get("model") or "")
            for entry in response.json().get("models", [])
            if entry
        ]

    def has_model(self, name: str | None = None) -> bool:
        """True when the configured model (or *name*) is available locally."""
        wanted = (name or self._config.model).strip()
        models = self.list_models()
        if not models:
            return False
        base = wanted.split(":")[0]
        return any(model == wanted or model.split(":")[0] == base for model in models)

    # -- transcription --------------------------------------------------

    def transcribe(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Send one page image and return the model's raw response text."""
        import httpx

        payload: dict = {
            "model": self._config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self._config.num_ctx,
            },
            "keep_alive": self._config.keep_alive,
        }
        if schema is not None:
            payload["format"] = schema

        timeout = httpx.Timeout(self._config.timeout, connect=15.0)
        try:
            response = httpx.post(
                f"{self._base}/api/chat",
                json=payload,
                headers=_headers(self._config.api_key),
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            raise OllamaError(f"request to {self._base} failed: {e}") from e

        if response.status_code != 200:
            body = response.text[:500]
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code}: {body}",
            )

        data = response.json()
        if data.get("error"):
            raise OllamaError(f"Ollama error: {data['error']}")
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError(f"unexpected Ollama response: {str(data)[:300]}")
        return content
