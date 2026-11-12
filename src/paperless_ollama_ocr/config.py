"""Configuration for the Ollama OCR plugin.

Read from environment variables only. A plugin must not hook into the
Paperless application configuration, which belongs to the core. Dedicated
variable names also avoid clashing with the built-in remote parser
(PAPERLESS_REMOTE_OCR_*).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from paperless_ollama_ocr.helpers import BBOX_SPACES

#: Default Ollama endpoint. Inside the paperless container "localhost" is the
#: container itself, so a host-side Ollama needs host.docker.internal (Docker
#: Desktop) or the LAN address of the GPU machine.
DEFAULT_ENDPOINT: str = "http://localhost:11434"

#: Default model. Any vision-capable Ollama model works; see the README for
#: the ones that transcribe reliably.
DEFAULT_MODEL: str = "deepseek-ocr"

#: The built-in remote parser (azureai) scores 20, Tesseract 10.
#: A higher value makes Ollama win when both are configured.
DEFAULT_SCORE: int = 30

#: Timeout (seconds) for a single page. Local models on CPU are slow.
DEFAULT_TIMEOUT: float = 600.0

#: Context window requested per page. Vision models need room for the image
#: tokens plus the transcription; 8192 is safe for the 3B-class OCR models.
DEFAULT_NUM_CTX: int = 8192

#: Truthy spellings accepted for the boolean environment variables.
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _bool_from_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable, falling back to *default*."""
    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class OllamaConfig:
    """Settings read from the environment."""

    endpoint: str
    model: str
    #: Optional bearer token, for an Ollama behind an authenticating proxy.
    api_key: str | None
    score: int
    #: Fall back to Tesseract when every page fails.
    fallback: bool
    #: Run ocrmypdf first to deskew/rotate/clean the pages.
    preprocess: bool
    #: Transcribe even when Paperless asks for no archive (PDFs that already
    #: contain text). Off by default, matching core behaviour.
    force: bool
    #: Let the geometry pass rewrite digitally signed PDFs. The stored original
    #: keeps its signature; this only affects the temporary working copy.
    invalidate_signatures: bool
    #: DPI used to rasterise pages before they are sent to the model.
    dpi: int
    #: Longest side in pixels of the image sent to the model; larger pages are
    #: downscaled for the request only.
    max_side: int
    #: Per-page timeout in seconds.
    timeout: float
    #: Context window per request.
    num_ctx: int
    #: Pages processed in parallel. Only useful when the Ollama server runs
    #: with OLLAMA_NUM_PARALLEL > 1 and has the VRAM for it.
    parallel: int
    #: Ask the model for line-level bounding boxes (JSON output). Off by
    #: default: most vision models either omit the boxes or collapse the page
    #: into one box, which costs several times the runtime for nothing.
    structured: bool
    #: How to read the model's bbox coordinates: auto, pixel, normalized1000
    #: or fraction.
    bbox_space: str
    #: How long the model stays loaded between documents.
    keep_alive: str
    #: Optional prompt override.
    prompt: str | None

    @classmethod
    def from_env(cls) -> OllamaConfig:
        space = (
            os.getenv("PAPERLESS_OLLAMA_OCR_BBOX_SPACE", "").strip().lower() or "auto"
        )
        if space not in BBOX_SPACES:
            space = "auto"
        return cls(
            endpoint=(
                os.getenv("PAPERLESS_OLLAMA_OCR_ENDPOINT", "").strip()
                or DEFAULT_ENDPOINT
            ),
            model=(
                os.getenv("PAPERLESS_OLLAMA_OCR_MODEL", "").strip() or DEFAULT_MODEL
            ),
            api_key=os.getenv("PAPERLESS_OLLAMA_OCR_API_KEY", "").strip() or None,
            score=_int_from_env("PAPERLESS_OLLAMA_OCR_SCORE", DEFAULT_SCORE),
            fallback=_bool_from_env("PAPERLESS_OLLAMA_OCR_FALLBACK", True),
            preprocess=_bool_from_env("PAPERLESS_OLLAMA_OCR_PREPROCESS", True),
            force=_bool_from_env("PAPERLESS_OLLAMA_OCR_FORCE", False),
            invalidate_signatures=_bool_from_env(
                "PAPERLESS_OLLAMA_OCR_INVALIDATE_SIGNATURES", False
            ),
            dpi=_int_from_env("PAPERLESS_OLLAMA_OCR_DPI", 200),
            max_side=_int_from_env("PAPERLESS_OLLAMA_OCR_MAX_SIDE", 2400),
            timeout=_float_from_env("PAPERLESS_OLLAMA_OCR_TIMEOUT", DEFAULT_TIMEOUT),
            num_ctx=_int_from_env("PAPERLESS_OLLAMA_OCR_NUM_CTX", DEFAULT_NUM_CTX),
            parallel=max(1, _int_from_env("PAPERLESS_OLLAMA_OCR_PARALLEL", 1)),
            structured=_bool_from_env("PAPERLESS_OLLAMA_OCR_STRUCTURED", False),
            bbox_space=space,
            keep_alive=os.getenv("PAPERLESS_OLLAMA_OCR_KEEP_ALIVE", "").strip() or "5m",
            prompt=os.getenv("PAPERLESS_OLLAMA_OCR_PROMPT", "").strip() or None,
        )

    def is_valid(self) -> bool:
        """True when the configuration is complete enough to try.

        No API key is required: a self-hosted Ollama usually has none. The
        parser separately checks that the endpoint answers (``score()``
        declines while it does not), so Paperless keeps using Tesseract
        instead of failing during processing.
        """
        return bool(self.endpoint and self.model)
