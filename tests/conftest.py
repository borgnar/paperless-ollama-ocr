"""Test fixtures.

Two things are stood in for when they are not installed:

* the ``paperless`` package (only present inside a paperless-ngx container),
  replaced by the minimum surface the plugin subclasses and calls;
* a fake Ollama HTTP server, so the client and the parser can be exercised
  end to end without a model or a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

# ----------------------------------------------------------------------
# paperless stub
# ----------------------------------------------------------------------


def _install_paperless_stub() -> None:
    if importlib.util.find_spec("paperless") is not None:
        return  # real package available, use it

    package = types.ModuleType("paperless")
    package.__path__ = []  # type: ignore[attr-defined]
    parsers = types.ModuleType("paperless.parsers")
    parsers.__path__ = []  # type: ignore[attr-defined]
    remote = types.ModuleType("paperless.parsers.remote")
    utils = types.ModuleType("paperless.parsers.utils")
    tesseract = types.ModuleType("paperless.parsers.tesseract")

    remote._SUPPORTED_MIME_TYPES = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tif",
        "image/bmp": ".bmp",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }

    class RemoteDocumentParser:
        """Stand-in with the attributes the plugin touches."""

        def __init__(self, logging_group=None):
            self._logging_group = logging_group
            self._tempdir = Path(tempfile.mkdtemp(prefix="ollama-ocr-test-"))
            self._text = ""
            self._archive_path = None

        def get_text(self):
            return self._text

        def get_archive_path(self):
            return self._archive_path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    remote.RemoteDocumentParser = RemoteDocumentParser

    def extract_pdf_text(path, log=None):
        return "text extracted locally from the pdf"

    def post_process_text(text):
        return (text or "").strip()

    utils.extract_pdf_text = extract_pdf_text
    utils.post_process_text = post_process_text

    class RasterisedDocumentParser:
        """Stand-in for the Tesseract parser used by the fallback path."""

        calls = []

        def __init__(self, logging_group=None):
            self._logging_group = logging_group

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def parse(self, document_path, mime_type, *, produce_archive=True):
            RasterisedDocumentParser.calls.append((document_path, mime_type))

        def get_archive_path(self):
            return None

        def get_text(self):
            return "tesseract fallback text"

    tesseract.RasterisedDocumentParser = RasterisedDocumentParser

    parsers.remote = remote
    parsers.utils = utils
    parsers.tesseract = tesseract
    package.parsers = parsers

    sys.modules["paperless"] = package
    sys.modules["paperless.parsers"] = parsers
    sys.modules["paperless.parsers.remote"] = remote
    sys.modules["paperless.parsers.utils"] = utils
    sys.modules["paperless.parsers.tesseract"] = tesseract


_install_paperless_stub()


# ----------------------------------------------------------------------
# fake Ollama
# ----------------------------------------------------------------------


class FakeOllama:
    """A tiny HTTP server that answers like Ollama's /api/tags and /api/chat."""

    def __init__(self):
        self.requests: list[dict] = []
        self.models = ["qwen2.5vl:3b", "granite3.2-vision:2b"]
        #: Response body per call; a string, a list of strings (consumed in
        #: order), or an int HTTP status code to fail with.
        self.replies: object = "hello"
        self.status = 200
        self.api_key_seen: list[str | None] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # keep pytest output clean
                return

            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.api_key_seen.append(self.headers.get("Authorization"))
                if self.path.rstrip("/") == "/api/tags":
                    self._send(
                        200,
                        {
                            "models": [
                                {"name": name, "model": name} for name in fake.models
                            ],
                        },
                    )
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                fake.api_key_seen.append(self.headers.get("Authorization"))
                fake.requests.append(payload)

                reply = fake.replies
                if isinstance(reply, list):
                    reply = reply.pop(0) if reply else ""
                if isinstance(reply, int):
                    self._send(reply, {"error": "boom"})
                    return
                self._send(200, {"message": {"role": "assistant", "content": reply}})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(autouse=True)
def _clear_probes():
    from paperless_ollama_ocr.client import clear_probe_cache

    clear_probe_cache()
    yield
    clear_probe_cache()
