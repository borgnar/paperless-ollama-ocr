"""End-to-end tests: parser -> HTTP -> transcribed text and archive PDF.

These exercise everything except the model itself: the fake server answers
like Ollama, the rasteriser and the PDF renderer are the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("pikepdf")
pytest.importorskip("ocrmypdf")

from PIL import Image  # noqa: E402
from PIL import ImageDraw  # noqa: E402

from paperless_ollama_ocr.parser import OllamaOcrParser  # noqa: E402

TEXT = "Rechnung Nr. 4711\nBetrag: 1.234,56 EUR"


@pytest.fixture
def page_png(tmp_path):
    image = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 60, 790, 140), fill="black")
    draw.rectangle((60, 200, 700, 260), fill="black")
    path = tmp_path / "page.png"
    image.save(path)
    return path


@pytest.fixture
def page_pdf(tmp_path, page_png):
    """A real one-page PDF, so the pdf2image path is exercised for real."""
    with Image.open(page_png) as image:
        path = tmp_path / "page.pdf"
        image.convert("RGB").save(path, "PDF", resolution=200.0)
    return path


@pytest.fixture
def page_tiff(tmp_path):
    first = Image.new("RGB", (600, 800), "white")
    second = Image.new("RGB", (600, 800), "white")
    ImageDraw.Draw(first).rectangle((40, 40, 500, 120), fill="black")
    ImageDraw.Draw(second).rectangle((40, 40, 500, 120), fill="black")
    path = tmp_path / "two-pages.tiff"
    first.save(path, save_all=True, append_images=[second])
    return path


@pytest.fixture
def big_page_png(tmp_path):
    """Large enough that the request image is actually downscaled."""
    image = Image.new("RGB", (3000, 2000), "white")
    ImageDraw.Draw(image).rectangle((100, 100, 2800, 400), fill="black")
    path = tmp_path / "big.png"
    image.save(path)
    return path


def configure(monkeypatch, fake_ollama, **env):
    monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_ENDPOINT", fake_ollama.endpoint)
    monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_MODEL", "qwen2.5vl:3b")
    monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_PREPROCESS", "false")
    for key, value in env.items():
        monkeypatch.setenv(f"PAPERLESS_OLLAMA_OCR_{key}", value)


def extract_pdf_text(path) -> str:
    from pdfminer.high_level import extract_text

    return extract_text(str(path))


class TestParse:
    def test_image_is_transcribed(self, monkeypatch, fake_ollama, page_png):
        fake_ollama.replies = json.dumps(
            {
                "lines": [
                    {"text": "Rechnung Nr. 4711", "bbox": [60, 60, 790, 140]},
                    {"text": "Betrag: 1.234,56 EUR", "bbox": [60, 200, 700, 260]},
                ]
            }
        )
        configure(monkeypatch, fake_ollama, STRUCTURED="true")
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png")

        assert parser.get_text() == TEXT
        assert len(fake_ollama.requests) == 1
        assert fake_ollama.requests[0]["format"]["required"] == ["lines"]

    def test_archive_pdf_is_searchable(self, monkeypatch, fake_ollama, page_png):
        fake_ollama.replies = json.dumps(
            {"lines": [{"text": TEXT.replace("\n", " "), "bbox": [60, 60, 790, 260]}]}
        )
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png")

        archive = parser.get_archive_path()
        assert archive is not None and archive.exists()

        import pikepdf

        with pikepdf.open(archive) as pdf:
            assert len(pdf.pages) == 1

        extracted = extract_pdf_text(archive)
        assert "4711" in extracted
        assert "1.234,56" in extracted

    def test_text_without_boxes_still_reaches_the_archive(
        self, monkeypatch, fake_ollama, page_png
    ):
        fake_ollama.replies = TEXT
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png")

        assert parser.get_text() == TEXT
        assert "4711" in extract_pdf_text(parser.get_archive_path())

    def test_plain_text_mode_sends_no_schema(self, monkeypatch, fake_ollama, page_png):
        fake_ollama.replies = TEXT
        configure(monkeypatch, fake_ollama, STRUCTURED="false")
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png")

        payload = fake_ollama.requests[0]
        assert "format" not in payload
        prompt = payload["messages"][0]["content"]
        assert "Transcribe all text" in prompt
        assert "JSON" not in prompt

    def test_custom_prompt_is_used(self, monkeypatch, fake_ollama, page_png):
        fake_ollama.replies = TEXT
        configure(
            monkeypatch,
            fake_ollama,
            PROMPT="Extract the invoice total.",
            STRUCTURED="false",
        )
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png")

        assert (
            fake_ollama.requests[0]["messages"][0]["content"]
            == "Extract the invoice total."
        )

    def test_multipage_document_makes_one_call_per_page(
        self, monkeypatch, fake_ollama, page_tiff
    ):
        fake_ollama.replies = ["page one", "page two"]
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(page_tiff, "image/tiff")

        assert len(fake_ollama.requests) == 2
        assert parser.get_text() == "page one\n\npage two"

        import pikepdf

        with pikepdf.open(parser.get_archive_path()) as pdf:
            assert len(pdf.pages) == 2

    def test_parallel_pages_are_all_transcribed(
        self, monkeypatch, fake_ollama, page_tiff
    ):
        # The fake server hands out replies in arrival order, so only the set
        # of results is stable here; that both pages reach the model is the
        # point. Order preservation of pool.map is covered by the sequential
        # test above.
        fake_ollama.replies = ["page one", "page two"]
        configure(monkeypatch, fake_ollama, PARALLEL="2")
        parser = OllamaOcrParser()
        parser.parse(page_tiff, "image/tiff")
        assert len(fake_ollama.requests) == 2
        assert set(parser.get_text().split("\n\n")) == {"page one", "page two"}

    def test_empty_page_does_not_kill_the_document(
        self, monkeypatch, fake_ollama, page_tiff
    ):
        fake_ollama.replies = ["page one", "   "]
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(page_tiff, "image/tiff")
        assert parser.get_text() == "page one"

    def test_empty_answer_is_retried_at_a_smaller_size(
        self, monkeypatch, fake_ollama, big_page_png
    ):
        # deepseek-ocr answers nothing when the image is too large; the retry is
        # the difference between a transcribed page and a Tesseract fallback.
        fake_ollama.replies = ["", TEXT]
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(big_page_png, "image/png")

        assert len(fake_ollama.requests) == 2
        assert parser.get_text() == TEXT

        import base64
        import io

        sizes = []
        for request in fake_ollama.requests:
            raw = base64.b64decode(request["messages"][0]["images"][0])
            with Image.open(io.BytesIO(raw)) as opened:
                sizes.append(opened.size)
        assert max(sizes[0]) == 2400
        assert max(sizes[1]) == 1600

    def test_two_empty_answers_fall_back_to_tesseract(
        self, monkeypatch, fake_ollama, big_page_png
    ):
        fake_ollama.replies = ["", ""]

        from paperless.parsers.tesseract import RasterisedDocumentParser

        RasterisedDocumentParser.calls.clear()
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(big_page_png, "image/png")

        assert len(fake_ollama.requests) == 2
        assert parser.get_text() == "tesseract fallback text"
        assert len(RasterisedDocumentParser.calls) == 1

    def test_born_digital_pdf_without_archive_is_not_sent(
        self, monkeypatch, fake_ollama, page_png
    ):
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(page_png, "application/pdf", produce_archive=False)

        assert fake_ollama.requests == []
        assert "text extracted locally" in parser.get_text()

    def test_force_transcribes_a_pdf_without_archive(
        self, monkeypatch, fake_ollama, page_pdf
    ):
        fake_ollama.replies = TEXT
        configure(monkeypatch, fake_ollama, FORCE="true")
        parser = OllamaOcrParser()
        parser.parse(page_pdf, "application/pdf", produce_archive=False)

        assert len(fake_ollama.requests) == 1
        assert "4711" in parser.get_text()
        # No archive was requested, so none is written.
        assert parser.get_archive_path() is None

    def test_no_archive_requested_skips_the_renderer(
        self, monkeypatch, fake_ollama, page_png
    ):
        fake_ollama.replies = TEXT
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png", produce_archive=False)

        assert parser.get_text() == TEXT
        assert parser.get_archive_path() is None


class TestPreprocess:
    """The geometry pass runs ocrmypdf as a renderer-only appliance."""

    def test_pdf_is_deskewed_without_adding_a_text_layer(self, monkeypatch):
        ocrmypdf = pytest.importorskip("ocrmypdf")
        calls = []

        def fake_ocr(source, target, **kwargs):
            calls.append({"source": source, "target": target, **kwargs})

        monkeypatch.setattr(ocrmypdf, "ocr", fake_ocr)
        parser = OllamaOcrParser()
        result = parser._preprocess(Path("/tmp/in.pdf"), "application/pdf")

        assert result == parser._tempdir / "preprocessed.pdf"
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["deskew"] is True
        assert kwargs["rotate_pages"] is True
        assert kwargs["rotate_pages_threshold"] == 6.0
        assert kwargs["tesseract_timeout"] == 0  # renderer only, no Tesseract
        assert kwargs["output_type"] == "pdf"
        assert kwargs["progress_bar"] is False

    def test_images_are_not_preprocessed(self, monkeypatch):
        ocrmypdf = pytest.importorskip("ocrmypdf")
        called = []
        monkeypatch.setattr(ocrmypdf, "ocr", lambda *a, **kw: called.append(1))
        parser = OllamaOcrParser()
        assert parser._preprocess(Path("/tmp/in.png"), "image/png") is None
        assert called == []

    def test_failure_falls_back_to_the_original(self, monkeypatch):
        ocrmypdf = pytest.importorskip("ocrmypdf")

        def boom(*args, **kwargs):
            raise RuntimeError("ghostscript missing")

        monkeypatch.setattr(ocrmypdf, "ocr", boom)
        parser = OllamaOcrParser()
        assert parser._preprocess(Path("/tmp/in.pdf"), "application/pdf") is None

    def test_parse_still_transcribes_when_preprocessing_fails(
        self, monkeypatch, fake_ollama, page_pdf
    ):
        fake_ollama.replies = TEXT
        configure(monkeypatch, fake_ollama, PREPROCESS="true")

        import ocrmypdf

        def boom(*args, **kwargs):
            raise RuntimeError("ghostscript missing")

        monkeypatch.setattr(ocrmypdf, "ocr", boom)
        parser = OllamaOcrParser()
        parser.parse(page_pdf, "application/pdf")

        assert "4711" in parser.get_text()


class TestFailures:
    def test_tesseract_takes_over_when_every_page_fails(
        self, monkeypatch, fake_ollama, page_png
    ):
        fake_ollama.replies = 500
        configure(monkeypatch, fake_ollama)

        from paperless.parsers.tesseract import RasterisedDocumentParser

        RasterisedDocumentParser.calls.clear()
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png")

        assert parser.get_text() == "tesseract fallback text"
        assert len(RasterisedDocumentParser.calls) == 1

    def test_no_fallback_means_empty_text(self, monkeypatch, fake_ollama, page_png):
        fake_ollama.replies = 500
        configure(monkeypatch, fake_ollama, FALLBACK="false")
        parser = OllamaOcrParser()
        parser.parse(page_png, "image/png")
        assert parser.get_text() == ""

    def test_unreachable_endpoint_declines_in_score(self, monkeypatch):
        monkeypatch.setenv("PAPERLESS_OLLAMA_OCR_ENDPOINT", "http://127.0.0.1:1")
        assert OllamaOcrParser.score("image/png", "x.png") is None

    def test_score_is_returned_when_reachable(self, monkeypatch, fake_ollama):
        configure(monkeypatch, fake_ollama, SCORE="42")
        assert OllamaOcrParser.score("image/png", "x.png") == 42

    def test_unsupported_mime_declines(self, monkeypatch, fake_ollama):
        configure(monkeypatch, fake_ollama)
        assert OllamaOcrParser.score("text/plain", "x.txt") is None

    def test_archive_failure_keeps_the_text(self, monkeypatch, fake_ollama, page_png):
        fake_ollama.replies = TEXT
        configure(monkeypatch, fake_ollama)
        parser = OllamaOcrParser()

        def boom(*args, **kwargs):
            raise RuntimeError("renderer exploded")

        monkeypatch.setattr(parser, "_build_searchable_pdf", boom)
        parser.parse(page_png, "image/png")

        assert parser.get_text() == TEXT
        assert parser.get_archive_path() is None
