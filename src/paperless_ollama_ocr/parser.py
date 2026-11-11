"""Ollama OCR parser plugin for Paperless-ngx.

Registers through the ``paperless_ngx.parsers`` entrypoint group and
subclasses the built-in RemoteDocumentParser to inherit its lifecycle,
thumbnail generation and metadata extraction. Only the identity,
``score()`` and ``parse()`` are overridden.

Derived from paperless-mistral-ocr (GPL-3.0-or-later, Jens Franke), which
took its OCR plumbing from paperless-ngx PR #13085. The difference: instead
of one document-level call to a hosted API, every page is rasterised and sent
to a vision model on a self-hosted Ollama instance, and the searchable
archive is assembled from the per-page results.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from paperless.parsers.remote import _SUPPORTED_MIME_TYPES
from paperless.parsers.remote import RemoteDocumentParser
from paperless.parsers.utils import extract_pdf_text
from paperless.parsers.utils import post_process_text

from paperless_ollama_ocr.client import OllamaClient
from paperless_ollama_ocr.client import OllamaError
from paperless_ollama_ocr.client import endpoint_available
from paperless_ollama_ocr.config import OllamaConfig
from paperless_ollama_ocr.helpers import LINE_SCHEMA
from paperless_ollama_ocr.helpers import OcrPage
from paperless_ollama_ocr.helpers import build_ocr_page
from paperless_ollama_ocr.helpers import build_prompt
from paperless_ollama_ocr.helpers import is_degenerate_text
from paperless_ollama_ocr.helpers import load_page_images
from paperless_ollama_ocr.helpers import parse_ocr_response
from paperless_ollama_ocr.helpers import prepare_request_image
from paperless_ollama_ocr.helpers import rescale_blocks

if TYPE_CHECKING:
    from paperless.parsers import ParserContext  # noqa: F401

logger = logging.getLogger("paperless.parsing.ollama")

__version__ = "0.1.6"

#: Confidence threshold for ocrmypdf's page rotation during preprocessing.
#: The ocrmypdf default of 12 is conservative and leaves many sideways
#: scans untouched; 6 catches noticeably more without flipping upright
#: pages in practice.
ROTATE_THRESHOLD: float = 6.0


class OllamaOcrParser(RemoteDocumentParser):
    """Transcribes documents with a vision model on a self-hosted Ollama.

    Each page is rasterised, sent to ``/api/chat``, and the replies are turned
    into plain text plus an invisible text layer over the original page
    images. No hosted service is involved and no API key is required.
    """

    name: str = "Ollama OCR Parser"
    version: str = __version__
    author: str = "paperless-ollama-ocr contributors"
    url: str = "https://github.com/borgnar/paperless-ollama-ocr"

    uses_remote_service: bool = True

    # ------------------------------------------------------------------
    # Registry interface
    # ------------------------------------------------------------------

    @classmethod
    def supported_mime_types(cls) -> dict[str, str]:
        """PDF and the usual image formats, same as the core remote parser."""
        return _SUPPORTED_MIME_TYPES

    @classmethod
    def score(
        cls,
        mime_type: str,
        filename: str,
        path: Path | None = None,
    ) -> int | None:
        """Priority for this file, or None when the plugin is not ready.

        No key is needed, but an endpoint that does not answer makes the
        parser decline so Paperless keeps using Tesseract for this run
        instead of failing every document. The probe result is cached, so
        this does not add an HTTP call per file.
        """
        config = OllamaConfig.from_env()
        if not config.is_valid():
            return None
        if mime_type not in _SUPPORTED_MIME_TYPES:
            return None
        if not endpoint_available(config.endpoint, config.api_key):
            logger.debug(
                "Ollama endpoint %s not reachable - declining.",
                config.endpoint,
            )
            return None
        return config.score

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def parse(
        self,
        document_path: Path,
        mime_type: str,
        *,
        produce_archive: bool = True,
    ) -> None:
        """Transcribe the document with Ollama and store the results."""
        config = OllamaConfig.from_env()

        if not config.is_valid():
            logger.warning(
                "Ollama OCR is not configured (PAPERLESS_OLLAMA_OCR_ENDPOINT "
                "missing), content will be empty.",
            )
            self._text = ""
            return

        # Born-digital PDFs for which Paperless requests no archive are not
        # sent to the model: the local text layer is reused instead. That is
        # Paperless' own decision (PAPERLESS_OCR_MODE / _ARCHIVE_FILE_GENERATION),
        # and with PAPERLESS_OLLAMA_OCR_FORCE it can be overridden per plugin.
        if not produce_archive and mime_type == "application/pdf" and not config.force:
            logger.info(
                "Ollama OCR skipped - Paperless requested no archive, so the "
                "existing text layer is used. Set PAPERLESS_OLLAMA_OCR_FORCE=true "
                "to transcribe these documents anyway.",
            )
            self._text = (
                post_process_text(extract_pdf_text(document_path, log=logger)) or ""
            )
            return

        source = document_path
        if config.preprocess:
            source = self._preprocess(document_path, mime_type) or document_path

        pages, images = self._ocr_document(source, mime_type, config)

        if pages is None:
            if config.fallback:
                logger.warning("Falling back to Tesseract after Ollama failure.")
                self._text = (
                    self._tesseract_fallback(
                        document_path,
                        mime_type,
                        produce_archive,
                    )
                    or ""
                )
            else:
                self._text = ""
            return

        self._text = post_process_text(
            "\n\n".join(page.text for page in pages if page.text.strip()),
        )

        if not produce_archive:
            return

        archive_path = self._tempdir / "archive.pdf"
        try:
            self._build_searchable_pdf(images, pages, archive_path, config)
        except Exception as e:
            # The transcription itself is fine; only the text layer failed.
            # Storing the text is still better than falling back to Tesseract.
            logger.exception("Building the searchable archive failed: %s", e)
            self._archive_path = None
            return
        self._archive_path = archive_path

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _ocr_document(
        self,
        source: Path,
        mime_type: str,
        config: OllamaConfig,
    ) -> tuple[list[OcrPage] | None, list]:
        """Transcribe every page. Returns ``(pages, images)`` or ``(None, [])``."""
        try:
            images = load_page_images(source, mime_type, config.dpi)
        except Exception as e:
            logger.exception("Rasterising the document failed: %s", e)
            return None, []

        if not images:
            logger.warning("No pages could be rasterised from %s.", source.name)
            return None, []

        schema = LINE_SCHEMA if config.structured else None
        client = OllamaClient(config)
        retry_side = min(config.max_side or 1600, 1600)

        def attempt(index: int, max_side: int) -> OcrPage | None:
            """One request for one page, or None when it yields no text."""
            image = images[index]
            data, (req_width, req_height) = prepare_request_image(image, max_side)
            payload_page = parse_ocr_response(
                client.transcribe(
                    data,
                    config.prompt
                    or build_prompt(config.structured, req_width, req_height),
                    schema=schema,
                ),
                req_width,
                req_height,
                config.bbox_space,
            )
            if not payload_page.text.strip():
                return None
            if is_degenerate_text(payload_page.text):
                logger.warning(
                    "Page %d came back as a repetition loop - discarding it.",
                    index + 1,
                )
                return None
            # Boxes refer to the (possibly downscaled) request image; map them
            # back onto the full-resolution page used for the archive.
            return rescale_blocks(
                payload_page,
                image.width / req_width,
                image.height / req_height,
            )

        def transcribe(index: int) -> OcrPage | None:
            started = time.monotonic()
            try:
                page = attempt(index, config.max_side)
            except OllamaError as e:
                logger.error("Page %d failed: %s", index + 1, e)
                return None
            if page is None:
                # An oversized request can leave a model with nothing to say
                # (deepseek-ocr returns an empty answer above ~2400 px). One
                # retry at a smaller size is cheaper than losing the page.
                logger.info(
                    "Page %d came back empty - retrying at %d px.",
                    index + 1,
                    retry_side,
                )
                try:
                    page = attempt(index, retry_side)
                except OllamaError as e:
                    logger.error("Page %d retry failed: %s", index + 1, e)
                    return None
            if page is None:
                logger.warning("Page %d came back empty twice.", index + 1)
                return None
            logger.debug(
                "Page %d: %d characters in %.1fs",
                index + 1,
                len(page.text),
                time.monotonic() - started,
            )
            return page

        started = time.monotonic()
        if config.parallel > 1 and len(images) > 1:
            with ThreadPoolExecutor(max_workers=config.parallel) as pool:
                results = list(pool.map(transcribe, range(len(images))))
        else:
            results = [transcribe(index) for index in range(len(images))]

        failed = sum(1 for page in results if page is None)
        if failed == len(images):
            logger.error(
                "Every page failed on %s (model %s, endpoint %s).",
                source.name,
                config.model,
                config.endpoint,
            )
            return None, []

        if failed:
            logger.warning(
                "%d of %d pages failed on %s; keeping the rest.",
                failed,
                len(images),
                source.name,
            )

        elapsed = time.monotonic() - started
        logger.info(
            "Ollama OCR: %d pages in %.1fs (model %s).",
            len(images),
            elapsed,
            config.model,
        )
        return [page or OcrPage() for page in results], images

    # ------------------------------------------------------------------
    # Preprocessing and fallback
    # ------------------------------------------------------------------

    def _preprocess(self, source: Path, mime_type: str) -> Path | None:
        """Deskew and rotate the pages before they are transcribed.

        A vision model reads rotated pages correctly either way, but the
        archive PDF would keep the original skew - pages then look sideways
        when browsing them in Paperless. Running ocrmypdf first fixes the
        geometry. ``tesseract_timeout=0`` keeps it from adding its own OCR
        text layer: only the page images are corrected, the text comes from
        the model.

        Returns the preprocessed file, or None to continue with the
        original when preprocessing is not applicable or fails.
        """
        if mime_type != "application/pdf":
            return None

        import ocrmypdf

        target = self._tempdir / "preprocessed.pdf"
        try:
            ocrmypdf.ocr(
                source,
                target,
                deskew=True,
                rotate_pages=True,
                rotate_pages_threshold=ROTATE_THRESHOLD,
                tesseract_timeout=0,
                progress_bar=False,
                output_type="pdf",
            )
        except Exception as e:
            logger.warning(
                "Preprocessing failed (%s), continuing with the original file.",
                type(e).__name__,
            )
            return None
        return target

    def _tesseract_fallback(
        self,
        document_path: Path,
        mime_type: str,
        produce_archive: bool,
    ) -> str:
        """Run the built-in Tesseract parser after an Ollama failure.

        Without this the document would be stored with empty content: the
        registry picks a parser before processing and does not retry with
        a different engine. The archive is copied into our own temporary
        directory because the Tesseract parser cleans up its own on exit.
        """
        import shutil

        from paperless.parsers.tesseract import RasterisedDocumentParser

        try:
            with RasterisedDocumentParser(self._logging_group) as parser:
                parser.parse(document_path, mime_type, produce_archive=produce_archive)
                archive = parser.get_archive_path()
                if archive and archive.exists():
                    target = self._tempdir / "archive_fallback.pdf"
                    shutil.copy2(archive, target)
                    self._archive_path = target
                return parser.get_text() or ""
        except Exception as e:
            logger.exception("Tesseract fallback failed as well: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Searchable PDF
    # ------------------------------------------------------------------

    def _build_searchable_pdf(
        self,
        images: list,
        pages: list[OcrPage],
        out_path: Path,
        config: OllamaConfig,
    ) -> None:
        """Assemble a searchable PDF from the page images and the transcriptions.

        Each page is written with an invisible text layer positioned from the
        model's bounding boxes (or across the whole page when it returned
        none), rendered with ocrmypdf's fpdf2 renderer and its bundled Unicode
        font. ocrmypdf is used purely as a renderer here, no OCR engine is
        involved.
        """
        import ocrmypdf
        import pikepdf
        from ocrmypdf.font import MultiFontManager
        from ocrmypdf.fpdf_renderer import Fpdf2PdfRenderer

        font_dir = Path(ocrmypdf.__file__).parent / "data"
        font_manager = MultiFontManager(font_dir)

        merged = pikepdf.Pdf.new()
        opened: list = []
        try:
            for index, image in enumerate(images):
                page = pages[index] if index < len(pages) else OcrPage()

                image_path = self._tempdir / f"page-{index}.png"
                image.save(image_path)

                ocr_page = build_ocr_page(
                    page,
                    image.width,
                    image.height,
                    config.dpi,
                    page_number=index,
                )

                page_pdf_path = self._tempdir / f"page-{index}.pdf"
                Fpdf2PdfRenderer(
                    page=ocr_page,
                    dpi=config.dpi,
                    multi_font_manager=font_manager,
                    invisible_text=True,
                    image=image_path,
                ).render(page_pdf_path)

                page_pdf = pikepdf.open(page_pdf_path)
                opened.append(page_pdf)
                merged.pages.extend(page_pdf.pages)

            merged.save(out_path)
        finally:
            for page_pdf in opened:
                page_pdf.close()
            merged.close()
