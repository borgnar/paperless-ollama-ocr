"""Pure helpers: prompts, response parsing, bbox geometry, PDF text layer.

The PDF text-layer code derives from paperless-mistral-ocr (GPL-3.0-or-later,
Jens Franke), which in turn came from paperless-ngx PR #13085. It is kept
close to the original so upstream fixes stay easy to merge.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocrmypdf.models.ocr_element import BoundingBox
    from ocrmypdf.models.ocr_element import OcrElement

logger = logging.getLogger("paperless.parsing.ollama")

#: DPI used to rasterise pages before overlaying the invisible text layer.
RENDER_DPI: int = 200

#: Instruction sent with every page. Transcribe-only: any "clean up" or
#: "summarise" phrasing silently loses content that full-text search needs.
PROMPT_TEXT: str = (
    "Transcribe all text on this page, exactly as printed. Keep the reading "
    "order and one line per text line. Do not translate, summarise, explain, "
    "correct or add anything. Preserve numbers, punctuation and diacritics. "
    "Output the transcription only, without commentary, markdown fences or "
    "code blocks."
)

PROMPT_STRUCTURED: str = (
    PROMPT_TEXT
    + " The image is {width}x{height} pixels. Return JSON only, in the form "
    '{{"lines":[{{"text":"<line as printed>","bbox":[x0,y0,x1,y1]}}]}} where '
    "the bbox is that line's box in absolute pixel coordinates of this image, "
    "origin at the top-left corner."
)

#: JSON schema for Ollama's ``format`` parameter (constrained decoding).
LINE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "prefixItems": [{"type": "number"}] * 4,
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["text"],
            },
        },
    },
    "required": ["lines"],
}

#: Accepted values for the bbox coordinate space.
BBOX_SPACES = ("auto", "pixel", "normalized1000", "fraction")


def build_prompt(structured: bool, width: int = 0, height: int = 0) -> str:
    """The instruction sent with a page, with its pixel size filled in."""
    if not structured:
        return PROMPT_TEXT
    if width <= 0 or height <= 0:
        return PROMPT_STRUCTURED.format(width="W", height="H")
    return PROMPT_STRUCTURED.format(width=width, height=height)


@dataclass
class OcrBlock:
    """One positioned (or unpositioned) chunk of transcribed text."""

    text: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class OcrPage:
    """Transcription of a single page."""

    text: str = ""
    blocks: list[OcrBlock] = field(default_factory=list)


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------

#: DeepSeek-OCR emits layout as ``<|ref|>label<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|>``
#: followed by the block's text.
_GROUNDING = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>(?P<det>.*?)<\|/det\|>",
    re.DOTALL,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_SPECIAL_TOKENS = re.compile(r"<\|[^|>]*\|>")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_FENCE_LINE = re.compile(r"^[ \t]*```.*$", re.MULTILINE)
_IMAGE_PLACEHOLDER = re.compile(r"!\[[^\]]*\]\([^)]*\)")

#: Keys a model may use for the text of a block, in order of preference.
_TEXT_KEYS = ("text", "content", "markdown", "line", "value", "transcription")
#: Keys a model may use for a block's box.
_BOX_KEYS = ("bbox", "bbox_2d", "box", "bounds", "coordinates", "rect", "polygon")

#: Dict layouts a model may use instead of a 4-element list.
_BOX_DICTS = (
    ("left", "top", "right", "bottom"),
    ("x0", "y0", "x1", "y1"),
    ("x1", "y1", "x2", "y2"),
    ("x", "y", "x2", "y2"),
)


#: DeepSeek-OCR emits HTML tables for some form layouts, and other vision models
#: sometimes do the same. Raw markup in the stored text is worse than useless:
#: full-text search sees tag soup, and anything downstream that classifies or
#: summarises the document has to read `<table><th colspan="1" rowspan="2">`
#: instead of the certificate. Tables are flattened to one row per line.
_HTML_TABLE = re.compile(r"<table\b.*?</table\s*>", re.DOTALL | re.IGNORECASE)
_HTML_ROW = re.compile(
    r"<tr\b[^>]*>(.*?)(?=<tr\b|</table\s*>|$)", re.DOTALL | re.IGNORECASE
)
_HTML_CELL = re.compile(
    r"<t[dh]\b[^>]*>(.*?)(?=<t[dh]\b|</tr\s*>|$)", re.DOTALL | re.IGNORECASE
)
_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
#: Only these tag names trigger the cleanup; a stray ``a < b`` in text is left alone.
_HTML_LIKE = re.compile(
    r"</?(?:table|tr|td|th|thead|tbody|tfoot|p|div|span|br|hr|b|i|u|s|strong|em|"
    r"ul|ol|li|h[1-6]|font|img|thead)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_BLOCK_BREAK = re.compile(
    r"<br\s*/?>|</p\s*>|</div\s*>|</li\s*>|</h[1-6]\s*>", re.IGNORECASE
)


def _strip_html_fragment(fragment: str) -> str:
    """Text of one HTML fragment: tags dropped, entities decoded, lines trimmed."""
    text = _HTML_BLOCK_BREAK.sub("\n", fragment)
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def html_tables_to_text(text: str) -> str:
    """Flatten HTML tables to `cell | cell` lines and drop leftover tags.

    Handles truncated markup too (a model that stops mid-table), which would
    otherwise leave an unterminated ``<table>`` in the stored text, and tables
    without ``<tr>`` wrappers, where the cells themselves carry the content.
    """

    def flatten(match: re.Match[str]) -> str:
        rows: list[str] = []
        for row in _HTML_ROW.findall(match.group(0)):
            cells = [_strip_html_fragment(cell) for cell in _HTML_CELL.findall(row)]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        if not rows:
            # No <tr> at all: keep the cell contents rather than dropping them.
            loose = [
                _strip_html_fragment(cell)
                for cell in _HTML_CELL.findall(match.group(0))
            ]
            rows = [cell for cell in loose if cell]
        return "\n".join(rows)

    converted = _HTML_TABLE.sub(flatten, text)
    # Whatever is left of the table (no closing tag, stray <th>) is tag soup.
    if "<" in converted and _HTML_TAG.search(converted):
        converted = _strip_html_fragment(converted)
    return converted


def _clean_text(text: str) -> str:
    """Strip model chatter, tokens and markdown noise from a transcription."""
    cleaned = _SPECIAL_TOKENS.sub("", text or "")
    cleaned = _IMAGE_PLACEHOLDER.sub("", cleaned)
    cleaned = _FENCE_LINE.sub("", cleaned)
    if _HTML_LIKE.search(cleaned):
        cleaned = html_tables_to_text(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _numbers(value: object) -> list[float] | None:
    """Flatten a bbox-ish value (dict, list, nested list, 4 points) to 4 floats."""
    if isinstance(value, dict):
        for keys in _BOX_DICTS:
            if all(key in value for key in keys):
                try:
                    return [float(value[key]) for key in keys]
                except (TypeError, ValueError):
                    return None
        if all(key in value for key in ("x", "y", "w", "h")):
            try:
                x, y, w, h = (float(value[key]) for key in ("x", "y", "w", "h"))
            except (TypeError, ValueError):
                return None
            return [x, y, x + w, y + h]
        return None
    if not isinstance(value, (list, tuple)) or not value:
        return None
    flat: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            if len(item) == 2 and all(isinstance(v, (int, float)) for v in item):
                flat.extend(float(v) for v in item)
            else:
                nested = _numbers(item)
                if nested is None:
                    return None
                flat.extend(nested)
        elif isinstance(item, (int, float)):
            flat.append(float(item))
        else:
            return None
    if len(flat) < 4:
        return None
    if len(flat) == 4:
        return flat
    # A polygon (corner points): collapse to its extent.
    xs = flat[0::2]
    ys = flat[1::2]
    if len(xs) >= 2 and len(ys) >= 2:
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _coerce_bbox(
    value: object,
    width: float,
    height: float,
    space: str = "auto",
) -> tuple[float, float, float, float] | None:
    """Normalise a model's box into image pixels, or return None.

    Models express boxes as image pixels, as a 0-1000 grid, or as fractions.
    ``space`` pins the interpretation; ``auto`` trusts pixels and only falls
    back to the other two when the numbers cannot be pixels. Boxes that still
    fall outside the page are rejected rather than clamped, because a
    misplaced text band looks fine in the index but ruins copy-paste.
    """
    nums = _numbers(value)
    if nums is None or width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = nums
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    largest = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if space == "fraction" or (space == "auto" and largest <= 1.001):
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    elif space == "normalized1000" or (
        space == "auto" and largest > max(width, height) * 1.02 and largest <= 1000
    ):
        x0, x1 = x0 * width / 1000.0, x1 * width / 1000.0
        y0, y1 = y0 * height / 1000.0, y1 * height / 1000.0

    tolerance_x = width * 0.02 + 2
    tolerance_y = height * 0.02 + 2
    if x1 > width + tolerance_x or y1 > height + tolerance_y:
        return None
    if x0 < -tolerance_x or y0 < -tolerance_y:
        return None
    x0, x1 = max(0.0, x0), min(width, x1)
    y0, y1 = max(0.0, y0), min(height, y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (x0, y0, x1, y1)


def _block_from_mapping(
    item: dict,
    width: float,
    height: float,
    space: str,
) -> OcrBlock | None:
    """Turn one JSON object from a model into an :class:`OcrBlock`."""
    text = ""
    for key in _TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if not text:
        return None
    bbox = None
    for key in _BOX_KEYS:
        if key in item:
            bbox = _coerce_bbox(item[key], width, height, space)
            if bbox is not None:
                break
    return OcrBlock(text=text, bbox=bbox)


def _json_payload(raw: str) -> object | None:
    """Extract a JSON value from a model response, fences and prose included."""
    candidates = [match.group(1) for match in _FENCE.finditer(raw)]
    candidates.append(raw)
    for candidate in candidates:
        text = candidate.strip()
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start == -1 or end <= start:
                continue
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _flatten_json_blocks(payload: object) -> list[dict]:
    """Collect the block-like objects out of a model's JSON structure."""
    found: list[dict] = []

    def visit(node: object, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(node, list):
            for entry in node:
                visit(entry, depth + 1)
            return
        if not isinstance(node, dict):
            return
        if any(isinstance(node.get(key), str) for key in _TEXT_KEYS):
            found.append(node)
            return
        for value in node.values():
            if isinstance(value, (list, dict)):
                visit(value, depth + 1)

    visit(payload)
    return found


def _parse_grounding(
    raw: str,
    width: float,
    height: float,
    space: str,
) -> OcrPage | None:
    """Parse DeepSeek-OCR style ``<|ref|>..<|det|>`` layout output."""
    matches = list(_GROUNDING.finditer(raw))
    if not matches:
        return None
    blocks: list[OcrBlock] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = _SPECIAL_TOKENS.sub("", raw[start:end]).strip()
        if not body:
            body = match.group("label").strip()
        if not body:
            continue
        # A detection may carry several boxes; the first one is the block.
        nums = [float(v) for v in _NUMBER.findall(match.group("det"))[:4]]
        bbox = _coerce_bbox(nums, width, height, space) if len(nums) == 4 else None
        blocks.append(OcrBlock(text=body, bbox=bbox))
    if not blocks:
        return None
    return OcrPage(text="\n".join(block.text for block in blocks), blocks=blocks)


def parse_ocr_response(
    raw: str,
    width: float = 0.0,
    height: float = 0.0,
    space: str = "auto",
) -> OcrPage:
    """Turn a model response into text plus positioned blocks.

    Tries the DeepSeek-OCR grounding format first, then JSON (with or without
    a fence), then falls back to the raw text with no geometry. ``width`` and
    ``height`` are the pixel dimensions of the image that was sent; when they
    are unknown, boxes are dropped instead of guessed.
    """
    page = _parse_grounding(raw, width, height, space)
    if page is not None:
        page.text = _clean_text(page.text)
        return page

    payload = _json_payload(raw)
    if payload is not None:
        blocks: list[OcrBlock] = []
        for item in _flatten_json_blocks(payload):
            block = _block_from_mapping(item, width, height, space)
            if block is not None:
                blocks.append(block)
        if blocks:
            return OcrPage(
                text=_clean_text("\n".join(block.text for block in blocks)),
                blocks=blocks,
            )

    return OcrPage(text=_clean_text(raw), blocks=[])


def strip_image_placeholders(text: str) -> str:
    """Remove markdown image references and tidy up the whitespace."""
    return _clean_text(text)


# ----------------------------------------------------------------------
# Searchable PDF assembly
# ----------------------------------------------------------------------


def _text_band_elements(
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
    max_width: float,
    max_height: float,
    dpi: float,
) -> list[OcrElement]:
    """Lay ``text`` into one thin invisible ``ocr_line`` band per text line.

    A single tall box makes the fpdf2 renderer scale text so aggressively that
    the glyphs become unrecoverable, so the box is divided into one horizontal
    band per line of text and each band's height is capped to roughly a single
    line.  Returns the resulting ``ocr_line`` elements (each wrapping one
    ``ocrx_word``); an empty list when there is no usable text.
    """
    from ocrmypdf.models.ocr_element import BoundingBox
    from ocrmypdf.models.ocr_element import OcrElement

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    def clamp(
        band_left: float, band_top: float, band_right: float, band_bottom: float
    ) -> BoundingBox | None:
        band_left = max(0.0, min(band_left, max_width))
        band_right = max(0.0, min(band_right, max_width))
        band_top = max(0.0, min(band_top, max_height))
        band_bottom = max(0.0, min(band_bottom, max_height))
        if band_right <= band_left or band_bottom <= band_top:
            return None
        return BoundingBox(band_left, band_top, band_right, band_bottom)

    band_cap = dpi * 0.3
    slot = (bottom - top) / len(lines)
    elements: list[OcrElement] = []
    for index, line in enumerate(lines):
        band_top = top + index * slot
        band_bottom = band_top + min(slot, band_cap)
        bbox = clamp(left, band_top, right, band_bottom)
        if bbox is None:
            continue
        word = OcrElement(ocr_class="ocrx_word", text=line, bbox=bbox)
        elements.append(OcrElement(ocr_class="ocr_line", bbox=bbox, children=[word]))
    return elements


def build_ocr_page(
    page: OcrPage,
    image_width: int,
    image_height: int,
    dpi: float,
    page_number: int = 0,
) -> OcrElement:
    """Build an ``OcrElement`` page tree for one transcribed page.

    Positioned blocks become bands at their coordinates. When the model gave
    no usable boxes (plain-text mode, or boxes that did not survive
    normalisation) the whole transcription is laid across the page instead, so
    the archive stays searchable either way.
    """
    from ocrmypdf.models.ocr_element import BoundingBox
    from ocrmypdf.models.ocr_element import OcrElement

    children: list[OcrElement] = []
    for block in page.blocks:
        if not block.text.strip() or block.bbox is None:
            continue
        left, top, right, bottom = block.bbox
        children.extend(
            _text_band_elements(
                block.text,
                left,
                top,
                right,
                bottom,
                image_width,
                image_height,
                dpi,
            )
        )

    if not children:
        children.extend(
            _text_band_elements(
                page.text,
                0,
                0,
                image_width,
                image_height,
                image_width,
                image_height,
                dpi,
            )
        )

    return OcrElement(
        ocr_class="ocr_page",
        bbox=BoundingBox(0, 0, image_width, image_height),
        dpi=dpi,
        page_number=page_number,
        children=children,
    )


def load_page_images(source: Path, mime_type: str, dpi: int) -> list:
    """Return one RGB ``PIL.Image`` per page of the source document.

    PDF input is rasterised with pdf2image; image input (including multi-frame
    TIFF) is loaded directly, one entry per frame.
    """
    from PIL import Image
    from PIL import ImageSequence

    if mime_type == "application/pdf":
        from pdf2image import convert_from_path

        return convert_from_path(source, dpi=dpi)

    with Image.open(source) as opened:
        return [frame.convert("RGB") for frame in ImageSequence.Iterator(opened)]


def encode_image(image, quality: int = 90) -> bytes:
    """Encode a page image as JPEG bytes for the model request."""
    import io

    image.save(buffer := io.BytesIO(), format="JPEG", quality=quality)
    return buffer.getvalue()


def prepare_request_image(
    image,
    max_side: int = 2400,
    quality: int = 90,
) -> tuple[bytes, tuple[int, int]]:
    """Encode a page for the request, downscaling if it exceeds ``max_side``.

    Returns the JPEG bytes and the pixel size they were produced at, so
    bounding boxes from the model can be mapped back onto the full-resolution
    page. Only the copy sent to the model is reduced: the archive PDF keeps the
    full rasterisation, so a 600 dpi scan costs a smaller request without
    losing archive quality.
    """
    longest = max(image.width, image.height)
    if max_side and longest > max_side:
        from PIL import Image

        scale = max_side / longest
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(size, Image.LANCZOS)
    return encode_image(image, quality), (image.width, image.height)


def rescale_blocks(page: OcrPage, scale_x: float, scale_y: float) -> OcrPage:
    """Map a page's boxes from the request image back to the full page."""
    if scale_x == 1 and scale_y == 1:
        return page
    blocks = [
        OcrBlock(
            text=block.text,
            bbox=(
                (
                    block.bbox[0] * scale_x,
                    block.bbox[1] * scale_y,
                    block.bbox[2] * scale_x,
                    block.bbox[3] * scale_y,
                )
                if block.bbox
                else None
            ),
        )
        for block in page.blocks
    ]
    return OcrPage(text=page.text, blocks=blocks)
