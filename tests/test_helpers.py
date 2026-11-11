"""Tests for response parsing, bbox normalisation and the text layer."""

from __future__ import annotations

import pytest

from paperless_ollama_ocr.helpers import OcrBlock
from paperless_ollama_ocr.helpers import OcrPage
from paperless_ollama_ocr.helpers import _coerce_bbox
from paperless_ollama_ocr.helpers import is_degenerate_text
from paperless_ollama_ocr.helpers import parse_ocr_response
from paperless_ollama_ocr.helpers import prepare_request_image
from paperless_ollama_ocr.helpers import rescale_blocks
from paperless_ollama_ocr.helpers import strip_image_placeholders

W, H = 1700, 2200  # a 200 dpi A4 page


class TestBboxNormalisation:
    def test_pixels_pass_through(self):
        assert _coerce_bbox([100, 200, 800, 260], W, H) == (100, 200, 800, 260)

    def test_zero_to_thousand_grid_is_scaled_when_pinned(self):
        box = _coerce_bbox([100, 100, 500, 150], W, H, "normalized1000")
        assert box == (170, 220, 850, 330)

    def test_auto_scales_a_grid_that_cannot_be_pixels(self):
        # 1000 exceeds both dimensions of this small page, so it is a grid.
        assert _coerce_bbox([100, 100, 1000, 1000], 800, 600) == (80, 60, 800, 600)

    def test_auto_trusts_pixels(self):
        assert _coerce_bbox([100, 200, 800, 260], W, H) == (100, 200, 800, 260)

    def test_fractions_are_scaled(self):
        box = _coerce_bbox([0.1, 0.1, 0.5, 0.2], W, H)
        assert box == pytest.approx((170, 220, 850, 440))

    def test_inverted_box_is_ordered(self):
        assert _coerce_bbox([800, 260, 100, 200], W, H) == (100, 200, 800, 260)

    def test_polygon_collapses_to_extent(self):
        polygon = [[100, 200], [800, 200], [800, 260], [100, 260]]
        assert _coerce_bbox(polygon, W, H) == (100, 200, 800, 260)

    def test_out_of_bounds_is_rejected(self):
        assert _coerce_bbox([100, 200, 4000, 5000], W, H) is None

    def test_degenerate_box_is_rejected(self):
        assert _coerce_bbox([100, 200, 100, 260], W, H) is None

    def test_garbage_is_rejected(self):
        assert _coerce_bbox("bottom left", W, H) is None
        assert _coerce_bbox([1, 2], W, H) is None
        assert _coerce_bbox([1, 2, 3, 4], 0, 0) is None

    def test_box_is_clipped_to_the_page(self):
        assert _coerce_bbox([-10, -10, 900, 900], W, H) == (0, 0, 900, 900)


class TestParsePlainText:
    def test_plain_transcription(self):
        page = parse_ocr_response("Rechnung Nr. 4711\nBetrag: 1.234,56 EUR")
        assert "Rechnung Nr. 4711" in page.text
        assert page.blocks == []

    def test_model_chatter_tokens_are_removed(self):
        page = parse_ocr_response("Text<|endoftext|>\n\n\n\nMore<|ref|>")
        assert page.text == "Text\n\nMore"

    def test_image_placeholders_are_removed(self):
        assert strip_image_placeholders("a ![img-0.jpeg](img-0.jpeg) b") == "a b"

    def test_empty_response(self):
        assert parse_ocr_response("").text == ""

    def test_code_fence_is_stripped_from_plain_text(self):
        page = parse_ocr_response("```text\nhello\nworld\n```")
        assert "hello" in page.text
        assert "```" not in page.text


class TestParseJson:
    def test_lines_with_pixel_boxes(self):
        raw = (
            '{"lines":[{"text":"Hello","bbox":[100,200,800,260]},'
            '{"text":"World","bbox":[100,300,900,360]}]}'
        )
        page = parse_ocr_response(raw, W, H)
        assert page.text == "Hello\nWorld"
        assert [block.bbox for block in page.blocks] == [
            (100.0, 200.0, 800.0, 260.0),
            (100.0, 300.0, 900.0, 360.0),
        ]

    def test_fenced_json(self):
        raw = '```json\n{"lines":[{"text":"Hi","bbox":[0,0,10,10]}]}\n```'
        page = parse_ocr_response(raw, W, H)
        assert page.text == "Hi"
        assert len(page.blocks) == 1

    def test_json_inside_prose(self):
        raw = 'Sure! {"lines":[{"text":"Hi","bbox":[0,0,10,10]}]} Hope that helps.'
        assert parse_ocr_response(raw, W, H).text == "Hi"

    def test_bbox_2d_key_is_understood(self):
        raw = '{"lines":[{"text":"Hi","bbox_2d":[10,20,30,40]}]}'
        assert parse_ocr_response(raw, W, H).blocks[0].bbox == (10, 20, 30, 40)

    def test_scaled_coordinates_are_understood(self):
        raw = '{"lines":[{"text":"Hi","bbox":[100,100,500,200]}]}'
        page = parse_ocr_response(raw, W, H, "normalized1000")
        assert page.blocks[0].bbox == (170, 220, 850, 440)

    def test_page_dimensions_of_the_model_are_ignored(self):
        # A model may report its own resized dimensions; our page image wins.
        raw = (
            '{"dimensions":{"width":1024,"height":1325},'
            '"lines":[{"text":"Hi","bbox":[100,100,500,200]}]}'
        )
        page = parse_ocr_response(raw, W, H, "normalized1000")
        assert page.blocks[0].bbox == (170, 220, 850, 440)

    def test_fractional_coordinates_are_understood(self):
        raw = '{"lines":[{"text":"Hi","bbox":[0.1,0.1,0.5,0.2]}]}'
        page = parse_ocr_response(raw, W, H)
        assert page.blocks[0].bbox == pytest.approx((170, 220, 850, 440))

    def test_text_without_box_still_counts(self):
        raw = '{"lines":[{"text":"first"},{"text":"second"}]}'
        page = parse_ocr_response(raw, W, H)
        assert page.text == "first\nsecond"
        assert all(block.bbox is None for block in page.blocks)

    def test_malformed_json_falls_back_to_text(self):
        raw = '{"lines": [{"text": "broken"'
        page = parse_ocr_response(raw, W, H)
        assert page.text
        assert page.blocks == []

    def test_nested_blocks_are_found(self):
        raw = (
            '{"pages":[{"blocks":[{"content":"nested text",'
            '"bbox":{"left":1,"top":2,"right":30,"bottom":40}}]}]}'
        )
        page = parse_ocr_response(raw, W, H)
        assert page.text == "nested text"
        assert page.blocks[0].bbox == (1, 2, 30, 40)


class TestParseGrounding:
    def test_deepseek_ocr_layout_output(self):
        raw = (
            "<|ref|>heading<|/ref|><|det|>[[10, 20, 300, 60]]<|/det|>Rechnung\n"
            "<|ref|>amount<|/ref|><|det|>[[10, 100, 300, 140]]<|/det|>1.234,56"
        )
        page = parse_ocr_response(raw, W, H)
        assert page.text == "Rechnung\n1.234,56"
        assert [block.bbox for block in page.blocks] == [
            (10.0, 20.0, 300.0, 60.0),
            (10.0, 100.0, 300.0, 140.0),
        ]

    def test_grounding_with_several_boxes_uses_the_first(self):
        raw = "<|ref|>t<|/ref|><|det|>[[1, 2, 30, 40], [50, 60, 70, 80]]<|/det|>body"
        assert parse_ocr_response(raw, W, H).blocks[0].bbox == (1, 2, 30, 40)

    def test_grounding_without_text_uses_the_label(self):
        raw = "<|ref|>figure<|/ref|><|det|>[[1, 2, 30, 40]]<|/det|>"
        assert parse_ocr_response(raw, W, H).text == "figure"


class TestHtmlCleanup:
    """DeepSeek-OCR emits HTML tables for some form layouts."""

    def test_table_becomes_one_row_per_line(self):
        raw = (
            "<table><tr><th>Certificate Number</th><td>2026-04417</td></tr>"
            "<tr><th>Date of Marriage</th><td>14 June 2026</td></tr></table>"
        )
        page = parse_ocr_response(raw)
        assert (
            page.text
            == "Certificate Number | 2026-04417\nDate of Marriage | 14 June 2026"
        )

    def test_truncated_table_leaves_no_markup(self):
        # Exactly what showed up in the wild: a model that stopped after the
        # first header cell.
        page = parse_ocr_response('<table><th colspan="1" rowspan="2">')
        assert "<" not in page.text
        assert page.text == ""

    def test_inline_tags_are_stripped(self):
        page = parse_ocr_response("Amount: <b>960,33 EUR</b><br>paid")
        assert page.text == "Amount: 960,33 EUR\npaid"

    def test_entities_are_decoded(self):
        page = parse_ocr_response("<td>M&#252;ller &amp; S&#246;hne</td>")
        assert page.text == "Müller & Söhne"

    def test_plain_text_with_angle_brackets_is_untouched(self):
        page = parse_ocr_response("if a < b and c > d then stop")
        assert page.text == "if a < b and c > d then stop"

    def test_table_without_rows_keeps_the_cell_text(self):
        page = parse_ocr_response(
            '<table><th colspan="1">Certificate Number</th><td>2026-04417</td></table>'
        )
        assert "2026-04417" in page.text
        assert "<" not in page.text


class TestDegenerateText:
    """Repetition loops must not be stored as if they were text."""

    def test_repetition_loop_is_detected(self):
        loop = "\n".join(["1. C"] * 400 + ["2. D"] * 200)
        assert is_degenerate_text(loop) is True

    def test_multi_line_cycle_is_detected(self):
        # The real deepseek-ocr loop: 655 lines cycling through distinct labels,
        # 50 % unique, so only the line-length signal catches it.
        cycle = [f"{i}. {chr(65 + i % 20)}" for i in range(30)]
        loop = "\n".join(cycle * 22)
        assert is_degenerate_text(loop) is True

    def test_bank_statement_is_not_flagged(self):
        statement = "\n".join(
            f"{i:02d}.11.2026  Lastschrift  {i},00 EUR  Saldo 1.{i:03d},00 EUR"
            for i in range(120)
        )
        assert is_degenerate_text(statement) is False

    def test_dense_price_table_is_not_flagged(self):
        table = "\n".join(f"{i}\tPosition {i}\t{i * 12.5:.2f}" for i in range(100))
        assert is_degenerate_text(table) is False

    def test_real_form_text_is_not_flagged(self):
        text = "\n".join(
            [
                "CERTIFICATE OF MARRIAGE",
                "1A. CERTIFICATE NUMBER",
                "2026-04417",
                "1B. DATE OF MARRIAGE",
                "14 June 2026",
                "1C. FULL NAME OF PARTY A",
                "Jordan A. Reyes",
                "1D. FULL NAME OF PARTY B",
                "Samuel T. Whitfield",
                "2A. OFFICIANT",
                "Rev. Marguerite Okafor",
                "2B. OFFICIAL TITLE",
                "Ordained Minister",
            ]
        )
        assert is_degenerate_text(text) is False

    def test_short_repetitive_text_is_not_flagged(self):
        # A small form that legitimately repeats "N/A" is not a loop.
        assert is_degenerate_text("\n".join(["N/A"] * 6)) is False

    def test_repeated_but_varied_lines_are_not_flagged(self):
        text = "\n".join(f"Row {index}: value {index}" for index in range(40))
        assert is_degenerate_text(text) is False


class TestRequestImage:
    def test_small_page_is_not_resized(self):
        Image = pytest.importorskip("PIL.Image")
        image = Image.new("RGB", (800, 1000), "white")
        data, size = prepare_request_image(image, 2400)
        assert size == (800, 1000)
        assert data[:2] == b"\xff\xd8"  # JPEG magic

    def test_large_page_is_downscaled_for_the_request(self):
        Image = pytest.importorskip("PIL.Image")
        image = Image.new("RGB", (3446, 4873), "white")
        data, size = prepare_request_image(image, 2400)
        assert max(size) == 2400
        assert size[0] / size[1] == pytest.approx(3446 / 4873, rel=1e-3)
        assert len(data) < 400_000

    def test_zero_disables_the_cap(self):
        Image = pytest.importorskip("PIL.Image")
        image = Image.new("RGB", (3000, 4000), "white")
        _, size = prepare_request_image(image, 0)
        assert size == (3000, 4000)


class TestRescaleBlocks:
    def test_boxes_are_mapped_back_to_the_full_page(self):
        page = OcrPage(
            text="hello",
            blocks=[OcrBlock(text="hello", bbox=(10, 20, 30, 40))],
        )
        scaled = rescale_blocks(page, 2.0, 3.0)
        assert scaled.blocks[0].bbox == (20, 60, 60, 120)
        assert scaled.text == "hello"

    def test_missing_boxes_stay_missing(self):
        page = OcrPage(text="hello", blocks=[OcrBlock(text="hello")])
        assert rescale_blocks(page, 2.0, 2.0).blocks[0].bbox is None

    def test_identity_returns_the_same_page(self):
        page = OcrPage(text="hello")
        assert rescale_blocks(page, 1, 1) is page
