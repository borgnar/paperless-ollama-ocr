# Changelog

Written by an AI agent; see the warning at the top of the README.

## 0.1.4

- **HTML tables in model output are flattened.** DeepSeek-OCR (and sometimes other
  vision models) answer form-like pages with `<table>`/`<th>`/`<td>` markup instead
  of text. Stored verbatim, that markup is what any downstream consumer — search,
  tags, classification — then reads. Tables are now converted to one `cell | cell`
  line per row, inline tags stripped, entities decoded. Truncated markup (a model
  that stops after `<table><th colspan="1" rowspan="2">`) yields empty text rather
  than tag soup, and pages that were entirely markup fall through to the per-page
  failure path. Plain text containing `<` or `>` is untouched.
- Test count: 90.

## 0.1.3

- **`deepseek-ocr` becomes the default model.** Measured against real Ollama
  output: 8.2 s on a skewed page with a low-contrast stamp, against 27.6 s for
  `qwen2.5vl:3b`, with the same six values transcribed correctly.
- **Removed an unverified claim.** The README had said that a `<|grounding|>`
  prompt makes `deepseek-ocr` emit per-block boxes. Three prompts were tried on
  Ollama 0.40.15 (plain transcribe, `<|grounding|>Convert the document to
  markdown.`, `<|grounding|>Given the layout of the image.`) and none produced
  `<|det|>` output, so the claim is gone. The parser still understands the
  format for builds that do emit it.
- Documented that `Parse the text in the image.` returns an empty answer from
  `deepseek-ocr`; the explicit transcribe instruction is required.

## 0.1.2

- `parser.url` pointed at the upstream Mistral repository, so Paperless printed
  the wrong link in its "handled by third-party parser" line. Fixed.
- Added tests for the geometry pass: ocrmypdf is invoked with
  `tesseract_timeout=0` (renderer only, no Tesseract text layer), images are
  skipped, and a failed pass still transcribes the original.
- Added a CI workflow running ruff and pytest.
- Test count: 84.

## 0.1.1

- Added `PAPERLESS_OLLAMA_OCR_FORCE`, which transcribes PDFs that already carry a
  text layer. Paperless passes `produce_archive=False` for those documents
  (`PAPERLESS_OCR_MODE=auto`, `PAPERLESS_ARCHIVE_FILE_GENERATION=auto`), and the
  inherited shortcut reused the local text layer instead of calling the model —
  sensible for a paid API, wrong for local inference.
- The skip is now logged at INFO instead of DEBUG, so it is visible at the
  default log level, and says which setting overrides it.
- The plugin version appears in the registry log line, so an update can be
  verified from the log.

## 0.1.0

First version, forked from [paperless-mistral-ocr](https://github.com/jensfr1/paperless-mistral-ocr):

- Per-page rasterisation and one `/api/chat` request per page against a
  self-hosted Ollama, replacing the single document-level Mistral API call.
- Parsing of three response shapes: plain text, JSON (fenced or not, several
  key spellings), and DeepSeek-OCR's `<|ref|>…<|det|>` grounding format.
- Bounding-box normalisation for pixels, 0–1000 grids and fractions, with
  out-of-range boxes rejected rather than clamped.
- Request-side downscaling (`PAPERLESS_OLLAMA_OCR_MAX_SIDE`) with bounding boxes
  mapped back onto the full-resolution page.
- Cached reachability probe so `score()` declines while Ollama is down instead
  of failing every document.
- Kept from upstream: the fpdf2 invisible text layer, the ocrmypdf geometry pass,
  the Tesseract fallback, and the plugin packaging.
