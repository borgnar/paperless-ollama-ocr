#!/bin/bash
# Installs the Ollama OCR plugin on every container start.
#
# Runs through the official paperless-ngx image's custom-init hook
# (/custom-cont-init.d), so the official image stays in use - "docker compose
# pull" keeps working as usual and the plugin is reinstalled after every
# update.
#
# The image checks permissions: the directory and the script must be owned by
# root and must not be writable by "others", otherwise nothing is executed.

set -uo pipefail

SOURCE="/opt/plugins/ollama-ocr-plugin"

if [ ! -d "$SOURCE" ]; then
    echo "[ollama-ocr] $SOURCE not found - skipped."
    exit 0
fi

# Always install: pip is idempotent and notices on its own whether anything
# changed. An import check would leave an old version in place when the plugin
# code was updated.
echo "[ollama-ocr] installing plugin from $SOURCE ..."
if ! pip install --no-cache-dir --no-deps --force-reinstall --quiet "$SOURCE"; then
    echo "[ollama-ocr] ERROR during installation - Paperless continues without the plugin."
    exit 0
fi
echo "[ollama-ocr] Installation successful."

# Verification: the plugin inherits from a core class. If Paperless renames or
# moves it, the import fails and Paperless would silently fall back to
# Tesseract. This makes it visible.
if (cd /usr/src/paperless/src && python3 -c "from paperless_ollama_ocr.parser import OllamaOcrParser") 2>/tmp/ollama-import.err; then
    echo "[ollama-ocr] OK - parser loads."
else
    echo "[ollama-ocr] ####################################################"
    echo "[ollama-ocr] WARNING: parser NOT loadable - the Paperless version"
    echo "[ollama-ocr] probably changed. OCR now runs on Tesseract again"
    echo "[ollama-ocr] (lower quality)."
    echo "[ollama-ocr] Error:"
    sed "s/^/[ollama-ocr]   /" /tmp/ollama-import.err | tail -5
    echo "[ollama-ocr] ####################################################"
fi

# The endpoint is not reachable from inside the container more often than
# not: "localhost" in docker-compose means the container itself, and a model
# that was never pulled fails on the first document. Report both up front.
endpoint="${PAPERLESS_OLLAMA_OCR_ENDPOINT:-http://localhost:11434}"
endpoint="${endpoint%/}"
model="${PAPERLESS_OLLAMA_OCR_MODEL:-deepseek-ocr}"

if python3 - "$endpoint" "$model" <<'PY'
import json, sys, urllib.error, urllib.request

endpoint, model = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=5) as response:
        names = [m.get("name") or m.get("model") for m in json.load(response).get("models", [])]
except (urllib.error.URLError, OSError, ValueError) as error:
    print(f"[ollama-ocr] endpoint {endpoint} NOT reachable ({error}).")
    print("[ollama-ocr] -> OCR falls back to Tesseract until it is.")
    sys.exit(0)

print(f"[ollama-ocr] endpoint {endpoint} reachable, {len(names)} model(s).")
if not any(name == model or (name or "").split(":")[0] == model.split(":")[0] for name in names):
    print(f"[ollama-ocr] model '{model}' NOT pulled -> run: ollama pull {model}")
PY
then :; fi
