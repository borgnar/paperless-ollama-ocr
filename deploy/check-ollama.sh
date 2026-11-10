#!/bin/bash
# Checks whether the Ollama OCR plugin is active.
# Usage: ./deploy/check-ollama.sh
#        HOST=your-server ./deploy/check-ollama.sh      # over SSH
# Exit code 0 = everything fine, 1 = plugin inactive (Paperless uses Tesseract).
#
# Environment: CONTAINER (default paperless-webserver-1),
#              HOST (default empty = docker runs locally).

set -uo pipefail

CONTAINER="${CONTAINER:-paperless-webserver-1}"
HOST="${HOST:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
errors=0

echo "=== Ollama OCR status $(date '+%F %T') ==="

if [ -n "$HOST" ]; then
    run() { ssh "$HOST" "$1"; }
    run_stdin() { ssh "$HOST" "$1"; }
else
    run() { bash -c "$1"; }
    run_stdin() { bash -c "$1"; }
fi

if ! run "docker ps --filter 'name=$CONTAINER' --format '{{.Names}}'" | grep -q .; then
    echo "  ERROR: container $CONTAINER is not running."
    exit 1
fi

version=$(run "docker exec $CONTAINER pip show paperless-ollama-ocr 2>/dev/null" \
    | awk '/^Version:/{print $2}')
if [ -n "$version" ]; then
    echo "  Plugin installed:  yes (v$version)"
else
    echo "  Plugin installed:  NO"
    errors=1
fi

if run "docker exec -w /usr/src/paperless/src $CONTAINER python3 -c 'from paperless_ollama_ocr.parser import OllamaOcrParser'" 2>/dev/null; then
    echo "  Parser loads:      yes"
else
    echo "  Parser loads:      NO   <-- probably broken by a Paperless update"
    errors=1
fi

endpoint=$(run "docker exec $CONTAINER printenv PAPERLESS_OLLAMA_OCR_ENDPOINT" 2>/dev/null)
endpoint="${endpoint:-http://localhost:11434}"
model=$(run "docker exec $CONTAINER printenv PAPERLESS_OLLAMA_OCR_MODEL" 2>/dev/null)
model="${model:-deepseek-ocr}"
echo "  Endpoint:          $endpoint"
echo "  Model:             $model"

# Probe from inside the container: that is where the request comes from, and
# "localhost" there is not the host.
if run_stdin "docker exec -i $CONTAINER python3 - $endpoint $model" < "$HERE/probe_ollama.py"; then
    :
else
    echo "  Endpoint answers:  NO   <-- the container cannot reach Ollama"
    errors=1
fi

handled=$(run "docker logs $CONTAINER 2>&1" | grep -c "third-party parser 'Ollama OCR Parser'")
echo "  Documents handled (since container start): $handled"

if [ "$errors" -eq 0 ]; then
    echo "  RESULT: Ollama OCR is active."
else
    echo "  RESULT: Ollama OCR is INACTIVE - Paperless uses Tesseract."
    echo "          Setup: see /opt/paperless-ngx/plugins/ollama-ocr-plugin/README.md"
fi
exit $errors
