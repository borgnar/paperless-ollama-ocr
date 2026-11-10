"""Probe a configured Ollama endpoint from inside the webserver container.

Read from stdin by check-ollama.sh:
    docker exec -i paperless-webserver-1 python3 - URL MODEL < deploy/probe_ollama.py
"""

import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    endpoint = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434").rstrip(
        "/"
    )
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ocr"
    try:
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=5) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as error:
        print(f"  Endpoint answers:  NO   ({error})")
        print("  -> run from inside this container: curl " + endpoint + "/api/tags")
        return 1

    names = [
        entry.get("name") or entry.get("model")
        for entry in payload.get("models", [])
        if entry
    ]
    print(f"  Endpoint answers:  yes ({len(names)} model(s))")
    base = model.split(":")[0]
    if any((name or "").split(":")[0] == base for name in names):
        print("  Model pulled:      yes")
        return 0
    print(f"  Model pulled:      NO   <-- ollama pull {model}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
