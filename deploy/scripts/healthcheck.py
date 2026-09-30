"""Check service readiness, including model and storage dependencies."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any

URLS = {
    "gateway": "http://127.0.0.1:8000/api/v1/status",
    "asr": "http://127.0.0.1:8001/health",
    "llm": "http://127.0.0.1:8002/health",
    "knowledge": "http://127.0.0.1:8003/health",
    "tts": "http://127.0.0.1:19000/api/v1/status",
}


def is_ready(service: str, body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    if service == "tts":
        return body.get("loaded") is True
    if service == "gateway":
        dependencies = body.get("services", {})
        return body.get("status") == "ready" and isinstance(dependencies, dict) and all(
            isinstance(dependencies.get(name), dict) and dependencies[name].get("ready") is True
            for name in ("asr", "llm", "tts")
        )
    if body.get("status") != "ok":
        return False
    if service == "knowledge":
        return body.get("ready") is True and all(body.get(name) == "ok" for name in ("postgres", "redis", "qdrant", "neo4j"))
    if service == "llm":
        runtime = body.get("runtime", {})
        if not isinstance(runtime, dict):
            return False
        memory = runtime.get("memory", {})
        knowledge = runtime.get("knowledge", {})
        return (
            body.get("provider") == "deepseek" and isinstance(memory, dict) and isinstance(knowledge, dict)
            and all(memory.get(name) == "ok" for name in ("postgres", "redis"))
            and is_ready("knowledge", knowledge.get("detail"))
        )
    return service == "asr"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=URLS)
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(URLS[args.service], timeout=15 if args.service == "gateway" else 8) as response:
            body = json.load(response)
        ready = isinstance(body, dict) and is_ready(args.service, body)
    except Exception:
        ready = False
    if not ready:
        raise SystemExit(f"{args.service}: not ready")
    print(f"{args.service}: ready")


if __name__ == "__main__":
    main()
