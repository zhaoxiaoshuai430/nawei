from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.knowledge.schemas import KnowledgeStructuredDocumentRequest


DEFAULT_FILE = PROJECT_ROOT / "knowledge_data" / "math" / "trigonometry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import one structured Knowledge GraphRAG document.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--api-url", default="http://127.0.0.1:8003")
    parser.add_argument(
        "--admin-api-key",
        default=os.getenv("KNOWLEDGE_ADMIN_API_KEY", ""),
    )
    parser.add_argument("--knowledge-base-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.file.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if args.knowledge_base_id:
        payload["knowledge_base_id"] = args.knowledge_base_id
    document = KnowledgeStructuredDocumentRequest(**payload)
    body = document.model_dump(mode="json")
    summary = {
        "file": str(source_path),
        "knowledge_base_id": document.knowledge_base_id,
        "title": document.title,
        "chunk_count": len(document.chunks),
        "entity_mentions": sum(len(chunk.entities) for chunk in document.chunks),
        "relation_count": sum(len(chunk.relations) for chunk in document.chunks),
    }
    if args.dry_run:
        print(json.dumps({"status": "dry-run", **summary}, ensure_ascii=False, indent=2))
        return

    headers = {"X-Admin-API-Key": args.admin_api_key} if args.admin_api_key else {}
    endpoint = args.api_url.rstrip("/") + "/api/v1/knowledge/documents/structured"
    with httpx.Client(timeout=300.0, trust_env=False) as client:
        response = client.post(endpoint, json=body, headers=headers)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
    print(json.dumps({"status": "completed", **summary, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
