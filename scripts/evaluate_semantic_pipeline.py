"""Compare existing indexes through the real hybrid retrieval and graph pipeline."""

import argparse
import asyncio
from dataclasses import replace
import json
import sys
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.knowledge.config import load_settings
from services.knowledge.schemas import KnowledgeRetrieveRequest
from services.knowledge.service import KnowledgeService


async def evaluate(args):
    settings = load_settings()
    if args.source == args.target:
        raise ValueError("Source and target must be separate existing indexes")
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    try:
        for collection in (args.source, args.target):
            if not await client.collection_exists(collection):
                raise ValueError(f"Evaluation requires an existing collection: {collection}")
    finally:
        await client.close()
    report = {"scope": "curated concept-term smoke labels, hybrid+graph", "cases": []}
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "tests/fixtures/semantic_queries.json").read_text(encoding="utf-8"))
    for label, config in [
        ("baseline", replace(settings, qdrant_collection=args.source, embedding_provider="hash", embedding_model="hash-embedding-v1", embedding_dimension=128)),
        ("semantic", replace(settings, qdrant_collection=args.target, embedding_provider="fastembed", embedding_model="BAAI/bge-small-zh-v1.5", embedding_dimension=512)),
    ]:
        service = KnowledgeService(config)
        try:
            await service.initialize()
            if service.last_error:
                raise RuntimeError(service.last_error)
            for index, case in enumerate(cases):
                if len(report["cases"]) <= index:
                    report["cases"].append({**case, "results": {}})
                started = time.perf_counter()
                context = await service.retrieve(KnowledgeRetrieveRequest(query=case["query"], top_k=5))
                relevant = [rank + 1 for rank, chunk in enumerate(context.chunks)
                            if any(term in chunk.text for term in case["expected_terms"])]
                report["cases"][index]["results"][label] = {
                    "first_relevant_rank": min(relevant) if relevant else None,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000), "context": context.model_dump(),
                }
                (args.output / "pipeline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            await service.close()
    answerable = [case for case in report["cases"] if case["expected_terms"]]
    report["summary"] = {label: {
        "hit_rate_at_5": sum(bool(case["results"][label]["first_relevant_rank"]) for case in answerable) / len(answerable),
        "mrr_at_5": sum(1 / case["results"][label]["first_relevant_rank"] if case["results"][label]["first_relevant_rank"] else 0 for case in answerable) / len(answerable),
        "max_ms": max(case["results"][label]["elapsed_ms"] for case in report["cases"]),
    } for label in ["baseline", "semantic"]}
    report["no_positive_regressions"] = all(
        not case["results"]["baseline"]["first_relevant_rank"] or case["results"]["semantic"]["first_relevant_rank"]
        for case in answerable
    )
    (args.output / "pipeline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "no_positive_regressions": report["no_positive_regressions"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="knowledge_chunks_v1")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/semantic-evaluation")
    asyncio.run(evaluate(parser.parse_args()))
