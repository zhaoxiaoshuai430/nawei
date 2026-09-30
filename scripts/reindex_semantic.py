"""Build a separate semantic index from existing ready payloads; never replace the source."""

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import AsyncQdrantClient, models
from services.knowledge.config import load_settings
from services.knowledge.embeddings import FastEmbedProvider, HashEmbeddingProvider
from services.knowledge.qdrant_store import QdrantKnowledgeStore


async def run(args):
    settings = load_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60)
    try:
        source = args.source or settings.qdrant_collection
        source_info = await client.get_collection(source)
        points, offset = [], None
        while True:
            batch, offset = await client.scroll(source, limit=256, offset=offset, with_payload=True, with_vectors=False)
            points.extend(batch)
            if offset is None:
                break
        overview = {"source": source, "points": len(points), "titles": sorted({point.payload.get("title", "") for point in points})}
        if args.inspect:
            print(json.dumps(overview, ensure_ascii=True), flush=True)
            return
        assert args.target and source != args.target, "A separate target collection is required"
        assert not await client.collection_exists(args.target), "Target already exists; use a new name"
        assert points and all(point.payload.get("status") == "ready" for point in points), "Source must contain only ready documents"
        assert all(point.payload.get("embedding_version") == "hash-embedding-v1" for point in points), "This evaluator expects a hash-embedding-v1 baseline"
        semantic = FastEmbedProvider(replace(settings, embedding_provider="fastembed", embedding_model="BAAI/bge-small-zh-v1.5", embedding_dimension=512))
        target = QdrantKnowledgeStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key, collection=args.target, dimension=512)
        args.output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            await semantic.embed(["知识检索预热"])
            await target.initialize()
            for start in range(0, len(points), 32):
                batch = points[start:start + 32]
                vectors = await semantic.embed([point.payload["text"] for point in batch])
                await client.upsert(args.target, points=[
                    models.PointStruct(id=point.id, vector=vector, payload={**point.payload, "embedding_version": semantic.model_id})
                    for point, vector in zip(batch, vectors, strict=True)
                ], wait=True)
                print(json.dumps({"indexed": min(start + 32, len(points)), "total": len(points)}), flush=True)
            assert (await client.count(source)).count == len(points), "Source changed during migration; do not activate target"
            assert (await client.count(args.target)).count == len(points)
            report = {**overview, "target": args.target, "model": semantic.model_id,
                      "build_seconds": round(time.perf_counter() - started, 2), "activated": False,
                      "source_ids_sha256": hashlib.sha256("\n".join(sorted(str(point.id) for point in points)).encode()).hexdigest(),
                      "queries": []}
            baseline = HashEmbeddingProvider(dimension=source_info.config.params.vectors.size)
            cases = json.loads(args.cases.read_text(encoding="utf-8"))
            for case in cases:
                result = {**case, "results": {}}
                for label, collection, provider in [("baseline", source, baseline), ("semantic", args.target, semantic)]:
                    query_started = time.perf_counter()
                    vector = (await provider.embed([case["query"]]))[0]
                    hits = await client.query_points(collection, query=vector, query_filter=models.Filter(must=[
                        models.FieldCondition(key="status", match=models.MatchValue(value="ready")),
                        models.FieldCondition(key="knowledge_base_id", match=models.MatchValue(value=case.get("knowledge_base_id", "default"))),
                    ]), limit=5, with_payload=True)
                    ranked = [{"id": str(hit.id), "score": hit.score, "text": hit.payload["text"], "title": hit.payload.get("title")} for hit in hits.points]
                    relevant = [index + 1 for index, hit in enumerate(ranked)
                                if any(term in hit["text"] for term in case["expected_terms"])]
                    result["results"][label] = {"first_relevant_rank": min(relevant) if relevant else None,
                                               "elapsed_ms": round((time.perf_counter() - query_started) * 1000), "hits": ranked}
                report["queries"].append(result)
                (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            answerable = [case for case in report["queries"] if case["expected_terms"]]
            report["summary"] = {label: {
                "hit_rate_at_5": sum(bool(case["results"][label]["first_relevant_rank"]) for case in answerable) / len(answerable),
                "mrr_at_5": sum(1 / case["results"][label]["first_relevant_rank"] if case["results"][label]["first_relevant_rank"] else 0 for case in answerable) / len(answerable),
            } for label in ["baseline", "semantic"]}
            report["evaluation_scope"] = "Curated concept-term smoke labels, vector-only retrieval, not an exhaustive relevance benchmark"
            (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report["summary"]), flush=True)
        finally:
            await semantic.close()
            await target.close()
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/semantic-evaluation")
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/fixtures/semantic_queries.json")
    asyncio.run(run(parser.parse_args()))
