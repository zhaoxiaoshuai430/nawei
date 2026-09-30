"""Application service coordinating catalog, Qdrant, Neo4j, and caching."""

# GraphRAG 业务核心：文档入库将文本与关系写到不同存储，检索时再组合成可引用证据。
# catalog(PostgreSQL) 跟踪文档状态，cache(Redis) 缓存检索结果，Qdrant 存向量与片段，Neo4j 存图谱。
# 阅读顺序：initialize -> ingest/ingest_structured -> retrieve -> delete_document。
# 所有索引操作都受 knowledge_base_id 约束；这里的公共知识与 LLM 用户记忆分属不同业务边界。

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from typing import Any

from services.knowledge.cache import KnowledgeCache
from services.knowledge.catalog import KnowledgeCatalog
from services.knowledge.chunker import chunk_text
from services.knowledge.config import KnowledgeSettings
from services.knowledge.embeddings import EmbeddingProvider, FastEmbedProvider, build_embedding_provider
from services.knowledge.entity_resolver import GraphEntityResolver
from services.knowledge.expansion_planner import KnowledgeExpansionPlanner
from services.knowledge.extractor import GraphExtractor, build_graph_extractor
from services.knowledge.grounding_assembler import GroundingAssembler
from services.knowledge.neo4j_store import Neo4jKnowledgeStore
from services.knowledge.qdrant_store import QdrantKnowledgeStore
from services.knowledge.schemas import (
    KnowledgeContext,
    KnowledgeDocumentRequest,
    KnowledgeDocumentResponse,
    KnowledgeRetrieveRequest,
    GraphCandidate,
    KnowledgeStructuredDocumentRequest,
)


_RETRIEVAL_CACHE_VERSION = "path-query-v5"


class KnowledgeService:
    def __init__(self, settings: KnowledgeSettings) -> None:
        self.settings = settings
        self.catalog = KnowledgeCatalog(settings.database_url)
        self.cache = KnowledgeCache(settings.redis_url, settings.cache_ttl_seconds)
        self.embeddings: EmbeddingProvider = build_embedding_provider(settings)
        self.extractor: GraphExtractor = build_graph_extractor(settings)
        self.qdrant: QdrantKnowledgeStore | None = None
        self.neo4j: Neo4jKnowledgeStore | None = None
        self.last_error: str | None = None
        self.startup_warmup: dict[str, Any] = {"status": "pending"}

    async def initialize(self) -> None:
        # FastEmbed 在就绪前预热，避免第一条正式查询承担模型首次加载开销。
        # 向量库初始化会检查模型版本/维度，不能让旧索引与新模型悄悄混用。
        await self.catalog.initialize()
        await self.cache.initialize()
        if not self.settings.enabled:
            return
        try:
            if isinstance(self.embeddings, FastEmbedProvider):
                await self.embeddings.embed(["知识检索预热"])
            self.qdrant = QdrantKnowledgeStore(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                collection=self.settings.qdrant_collection,
                dimension=self.settings.embedding_dimension,
                embedding_version=self.embeddings.model_id,
            )
            self.neo4j = Neo4jKnowledgeStore(
                uri=self.settings.neo4j_uri,
                user=self.settings.neo4j_user,
                password=self.settings.neo4j_password,
                database=self.settings.neo4j_database,
            )
            await self.qdrant.initialize()
            await self.neo4j.initialize()
            self.last_error = None
            await self._warm_retrieval()
        except Exception as exc:
            self.last_error = str(exc)
            if self.settings.required:
                raise RuntimeError(f"Knowledge dependencies are unavailable: {exc}") from exc

    async def _warm_retrieval(self) -> None:
        if os.getenv("KNOWLEDGE_STARTUP_WARMUP_ENABLED", "true").strip().lower() in {"false", "0", "off", "no"}:
            self.startup_warmup = {"status": "disabled"}
            return
        started = time.perf_counter()
        timeout = float(os.getenv("KNOWLEDGE_STARTUP_WARMUP_TIMEOUT_SECONDS", "60"))
        if not 0 < timeout <= 180:
            raise ValueError("KNOWLEDGE_STARTUP_WARMUP_TIMEOUT_SECONDS must be in (0, 180]")
        try:
            async with asyncio.timeout(timeout):
                # A unique query avoids a Redis hit hiding cold vector/graph connections.
                context = await self.retrieve(KnowledgeRetrieveRequest(
                    knowledge_base_id=self.settings.default_base_id,
                    query=f"知识检索预热 {uuid.uuid4().hex}",
                    top_k=8, graph_depth=2,
                    graph_candidates=[GraphCandidate(name="知识检索预热", entity_types=["Concept"], priority=1)],
                ))
            self.startup_warmup = {
                "status": "ready", "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "timings_ms": context.metadata.get("timings_ms", {}),
            }
        except Exception as exc:
            self.startup_warmup = {"status": "failed", "error_type": type(exc).__name__}
            raise RuntimeError(f"Knowledge retrieval warmup failed: {type(exc).__name__}") from exc

    async def close(self) -> None:
        await self.catalog.close()
        await self.cache.close()
        await self.embeddings.close()
        await self.extractor.close()
        if self.qdrant is not None:
            await self.qdrant.close()
        if self.neo4j is not None:
            await self.neo4j.close()

    async def ingest(self, request: KnowledgeDocumentRequest) -> KnowledgeDocumentResponse:
        # 同一知识库按内容哈希检测已完成的重复导入；文档及片段使用可重复生成的 UUID。
        # 流程为切块 -> 向量化 -> 抽取实体关系 -> 写双库 -> 标记就绪 -> 失效检索缓存。
        qdrant, neo4j = self._require_stores()
        content_hash = hashlib.sha256(request.content.strip().encode("utf-8")).hexdigest()
        duplicate = await self.catalog.find_ready_by_hash(
            knowledge_base_id=request.knowledge_base_id,
            content_hash=content_hash,
        )
        if duplicate:
            return KnowledgeDocumentResponse(
                document_id=str(duplicate["document_id"]),
                knowledge_base_id=request.knowledge_base_id,
                status="ready",
                chunk_count=int(duplicate.get("chunk_count") or 0),
                entity_count=int(duplicate.get("entity_count") or 0),
                relation_count=int(duplicate.get("relation_count") or 0),
                duplicate=True,
            )

        document_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"nawei:{request.knowledge_base_id}:{content_hash}")
        )
        await self.catalog.start_document(
            document_id=document_id,
            knowledge_base_id=request.knowledge_base_id,
            title=request.title,
            source_name=request.source_name,
            content_hash=content_hash,
            metadata=request.metadata,
        )
        try:
            texts = chunk_text(
                request.content,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            if not texts:
                raise ValueError("Document did not contain indexable text.")
            vectors = await self.embeddings.embed(texts)
            chunks: list[dict[str, Any]] = []
            entities_by_id: dict[str, dict[str, Any]] = {}
            mentions: list[dict[str, str]] = []
            relations: list[dict[str, Any]] = []
            for index, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
                # 同一个 chunk_id 同时用于向量点和图谱 Chunk，检索命中可直接作为图遍历起点。
                chunk_id = str(uuid.uuid5(uuid.UUID(document_id), f"chunk:{index}"))
                extraction = await self.extractor.extract(
                    title=request.title,
                    text=text,
                    chunk_id=chunk_id,
                )
                name_to_id: dict[str, str] = {}
                for entity in extraction.entities:
                    canonical_name = _canonical_name(entity.name)
                    entity_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"nawei:{request.knowledge_base_id}:entity:{entity.entity_type}:{canonical_name}",
                        )
                    )
                    name_to_id[entity.name] = entity_id
                    entities_by_id[entity_id] = {
                        "entity_id": entity_id,
                        "name": entity.name,
                        "canonical_name": canonical_name,
                        "entity_type": entity.entity_type,
                        "description": entity.description,
                        "aliases": [],
                        "discipline_level": None,
                    }
                    mentions.append({"chunk_id": chunk_id, "entity_id": entity_id})
                for relation in extraction.relations:
                    source_id = name_to_id.get(relation.source)
                    target_id = name_to_id.get(relation.target)
                    if not source_id or not target_id or source_id == target_id:
                        continue
                    relation_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{source_id}:{relation.relation_type}:{target_id}:{chunk_id}",
                        )
                    )
                    relations.append(
                        {
                            "relation_id": relation_id,
                            "source_id": source_id,
                            "target_id": target_id,
                            "relation_type": _relation_type(relation.relation_type),
                            "confidence": relation.confidence,
                            "evidence": relation.evidence,
                            "evidence_chunk_id": chunk_id,
                        }
                    )
                entity_ids = sorted(set(name_to_id.values()))
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": index,
                        "title": request.title,
                        "source_name": request.source_name,
                        "qdrant_collection": self.settings.qdrant_collection,
                        "embedding": vector,
                        "payload": {
                            "knowledge_base_id": request.knowledge_base_id,
                            "document_id": document_id,
                            "chunk_id": chunk_id,
                            "chunk_index": index,
                            "title": request.title,
                            "text": text,
                            "source_name": request.source_name,
                            "page": None,
                            "entity_ids": entity_ids,
                            "embedding_version": self.embeddings.model_id,
                            "content_hash": content_hash,
                            "status": "indexing",
                        },
                    }
                )

            # 先以 indexing 状态写入，再完成图谱写入后置为 ready，避免正常检索使用半成品。
            # 这不是跨数据库事务；失败时依靠补偿清理，见 _rollback_document。
            await qdrant.upsert_chunks(chunks)
            await neo4j.upsert_document(
                knowledge_base_id=request.knowledge_base_id,
                document={
                    "document_id": document_id,
                    "title": request.title,
                    "source_name": request.source_name,
                    "content_hash": content_hash,
                },
                chunks=chunks,
                entities=list(entities_by_id.values()),
                mentions=_dedupe_dicts(mentions, key_fields=("chunk_id", "entity_id")),
                relations=_dedupe_dicts(relations, key_fields=("relation_id",)),
            )
            await qdrant.mark_document_ready(document_id)
            await self.catalog.complete_document(
                document_id=document_id,
                chunk_count=len(chunks),
                entity_count=len(entities_by_id),
                relation_count=len(relations),
            )
            await self.cache.invalidate_base(request.knowledge_base_id)
            return KnowledgeDocumentResponse(
                document_id=document_id,
                knowledge_base_id=request.knowledge_base_id,
                status="ready",
                chunk_count=len(chunks),
                entity_count=len(entities_by_id),
                relation_count=len(relations),
            )
        except Exception as exc:
            await self._rollback_document(document_id, qdrant=qdrant, neo4j=neo4j, error=exc)
            raise

    async def ingest_structured(
        self,
        request: KnowledgeStructuredDocumentRequest,
    ) -> KnowledgeDocumentResponse:
        qdrant, neo4j = self._require_stores()
        serialized = json.dumps(_model_to_dict(request), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        duplicate = await self.catalog.find_ready_by_hash(
            knowledge_base_id=request.knowledge_base_id,
            content_hash=content_hash,
        )
        if duplicate:
            return KnowledgeDocumentResponse(
                document_id=str(duplicate["document_id"]),
                knowledge_base_id=request.knowledge_base_id,
                status="ready",
                chunk_count=int(duplicate.get("chunk_count") or 0),
                entity_count=int(duplicate.get("entity_count") or 0),
                relation_count=int(duplicate.get("relation_count") or 0),
                duplicate=True,
            )

        document_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"nawei:{request.knowledge_base_id}:{content_hash}")
        )
        await self.catalog.start_document(
            document_id=document_id,
            knowledge_base_id=request.knowledge_base_id,
            title=request.title,
            source_name=request.source_name,
            content_hash=content_hash,
            metadata=request.metadata,
        )
        try:
            texts = [chunk.text.strip() for chunk in request.chunks]
            if any(not text for text in texts):
                raise ValueError("Structured chunks must contain non-empty text.")
            vectors = await self._embed_in_batches(texts)
            chunks: list[dict[str, Any]] = []
            entities_by_id: dict[str, dict[str, Any]] = {}
            mentions: list[dict[str, str]] = []
            relations: list[dict[str, Any]] = []

            for index, (structured, text, vector) in enumerate(zip(request.chunks, texts, vectors, strict=True)):
                chunk_id = str(uuid.uuid5(uuid.UUID(document_id), f"chunk:{index}:{structured.chunk_key}"))
                entity_lookup: dict[tuple[str, str], str] = {}
                chunk_entity_ids: set[str] = set()

                def ensure_entity(
                    name: str,
                    entity_type: str,
                    description: str = "",
                    aliases: list[str] | None = None,
                    discipline_level: int | None = None,
                ) -> str:
                    normalized_name = name.strip()
                    normalized_type = entity_type.strip() or "Concept"
                    lookup_key = (normalized_name, normalized_type)
                    existing = entity_lookup.get(lookup_key)
                    if existing:
                        return existing
                    canonical_name = _canonical_name(normalized_name)
                    entity_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"nawei:{request.knowledge_base_id}:entity:{normalized_type}:{canonical_name}",
                        )
                    )
                    entity_lookup[lookup_key] = entity_id
                    current = entities_by_id.get(entity_id)
                    if current is None or (not current.get("description") and description):
                        entities_by_id[entity_id] = {
                            "entity_id": entity_id,
                            "name": normalized_name,
                            "canonical_name": canonical_name,
                            "entity_type": normalized_type,
                            "description": description[:500],
                            "aliases": _normalize_aliases(aliases or []),
                            "discipline_level": discipline_level,
                        }
                    elif current is not None:
                        current["aliases"] = _normalize_aliases(
                            [*current.get("aliases", []), *(aliases or [])]
                        )
                        if current.get("discipline_level") is None and discipline_level is not None:
                            current["discipline_level"] = discipline_level
                    return entity_id

                for entity in structured.entities:
                    entity_id = ensure_entity(
                        entity.name,
                        entity.entity_type,
                        entity.description,
                        entity.aliases,
                        entity.discipline_level,
                    )
                    chunk_entity_ids.add(entity_id)
                    mentions.append({"chunk_id": chunk_id, "entity_id": entity_id})

                for relation in structured.relations:
                    source_id = ensure_entity(relation.source, relation.source_type)
                    target_id = ensure_entity(relation.target, relation.target_type)
                    chunk_entity_ids.update((source_id, target_id))
                    mentions.extend(
                        (
                            {"chunk_id": chunk_id, "entity_id": source_id},
                            {"chunk_id": chunk_id, "entity_id": target_id},
                        )
                    )
                    relation_type = _relation_type(relation.relation_type)
                    relation_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{source_id}:{relation_type}:{target_id}:{chunk_id}",
                        )
                    )
                    relations.append(
                        {
                            "relation_id": relation_id,
                            "source_id": source_id,
                            "target_id": target_id,
                            "relation_type": relation_type,
                            "confidence": relation.confidence,
                            "evidence": (relation.evidence or text)[:1000],
                            "evidence_chunk_id": chunk_id,
                        }
                    )

                entity_ids = sorted(chunk_entity_ids)
                source_name = structured.source_name or request.source_name
                page = structured.metadata.get("page")
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "chunk_index": index,
                        "title": request.title,
                        "source_name": source_name,
                        "source_url": structured.source_url,
                        "source_locator": structured.source_locator,
                        "qdrant_collection": self.settings.qdrant_collection,
                        "embedding": vector,
                        "payload": {
                            "knowledge_base_id": request.knowledge_base_id,
                            "document_id": document_id,
                            "chunk_id": chunk_id,
                            "chunk_index": index,
                            "chunk_key": structured.chunk_key,
                            "title": request.title,
                            "text": text,
                            "source_name": source_name,
                            "source_url": structured.source_url,
                            "source_locator": structured.source_locator,
                            "page": int(page) if page is not None else None,
                            "entity_ids": entity_ids,
                            "metadata": structured.metadata,
                            **_flatten_knowledge_metadata(structured.metadata),
                            "embedding_version": self.embeddings.model_id,
                            "content_hash": content_hash,
                            "status": "indexing",
                        },
                    }
                )

            deduped_mentions = _dedupe_dicts(mentions, key_fields=("chunk_id", "entity_id"))
            deduped_relations = _dedupe_dicts(relations, key_fields=("relation_id",))
            await qdrant.upsert_chunks(chunks)
            await neo4j.upsert_document(
                knowledge_base_id=request.knowledge_base_id,
                document={
                    "document_id": document_id,
                    "title": request.title,
                    "source_name": request.source_name,
                    "content_hash": content_hash,
                },
                chunks=chunks,
                entities=list(entities_by_id.values()),
                mentions=deduped_mentions,
                relations=deduped_relations,
            )
            await qdrant.mark_document_ready(document_id)
            await self.catalog.complete_document(
                document_id=document_id,
                chunk_count=len(chunks),
                entity_count=len(entities_by_id),
                relation_count=len(deduped_relations),
            )
            await self.cache.invalidate_base(request.knowledge_base_id)
            return KnowledgeDocumentResponse(
                document_id=document_id,
                knowledge_base_id=request.knowledge_base_id,
                status="ready",
                chunk_count=len(chunks),
                entity_count=len(entities_by_id),
                relation_count=len(deduped_relations),
            )
        except Exception as exc:
            await self._rollback_document(document_id, qdrant=qdrant, neo4j=neo4j, error=exc)
            raise

    async def _rollback_document(
        self,
        document_id: str,
        *,
        qdrant: QdrantKnowledgeStore,
        neo4j: Neo4jKnowledgeStore,
        error: Exception,
    ) -> None:
        # 尽力标记失败并分别清理双库，不让清理异常覆盖原始入库错误。
        # 补偿本身也可能失败，因此失败状态和日志仍需保留供后续排查。
        try:
            await self.catalog.fail_document(document_id=document_id, error=str(error))
        except Exception:
            pass
        for delete_document in (qdrant.delete_document, neo4j.delete_document):
            try:
                await delete_document(document_id)
            except Exception:
                pass

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        batch_size = max(1, int(os.getenv("KNOWLEDGE_EMBEDDING_BATCH_SIZE", "64")))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(await self.embeddings.embed(texts[start : start + batch_size]))
        return vectors

    async def retrieve(self, request: KnowledgeRetrieveRequest) -> KnowledgeContext:
        # 主路径：缓存 -> 图谱候选解析 -> 问题向量化 -> 片段召回 -> 锚点校验 -> 图关系扩展。
        # 返回的是证据上下文，不是模型生成的答案；来源与匹配状态交给 LLM 组织引用。
        started = time.perf_counter()
        qdrant, neo4j = self._require_stores()
        anchor_terms = _normalize_anchor_terms(request.anchor_terms)
        # 缓存键包含索引、模型、维度与扩展策略，避免切换模型或检索逻辑后复用旧结果。
        route_context = {
            "collection": self.settings.qdrant_collection,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.embeddings.model_id,
            "embedding_dimension": self.embeddings.dimension,
            "retrieval_version": _RETRIEVAL_CACHE_VERSION,
            "graph_candidates": [item.model_dump() for item in request.graph_candidates],
            "expansion": request.expansion.model_dump(),
        }
        cached = await self.cache.get(
            knowledge_base_id=request.knowledge_base_id,
            query=request.query,
            top_k=request.top_k,
            graph_depth=request.graph_depth,
            anchor_terms=anchor_terms,
            route_context=route_context,
        )
        if cached is not None:
            cached.metadata["cache"] = "hit"
            cached.metadata["timings_ms"] = {"cache_lookup": round((time.perf_counter() - started) * 1000)}
            return cached

        timings = {"cache_lookup": round((time.perf_counter() - started) * 1000)}
        stage_started = time.perf_counter()
        resolver = GraphEntityResolver(neo4j)
        planner = KnowledgeExpansionPlanner()
        resolved_entities = await resolver.resolve(
            knowledge_base_id=request.knowledge_base_id,
            candidates=request.graph_candidates,
        )
        qdrant_filters = planner.qdrant_filters(resolved_entities)
        timings["entity_resolution"] = round((time.perf_counter() - stage_started) * 1000)
        stage_started = time.perf_counter()
        vector = (await self.embeddings.embed([request.query]))[0]
        timings["embedding"] = round((time.perf_counter() - stage_started) * 1000)
        stage_started = time.perf_counter()
        chunks = await qdrant.search(
            knowledge_base_id=request.knowledge_base_id,
            query=request.query,
            vector=vector,
            top_k=request.top_k,
            filters=qdrant_filters,
        )
        if qdrant_filters and not chunks:
            # 图谱限定范围无结果时放宽过滤，但仍限制在同一 knowledge_base_id 内。
            chunks = await qdrant.search(
                knowledge_base_id=request.knowledge_base_id,
                query=request.query,
                vector=vector,
                top_k=request.top_k,
                filters=None,
            )
        retrieved_chunk_count = len(chunks)
        timings["vector_and_lexical_search"] = round((time.perf_counter() - stage_started) * 1000)
        stage_started = time.perf_counter()
        if anchor_terms:
            # 片段必须实际提到锚点概念；仅命中宽泛学科节点，不等于有具体公式的文本证据。
            chunks = [chunk for chunk in chunks if _chunk_matches_anchor(chunk, anchor_terms)]
        relation_groups = (
            await neo4j.expand_entities(
                entity_ids=[item.entity_id for item in resolved_entities],
                relation_groups=planner.relation_groups(request.expansion),
                depth=request.graph_depth,
            )
            if resolved_entities
            else {}
        )
        chunk_relations = await neo4j.expand(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            depth=request.graph_depth,
        )
        if chunk_relations:
            relation_groups["chunk_relations"] = chunk_relations
        timings["graph_expansion"] = round((time.perf_counter() - stage_started) * 1000)
        context = GroundingAssembler().assemble(
            knowledge_base_id=request.knowledge_base_id,
            query=request.query,
            chunks=chunks,
            resolved_entities=resolved_entities,
            relation_groups=relation_groups,
            anchor_terms=anchor_terms,
            retrieved_chunk_count=retrieved_chunk_count,
            metadata={
                "cache": "miss",
                "timings_ms": timings,
                "embedding_model": self.embeddings.model_id,
                "qdrant_filters": qdrant_filters,
                "graph_candidate_count": len(request.graph_candidates),
            },
        )
        await self.cache.set(
            context,
            top_k=request.top_k,
            graph_depth=request.graph_depth,
            anchor_terms=anchor_terms,
            route_context=route_context,
        )
        return context

    async def delete_document(self, document_id: str) -> bool:
        # 删除需同时处理向量、图谱、目录及缓存；只删一处会留下可被检索到的旧证据。
        qdrant, neo4j = self._require_stores()
        document = await self.catalog.get_document(document_id)
        if document is None:
            return False
        await qdrant.delete_document(document_id)
        await neo4j.delete_document(document_id)
        await self.catalog.delete_document(document_id)
        await self.cache.invalidate_base(str(document["knowledge_base_id"]))
        return True

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        return await self.catalog.get_document(document_id)

    async def health(self) -> dict[str, Any]:
        qdrant_status = await self.qdrant.health() if self.qdrant is not None else "unavailable"
        neo4j_status = await self.neo4j.health() if self.neo4j is not None else "unavailable"
        catalog_status = await self.catalog.health()
        catalog_ready = catalog_status == "ok" if self.settings.required else catalog_status in {"ok", "fallback"}
        ready = not self.settings.enabled or (
            qdrant_status == "ok" and neo4j_status == "ok" and catalog_ready
        )
        return {
            "ready": ready,
            "enabled": self.settings.enabled,
            "qdrant": qdrant_status,
            "neo4j": neo4j_status,
            "postgres": catalog_status,
            "redis": "ok" if self.cache.redis is not None else "fallback",
            "embedding_model": self.embeddings.model_id,
            "startup_warmup": self.startup_warmup,
        }

    def _require_stores(self) -> tuple[QdrantKnowledgeStore, Neo4jKnowledgeStore]:
        if not self.settings.enabled:
            raise RuntimeError("Knowledge service is disabled.")
        if self.qdrant is None or self.neo4j is None:
            raise RuntimeError(self.last_error or "Knowledge stores are unavailable.")
        return self.qdrant, self.neo4j


def _canonical_name(value: str) -> str:
    return "".join(value.strip().lower().split())[:120]


def _normalize_anchor_terms(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = " ".join(str(value).strip().lower().split())[:120]
        if not term or term in seen:
            continue
        seen.add(term)
        result.append(term)
    return result[:20]


def _normalize_aliases(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = " ".join(str(value).strip().split())[:120]
        key = alias.lower()
        if not alias or key in seen:
            continue
        seen.add(key)
        result.append(alias)
    return result[:20]


def _flatten_knowledge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "subject",
        "domain",
        "concept",
        "aliases",
        "content_dimension",
        "content_type",
        "grade",
        "difficulty",
    )
    return {field: metadata[field] for field in fields if metadata.get(field) is not None}


def _chunk_matches_anchor(chunk: Any, anchor_terms: list[str]) -> bool:
    text = f"{chunk.title}\n{chunk.text}".lower()
    compact_text = "".join(text.split())
    for term in anchor_terms:
        compact_term = "".join(term.split())
        if not compact_term:
            continue
        if compact_term.isascii() and compact_term.replace("_", "").isalnum():
            if re.search(rf"(?<![a-z0-9_]){re.escape(compact_term)}(?![a-z0-9_])", text):
                return True
        elif compact_term in compact_text:
            return True
    return False


def _relation_type(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value.upper())
    return normalized.strip("_")[:60] or "RELATED_TO"


def _dedupe_dicts(items: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        key = tuple(item[field] for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()
