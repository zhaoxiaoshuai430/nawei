"""Qdrant persistence for knowledge chunks and embeddings."""

from __future__ import annotations

import re
from typing import Any

from services.knowledge.schemas import KnowledgeChunkHit

try:
    from qdrant_client import AsyncQdrantClient, models
except ImportError:  # pragma: no cover - dependency validated by live health
    AsyncQdrantClient = None  # type: ignore[misc,assignment]
    models = None  # type: ignore[misc,assignment]


class QdrantKnowledgeStore:
    def __init__(self, *, url: str, api_key: str, collection: str, dimension: int, embedding_version: str | None = None) -> None:
        if AsyncQdrantClient is None or models is None:
            raise RuntimeError("qdrant-client is not installed.")
        self.collection = collection
        self.dimension = dimension
        self.embedding_version = embedding_version
        self.client = AsyncQdrantClient(url=url, api_key=api_key or None, timeout=30)

    async def initialize(self) -> None:
        if await self.client.collection_exists(self.collection):
            info = await self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if not isinstance(vectors, models.VectorParams) or vectors.size != self.dimension or vectors.distance != models.Distance.COSINE:
                raise ValueError("Existing Qdrant collection is incompatible; create a separate embedding index")
            if self.embedding_version:
                mismatched = await self.client.count(self.collection, count_filter=models.Filter(must_not=[
                    models.FieldCondition(key="embedding_version", match=models.MatchValue(value=self.embedding_version)),
                ]), exact=True)
                if mismatched.count:
                    raise ValueError("Existing Qdrant collection contains a different embedding version")
        else:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
            )
        for field_name in (
            "knowledge_base_id",
            "document_id",
            "status",
            "embedding_version",
            "subject",
            "domain",
            "concept",
            "content_dimension",
            "content_type",
            "entity_ids",
        ):
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise
        try:
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    min_token_len=1,
                    max_token_len=30,
                    lowercase=True,
                ),
                wait=True,
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise

    async def close(self) -> None:
        await self.client.close()

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        for start in range(0, len(chunks), 256):
            batch = chunks[start : start + 256]
            points = [
                models.PointStruct(id=item["chunk_id"], vector=item["embedding"], payload=item["payload"])
                for item in batch
            ]
            await self.client.upsert(collection_name=self.collection, points=points, wait=True)

    async def mark_document_ready(self, document_id: str) -> None:
        await self.client.set_payload(
            collection_name=self.collection,
            payload={"status": "ready"},
            points=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            ),
            wait=True,
        )

    async def search(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[KnowledgeChunkHit]:
        payload_conditions = _payload_filter_conditions(filters or {})
        response = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="knowledge_base_id",
                        match=models.MatchValue(value=knowledge_base_id),
                    ),
                    models.FieldCondition(key="status", match=models.MatchValue(value="ready")),
                    *payload_conditions,
                ]
            ),
            limit=min(max(top_k * 3, top_k), 90),
            with_payload=True,
        )
        hits_by_id: dict[str, KnowledgeChunkHit] = {}
        for point in response.points:
            payload = point.payload or {}
            hit = _payload_hit(point.id, payload, score=float(point.score or 0.0))
            hits_by_id[hit.chunk_id] = hit

        terms = _lexical_terms(query)
        for term in terms:
            next_offset: Any = None
            for _ in range(5):
                records, next_offset = await self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="knowledge_base_id",
                                match=models.MatchValue(value=knowledge_base_id),
                            ),
                            models.FieldCondition(
                                key="status",
                                match=models.MatchValue(value="ready"),
                            ),
                            models.FieldCondition(key="text", match=models.MatchText(text=term)),
                            *payload_conditions,
                        ]
                    ),
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    text = str(payload.get("text") or "")
                    lexical_score = _lexical_score(text, terms)
                    if lexical_score <= 0:
                        continue
                    chunk_id = str(payload.get("chunk_id") or record.id)
                    current = hits_by_id.get(chunk_id)
                    combined_score = max(
                        current.score if current else 0.0,
                        0.65 + 0.35 * lexical_score,
                    )
                    hits_by_id[chunk_id] = _payload_hit(record.id, payload, score=combined_score)
                if next_offset is None:
                    break

        return sorted(hits_by_id.values(), key=lambda item: item.score, reverse=True)[:top_k]

    async def delete_document(self, document_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                )
            ),
            wait=True,
        )

    async def health(self) -> str:
        try:
            await self.client.get_collection(self.collection)
            return "ok"
        except Exception:
            return "error"


_QUERY_FILLERS = (
    "根据知识库说明",
    "根据知识库",
    "请给出引用",
    "请提供引用",
    "在知识图谱中的",
    "在图谱中的",
    "属于什么知识路径",
    "属于哪个知识路径",
    "完整知识路径是什么",
    "知识路径是什么",
    "在知识图谱中",
    "在图谱中",
    "属于哪个学科",
    "属于什么学科",
    "分类来源",
    "有哪些内容",
    "有哪些",
    "是多少",
    "是什么",
    "什么是",
    "请问",
    "请介绍",
    "介绍一下",
    "告诉我",
    "怎么理解",
    "如何理解",
    "为什么",
    "怎么样",
)
_GENERIC_TERMS = {
    "知识库",
    "资料",
    "内容",
    "引用",
    "图谱",
    "完整知识路径",
    "知识路径",
    "分类来源",
    "哪个学科",
    "什么学科",
}


def _lexical_terms(query: str) -> list[str]:
    normalized = query.strip().lower()
    age_ranges = re.findall(r"(\d+)\s*(?:到|至|[-–—])\s*(\d+)\s*岁", normalized)
    normalized = re.sub(r"\d+\s*(?:到|至|[-–—])\s*\d+\s*岁", " ", normalized)
    for filler in _QUERY_FILLERS:
        normalized = normalized.replace(filler, " ")
    segments = re.findall(r"[a-z0-9_\-–—]+|[\u4e00-\u9fff]+", normalized)
    candidates = [f"{start}–{end}岁" for start, end in age_ranges]
    for segment in segments:
        if len(segment) >= 2 and segment not in _GENERIC_TERMS:
            candidates.append(segment)
        for part in re.split(r"的|和|与|及|关于|其中|里面|中的", segment):
            if len(part) >= 2 and part not in _GENERIC_TERMS:
                candidates.append(part)
    unique = sorted(set(candidates), key=lambda item: (-len(item), item))
    return unique[:6]


def _lexical_score(text: str, terms: list[str]) -> float:
    primary_field_match = any(_has_primary_field_match(text, term) for term in terms)
    normalized_text = re.sub(r"\s+", "", text.lower())
    normalized_terms = [re.sub(r"\s+", "", term.lower()) for term in terms if term.strip()]
    if not normalized_terms:
        return 0.0
    matched = [term for term in normalized_terms if term in normalized_text]
    if not matched:
        return 0.0
    matched_length = sum(len(term) for term in set(matched))
    total_length = sum(len(term) for term in set(normalized_terms))
    longest_match = max(map(len, matched))
    longest_term = max(map(len, normalized_terms))
    coverage = matched_length / max(1, total_length)
    specificity = longest_match / max(1, longest_term)
    score = coverage * 0.65 + specificity * 0.35
    if primary_field_match:
        return min(1.0, 0.9 + score * 0.1)
    return min(0.89, score)


def _has_primary_field_match(text: str, term: str) -> bool:
    field_names = (
        "当前节点",
        "学校中文名",
        "学校英文/原文名",
        "教育知识领域",
        "课程类别",
        "文件/网页标题",
        "发现",
        "网页标题",
        "知识图谱节点数",
        "教育课程映射数",
        "国家数",
        "年龄段数",
        "教育领域数",
        "年龄进阶关系数",
        "学校案例数",
        "学校课程关系数",
    )
    if term in field_names and re.search(
        rf"^{re.escape(term)}[：:]",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        return True
    pattern = rf"^(?:{'|'.join(map(re.escape, field_names))})[：:]\s*{re.escape(term)}\s*$"
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def _payload_hit(point_id: Any, payload: dict[str, Any], *, score: float) -> KnowledgeChunkHit:
    return KnowledgeChunkHit(
        chunk_id=str(payload.get("chunk_id") or point_id),
        document_id=str(payload.get("document_id") or ""),
        title=str(payload.get("title") or ""),
        text=str(payload.get("text") or ""),
        source_name=str(payload["source_name"]) if payload.get("source_name") else None,
        source_url=str(payload["source_url"]) if payload.get("source_url") else None,
        source_locator=str(payload["source_locator"]) if payload.get("source_locator") else None,
        page=int(payload["page"]) if payload.get("page") is not None else None,
        score=score,
        entity_ids=[str(item) for item in payload.get("entity_ids") or []],
        subject=str(payload["subject"]) if payload.get("subject") else None,
        domain=str(payload["domain"]) if payload.get("domain") else None,
        concept=str(payload["concept"]) if payload.get("concept") else None,
        content_dimension=(
            str(payload["content_dimension"]) if payload.get("content_dimension") else None
        ),
        content_type=str(payload["content_type"]) if payload.get("content_type") else None,
        metadata=dict(payload.get("metadata") or {}),
    )


def _payload_filter_conditions(filters: dict[str, str]) -> list[Any]:
    if models is None:
        return []
    return [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in filters.items()
        if key in {"subject", "domain", "concept", "content_dimension", "content_type"}
        and value
    ]
