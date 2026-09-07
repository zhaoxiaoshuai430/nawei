"""Knowledge-graph extraction using rules or the configured internal LLM."""

from __future__ import annotations

import json
import re
import uuid
from typing import Protocol

import httpx

from services.knowledge.config import KnowledgeSettings
from services.knowledge.schemas import ExtractedEntity, ExtractedRelation, GraphExtraction


class GraphExtractor(Protocol):
    async def extract(self, *, title: str, text: str, chunk_id: str) -> GraphExtraction:
        """Extract normalized entities and relations from one chunk."""

    async def close(self) -> None:
        """Release extractor resources."""


class RuleGraphExtractor:
    _RELATIONS = {
        "属于": "BELONGS_TO",
        "包括": "CONTAINS",
        "导致": "CAUSES",
        "依赖": "DEPENDS_ON",
        "需要": "REQUIRES",
        "解释": "EXPLAINS",
        "定义为": "DEFINES",
        "是": "IS_A",
    }
    _PATTERN = re.compile(
        r"(?P<source>[\u4e00-\u9fffA-Za-z0-9·_-]{2,30})"
        r"(?P<relation>定义为|属于|包括|导致|依赖|需要|解释|是)"
        r"(?P<target>[^。；;，,\n]{2,40})"
    )

    async def extract(self, *, title: str, text: str, chunk_id: str) -> GraphExtraction:
        entities: dict[str, ExtractedEntity] = {}
        relations: list[ExtractedRelation] = []
        title_name = title.strip()[:120]
        if title_name:
            entities[title_name] = ExtractedEntity(
                name=title_name,
                entity_type="DocumentTopic",
                description="",
            )
        for match in self._PATTERN.finditer(text):
            source = _clean_entity(match.group("source"))
            target = _clean_entity(match.group("target"))
            if not source or not target or source == target:
                continue
            relation_type = self._RELATIONS[match.group("relation")]
            entities.setdefault(
                source,
                ExtractedEntity(name=source, entity_type="Concept", description=""),
            )
            entities.setdefault(
                target,
                ExtractedEntity(name=target, entity_type="Concept", description=""),
            )
            relations.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    confidence=0.65,
                    evidence=match.group(0)[:500],
                )
            )
        return GraphExtraction(entities=list(entities.values()), relations=relations)

    async def close(self) -> None:
        return None


class LLMGraphExtractor:
    def __init__(self, settings: KnowledgeSettings, fallback: GraphExtractor) -> None:
        self.url = settings.llm_extractor_url
        self.api_key = settings.llm_internal_api_key
        self.fallback = fallback
        self.client = httpx.AsyncClient(timeout=60.0, trust_env=False)

    async def extract(self, *, title: str, text: str, chunk_id: str) -> GraphExtraction:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Internal-API-Key"] = self.api_key
        request_id = str(uuid.uuid4())
        prompt = (
            "从以下知识文本提取实体和关系，只返回严格 JSON，不要 Markdown。"
            "实体类型优先使用 Subject、Chapter、Concept、Definition、Formula、Theorem、Example、Exercise、Person、Event。"
            "关系类型只能使用 BELONGS_TO、PREREQUISITE_OF、DEFINES、DESCRIBES、EXPLAINS、TESTS、CAUSES、RELATED_TO、CONTAINS。"
            "JSON 格式：{\"entities\":[{\"name\":\"\",\"entity_type\":\"Concept\",\"description\":\"\"}],"
            "\"relations\":[{\"source\":\"\",\"target\":\"\",\"relation_type\":\"RELATED_TO\","
            "\"confidence\":0.8,\"evidence\":\"\"}]}。"
            f"\n标题：{title}\n正文：{text}"
        )
        payload = {
            "request_id": request_id,
            "trace_id": request_id,
            "session_id": f"knowledge-{chunk_id}",
            "user_id": "knowledge-indexer",
            "messages": [{"role": "user", "content": prompt, "metadata": {}}],
            "stream": False,
            "tools": [],
            "generation_config": {"agent_enabled": False, "temperature": 0},
            "metadata": {"purpose": "knowledge_graph_extraction"},
        }
        try:
            response = await self.client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            raw = str(response.json().get("text") or "")
            return _parse_extraction(raw)
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return await self.fallback.extract(title=title, text=text, chunk_id=chunk_id)

    async def close(self) -> None:
        await self.client.aclose()
        await self.fallback.close()


def build_graph_extractor(settings: KnowledgeSettings) -> GraphExtractor:
    fallback = RuleGraphExtractor()
    if settings.graph_extractor in {"llm", "deepseek"}:
        return LLMGraphExtractor(settings, fallback)
    return fallback


def _parse_extraction(raw: str) -> GraphExtraction:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    body = json.loads(value)
    return GraphExtraction(**body)


def _clean_entity(value: str) -> str:
    return value.strip(" ：:，,。；;、\t\r\n")[:120]
