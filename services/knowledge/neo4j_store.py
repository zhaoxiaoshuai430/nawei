"""Neo4j persistence and graph expansion for knowledge entities."""

from __future__ import annotations

from typing import Any

from services.knowledge.concurrency import gather_cancel_on_error
from services.knowledge.schemas import GraphCandidate, KnowledgeGraphRelation, ResolvedKnowledgeEntity

try:
    from neo4j import AsyncGraphDatabase
except ImportError:  # pragma: no cover - dependency validated by live health
    AsyncGraphDatabase = None  # type: ignore[misc,assignment]


class Neo4jKnowledgeStore:
    def __init__(self, *, uri: str, user: str, password: str, database: str) -> None:
        if AsyncGraphDatabase is None:
            raise RuntimeError("neo4j Python driver is not installed.")
        self.database = database
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def initialize(self) -> None:
        await self.driver.verify_connectivity()
        for query in _CONSTRAINTS:
            await self.driver.execute_query(query, database_=self.database)

    async def close(self) -> None:
        await self.driver.close()

    async def upsert_document(
        self,
        *,
        knowledge_base_id: str,
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
        entities: list[dict[str, Any]],
        mentions: list[dict[str, str]],
        relations: list[dict[str, Any]],
    ) -> None:
        await self.driver.execute_query(
            """
            MERGE (kb:KnowledgeBase {id: $knowledge_base_id})
            MERGE (doc:Document {id: $document.document_id})
            SET doc.title = $document.title,
                doc.source_name = $document.source_name,
                doc.content_hash = $document.content_hash,
                doc.status = 'ready',
                doc.updated_at = datetime()
            MERGE (kb)-[:CONTAINS]->(doc)
            """,
            knowledge_base_id=knowledge_base_id,
            document=document,
            database_=self.database,
        )
        for batch in _batches(chunks):
            await self.driver.execute_query(
                """
                MATCH (doc:Document {id: $document_id})
                UNWIND $chunks AS item
                MERGE (chunk:Chunk {id: item.chunk_id})
                SET chunk.document_id = item.document_id,
                    chunk.chunk_index = item.chunk_index,
                    chunk.title = item.title,
                    chunk.source_name = item.source_name,
                    chunk.source_url = item.source_url,
                    chunk.source_locator = item.source_locator,
                    chunk.qdrant_collection = item.qdrant_collection,
                    chunk.qdrant_point_id = item.chunk_id
                MERGE (doc)-[:HAS_CHUNK]->(chunk)
                """,
                document_id=document["document_id"],
                chunks=batch,
                database_=self.database,
            )
        for batch in _batches(entities):
            await self.driver.execute_query(
                """
                UNWIND $entities AS item
                MERGE (entity:Entity {id: item.entity_id})
                SET entity.knowledge_base_id = $knowledge_base_id,
                    entity.name = item.name,
                    entity.canonical_name = item.canonical_name,
                    entity.entity_type = item.entity_type,
                    entity.description = item.description,
                    entity.aliases = item.aliases,
                    entity.discipline_level = item.discipline_level,
                    entity.updated_at = datetime()
                """,
                knowledge_base_id=knowledge_base_id,
                entities=batch,
                database_=self.database,
            )
        for batch in _batches(mentions):
            await self.driver.execute_query(
                """
                UNWIND $mentions AS item
                MATCH (chunk:Chunk {id: item.chunk_id})
                MATCH (entity:Entity {id: item.entity_id})
                MERGE (chunk)-[:MENTIONS]->(entity)
                """,
                mentions=batch,
                database_=self.database,
            )
        for batch in _batches(relations):
            await self.driver.execute_query(
                """
                UNWIND $relations AS item
                MATCH (source:Entity {id: item.source_id})
                MATCH (target:Entity {id: item.target_id})
                MERGE (source)-[relation:RELATED {relation_id: item.relation_id}]->(target)
                SET relation.relation_type = item.relation_type,
                    relation.confidence = item.confidence,
                    relation.evidence = item.evidence,
                    relation.evidence_chunk_id = item.evidence_chunk_id,
                    relation.updated_at = datetime()
                """,
                relations=batch,
                database_=self.database,
            )

    async def resolve_entities(
        self,
        *,
        knowledge_base_id: str,
        candidates: list[GraphCandidate],
    ) -> list[ResolvedKnowledgeEntity]:
        if not candidates:
            return []

        ordered_candidates = sorted(candidates, key=lambda item: item.priority)
        candidate_payloads = []
        for position, candidate in enumerate(ordered_candidates):
            canonical_names = {
                _canonical_name(candidate.name),
                *(_canonical_name(alias) for alias in candidate.aliases),
            }
            raw_names = {candidate.name.strip().lower(), *(alias.strip().lower() for alias in candidate.aliases)}
            candidate_payloads.append(
                {
                    "position": position,
                    "name": candidate.name,
                    "canonical_name": _canonical_name(candidate.name),
                    "entity_types": candidate.entity_types,
                    "canonical_names": sorted(canonical_names),
                    "raw_names": sorted(raw_names),
                }
            )

        records, _, _ = await self.driver.execute_query(
            """
            UNWIND $candidates AS candidate
            MATCH (entity:Entity)
            WHERE entity.knowledge_base_id = $knowledge_base_id
              AND (
                size(candidate.entity_types) = 0
                OR entity.entity_type IN candidate.entity_types
              )
              AND (
                entity.canonical_name IN candidate.canonical_names
                OR toLower(entity.name) IN candidate.raw_names
                OR any(alias IN coalesce(entity.aliases, []) WHERE toLower(alias) IN candidate.raw_names)
              )
            RETURN entity.id AS entity_id,
                   entity.name AS name,
                   entity.canonical_name AS canonical_name,
                   entity.entity_type AS entity_type,
                   coalesce(entity.aliases, []) AS aliases,
                   candidate.name AS candidate_name,
                   candidate.canonical_name AS candidate_canonical_name
            ORDER BY candidate.position,
                     CASE WHEN entity.canonical_name = candidate.canonical_name THEN 0 ELSE 1 END,
                     entity.name
            LIMIT 1
            """,
            knowledge_base_id=knowledge_base_id,
            candidates=candidate_payloads,
            database_=self.database,
        )
        if not records:
            return []

        record = records[0]
        canonical_name = str(record["canonical_name"] or "")
        match_level = "exact_concept" if canonical_name == str(record["candidate_canonical_name"]) else "exact_alias"
        return [
            ResolvedKnowledgeEntity(
                entity_id=str(record["entity_id"]),
                name=str(record["name"]),
                canonical_name=canonical_name,
                entity_type=str(record["entity_type"]),
                aliases=[str(item) for item in record["aliases"] or []],
                match_level=match_level,
                candidate_name=str(record["candidate_name"]),
                confidence=1.0 if match_level == "exact_concept" else 0.95,
            )
        ]

    async def expand_entities(
        self,
        *,
        entity_ids: list[str],
        relation_groups: dict[str, tuple[list[str], int]],
        depth: int = 2,
    ) -> dict[str, list[KnowledgeGraphRelation]]:
        if not entity_ids or depth <= 0:
            return {}
        max_depth = 1 if depth == 1 else 2
        async def expand_group(
            group_name: str,
            relation_types: list[str],
            limit: int,
        ) -> tuple[str, list[KnowledgeGraphRelation]]:
            records, _, _ = await self.driver.execute_query(
                f"""
                MATCH (seed:Entity)
                WHERE seed.id IN $entity_ids
                MATCH path = (seed)-[:RELATED*1..{max_depth}]-(related:Entity)
                UNWIND relationships(path) AS relation
                WITH DISTINCT relation
                WHERE relation.relation_type IN $relation_types
                RETURN startNode(relation).id AS source_id,
                       startNode(relation).name AS source_name,
                       relation.relation_type AS relation_type,
                       endNode(relation).id AS target_id,
                       endNode(relation).name AS target_name,
                       relation.confidence AS confidence,
                       relation.evidence_chunk_id AS evidence_chunk_id
                LIMIT $limit
                """,
                entity_ids=entity_ids,
                relation_types=relation_types,
                limit=limit,
                database_=self.database,
            )
            return group_name, [
                KnowledgeGraphRelation(
                    source_id=str(record["source_id"]),
                    source_name=str(record["source_name"]),
                    relation_type=str(record["relation_type"]),
                    target_id=str(record["target_id"]),
                    target_name=str(record["target_name"]),
                    confidence=float(record["confidence"] or 0.75),
                    evidence_chunk_id=(
                        str(record["evidence_chunk_id"])
                        if record["evidence_chunk_id"]
                        else None
                    ),
                )
                for record in records
            ]

        jobs = [
            expand_group(group_name, relation_types, limit)
            for group_name, (relation_types, limit) in relation_groups.items()
            if relation_types and limit > 0
        ]
        if not jobs:
            return {}
        expanded = await gather_cancel_on_error(*jobs)
        return dict(expanded)

    async def expand(self, *, chunk_ids: list[str], depth: int = 2) -> list[KnowledgeGraphRelation]:
        if not chunk_ids or depth <= 0:
            return []
        max_depth = 1 if depth == 1 else 2
        records, _, _ = await self.driver.execute_query(
            f"""
            MATCH (chunk:Chunk)-[:MENTIONS]->(seed:Entity)
            WHERE chunk.id IN $chunk_ids
            MATCH path = (seed)-[:RELATED*1..{max_depth}]-(related:Entity)
            UNWIND relationships(path) AS relation
            WITH DISTINCT relation
            WHERE relation.evidence_chunk_id IN $chunk_ids
            RETURN DISTINCT
                startNode(relation).id AS source_id,
                startNode(relation).name AS source_name,
                relation.relation_type AS relation_type,
                endNode(relation).id AS target_id,
                endNode(relation).name AS target_name,
                relation.confidence AS confidence,
                relation.evidence_chunk_id AS evidence_chunk_id
            LIMIT 100
            """,
            chunk_ids=chunk_ids,
            database_=self.database,
        )
        return [
            KnowledgeGraphRelation(
                source_id=str(record["source_id"]),
                source_name=str(record["source_name"]),
                relation_type=str(record["relation_type"] or "RELATED_TO"),
                target_id=str(record["target_id"]),
                target_name=str(record["target_name"]),
                confidence=float(record["confidence"] or 0.75),
                evidence_chunk_id=(
                    str(record["evidence_chunk_id"]) if record["evidence_chunk_id"] else None
                ),
            )
            for record in records
        ]

    async def delete_document(self, document_id: str) -> None:
        await self.driver.execute_query(
            """
            MATCH (document:Document {id: $document_id})
            MATCH (document)-[:HAS_CHUNK]->(chunk:Chunk)
            DETACH DELETE chunk
            """,
            document_id=document_id,
            database_=self.database,
        )
        await self.driver.execute_query(
            "MATCH (document:Document {id: $document_id}) DETACH DELETE document",
            document_id=document_id,
            database_=self.database,
        )
        await self.driver.execute_query(
            "MATCH (entity:Entity) WHERE NOT (entity)<-[:MENTIONS]-() DETACH DELETE entity",
            database_=self.database,
        )

    async def health(self) -> str:
        try:
            await self.driver.verify_connectivity()
            return "ok"
        except Exception:
            return "error"


_CONSTRAINTS = (
    "CREATE CONSTRAINT knowledge_base_id IF NOT EXISTS FOR (node:KnowledgeBase) REQUIRE node.id IS UNIQUE",
    "CREATE CONSTRAINT knowledge_document_id IF NOT EXISTS FOR (node:Document) REQUIRE node.id IS UNIQUE",
    "CREATE CONSTRAINT knowledge_chunk_id IF NOT EXISTS FOR (node:Chunk) REQUIRE node.id IS UNIQUE",
    "CREATE CONSTRAINT knowledge_entity_id IF NOT EXISTS FOR (node:Entity) REQUIRE node.id IS UNIQUE",
    "CREATE INDEX knowledge_entity_lookup IF NOT EXISTS FOR (node:Entity) ON (node.knowledge_base_id, node.canonical_name, node.entity_type)",
    "CREATE INDEX knowledge_entity_name IF NOT EXISTS FOR (node:Entity) ON (node.knowledge_base_id, node.name)",
)


def _batches(items: list[dict[str, Any]], size: int = 500) -> list[list[dict[str, Any]]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _canonical_name(value: str) -> str:
    return "".join(value.strip().lower().split())[:120]
