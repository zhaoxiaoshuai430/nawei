"""PostgreSQL control-plane metadata for knowledge ingestion."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

try:
    import asyncpg
except ImportError:  # pragma: no cover - optional in lightweight tests
    asyncpg = None  # type: ignore[assignment]


class KnowledgeCatalog:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.pool: Any = None
        self.memory: dict[str, dict[str, Any]] = {}
        self.last_error: str | None = None
        self._active_documents: set[str] = set()

    @asynccontextmanager
    async def document_operation(self, document_id: str) -> AsyncIterator[None]:
        """Exclude competing imports/deletes, including across service processes.

        A dedicated session keeps the advisory lock out of the small query pool.
        PostgreSQL releases the lock when this session closes, including crashes.
        Reject contention rather than queue unbounded document uploads.
        """
        if document_id in self._active_documents:
            raise RuntimeError("Document operation already in progress; retry later.")
        self._active_documents.add(document_id)
        connection = None
        try:
            if self.pool is not None:
                connection = await asyncpg.connect(self.database_url, timeout=4.0)
                acquired = await connection.fetchval(
                    "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
                    f"knowledge-document:{document_id}",
                    timeout=4.0,
                )
                if not acquired:
                    raise RuntimeError("Document operation already in progress; retry later.")
            yield
        finally:
            if connection is not None:
                # No await: cancellation must not release the local guard first.
                connection.terminate()
            self._active_documents.discard(document_id)

    async def initialize(self) -> None:
        if self.pool is not None:
            return
        if asyncpg is None:
            self.last_error = "asyncpg is not installed; using in-memory knowledge catalog."
            return
        pool = None
        try:
            pool = await asyncio.wait_for(
                asyncpg.create_pool(self.database_url, min_size=1, max_size=3, timeout=3.0),
                timeout=4.0,
            )
            async with pool.acquire() as connection:
                await connection.execute(_SCHEMA_SQL)
            self.pool = pool
            self.last_error = None
        except Exception as exc:
            self.last_error = f"PostgreSQL knowledge catalog unavailable: {exc}"
        finally:
            if pool is not None and pool is not self.pool:
                # A failed/cancelled schema setup must not leak an unpublished pool.
                pool.terminate()

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def find_ready_by_hash(self, *, knowledge_base_id: str, content_hash: str) -> dict[str, Any] | None:
        if self.pool is not None:
            row = await self.pool.fetchrow(
                """
                SELECT document_id, knowledge_base_id, status, chunk_count, entity_count, relation_count
                FROM knowledge_documents
                WHERE knowledge_base_id = $1 AND content_hash = $2 AND status = 'ready'
                """,
                knowledge_base_id,
                content_hash,
            )
            return dict(row) if row else None
        for item in self.memory.values():
            if (
                item["knowledge_base_id"] == knowledge_base_id
                and item["content_hash"] == content_hash
                and item["status"] == "ready"
            ):
                return dict(item)
        return None

    async def start_document(
        self,
        *,
        document_id: str,
        knowledge_base_id: str,
        title: str,
        source_name: str | None,
        content_hash: str,
        metadata: dict[str, Any],
    ) -> None:
        record = {
            "document_id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "title": title,
            "source_name": source_name,
            "content_hash": content_hash,
            "status": "indexing",
            "chunk_count": 0,
            "entity_count": 0,
            "relation_count": 0,
            "error": None,
            "metadata": metadata,
            "updated_at": time.time(),
        }
        self.memory[document_id] = record
        if self.pool is not None:
            await self.pool.execute(
                """
                INSERT INTO knowledge_documents (
                    document_id, knowledge_base_id, title, source_name, content_hash, status, metadata, updated_at
                ) VALUES ($1, $2, $3, $4, $5, 'indexing', $6::jsonb, now())
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_name = EXCLUDED.source_name,
                    status = 'indexing',
                    error = NULL,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                document_id,
                knowledge_base_id,
                title,
                source_name,
                content_hash,
                json.dumps(metadata, ensure_ascii=False),
            )

    async def complete_document(
        self,
        *,
        document_id: str,
        chunk_count: int,
        entity_count: int,
        relation_count: int,
    ) -> None:
        if document_id in self.memory:
            self.memory[document_id].update(
                status="ready",
                chunk_count=chunk_count,
                entity_count=entity_count,
                relation_count=relation_count,
                updated_at=time.time(),
            )
        if self.pool is not None:
            await self.pool.execute(
                """
                UPDATE knowledge_documents
                SET status = 'ready', chunk_count = $2, entity_count = $3,
                    relation_count = $4, error = NULL, updated_at = now()
                WHERE document_id = $1
                """,
                document_id,
                chunk_count,
                entity_count,
                relation_count,
            )

    async def fail_document(self, *, document_id: str, error: str) -> None:
        if document_id in self.memory:
            self.memory[document_id].update(status="failed", error=error[:1000], updated_at=time.time())
        if self.pool is not None:
            await self.pool.execute(
                "UPDATE knowledge_documents SET status = 'failed', error = $2, updated_at = now() WHERE document_id = $1",
                document_id,
                error[:1000],
            )

    async def delete_document(self, document_id: str) -> None:
        self.memory.pop(document_id, None)
        if self.pool is not None:
            await self.pool.execute("DELETE FROM knowledge_documents WHERE document_id = $1", document_id)

    async def get_document(self, document_id: str) -> dict[str, Any] | None:
        if self.pool is not None:
            row = await self.pool.fetchrow(
                "SELECT * FROM knowledge_documents WHERE document_id = $1",
                document_id,
            )
            return dict(row) if row else None
        value = self.memory.get(document_id)
        return dict(value) if value else None

    async def health(self) -> str:
        if self.pool is None:
            return "fallback"
        try:
            await self.pool.fetchval("SELECT 1")
            return "ok"
        except Exception:
            return "error"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id UUID PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_name TEXT,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded',
    chunk_count INT NOT NULL DEFAULT 0,
    entity_count INT NOT NULL DEFAULT 0,
    relation_count INT NOT NULL DEFAULT 0,
    error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (knowledge_base_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_base_status
ON knowledge_documents(knowledge_base_id, status, updated_at DESC);
"""
