"""Environment-backed configuration for the knowledge service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env.knowledge", override=False)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass(frozen=True)
class KnowledgeSettings:
    enabled: bool
    required: bool
    host: str
    port: int
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    database_url: str
    redis_url: str
    embedding_provider: str
    embedding_dimension: int
    embedding_api_base_url: str
    embedding_api_key: str
    embedding_model: str
    graph_extractor: str
    llm_extractor_url: str
    llm_internal_api_key: str
    chunk_size: int
    chunk_overlap: int
    cache_ttl_seconds: int
    max_context_chars: int
    default_base_id: str
    embedding_cache_dir: str = ""
    embedding_threads: int = 2


def load_settings() -> KnowledgeSettings:
    return KnowledgeSettings(
        enabled=_read_bool("KNOWLEDGE_ENABLED", True),
        required=_read_bool("KNOWLEDGE_REQUIRED", False),
        host=os.getenv("KNOWLEDGE_HOST", "127.0.0.1").strip(),
        port=_read_int("KNOWLEDGE_PORT", 8003),
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333").strip(),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "knowledge_chunks_v1").strip(),
        neo4j_uri=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687").strip(),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j").strip(),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "").strip(),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j").strip(),
        database_url=os.getenv("DATABASE_URL", "postgresql://nawei:nawei@127.0.0.1:5432/nawei").strip(),
        redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hash").strip().lower(),
        embedding_dimension=_read_int("EMBEDDING_DIMENSION", 128),
        embedding_api_base_url=os.getenv("EMBEDDING_API_BASE_URL", "").strip(),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "hash-embedding-v1").strip(),
        graph_extractor=os.getenv("KNOWLEDGE_GRAPH_EXTRACTOR", "rules").strip().lower(),
        llm_extractor_url=os.getenv(
            "KNOWLEDGE_LLM_EXTRACTOR_URL",
            "http://127.0.0.1:8002/api/v1/llm",
        ).strip(),
        llm_internal_api_key=os.getenv("LLM_INTERNAL_API_KEY", "").strip(),
        chunk_size=_read_int("KNOWLEDGE_CHUNK_SIZE", 600),
        chunk_overlap=_read_int("KNOWLEDGE_CHUNK_OVERLAP", 100),
        cache_ttl_seconds=_read_int("KNOWLEDGE_CACHE_TTL_SECONDS", 600),
        max_context_chars=_read_int("KNOWLEDGE_MAX_CONTEXT_CHARS", 6000),
        default_base_id=os.getenv("KNOWLEDGE_DEFAULT_BASE_ID", "default").strip(),
        embedding_cache_dir=os.getenv("EMBEDDING_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".runtime/embedding-cache")),
        embedding_threads=_read_int("EMBEDDING_THREADS", 2),
    )


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    try:
        return int(value) if value else default
    except ValueError:
        return default
