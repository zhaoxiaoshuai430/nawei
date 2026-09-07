"""Persistent memory support for the LLM Agent runtime.

The module intentionally uses optional dependencies. When PostgreSQL or Redis
client packages are unavailable, development keeps running with an in-memory
fallback so local protocol tests remain fast and deterministic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by integration tests or live runtime
    import asyncpg  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised by integration tests or live runtime
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - optional dependency
    aioredis = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised by live LLM runtime
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore[assignment]


def _consume_background_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Background memory task failed")


DEFAULT_DATABASE_URL = "postgresql://llm:llm@127.0.0.1:5432/llm"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
MAX_HISTORY_MESSAGES = 20
HISTORY_TTL_SECONDS = 24 * 3600
PROFILE_CACHE_TTL_SECONDS = 3600
MIN_MEMORY_IMPORTANCE = 4
DEFAULT_PROMPT_VERSION = "navi-memory-v2"
DEFAULT_MEMORY_EXTRACTOR_URL = "http://127.0.0.1:8002/api/v1/llm"
DEFAULT_MEMORY_EXTRACTOR_TIMEOUT_SECONDS = 60.0


def _redis_key_component(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_MEMORY_EXTRACTION_SYSTEM_PROMPT = """
You extract memory for Xiaowei, an AI education voice robot.
Return only one strict JSON object. No markdown, no explanation.
The next user message is one raw student utterance. Extract useful memory from that utterance.

Output schema:
{
  "profile": {
    "nickname": string optional,
    "current_focus": string optional,
    "preferences": {"topics_enjoy": string[] optional, "topics_avoid": string[] optional} optional,
    "learning_state": object optional,
    "grade": string optional,
    "subjects": string[] optional
  },
  "memories": [
    {
      "content": string,
      "memory_type": "profile_fact" | "learning_state" | "preference" | "learning_event" | "plan" | "weakness",
      "importance": integer 1-10,
      "confidence": number 0-1,
      "sensitivity": "normal" | "sensitive" | "private",
      "tags": string[],
      "summary": string or null,
      "expires_at": string or null
    }
  ]
}

Must extract these when present:
- name or nickname -> profile.nickname and a profile_fact memory.
- grade, subject, current topic -> profile.grade/current_focus and a learning_state memory.
- phrases like "I don't understand", "I always get stuck", "weak at" -> weakness memory.
- likes/dislikes about teaching style or topics -> profile.preferences and preference memory.
- homework, exam, deadline, study plan -> plan or learning_event memory.

Do not extract greetings, thanks, filler, or questions asking what you remember.
If the utterance includes secrets, passwords, phone numbers, addresses, or ID numbers, omit them unless education-critical; if kept, mark sensitive/private.
Keep content concise and in the student's language. Return at most 5 memories. Each content must be under 40 Chinese characters or 80 English characters.
""".strip()


@dataclass(frozen=True)
class MemoryHit:
    """Memory item returned by recall."""

    id: str
    memory_type: str
    content: str
    importance: int
    score: float
    summary: str | None = None
    sensitivity: str = "normal"
    confidence: float = 0.75
    tags: list[str] = field(default_factory=list)
    usage_count: int = 0


@dataclass(frozen=True)
class MemoryContext:
    """Context block injected into the education LLM prompt."""

    user_id: str
    profile: dict[str, Any]
    history: list[dict[str, str]]
    memories: list[MemoryHit]


@dataclass(frozen=True)
class LearningConcept:
    """Tracked education concept state for personalization."""

    id: str
    subject: str
    concept: str
    mastery_score: float
    confidence: float
    exposure_count: int
    weakness_count: int
    correct_count: int = 0
    incorrect_count: int = 0
    last_event_at: str | None = None
    last_practiced_at: str | None = None
    effective_strategy: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """PostgreSQL + Redis memory facade without vector storage."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        redis_url: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        raw_enabled = os.getenv("MEMORY_ENABLED", "true").strip().lower()
        self.enabled = enabled if enabled is not None else raw_enabled not in {"0", "false", "no", "off"}
        self.database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.redis_url = redis_url or os.getenv("REDIS_URL", DEFAULT_REDIS_URL)
        self._pool: Any = None
        self._pool_loop: asyncio.AbstractEventLoop | None = None
        self._redis: Any = None
        self._init_lock = asyncio.Lock()
        self._init_lock_loop: asyncio.AbstractEventLoop | None = None
        self._memory_messages: dict[str, list[dict[str, Any]]] = {}
        self._memory_profiles: dict[str, dict[str, Any]] = {}
        self._memory_items: dict[str, list[dict[str, Any]]] = {}
        self._memory_concepts: dict[str, list[dict[str, Any]]] = {}
        self._profile_process_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._postgres_retry_after = 0.0
        self._redis_retry_after = 0.0
        self._redis_last_error: str | None = None
        self.profile_redis_cache_enabled = os.getenv("MEMORY_PROFILE_REDIS_CACHE", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.extractor_enabled = os.getenv("MEMORY_EXTRACTOR_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        self.extractor_url = os.getenv("MEMORY_EXTRACTOR_URL", DEFAULT_MEMORY_EXTRACTOR_URL).strip()
        self.extractor_timeout_seconds = _read_float_env("MEMORY_EXTRACTOR_TIMEOUT_SECONDS", DEFAULT_MEMORY_EXTRACTOR_TIMEOUT_SECONDS)
        self.last_error: str | None = None

    async def initialize(self) -> None:
        """Create database objects and connect Redis when available."""

        if not self.enabled:
            return
        current_loop = asyncio.get_running_loop()
        if self._init_lock_loop is not current_loop:
            self._init_lock = asyncio.Lock()
            self._init_lock_loop = current_loop
        async with self._init_lock:
            if self._pool is not None and self._pool_loop is not current_loop:
                self._pool = None
                self._pool_loop = None
            if self._pool is not None:
                await self._ensure_redis()
                return
            if self.last_error and time.monotonic() < self._postgres_retry_after:
                await self._ensure_redis()
                return
            if asyncpg is None:
                self.last_error = "asyncpg is not installed; using in-memory memory fallback."
                await self._ensure_redis()
                return
            try:
                connect_timeout = _read_float_env("MEMORY_CONNECT_TIMEOUT_SECONDS", 3.0)
                self._pool = await asyncio.wait_for(
                    asyncpg.create_pool(
                        self.database_url,
                        min_size=1,
                        max_size=_read_int_env("MEMORY_POSTGRES_POOL_SIZE", 5),
                        timeout=connect_timeout,
                    ),
                    timeout=connect_timeout + 0.5,
                )
                self._pool_loop = current_loop
                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT pg_advisory_xact_lock(hashtext('nawei_llm_memory_schema'))"
                        )
                        await conn.execute(_SCHEMA_SQL)
                self.last_error = None
                self._postgres_retry_after = 0.0
            except Exception as exc:  # pragma: no cover - depends on local services
                self.last_error = f"PostgreSQL memory disabled: {exc}"
                self._postgres_retry_after = time.monotonic() + _read_float_env("MEMORY_RETRY_INTERVAL_SECONDS", 10.0)
                self._pool = None
                self._pool_loop = None

            await self._ensure_redis()

    async def _ensure_redis(self) -> None:
        if self._redis is not None or aioredis is None:
            return
        if time.monotonic() < self._redis_retry_after:
            return
        try:
            connect_timeout = _read_float_env("MEMORY_CONNECT_TIMEOUT_SECONDS", 3.0)
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=connect_timeout,
                socket_timeout=connect_timeout,
            )
            await self._redis.ping()
            self._redis_retry_after = 0.0
            if self._redis_last_error is not None:
                logger.info("Redis memory cache connection restored")
                self._redis_last_error = None
        except Exception as exc:
            self._redis = None
            self._redis_retry_after = time.monotonic() + _read_float_env("MEMORY_RETRY_INTERVAL_SECONDS", 10.0)
            error = f"{type(exc).__name__}: {exc}"
            if error != self._redis_last_error:
                logger.warning("Redis memory cache unavailable; retrying with fallback: %s", error)
                self._redis_last_error = error

    async def close(self) -> None:
        """Close optional PostgreSQL and Redis clients owned by this store."""

        if self._redis is not None:
            redis_client = self._redis
            self._redis = None
            try:
                close = getattr(redis_client, "aclose", None) or getattr(redis_client, "close", None)
                if close is not None:
                    try:
                        result = close(close_connection_pool=True)
                    except TypeError:
                        result = close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                pass
            try:
                pool = getattr(redis_client, "connection_pool", None)
                disconnect = getattr(pool, "disconnect", None)
                if disconnect is not None:
                    try:
                        result = disconnect(inuse_connections=True)
                    except TypeError:
                        result = disconnect()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                pass

        if self._pool is not None:
            pool = self._pool
            self._pool = None
            self._pool_loop = None
            try:
                await pool.close()
            except Exception:
                pass

    async def health(self) -> dict[str, Any]:
        """Return memory dependency status."""

        await self.initialize()
        if not self.enabled:
            return {
                "enabled": False,
                "postgres": "disabled",
                "redis": "disabled",
                "vector": "disabled",
                "last_error": self.last_error,
            }
        pg_ok = self._pool is not None
        redis_ok = False
        if self._redis is not None:
            try:
                redis_ok = bool(await self._redis.ping())
            except Exception:
                self._redis = None
                redis_ok = False
        if not redis_ok:
            await self._ensure_redis()
            if self._redis is not None:
                try:
                    redis_ok = bool(await self._redis.ping())
                except Exception:
                    self._redis = None
                    redis_ok = False
        return {
            "enabled": self.enabled,
            "postgres": "ok" if pg_ok else "fallback",
            "redis": "ok" if redis_ok else "fallback",
            "vector": "removed",
            "last_error": self.last_error,
        }

    async def build_context(self, *, user_id: str, session_id: str, query: str, lightweight: bool = False) -> MemoryContext:
        """Load profile, short-term history, and relevant long-term memories."""

        await self.initialize()
        if lightweight:
            profile, history, memories = await asyncio.gather(
                self.get_profile(user_id),
                self.get_history(user_id=user_id, session_id=session_id, limit=4),
                self.recall_foundation(user_id=user_id, query=query, top_k=6),
            )
            profile = _profile_for_query(profile, query)
            return MemoryContext(user_id=user_id, profile=profile, history=history, memories=memories)

        profile, history, memories = await asyncio.gather(
            self.get_profile(user_id),
            self.get_history(user_id=user_id, session_id=session_id),
            self.recall_layered(user_id=user_id, query=query),
        )
        profile = _profile_for_query(profile, query)
        return MemoryContext(user_id=user_id, profile=profile, history=history, memories=memories)

    async def build_short_term_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str = "",
        history_limit: int | None = None,
        include_profile: bool = True,
        prefer_local_history: bool = False,
    ) -> MemoryContext:
        """Load only profile and recent history for the current turn."""

        limit = history_limit if history_limit is not None else _read_int_env("MEMORY_SHORT_TERM_HISTORY_LIMIT", 8)
        limit = max(1, min(limit, MAX_HISTORY_MESSAGES))
        if prefer_local_history:
            local_history = self._local_history(user_id=user_id, session_id=session_id, limit=limit)
            if local_history:
                return MemoryContext(user_id=user_id, profile={}, history=local_history, memories=[])

        await self.initialize()
        if not include_profile:
            history = await self.get_history(user_id=user_id, session_id=session_id, limit=limit)
            return MemoryContext(user_id=user_id, profile={}, history=history, memories=[])

        profile, history = await asyncio.gather(
            self.get_profile(user_id),
            self.get_history(user_id=user_id, session_id=session_id, limit=limit),
        )
        return MemoryContext(
            user_id=user_id,
            profile=_profile_for_query(profile, query),
            history=history,
            memories=[],
        )

    async def recall_foundation(self, *, user_id: str, query: str, top_k: int = 6) -> list[MemoryHit]:
        """Recall stable profile/preference memories with text filters."""

        query = query.strip()
        await self.initialize()
        if self._pool is None:
            hits = [
                _memory_hit_from_item(item, score=0.95)
                for item in self._memory_items.get(user_id, [])
                if not item.get("is_deleted") and item.get("memory_type") in {"profile_fact", "preference"}
            ]
            return [hit for hit in hits if not query or _foundation_memory_relevant(hit, query)][:top_k]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text AS id, memory_type, content, summary, importance, sensitivity, confidence,
                       COALESCE(tags, ARRAY[]::text[]) AS tags, usage_count,
                       0.95::float AS score
                FROM navi_memories
                WHERE user_id = $1
                  AND memory_type IN ('profile_fact', 'preference')
                  AND NOT is_deleted
                  AND importance >= 4
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY importance DESC, updated_at DESC
                LIMIT $2
                """,
                user_id,
                top_k,
            )
        hits = [_memory_hit_from_row(row) for row in rows]
        return [hit for hit in hits if not query or _foundation_memory_relevant(hit, query)][:top_k]

    async def list_concepts(self, *, user_id: str, query: str = "", limit: int = 8) -> list[LearningConcept]:
        """Return recently relevant concept mastery states."""

        await self.initialize()
        query_terms = set(_tokenize(query))
        if self._pool is None:
            items = [item for item in self._memory_concepts.get(user_id, []) if not item.get("is_deleted")]
            ranked = sorted(
                items,
                key=lambda item: (
                    _concept_query_overlap(item, query_terms),
                    int(item.get("weakness_count") or 0),
                    int(item.get("exposure_count") or 0),
                ),
                reverse=True,
            )
            return [_concept_from_item(item) for item in ranked[:limit]]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text AS id, subject, concept, mastery_score, confidence,
                       exposure_count, weakness_count, correct_count, incorrect_count,
                       last_event_at, last_practiced_at, effective_strategy, evidence
                FROM navi_learning_concepts
                WHERE user_id = $1 AND NOT is_deleted
                ORDER BY
                    CASE
                        WHEN $2 = '' THEN 0
                        WHEN concept ILIKE '%' || $2 || '%' THEN 3
                        WHEN subject ILIKE '%' || $2 || '%' THEN 2
                        ELSE 0
                    END DESC,
                    weakness_count DESC,
                    last_event_at DESC NULLS LAST,
                    updated_at DESC
                LIMIT $3
                """,
                user_id,
                query.strip()[:80],
                limit,
            )
        return [_concept_from_row(row) for row in rows]

    async def update_concepts_from_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str = "",
    ) -> list[LearningConcept]:
        """Extract concept signals from one turn and update mastery state."""

        await self.initialize()
        signals = _extract_concept_signals(user_text, assistant_text)
        if not signals:
            return []

        if self._pool is None:
            updated: list[LearningConcept] = []
            bucket = self._memory_concepts.setdefault(user_id, [])
            for signal in signals:
                existing = next(
                    (
                        item
                        for item in bucket
                        if item.get("subject") == signal["subject"]
                        and item.get("concept") == signal["concept"]
                        and not item.get("is_deleted")
                    ),
                    None,
                )
                if existing is None:
                    existing = {
                        "id": hashlib.sha1(f"{user_id}:{signal['subject']}:{signal['concept']}".encode("utf-8")).hexdigest()[:16],
                        "subject": signal["subject"],
                        "concept": signal["concept"],
                        "mastery_score": signal["initial_mastery"],
                        "confidence": signal["confidence"],
                        "exposure_count": 0,
                        "weakness_count": 0,
                        "correct_count": 0,
                        "incorrect_count": 0,
                        "last_practiced_at": None,
                        "effective_strategy": signal.get("effective_strategy"),
                        "evidence": {},
                        "is_deleted": False,
                    }
                    bucket.append(existing)
                existing["mastery_score"] = _clamp_float(float(existing.get("mastery_score", 0.35)) + signal["mastery_delta"], 0.0, 1.0)
                existing["confidence"] = max(float(existing.get("confidence", 0.5)), signal["confidence"])
                existing["exposure_count"] = int(existing.get("exposure_count") or 0) + 1
                if signal["event_type"] == "weakness":
                    existing["weakness_count"] = int(existing.get("weakness_count") or 0) + 1
                    existing["incorrect_count"] = int(existing.get("incorrect_count") or 0) + 1
                if signal["event_type"] == "progress":
                    existing["correct_count"] = int(existing.get("correct_count") or 0) + 1
                if signal["event_type"] in {"practice", "progress"}:
                    existing["last_practiced_at"] = "memory"
                existing["effective_strategy"] = signal.get("effective_strategy") or existing.get("effective_strategy")
                existing["evidence"] = signal["evidence"]
                updated.append(_concept_from_item(existing))
            return updated

        updated_rows: list[Any] = []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for signal in signals:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO navi_learning_concepts(
                            user_id, subject, concept, mastery_score, confidence,
                            exposure_count, weakness_count, correct_count, incorrect_count,
                            last_event_at, last_practiced_at, effective_strategy, evidence
                        )
                        VALUES ($1, $2, $3, $4, $5, 1, $6, $9, $10, now(), CASE WHEN $11 THEN now() ELSE NULL END, $12, $7::jsonb)
                        ON CONFLICT (user_id, subject, concept) DO UPDATE SET
                            mastery_score = LEAST(1.0, GREATEST(0.0, navi_learning_concepts.mastery_score + $8)),
                            confidence = GREATEST(navi_learning_concepts.confidence, EXCLUDED.confidence),
                            exposure_count = navi_learning_concepts.exposure_count + 1,
                            weakness_count = navi_learning_concepts.weakness_count + $6,
                            correct_count = navi_learning_concepts.correct_count + $9,
                            incorrect_count = navi_learning_concepts.incorrect_count + $10,
                            last_event_at = now(),
                            last_practiced_at = COALESCE(EXCLUDED.last_practiced_at, navi_learning_concepts.last_practiced_at),
                            effective_strategy = COALESCE(EXCLUDED.effective_strategy, navi_learning_concepts.effective_strategy),
                            evidence = EXCLUDED.evidence,
                            updated_at = now(),
                            is_deleted = FALSE
                        RETURNING id::text AS id, subject, concept, mastery_score, confidence,
                                  exposure_count, weakness_count, correct_count, incorrect_count,
                                  last_event_at, last_practiced_at, effective_strategy, evidence
                        """,
                        user_id,
                        signal["subject"],
                        signal["concept"],
                        signal["initial_mastery"],
                        signal["confidence"],
                        1 if signal["event_type"] == "weakness" else 0,
                        json.dumps(signal["evidence"], ensure_ascii=False),
                        signal["mastery_delta"],
                        1 if signal["event_type"] == "progress" else 0,
                        1 if signal["event_type"] == "weakness" else 0,
                        signal["event_type"] in {"practice", "progress"},
                        signal.get("effective_strategy"),
                    )
                    updated_rows.append(row)
        return [_concept_from_row(row) for row in updated_rows]

    async def add_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        persist_messages: bool = True,
        extract_memory: bool = True,
    ) -> None:
        """Persist a completed turn and asynchronously extract useful memory."""

        await self.initialize()
        if persist_messages:
            await self.add_message(user_id=user_id, session_id=session_id, role="user", content=user_text)
            await self.add_message(user_id=user_id, session_id=session_id, role="assistant", content=assistant_text)
        if not extract_memory:
            return
        await self.extract_and_store(user_id=user_id, session_id=session_id, user_text=user_text, use_llm=False)
        if self._should_run_llm_extractor(user_text):
            task = asyncio.create_task(self.extract_and_store(user_id=user_id, session_id=session_id, user_text=user_text, use_rules=False))
            task.add_done_callback(_consume_background_task_exception)

    async def add_message(self, *, user_id: str, session_id: str, role: str, content: str) -> None:
        """Save a message to Redis and PostgreSQL."""

        content = self.cache_message(user_id=user_id, session_id=session_id, role=role, content=content)
        if not content:
            return
        await self.persist_message(user_id=user_id, session_id=session_id, role=role, content=content)

    def cache_message(self, *, user_id: str, session_id: str, role: str, content: str) -> str:
        """Make a message immediately available to same-process follow-up turns."""

        content = content.strip()
        if not content:
            return ""
        key = self._history_key(user_id, session_id)
        items = self._memory_messages.setdefault(key, [])
        items.append({"role": role, "content": content, "ts": time.time()})
        del items[:-MAX_HISTORY_MESSAGES]
        return content

    async def persist_message(self, *, user_id: str, session_id: str, role: str, content: str) -> None:
        """Persist an already locally cached message to Redis and PostgreSQL."""

        content = content.strip()
        if not content:
            return
        key = self._history_key(user_id, session_id)
        if self._redis is not None:
            try:
                payload = json.dumps({"role": role, "content": content, "ts": time.time()}, ensure_ascii=False)
                await self._redis.rpush(key, payload)
                await self._redis.ltrim(key, -MAX_HISTORY_MESSAGES, -1)
                await self._redis.expire(key, HISTORY_TTL_SECONDS)
            except Exception:
                self._redis = None

        if self._pool is None:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO navi_messages(session_id, user_id, role, content, input_type)
                VALUES ($1, $2, $3, $4, 'voice')
                """,
                session_id,
                user_id,
                role,
                content,
            )

    async def list_user_questions_on_date(
        self,
        *,
        user_id: str,
        target_date: date,
        timezone: str = "Asia/Shanghai",
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Return user questions asked on one local calendar date across sessions."""

        await self.initialize()
        tz = _safe_zoneinfo(timezone)
        start = datetime.combine(target_date, datetime_time.min, tzinfo=tz)
        end = start + timedelta(days=1)

        if self._pool is None:
            rows: list[dict[str, str]] = []
            for key, items in self._memory_messages.items():
                if not key.startswith(f"hist:{user_id}:"):
                    continue
                for item in items:
                    if item.get("role") != "user":
                        continue
                    ts = float(item.get("ts") or 0)
                    created_at = datetime.fromtimestamp(ts, tz)
                    if start <= created_at < end:
                        rows.append({"content": str(item.get("content") or ""), "created_at": created_at.isoformat(), "session_id": key.rsplit(":", 1)[-1]})
            return rows[-limit:]

        async with self._pool.acquire() as conn:
            db_rows = await conn.fetch(
                """
                SELECT session_id, content, created_at
                FROM navi_messages
                WHERE user_id = $1
                  AND role = 'user'
                  AND created_at >= $2
                  AND created_at < $3
                ORDER BY created_at ASC
                LIMIT $4
                """,
                user_id,
                start,
                end,
                limit,
            )
        return [
            {
                "session_id": str(row["session_id"]),
                "content": str(row["content"]),
                "created_at": str(row["created_at"]),
            }
            for row in db_rows
        ]

    async def get_history(self, *, user_id: str, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        """Return recent short-term messages in chronological order."""

        if self._redis is not None:
            try:
                key = self._history_key(user_id, session_id)
                raw_items = await self._redis.lrange(key, -limit, -1)
                if raw_items:
                    return [_compact_message(json.loads(item)) for item in raw_items]
            except Exception:
                self._redis = None
        key = self._history_key(user_id, session_id)
        memory_items = self._memory_messages.get(key, [])[-limit:]
        if memory_items:
            return [_compact_message(item) for item in memory_items]
        if self._pool is None:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content
                FROM (
                    SELECT role, content, created_at
                    FROM navi_messages
                    WHERE user_id = $1 AND session_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                ) recent
                ORDER BY created_at ASC
                """,
                user_id,
                session_id,
                limit,
            )
        return [_compact_message(dict(row)) for row in rows]

    def _local_history(self, *, user_id: str, session_id: str, limit: int) -> list[dict[str, str]]:
        items = self._memory_messages.get(self._history_key(user_id, session_id), [])[-limit:]
        return [_compact_message(item) for item in items]

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        """Read a user profile from Redis cache or PostgreSQL."""

        cached_profile = self._profile_process_cache.get(user_id)
        if cached_profile and time.time() - cached_profile[0] <= PROFILE_CACHE_TTL_SECONDS:
            return dict(cached_profile[1])

        if self.profile_redis_cache_enabled and self._redis is not None:
            try:
                raw = await self._redis.get(self._profile_key(user_id))
                if raw:
                    profile = dict(json.loads(raw))
                    self._profile_process_cache[user_id] = (time.time(), profile)
                    return dict(profile)
            except Exception:
                self._redis = None

        if self._pool is not None:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("SELECT profile_data FROM navi_user_profiles WHERE user_id = $1", user_id)
            profile = _jsonb_to_dict(row["profile_data"]) if row and row["profile_data"] else {}
        else:
            profile = dict(self._memory_profiles.get(user_id, {}))
        profile = _normalize_education_profile(profile)

        if self.profile_redis_cache_enabled and self._redis is not None:
            try:
                await self._redis.setex(self._profile_key(user_id), PROFILE_CACHE_TTL_SECONDS, json.dumps(profile))
            except Exception:
                self._redis = None
        self._profile_process_cache[user_id] = (time.time(), profile)
        return profile

    async def update_profile(self, *, user_id: str, changes: dict[str, Any], reason: str, session_id: str) -> None:
        """Merge profile changes into the simplified user profile table."""

        if not changes:
            return
        normalized_changes = _normalize_education_profile(changes)
        self._profile_process_cache.pop(user_id, None)

        if self._pool is not None:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", user_id)
                    row = await conn.fetchrow(
                        "SELECT profile_data FROM navi_user_profiles WHERE user_id = $1",
                        user_id,
                    )
                    current = _normalize_education_profile(
                        _jsonb_to_dict(row["profile_data"]) if row and row["profile_data"] else {}
                    )
                    merged = _normalize_education_profile(_merge_profile(current, normalized_changes))
                    if merged == current:
                        return
                    await conn.execute(
                        """
                        INSERT INTO navi_user_profiles(user_id, profile_data, field_count, completeness, last_updated_at)
                        VALUES ($1, $2::jsonb, $3, $4, now())
                        ON CONFLICT (user_id) DO UPDATE SET
                            profile_data = EXCLUDED.profile_data,
                            field_count = EXCLUDED.field_count,
                            completeness = EXCLUDED.completeness,
                            last_updated_at = now(),
                            updated_at = now()
                        """,
                        user_id,
                        json.dumps(merged, ensure_ascii=False),
                        len(merged),
                        min(len(merged) / 20.0, 1.0),
                    )
                    await self._obsolete_profile_conflicts(
                        conn=conn,
                        user_id=user_id,
                        changes=normalized_changes,
                        session_id=session_id,
                    )
        else:
            current = await self.get_profile(user_id)
            merged = _normalize_education_profile(_merge_profile(current, normalized_changes))
            if merged == current:
                return
            self._memory_profiles[user_id] = merged
            self._obsolete_profile_conflicts_in_memory(user_id=user_id, changes=normalized_changes)

        if self._redis is not None:
            try:
                await self._redis.delete(self._profile_key(user_id))
            except Exception:
                self._redis = None

    async def _obsolete_profile_conflicts(self, *, conn: Any, user_id: str, changes: dict[str, Any], session_id: str) -> None:
        signatures = _profile_signatures_from_changes(changes)
        if not signatures:
            return
        for profile_field, value in signatures.items():
            await conn.execute(
                """
                UPDATE navi_memories
                SET is_deleted = TRUE,
                    deleted_at = now(),
                    deleted_reason = $4,
                    updated_at = now()
                WHERE user_id = $1
                  AND memory_type = 'profile_fact'
                  AND NOT is_deleted
                  AND (
                      evidence->>'profile_field' = $2
                      OR $2 = ANY(COALESCE(tags, ARRAY[]::text[]))
                      OR ($2 = 'grade' AND (content LIKE '%初%' OR content LIKE '%高%' OR content LIKE '%年级%'))
                      OR ($2 = 'current_focus' AND (content LIKE '%学习%' OR content LIKE '%学%'))
                  )
                  AND content NOT ILIKE '%' || $3 || '%'
                """,
                user_id,
                profile_field,
                str(value),
                f"obsolete_profile_fact:{profile_field}:{session_id}",
            )

    def _obsolete_profile_conflicts_in_memory(self, *, user_id: str, changes: dict[str, Any]) -> None:
        signatures = _profile_signatures_from_changes(changes)
        if not signatures:
            return
        for item in self._memory_items.get(user_id, []):
            if item.get("is_deleted") or item.get("memory_type") != "profile_fact":
                continue
            field = str((item.get("evidence") or {}).get("profile_field") or "")
            if not field:
                signature = _profile_fact_signature(str(item.get("content") or ""))
                field = signature[0] if signature else ""
            content = str(item.get("content") or "")
            mixed_grade = "grade" in signatures and any(marker in content for marker in ("初", "高", "年级"))
            mixed_focus = "current_focus" in signatures and any(marker in content for marker in ("学习", "学"))
            if (
                field in signatures
                and str(signatures[field]) not in content
                or mixed_grade
                and str(signatures.get("grade")) not in content
                or mixed_focus
                and str(signatures.get("current_focus")) not in content
            ):
                item["is_deleted"] = True
                item["deleted_reason"] = f"obsolete_profile_fact:{field}"

    async def recall(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_importance: int = MIN_MEMORY_IMPORTANCE,
    ) -> list[MemoryHit]:
        """Recall relevant active memories with deterministic text scoring."""

        query = query.strip()
        if not query:
            return []
        await self.apply_memory_decay(user_id=user_id)
        if self._pool is None:
            return self._recall_in_memory(user_id=user_id, query=query, top_k=top_k, min_importance=min_importance)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text AS id, memory_type, content, summary, importance, sensitivity,
                       confidence, COALESCE(tags, ARRAY[]::text[]) AS tags, usage_count, decay_score
                FROM navi_memories
                WHERE user_id = $1
                  AND NOT is_deleted
                  AND importance >= $2
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY
                    CASE
                        WHEN content ILIKE '%' || $3 || '%' THEN 3
                        WHEN array_to_string(COALESCE(tags, ARRAY[]::text[]), ',') ILIKE '%' || $3 || '%' THEN 2
                        ELSE 0
                    END DESC,
                    importance DESC,
                    updated_at DESC
                LIMIT 80
                """,
                user_id,
                min_importance,
                query[:80],
            )
            hits = sorted(
                (_memory_hit_from_row(row, score=_score_memory_text(query=query, row=row)) for row in rows),
                key=lambda hit: hit.score,
                reverse=True,
            )[:top_k]
            ids = [hit.id for hit in hits]
            if ids:
                await conn.execute(
                    "UPDATE navi_memories SET usage_count = usage_count + 1, last_used_at = now(), updated_at = now() WHERE id = ANY($1::uuid[])",
                    ids,
                )
        return hits

    async def recall_layered(self, *, user_id: str, query: str, top_k: int = 8) -> list[MemoryHit]:
        """Recall with stable memory layers for education personalization."""

        query = query.strip()
        if not query:
            return []
        await self.apply_memory_decay(user_id=user_id)
        foundation = await self.recall_foundation(user_id=user_id, query=query, top_k=3)
        relevant = await self.recall(user_id=user_id, query=query, top_k=top_k, min_importance=MIN_MEMORY_IMPORTANCE)
        return _dedupe_memory_hits(foundation + relevant)[:top_k]

    async def apply_memory_decay(self, *, user_id: str) -> None:
        """Apply automatic expiry and lightweight decay without user confirmation."""

        await self.initialize()
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE navi_memories
                    SET is_deleted = TRUE,
                        deleted_at = now(),
                        deleted_reason = COALESCE(deleted_reason, 'expired'),
                        updated_at = now()
                    WHERE user_id = $1
                      AND NOT is_deleted
                      AND expires_at IS NOT NULL
                      AND expires_at <= now()
                    """,
                    user_id,
                )
                await conn.execute(
                    """
                    UPDATE navi_memories
                    SET decay_score = GREATEST(
                            0.15,
                            LEAST(
                                1.0,
                                CASE decay_category
                                    WHEN 'high' THEN 1.0 - EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0 * 0.002
                                    WHEN 'medium' THEN 1.0 - EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0 * 0.010
                                    ELSE 1.0 - EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0 * 0.030
                                END + LEAST(COALESCE(usage_count, 0), 10) * 0.01
                            )
                        ),
                        updated_at = now()
                    WHERE user_id = $1
                      AND NOT is_deleted
                      AND memory_type NOT IN ('profile_fact', 'preference')
                    """,
                    user_id,
                )
        except Exception as exc:
            logger.warning("Memory decay failed; continuing without decay: %s", exc)

    async def remember(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        memory_type: str = "episodic",
        importance: int = 5,
        source_agent: str = "memory_agent",
        summary: str | None = None,
        sensitivity: str = "normal",
        confidence: float = 0.75,
        tags: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> str | None:
        """Write or merge one governed long-term memory."""

        content = _clean_space(content)
        if not _should_remember(content, importance):
            return None
        importance = max(1, min(int(importance), 10))
        confidence = max(0.0, min(float(confidence), 1.0))
        sensitivity = _normalize_sensitivity(sensitivity, content)
        tags = _normalize_tags(tags or _infer_memory_tags(content, memory_type))
        summary = _clean_space(summary or "") or None
        evidence = dict(evidence or {})
        profile_signature = _profile_fact_signature(content) if memory_type == "profile_fact" else None
        if profile_signature:
            evidence.setdefault("profile_field", profile_signature[0])
            tags = _normalize_tags(tags + ["profile_fact", profile_signature[0]])
        content_hash = _memory_content_hash(content)
        decay_category = "high" if importance >= 8 else "low" if importance <= 5 else "medium"

        if self._pool is None:
            incoming_record = {
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "confidence": confidence,
            }
            existing = next(
                (
                    item
                    for item in self._memory_items.get(user_id, [])
                    if item.get("content_hash") == content_hash or _records_refer_to_same_memory(item, incoming_record)
                ),
                None,
            )
            if existing:
                if _memory_record_rank(incoming_record) > _memory_record_rank(existing):
                    existing["content"] = content
                    existing["content_hash"] = content_hash
                existing["importance"] = max(int(existing["importance"]), importance)
                existing["confidence"] = max(float(existing.get("confidence", 0.75)), confidence)
                existing["tags"] = _normalize_tags(list(existing.get("tags", [])) + tags)
                existing["summary"] = existing.get("summary") or summary
                return str(existing["id"])
            if profile_signature:
                self._obsolete_profile_conflicts_in_memory(
                    user_id=user_id,
                    changes={profile_signature[0]: profile_signature[1]},
                )
            memory_id = hashlib.sha1(f"{user_id}:{content}".encode("utf-8")).hexdigest()[:16]
            self._memory_items.setdefault(user_id, []).append(
                {
                    "id": memory_id,
                    "memory_type": memory_type,
                    "content": content,
                    "summary": summary,
                    "importance": importance,
                    "sensitivity": sensitivity,
                    "confidence": confidence,
                    "tags": tags,
                    "evidence": evidence,
                    "usage_count": 0,
                    "content_hash": content_hash,
                    "is_deleted": False,
                }
            )
            return memory_id

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if profile_signature:
                    await self._obsolete_profile_conflicts(
                        conn=conn,
                        user_id=user_id,
                        changes={profile_signature[0]: profile_signature[1]},
                        session_id=session_id,
                    )
                existing_id = await conn.fetchval(
                    "SELECT id FROM navi_memories WHERE user_id = $1 AND content_hash = $2 AND NOT is_deleted",
                    user_id,
                    content_hash,
                )
                if existing_id:
                    await conn.execute(
                        """
                        UPDATE navi_memories
                        SET importance = GREATEST(importance, $1),
                            confidence = GREATEST(COALESCE(confidence, 0.0), $2),
                            tags = (SELECT ARRAY(SELECT DISTINCT unnest(COALESCE(tags, ARRAY[]::text[]) || $3::text[]))),
                            summary = COALESCE(navi_memories.summary, $4),
                            last_used_at = now(),
                            updated_at = now()
                        WHERE id = $5
                        """,
                        importance,
                        confidence,
                        tags,
                        summary,
                        existing_id,
                    )
                    return str(existing_id)

                candidates = await conn.fetch(
                    """
                    SELECT id, content, memory_type, importance, confidence
                    FROM navi_memories
                    WHERE user_id = $1
                      AND memory_type = $2
                      AND NOT is_deleted
                      AND (expires_at IS NULL OR expires_at > now())
                    ORDER BY updated_at DESC
                    LIMIT 30
                    """,
                    user_id,
                    memory_type,
                )
                incoming_record = {
                    "content": content,
                    "memory_type": memory_type,
                    "importance": importance,
                    "confidence": confidence,
                }
                for candidate in candidates:
                    existing_record = {
                        "content": str(candidate["content"]),
                        "memory_type": str(candidate["memory_type"]),
                        "importance": int(candidate["importance"] or 5),
                        "confidence": float(candidate["confidence"] or 0.75),
                    }
                    if not _records_refer_to_same_memory(existing_record, incoming_record):
                        continue
                    should_replace_content = _memory_record_rank(incoming_record) > _memory_record_rank(existing_record)
                    merged_content = content if should_replace_content else str(candidate["content"])
                    merged_hash = _memory_content_hash(merged_content)
                    await conn.execute(
                        """
                        UPDATE navi_memories
                        SET content = $1,
                            content_hash = $2,
                            importance = GREATEST(importance, $3),
                            confidence = GREATEST(COALESCE(confidence, 0.0), $4),
                            tags = (SELECT ARRAY(SELECT DISTINCT unnest(COALESCE(tags, ARRAY[]::text[]) || $5::text[]))),
                            summary = COALESCE(navi_memories.summary, $6),
                            sensitivity = CASE WHEN sensitivity = 'normal' THEN $7 ELSE sensitivity END,
                            last_used_at = now(),
                            updated_at = now()
                        WHERE id = $8
                        """,
                        merged_content,
                        merged_hash,
                        importance,
                        confidence,
                        tags,
                        summary,
                        sensitivity,
                        candidate["id"],
                    )
                    return str(candidate["id"])

                memory_id = await conn.fetchval(
                    """
                    INSERT INTO navi_memories(
                        user_id, memory_type, content, summary, importance, sensitivity, confidence,
                        tags, evidence, decay_category, source_agent, source_session_id, expires_at, content_hash
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text[], $9::jsonb, $10, $11, $12, $13::timestamptz, $14)
                    RETURNING id
                    """,
                    user_id,
                    memory_type,
                    content,
                    summary,
                    importance,
                    sensitivity,
                    confidence,
                    tags,
                    json.dumps(evidence, ensure_ascii=False),
                    decay_category,
                    source_agent,
                    session_id,
                    expires_at,
                    content_hash,
                )
                return str(memory_id)

    async def extract_and_store(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        use_rules: bool = True,
        use_llm: bool = True,
    ) -> None:
        """Extract profile changes and governed memory candidates from a user utterance."""

        profile_changes = _extract_profile_changes(user_text) if use_rules else {}
        memory_records = _extract_memory_records(user_text) if use_rules else []
        extraction = await self.extract_with_llm(user_id=user_id, session_id=session_id, user_text=user_text) if use_llm else None
        if extraction:
            profile_changes = _merge_profile(profile_changes, extraction.get("profile", {}))
            memory_records = _merge_memory_records(memory_records, extraction.get("memories", []))

        if not profile_changes and not memory_records:
            return

        await self.update_profile(user_id=user_id, changes=profile_changes, reason="memory_extraction", session_id=session_id)
        for record in memory_records:
            await self.remember(
                user_id=user_id,
                session_id=session_id,
                content=record["content"],
                memory_type=record["memory_type"],
                importance=record["importance"],
                confidence=record["confidence"],
                sensitivity=record.get("sensitivity", "normal"),
                summary=record.get("summary"),
                tags=record["tags"],
                evidence={"source_text": user_text[:240], "extractor": record.get("extractor", "rules")},
                expires_at=record.get("expires_at"),
            )

    def _should_run_llm_extractor(self, user_text: str) -> bool:
        return (
            self._pool is not None
            and self.extractor_enabled
            and httpx is not None
            and bool(self.extractor_url)
            and not _looks_like_memory_question(user_text)
        )

    async def extract_with_llm(self, *, user_id: str, session_id: str, user_text: str) -> dict[str, Any] | None:
        """Use the configured LLM service to produce structured memory JSON."""

        if not self._should_run_llm_extractor(user_text):
            return None
        payload = {
            "request_id": f"memory-extract-{hashlib.sha1(f'{user_id}:{session_id}:{time.time()}'.encode('utf-8')).hexdigest()[:12]}",
            "trace_id": f"memory-extract-{session_id}",
            "session_id": session_id,
            "user_id": user_id,
            "stream": False,
            "generation_config": {
                "temperature": 0.0,
                "max_tokens": 1200,
                "agent_enabled": False,
            },
            "messages": [
                {"role": "system", "content": _MEMORY_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        }
        raw_text = ""
        try:
            headers: dict[str, str] = {}
            internal_api_key = os.getenv("LLM_INTERNAL_API_KEY", "").strip()
            if internal_api_key:
                headers["X-Internal-API-Key"] = internal_api_key
            async with httpx.AsyncClient(timeout=self.extractor_timeout_seconds) as client:
                response = await client.post(self.extractor_url, json=payload, headers=headers)
                response.raise_for_status()
                raw_text = str((response.json() or {}).get("text") or "")
            extraction = _parse_llm_memory_extraction(raw_text)
            return extraction
        except Exception:  # pragma: no cover - depends on live LLM service
            return None

    async def inspect(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return a complete memory snapshot for the test console."""

        await self.initialize()
        profile, history = await asyncio.gather(
            self.get_profile(user_id),
            self.get_history(user_id=user_id, session_id=session_id, limit=limit),
        )
        memories, concepts = await asyncio.gather(
            self.recall_layered(user_id=user_id, query=query, top_k=limit) if query else self.list_memories(user_id=user_id, limit=limit),
            self.list_concepts(user_id=user_id, query=query, limit=limit),
        )
        return {
            "user_id": user_id,
            "session_id": session_id,
            "profile": profile,
            "history": history,
            "memories": [memory.__dict__ for memory in memories],
            "concepts": [concept.__dict__ for concept in concepts],
            "stats": await self.memory_stats(user_id=user_id),
            "health": await self.health(),
        }

    async def list_memories(self, *, user_id: str, limit: int = 20) -> list[MemoryHit]:
        """List recent active long-term memories."""

        await self.initialize()
        await self.apply_memory_decay(user_id=user_id)
        if self._pool is None:
            items = [item for item in self._memory_items.get(user_id, []) if not item.get("is_deleted")][-limit:]
            return [_memory_hit_from_item(item, score=0.0) for item in reversed(items)]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text AS id, memory_type, content, summary, importance, sensitivity, confidence,
                       COALESCE(tags, ARRAY[]::text[]) AS tags, usage_count
                FROM navi_memories
                WHERE user_id = $1
                  AND NOT is_deleted
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
        return [_memory_hit_from_row(row, score=0.0) for row in rows]

    async def forget_memory(self, *, user_id: str, memory_id: str, reason: str = "manual") -> bool:
        """Soft-delete one memory in the simplified memory table."""

        await self.initialize()
        if self._pool is None:
            for item in self._memory_items.get(user_id, []):
                if str(item.get("id")) == memory_id and not item.get("is_deleted"):
                    item["is_deleted"] = True
                    item["deleted_reason"] = reason
                    return True
            return False

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE navi_memories
                SET is_deleted = TRUE, deleted_at = now(), deleted_reason = $3, updated_at = now()
                WHERE id = $1 AND user_id = $2 AND NOT is_deleted
                """,
                memory_id,
                user_id,
                reason,
            )
        return result.endswith(" 1")

    async def memory_stats(self, *, user_id: str) -> dict[str, Any]:
        """Return lightweight memory counts for the debug console."""

        if self._pool is None:
            active = [item for item in self._memory_items.get(user_id, []) if not item.get("is_deleted")]
            by_type: dict[str, int] = {}
            for item in active:
                by_type[str(item.get("memory_type") or "episodic")] = by_type.get(str(item.get("memory_type") or "episodic"), 0) + 1
            return {"active_memories": len(active), "by_type": by_type}

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_type, count(*)::int AS count
                FROM navi_memories
                WHERE user_id = $1 AND NOT is_deleted AND (expires_at IS NULL OR expires_at > now())
                GROUP BY memory_type
                """,
                user_id,
            )
        by_type = {row["memory_type"]: row["count"] for row in rows}
        return {"active_memories": sum(by_type.values()), "by_type": by_type}

    async def clear_user(self, *, user_id: str, session_id: str | None = None) -> None:
        """Soft-delete user memories and clear cached profile/history."""

        await self.initialize()
        self._memory_profiles.pop(user_id, None)
        self._profile_process_cache.pop(user_id, None)
        self._memory_items.pop(user_id, None)
        self._memory_concepts.pop(user_id, None)
        if session_id:
            self._memory_messages.pop(self._history_key(user_id, session_id), None)
        else:
            history_prefix = self._history_key(user_id, "")
            for key in [key for key in self._memory_messages if key.startswith(history_prefix)]:
                self._memory_messages.pop(key, None)

        if self._redis is not None:
            try:
                await self._redis.delete(self._profile_key(user_id))
                if session_id:
                    await self._redis.delete(self._history_key(user_id, session_id))
                else:
                    history_keys = [key async for key in self._redis.scan_iter(match=self._history_key(user_id, "*"))]
                    if history_keys:
                        await self._redis.delete(*history_keys)
            except Exception:
                self._redis = None

        if self._pool is None:
            return
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE navi_memories SET is_deleted = TRUE, deleted_at = now(), updated_at = now() WHERE user_id = $1",
                    user_id,
                )
                await conn.execute("DELETE FROM navi_user_profiles WHERE user_id = $1", user_id)
                if session_id:
                    await conn.execute("DELETE FROM navi_messages WHERE user_id = $1 AND session_id = $2", user_id, session_id)
                else:
                    await conn.execute("DELETE FROM navi_messages WHERE user_id = $1", user_id)
                await conn.execute("DELETE FROM navi_learning_concepts WHERE user_id = $1", user_id)

    def _recall_in_memory(self, *, user_id: str, query: str, top_k: int, min_importance: int) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for item in self._memory_items.get(user_id, []):
            if item.get("is_deleted") or int(item["importance"]) < min_importance:
                continue
            score = _score_memory_text(query=query, item=item)
            item["usage_count"] = int(item.get("usage_count", 0)) + 1
            hits.append(_memory_hit_from_item(item, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _history_key(user_id: str, session_id: str) -> str:
        return f"navi:history:v2:{_redis_key_component(user_id)}:{_redis_key_component(session_id)}"

    @staticmethod
    def _profile_key(user_id: str) -> str:
        return f"navi:profile:v2:{_redis_key_component(user_id)}"


def _memory_hit_from_row(row: Any, score: float | None = None) -> MemoryHit:
    return MemoryHit(
        id=str(row["id"]),
        memory_type=str(row["memory_type"]),
        content=str(row["content"]),
        summary=row["summary"],
        importance=int(row["importance"]),
        sensitivity=str(row["sensitivity"] or "normal"),
        confidence=float(row["confidence"] or 0.75),
        tags=list(row["tags"] or []),
        usage_count=int(row["usage_count"] or 0),
        score=float(score if score is not None else (row["score"] or 0)),
    )


def _memory_hit_from_item(item: dict[str, Any], *, score: float) -> MemoryHit:
    return MemoryHit(
        id=str(item["id"]),
        memory_type=str(item.get("memory_type") or "episodic"),
        content=str(item.get("content") or ""),
        summary=item.get("summary"),
        importance=int(item.get("importance") or 5),
        sensitivity=str(item.get("sensitivity") or "normal"),
        confidence=float(item.get("confidence") or 0.75),
        tags=list(item.get("tags") or []),
        usage_count=int(item.get("usage_count") or 0),
        score=score,
    )


def _dedupe_memory_hits(hits: list[MemoryHit]) -> list[MemoryHit]:
    best: dict[str, MemoryHit] = {}
    for hit in hits:
        current = best.get(hit.id)
        if current is None or hit.score > current.score:
            best[hit.id] = hit
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def _score_memory_text(*, query: str, row: Any | None = None, item: dict[str, Any] | None = None) -> float:
    """Rank a memory with local text signals instead of vector similarity."""

    source: Any = row if row is not None else item or {}

    def read(name: str, default: Any = None) -> Any:
        try:
            return source[name]
        except Exception:
            if isinstance(source, dict):
                return source.get(name, default)
            return default

    content = str(read("content", "") or "")
    memory_type = str(read("memory_type", "episodic") or "episodic")
    tags = [str(tag) for tag in (read("tags", []) or [])]
    importance = _clamp_float(read("importance", 5), 1.0, 10.0)
    confidence = _clamp_float(read("confidence", 0.75), 0.0, 1.0)
    decay_score = _clamp_float(read("decay_score", 1.0), 0.0, 1.0)

    query_terms = set(_tokenize(query))
    memory_terms = set(_tokenize(content))
    tag_terms = set(term for tag in tags for term in _tokenize(tag))
    all_terms = memory_terms | tag_terms
    overlap = len(query_terms & all_terms) / max(len(query_terms | all_terms), 1)
    substring = 0.0
    if query and query in content:
        substring = 1.0
    elif query_terms and any(term in content for term in query_terms):
        substring = 0.55
    sequence = SequenceMatcher(None, _clean_space(query), _clean_space(content)).ratio() if query and content else 0.0
    type_bonus = 0.08 if memory_type in {"weakness", "learning_state", "plan"} else 0.05 if memory_type in {"profile_fact", "preference"} else 0.0
    return round(
        overlap * 0.34
        + substring * 0.20
        + sequence * 0.18
        + (importance / 10.0) * 0.14
        + confidence * 0.09
        + decay_score * 0.05
        + type_bonus,
        6,
    )


def _concept_from_row(row: Any) -> LearningConcept:
    return LearningConcept(
        id=str(row["id"]),
        subject=str(row["subject"]),
        concept=str(row["concept"]),
        mastery_score=float(row["mastery_score"] or 0.0),
        confidence=float(row["confidence"] or 0.5),
        exposure_count=int(row["exposure_count"] or 0),
        weakness_count=int(row["weakness_count"] or 0),
        correct_count=int(row["correct_count"] or 0),
        incorrect_count=int(row["incorrect_count"] or 0),
        last_event_at=str(row["last_event_at"]) if row["last_event_at"] else None,
        last_practiced_at=str(row["last_practiced_at"]) if row["last_practiced_at"] else None,
        effective_strategy=str(row["effective_strategy"]) if row["effective_strategy"] else None,
        evidence=_jsonb_to_dict(row["evidence"]) if row["evidence"] else {},
    )


def _concept_from_item(item: dict[str, Any]) -> LearningConcept:
    return LearningConcept(
        id=str(item["id"]),
        subject=str(item.get("subject") or "general"),
        concept=str(item.get("concept") or ""),
        mastery_score=float(item.get("mastery_score") or 0.0),
        confidence=float(item.get("confidence") or 0.5),
        exposure_count=int(item.get("exposure_count") or 0),
        weakness_count=int(item.get("weakness_count") or 0),
        correct_count=int(item.get("correct_count") or 0),
        incorrect_count=int(item.get("incorrect_count") or 0),
        last_event_at=item.get("last_event_at"),
        last_practiced_at=item.get("last_practiced_at"),
        effective_strategy=item.get("effective_strategy"),
        evidence=dict(item.get("evidence") or {}),
    )


def _concept_query_overlap(item: dict[str, Any], query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    text = f"{item.get('subject', '')} {item.get('concept', '')}"
    terms = set(_tokenize(text))
    return len(terms & query_terms) / max(len(query_terms), 1)



def _current_date_context() -> str:
    now = datetime.now(timezone(timedelta(hours=8)))
    return f"{now:%Y-%m-%d} {now:%A} Asia/Shanghai"


def _compact_message(item: dict[str, Any]) -> dict[str, str]:
    return {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}


def _safe_zoneinfo(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("Asia/Shanghai")


def _jsonb_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _normalize_education_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep legacy profile keys while maintaining education-specific sections."""

    if not profile:
        return {}
    normalized = dict(profile)
    identity = dict(normalized.get("identity") or {})
    learning_profile = dict(normalized.get("learning_profile") or {})
    preferences = dict(normalized.get("preferences") or {})
    subject_states = dict(normalized.get("subject_states") or {})
    interaction_style = dict(normalized.get("interaction_style") or {})
    risk_flags = dict(normalized.get("risk_flags") or {})

    if normalized.get("nickname"):
        identity["nickname"] = normalized["nickname"]
    if normalized.get("grade"):
        identity["grade"] = normalized["grade"]
    if normalized.get("current_focus"):
        learning_profile["current_focus"] = normalized["current_focus"]
    if normalized.get("learning_state"):
        learning_profile["learning_state"] = normalized["learning_state"]
    if normalized.get("subjects"):
        learning_profile["subjects"] = normalized["subjects"]

    subjects = list(learning_profile.get("subjects") or normalized.get("subjects") or [])
    current_focus = str(learning_profile.get("current_focus") or normalized.get("current_focus") or "")
    for subject in subjects:
        key = str(subject)
        state = dict(subject_states.get(key) or {})
        if current_focus:
            state.setdefault("current_focus", current_focus)
        subject_states[key] = state

    enjoy_text = " ".join(str(item) for item in preferences.get("topics_enjoy", []) if item)
    avoid_text = " ".join(str(item) for item in preferences.get("topics_avoid", []) if item)
    if enjoy_text and not interaction_style.get("preferred_explanation"):
        if any(marker in enjoy_text for marker in ("\u751f\u6d3b", "\u4f8b\u5b50", "\u6848\u4f8b")):
            interaction_style["preferred_explanation"] = "life_examples"
        elif any(marker in enjoy_text for marker in ("\u6b65\u9aa4", "\u4e00\u6b65\u4e00\u6b65")):
            interaction_style["preferred_explanation"] = "step_by_step"
    if avoid_text and not interaction_style.get("avoid"):
        interaction_style["avoid"] = avoid_text[:80]

    if identity:
        normalized["identity"] = identity
        normalized.setdefault("nickname", identity.get("nickname"))
        normalized.setdefault("grade", identity.get("grade"))
    if learning_profile:
        normalized["learning_profile"] = learning_profile
        normalized.setdefault("current_focus", learning_profile.get("current_focus"))
        normalized.setdefault("learning_state", learning_profile.get("learning_state"))
        normalized.setdefault("subjects", learning_profile.get("subjects"))
    normalized["preferences"] = preferences
    normalized["subject_states"] = subject_states
    normalized["interaction_style"] = interaction_style
    normalized["risk_flags"] = risk_flags
    return {key: value for key, value in normalized.items() if value not in (None, "", [], {})}


def _profile_for_query(profile: dict[str, Any], query: str) -> dict[str, Any]:
    """Return only profile fields useful for this turn without mutating storage."""

    if not profile or not query.strip():
        return profile
    filtered: dict[str, Any] = {}
    for key in ("nickname", "grade", "identity", "interaction_style", "risk_flags"):
        if key in profile:
            filtered[key] = profile[key]
    preferences = _preferences_for_query(profile.get("preferences"), query)
    if preferences:
        filtered["preferences"] = preferences

    learning_profile_value = profile.get("learning_profile")
    learning_profile: dict[str, Any] = learning_profile_value if isinstance(learning_profile_value, dict) else {}
    filtered_learning: dict[str, Any] = {}

    subjects = list(learning_profile.get("subjects") or profile.get("subjects") or [])
    relevant_subjects = [item for item in subjects if _profile_value_relevant(item, query)]
    if relevant_subjects:
        filtered_learning["subjects"] = relevant_subjects
        filtered["subjects"] = relevant_subjects

    current_focus = learning_profile.get("current_focus") or profile.get("current_focus")
    if current_focus and _profile_value_relevant(current_focus, query):
        filtered_learning["current_focus"] = current_focus
        filtered["current_focus"] = current_focus

    learning_state = learning_profile.get("learning_state") or profile.get("learning_state")
    if learning_state and _profile_value_relevant(learning_state, query):
        filtered_learning["learning_state"] = learning_state
        filtered["learning_state"] = learning_state

    subject_states_value = profile.get("subject_states")
    subject_states: dict[str, Any] = subject_states_value if isinstance(subject_states_value, dict) else {}
    relevant_states: dict[str, Any] = {}
    for subject, state in subject_states.items():
        if _profile_value_relevant(subject, query) or _profile_value_relevant(state, query):
            relevant_states[str(subject)] = state
    if relevant_states:
        filtered["subject_states"] = relevant_states

    if filtered_learning:
        filtered["learning_profile"] = filtered_learning
    return filtered


def _preferences_for_query(preferences: Any, query: str) -> dict[str, Any]:
    if not isinstance(preferences, dict):
        return {}
    filtered: dict[str, Any] = {}
    style_markers = ("生活", "例子", "案例", "步骤", "一步", "少公式", "不要公式", "口语", "简单", "详细")
    for key, value in preferences.items():
        if isinstance(value, list):
            kept = [
                item
                for item in value
                if _profile_value_relevant(item, query) or any(marker in str(item) for marker in style_markers)
            ]
            if kept:
                filtered[key] = kept
        elif _profile_value_relevant(value, query) or any(marker in str(value) for marker in style_markers):
            filtered[key] = value
    return filtered


def _foundation_memory_relevant(memory: MemoryHit, query: str) -> bool:
    if _looks_like_memory_question(query):
        return True
    if memory.memory_type not in {"profile_fact", "preference"}:
        return True
    text = " ".join([memory.content, memory.summary or "", " ".join(memory.tags)])
    return _profile_value_relevant(text, query) or _looks_like_style_preference(text)


def _looks_like_style_preference(text: str) -> bool:
    return any(
        marker in text
        for marker in ("生活", "例子", "案例", "步骤", "一步", "少公式", "不要公式", "口语", "简单", "详细")
    )


def _profile_value_relevant(value: Any, query: str) -> bool:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "")
    if not text.strip():
        return False
    query_domains = _profile_domains(query)
    value_domains = _profile_domains(text)
    if query_domains and value_domains:
        return bool(query_domains & value_domains)
    if query_domains and not value_domains:
        return False
    query_terms = set(_content_terms(query))
    value_terms = set(_content_terms(text))
    return len(query_terms & value_terms) >= 2


def _profile_domains(text: str) -> set[str]:
    lowered = text.lower()
    domains: set[str] = set()
    markers = {
        "math": ("数学", "函数", "方程", "几何", "二次", "一次", "斜率", "顶点", "坐标", "抛物线"),
        "programming": ("python", "编程", "代码", "程序", "开发", "算法"),
        "english": ("英语", "单词", "语法", "作文", "阅读理解", "听力"),
        "physics": ("物理", "力学", "电路", "速度", "压强", "浮力"),
        "chemistry": ("化学", "元素", "反应", "酸碱", "化学方程"),
        "chinese": ("语文", "古诗", "文言文", "阅读", "作文"),
    }
    for domain, domain_markers in markers.items():
        if any(marker in lowered for marker in domain_markers):
            domains.add(domain)
    return domains


def _content_terms(text: str) -> list[str]:
    stop_chars = set("我你他她它的是了在和与或也就都很不太吗呢啊把被用给讲学")
    ascii_terms = re.findall(r"[A-Za-z0-9_]+", text.lower())
    cjk_terms = [char for char in re.findall(r"[\u4e00-\u9fff]", text) if char not in stop_chars]
    return ascii_terms + cjk_terms


def _profile_signatures_from_changes(changes: dict[str, Any]) -> dict[str, Any]:
    signatures: dict[str, Any] = {}
    identity_value = changes.get("identity")
    identity: dict[str, Any] = identity_value if isinstance(identity_value, dict) else {}
    learning_profile_value = changes.get("learning_profile")
    learning_profile: dict[str, Any] = learning_profile_value if isinstance(learning_profile_value, dict) else {}
    for key in ("nickname", "grade"):
        value = changes.get(key) or identity.get(key)
        if value not in (None, "", [], {}):
            signatures[key] = value
    current_focus = changes.get("current_focus") or learning_profile.get("current_focus")
    if current_focus:
        signatures["current_focus"] = current_focus
    return signatures


def _merge_profile(current: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in changes.items():
        if value in (None, "", [], {}):
            continue
        merged[key] = _merge_profile_value(merged.get(key), value)
    return merged


def _merge_profile_value(current: Any, value: Any) -> Any:
    if isinstance(current, dict) and isinstance(value, dict):
        merged = dict(current)
        for key, nested in value.items():
            if nested not in (None, "", [], {}):
                merged[key] = _merge_profile_value(merged.get(key), nested)
        return merged
    if isinstance(current, list) and isinstance(value, list):
        seen = set(str(item) for item in current)
        return list(current) + [item for item in value if str(item) not in seen]
    return value


def _parse_llm_memory_extraction(raw_text: str) -> dict[str, Any] | None:
    body = _extract_json_object(raw_text)
    if not body:
        return None
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    profile_value = decoded.get("profile")
    profile: dict[str, Any] = profile_value if isinstance(profile_value, dict) else {}
    memories_value = decoded.get("memories")
    memories: list[Any] = memories_value if isinstance(memories_value, list) else []
    records = []
    for item in memories:
        if isinstance(item, dict):
            record = _normalize_llm_memory_record(item)
            if record:
                records.append(record)
    return {"profile": _sanitize_profile_changes(profile), "memories": records[:8]}


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]


def _normalize_llm_memory_record(item: dict[str, Any]) -> dict[str, Any] | None:
    content = _clean_space(str(item.get("content") or ""))
    if not content or _looks_like_memory_question(content):
        return None
    importance = _clamp_int(item.get("importance", 5), 1, 10)
    if not _should_remember(content, importance):
        return None
    memory_type = str(item.get("memory_type") or "episodic").strip().lower()
    if memory_type not in {"profile_fact", "learning_state", "preference", "learning_event", "plan", "weakness", "episodic"}:
        memory_type = "episodic"
    confidence = _clamp_float(item.get("confidence", 0.7), 0.0, 1.0)
    sensitivity = _normalize_sensitivity(str(item.get("sensitivity") or "normal"), content)
    tags = _normalize_tags(list(item.get("tags") or []) + _infer_memory_tags(content, memory_type))
    summary = _clean_space(str(item.get("summary") or "")) or None
    expires_at = item.get("expires_at")
    return {
        "content": content[:240],
        "memory_type": memory_type,
        "importance": importance,
        "confidence": confidence,
        "sensitivity": sensitivity,
        "tags": tags,
        "summary": summary[:160] if summary else None,
        "expires_at": str(expires_at) if expires_at else None,
        "extractor": "llm",
    }


def _sanitize_profile_changes(profile: dict[str, Any]) -> dict[str, Any]:
    allowed = {"nickname", "current_focus", "preferences", "learning_state", "grade", "subjects"}
    cleaned: dict[str, Any] = {}
    for key, value in profile.items():
        if key not in allowed or value in (None, "", [], {}):
            continue
        if key == "nickname" and _looks_like_unknown_answer(str(value)):
            continue
        cleaned[key] = value
    return cleaned


def _merge_memory_records(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in left + right:
        normalized = _normalize_llm_memory_record(record) if record.get("extractor") == "llm" else dict(record)
        if not normalized:
            continue
        key = _find_similar_memory_key(merged, normalized) or _memory_content_hash(str(normalized["content"]))
        current = merged.get(key)
        if current is None:
            merged[key] = normalized
            continue
        _merge_into_memory_record(current, normalized)
    return list(merged.values())[:10]


def _find_similar_memory_key(merged: dict[str, dict[str, Any]], record: dict[str, Any]) -> str | None:
    exact_key = _memory_content_hash(str(record["content"]))
    if exact_key in merged:
        return exact_key
    for key, candidate in merged.items():
        if _records_refer_to_same_memory(candidate, record):
            return key
    return None


def _merge_into_memory_record(current: dict[str, Any], normalized: dict[str, Any]) -> None:
    if _memory_record_rank(normalized) > _memory_record_rank(current):
        current["content"] = normalized["content"]
        current["memory_type"] = normalized.get("memory_type", current.get("memory_type"))
        current["extractor"] = normalized.get("extractor", current.get("extractor"))
        current["expires_at"] = normalized.get("expires_at") or current.get("expires_at")
        current["importance"] = max(int(current.get("importance", 5)), int(normalized.get("importance", 5)))
    current["importance"] = max(int(current.get("importance", 5)), int(normalized.get("importance", 5)))
    current["confidence"] = max(float(current.get("confidence", 0.7)), float(normalized.get("confidence", 0.7)))
    current["tags"] = _normalize_tags(list(current.get("tags", [])) + list(normalized.get("tags", [])))
    current["summary"] = current.get("summary") or normalized.get("summary")
    if current.get("sensitivity") == "normal":
        current["sensitivity"] = normalized.get("sensitivity", "normal")


def _records_refer_to_same_memory(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("memory_type")) != str(right.get("memory_type")):
        return False
    left_fact_key = _profile_fact_signature_from_record(left) or _profile_fact_key(left)
    right_fact_key = _profile_fact_signature_from_record(right) or _profile_fact_key(right)
    if left_fact_key and left_fact_key == right_fact_key:
        return True
    return _memory_text_similarity(str(left.get("content", "")), str(right.get("content", ""))) >= 0.55


def _memory_record_rank(record: dict[str, Any]) -> tuple[int, float, int]:
    return (
        int(record.get("importance", 5)),
        float(record.get("confidence", 0.7)),
        len(str(record.get("content", ""))),
    )


def _memory_text_similarity(left: str, right: str) -> float:
    left = _clean_space(left)
    right = _clean_space(right)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    ratio = SequenceMatcher(None, left, right).ratio()
    left_terms = set(_memory_similarity_terms(left))
    right_terms = set(_memory_similarity_terms(right))
    overlap = 0.0
    if left_terms and right_terms:
        overlap = len(left_terms & right_terms) / len(left_terms | right_terms)
    return max(ratio, overlap)


def _memory_similarity_terms(content: str) -> list[str]:
    stop_chars = {"我", "的", "是", "在", "了", "和", "用", "讲", "学", "习", "总", "总是"}
    return [token for token in _tokenize(content) if token not in stop_chars]


def _profile_fact_signature_from_record(record: dict[str, Any]) -> tuple[str, str] | None:
    if str(record.get("memory_type")) != "profile_fact":
        return None
    return _profile_fact_signature(str(record.get("content") or ""))


def _profile_fact_signature(content: str) -> tuple[str, str] | None:
    content = _clean_space(content)
    nickname = _first_match(
        content,
        [
            r"(?:\u6211\u53eb|\u53eb\u6211|\u6211\u7684\u540d\u5b57\u662f|\u6635\u79f0[:\uff1a]?\s*)([\u4e00-\u9fffA-Za-z0-9_]{1,16})",
            r"(?:my name is|call me)\s+([A-Za-z0-9_]{1,24})",
        ],
    )
    if nickname and not _looks_like_unknown_answer(nickname):
        return ("nickname", nickname)
    grade = _first_match(
        content,
        [
            r"(\u521d[\u4e00\u4e8c\u4e09123]|\u9ad8[\u4e00\u4e8c\u4e09123]|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d1-6]\u5e74\u7ea7|\u5c0f\u5b66[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d123456]\u5e74\u7ea7?)",
        ],
    )
    if grade:
        return ("grade", grade)
    focus = _first_match(
        content,
        [
            r"(?:\u6b63\u5728\u5b66\u4e60|\u6700\u8fd1\u5728\u5b66|\u5f53\u524d\u5b66\u4e60\u7126\u70b9[:\uff1a]?)([^,\uff0c\u3002\uff01\uff1f;；]{1,40})",
        ],
    )
    if focus and not _looks_like_unknown_answer(focus):
        return ("current_focus", focus)
    return None


def _profile_fact_key(record: dict[str, Any]) -> tuple[str, str] | None:
    if str(record.get("memory_type")) != "profile_fact":
        return None
    content = _clean_space(str(record.get("content", "")))
    nickname = _first_match(content, [r"(?:我叫|叫我|昵称是|学生昵称[:：]\s*)([\u4e00-\u9fffA-Za-z0-9_]{1,12})"])
    if nickname and not _looks_like_unknown_answer(nickname):
        return ("nickname", nickname)
    grade = _first_match(content, [r"(初[一二三123]|高[一二三123]|[一二三四五六1-6]年级|小学[一二三四五六123456]年级?)"])
    if grade:
        return ("grade", grade)
    return None


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def _extract_profile_changes(text: str) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    nickname = _first_match(text, [r"(?:我叫|叫我|我的名字是)([\u4e00-\u9fffA-Za-z0-9_]{1,12})"])
    if nickname and not _looks_like_unknown_answer(nickname):
        changes["nickname"] = nickname
    grade = _first_match(text, [r"(初[一二三123]|高[一二三123]|[一二三四五六1-6]年级|小学[一二三四五六123456]年级?)"])
    if grade:
        changes["grade"] = grade
    subjects = _infer_profile_subjects(text)
    if subjects:
        changes["subjects"] = subjects
    likes = _first_match(
        text,
        [r"(?:我喜欢(?!的|什么|啥|哪种|哪个|哪些)|我爱(?!的)|我想学|我对)([^，。！？；;,.!?\n]{1,30})(?:感兴趣)?"],
    )
    if likes:
        changes["preferences"] = {"topics_enjoy": [_strip_interest_suffix(likes)]}
    avoids = _first_match(
        text,
        [r"(?:我不喜欢(?!的|什么|啥|哪种|哪个|哪些)|我讨厌|我害怕)([^，。！？；;,.!?\n]{1,30})"],
    )
    if avoids:
        changes["preferences"] = {"topics_avoid": [avoids.strip()]}
    focus = _first_match(
        text,
        [
            r"(?:我(?:最近|现在)?(?:正在)?(?:在)?(?:学|学习)|最近(?:在)?(?:学|学习)|正在(?:学|学习))([^，。！？；;,.!?\n]{1,40})"
        ],
    )
    if focus and not _looks_like_unknown_answer(focus):
        changes["current_focus"] = f"正在学习{focus.strip()}"
    return changes


def _extract_memory_records(text: str) -> list[dict[str, Any]]:
    text = _clean_space(text)
    if not text or len(text) < 6:
        return []
    boundary = "[^\uff0c\u3002\uff01\uff1f\uff1b;,.!?\\n]"
    patterns = [
        ("(?:\u6211(?:\u6700\u8fd1|\u73b0\u5728)?(?:\u6b63\u5728)?(?:\u5728)?(?:\u5b66|\u5b66\u4e60)|\u6700\u8fd1(?:\u5728)?(?:\u5b66|\u5b66\u4e60)|\u6b63\u5728(?:\u5b66|\u5b66\u4e60))" + boundary + "{1,40}", 6, "learning_state"),
        ("(?:\u6211)?(?:\u603b\u662f|\u4e00\u76f4|\u7ecf\u5e38|\u8001\u662f|\u6709\u70b9|\u6bd4\u8f83)?(?:\u4e0d\u592a\u4f1a|\u4e0d\u4f1a|\u4e0d\u61c2|\u6ca1\u5f04\u61c2|\u641e\u4e0d\u61c2|\u770b\u4e0d\u61c2|\u5361\u5728|\u5361\u4f4f|\u8584\u5f31|\u5f31\u5728)" + boundary + "{1,40}", 8, "weakness"),
        ("(?:\u6211\u559c\u6b22(?!\u7684|\u4ec0\u4e48|\u5565|\u54ea\u79cd|\u54ea\u4e2a|\u54ea\u4e9b)|\u6211\u7231(?!\u7684)|\u6211\u60f3\u5b66|\u6211\u4e0d\u559c\u6b22(?!\u7684|\u4ec0\u4e48|\u5565|\u54ea\u79cd|\u54ea\u4e2a|\u54ea\u4e9b)|\u6211\u8ba8\u538c|\u6211\u5bb3\u6015)" + boundary + "{1,40}", 6, "preference"),
        ("(?:\u6211\u8981|\u6211\u51c6\u5907|\u4e0b\u5468|\u660e\u5929|\u4eca\u5929)" + boundary + "{2,50}", 5, "plan"),
        ("(?:\u8003\u8bd5|\u4f5c\u4e1a|\u8bfe\u7a0b|\u8001\u5e08|\u5b66\u6821|\u9898\u76ee|\u77e5\u8bc6\u70b9)" + boundary + "{0,50}", 5, "learning_event"),
        ("(?:\u6211\u53eb|\u53eb\u6211|\u6211\u7684\u540d\u5b57\u662f)" + boundary + "{1,20}", 8, "profile_fact"),
        ("(?:初[一二三123]|高[一二三123]|[一二三四五六1-6]年级|小学[一二三四五六123456]年级?)", 7, "profile_fact"),
    ]
    records: list[dict[str, Any]] = []
    for pattern, importance, memory_type in patterns:
        for match in re.findall(pattern, text):
            content = _clean_space(match)
            if content and not _looks_like_memory_question(content):
                records.append(
                    {
                        "content": content,
                        "importance": importance,
                        "memory_type": memory_type,
                        "confidence": 0.85 if importance >= 8 else 0.75,
                        "tags": _infer_memory_tags(content, memory_type),
                    }
                )
    if not records and any(word in text for word in ("\u4e0d\u4f1a", "\u4e0d\u61c2", "\u9519\u9898", "\u590d\u4e60", "\u9884\u4e60")):
        records.append(
            {
                "content": text[:120],
                "importance": 5,
                "memory_type": "learning_event",
                "confidence": 0.65,
                "tags": _infer_memory_tags(text, "learning_event"),
            }
        )
    return records[:5]




def _memory_content_hash(content: str) -> str:
    return hashlib.sha1(_clean_space(content).lower().encode("utf-8")).hexdigest()


def _normalize_sensitivity(value: str, content: str) -> str:
    normalized = (value or "normal").strip().lower()
    if normalized not in {"public", "normal", "sensitive", "private"}:
        normalized = "normal"
    sensitive_markers = ("\u8eab\u4efd\u8bc1", "\u624b\u673a\u53f7", "\u7535\u8bdd", "\u4f4f\u5740", "\u5bb6\u5ead\u4f4f\u5740", "\u94f6\u884c\u5361", "\u5bc6\u7801")
    if any(marker in content for marker in sensitive_markers):
        return "sensitive"
    return normalized


def _normalize_tags(tags: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = _clean_space(str(tag)).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value[:32])
    return cleaned[:12]


def _infer_memory_tags(content: str, memory_type: str) -> list[str]:
    tags = [memory_type]
    subject_markers = {
        "math": ("\u6570\u5b66", "\u51fd\u6570", "\u65b9\u7a0b", "\u51e0\u4f55", "\u4e8c\u6b21", "\u5bfc\u6570"),
        "english": ("\u82f1\u8bed", "\u5355\u8bcd", "\u8bed\u6cd5", "\u4f5c\u6587"),
        "physics": ("\u7269\u7406", "\u529b\u5b66", "\u7535\u8def", "\u901f\u5ea6"),
        "chemistry": ("\u5316\u5b66", "\u5143\u7d20", "\u53cd\u5e94", "\u65b9\u7a0b\u5f0f"),
    }
    for tag, markers in subject_markers.items():
        if any(marker in content for marker in markers):
            tags.append(tag)
    if any(marker in content for marker in ("\u4e0d\u4f1a", "\u4e0d\u61c2", "\u8584\u5f31", "\u9519\u9898", "\u5361")):
        tags.append("weakness")
    if any(marker in content for marker in ("\u559c\u6b22", "\u8ba8\u538c", "\u5bb3\u6015", "\u504f\u597d")):
        tags.append("preference")
    return _normalize_tags(tags)


def _infer_profile_subjects(text: str) -> list[str]:
    subjects: list[str] = []
    subject_markers = {
        "数学": ("数学", "函数", "方程", "几何", "二次", "导数", "斜率"),
        "英语": ("英语", "单词", "语法", "作文"),
        "物理": ("物理", "力学", "电路", "速度"),
        "化学": ("化学", "元素", "反应", "方程式"),
    }
    for subject, markers in subject_markers.items():
        if any(marker in text for marker in markers):
            subjects.append(subject)
    return subjects


def _extract_concept_signals(user_text: str, assistant_text: str = "") -> list[dict[str, Any]]:
    """Extract lightweight concept mastery signals from a student turn."""

    text = _clean_space(user_text)
    if not text:
        return []
    subject = _infer_concept_subject(text)
    concepts = _infer_concepts(text)
    if not concepts:
        return []

    is_weakness = any(marker in text for marker in ("不会", "不懂", "没弄懂", "搞不懂", "看不懂", "卡住", "薄弱", "错题", "老是错", "总是"))
    is_progress = any(marker in text for marker in ("会了", "懂了", "明白了", "做对了", "掌握", "学会"))
    is_plan = any(marker in text for marker in ("复习", "预习", "练习", "作业", "考试", "明天", "下周"))
    if is_weakness:
        event_type = "weakness"
        delta = -0.08
        initial = 0.28
        confidence = 0.82
    elif is_progress:
        event_type = "progress"
        delta = 0.08
        initial = 0.52
        confidence = 0.74
    elif is_plan:
        event_type = "practice"
        delta = 0.02
        initial = 0.42
        confidence = 0.62
    else:
        event_type = "exposure"
        delta = 0.0
        initial = 0.4
        confidence = 0.58
    effective_strategy = _infer_effective_strategy(user_text, assistant_text)

    signals: list[dict[str, Any]] = []
    for concept in concepts[:4]:
        signals.append(
            {
                "subject": subject,
                "concept": concept,
                "event_type": event_type,
                "mastery_delta": delta,
                "initial_mastery": initial,
                "confidence": confidence,
                "effective_strategy": effective_strategy,
                "evidence": {
                    "source": "rules",
                    "source_text": text[:240],
                    "assistant_preview": assistant_text[:160],
                    "effective_strategy": effective_strategy,
                },
            }
        )
    return signals


def _infer_concept_subject(text: str) -> str:
    subject_markers = {
        "数学": ("数学", "函数", "方程", "几何", "斜率", "顶点式", "二次", "一次", "导数", "坐标"),
        "英语": ("英语", "单词", "语法", "作文", "阅读理解", "听力"),
        "物理": ("物理", "力学", "电路", "速度", "压强", "浮力"),
        "化学": ("化学", "元素", "反应", "方程式", "酸碱"),
        "语文": ("语文", "作文", "阅读", "古诗", "文言文"),
    }
    for subject, markers in subject_markers.items():
        if any(marker in text for marker in markers):
            return subject
    return "通用"


def _infer_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    known = [
        "一次函数图像",
        "一次函数",
        "二次函数",
        "函数图像",
        "斜率",
        "顶点式",
        "方程",
        "几何",
        "导数",
        "英语语法",
        "单词",
        "作文",
        "阅读理解",
        "电路",
        "力学",
        "化学方程式",
    ]
    for concept in known:
        if concept in text:
            concepts.append(concept)
    learned = _first_match(
        text,
        [
            r"(?:最近|现在)?(?:正在)?(?:在)?(?:学|学习|复习|预习)([^，。！？；;,.!?\n]{1,24})",
            r"(?:不会|不懂|看不懂|卡在|薄弱)([^，。！？；;,.!?\n]{1,24})",
        ],
    )
    if learned:
        cleaned = _clean_concept_name(learned)
        if cleaned:
            concepts.append(cleaned)
    deduped: list[str] = []
    seen: set[str] = set()
    for concept in concepts:
        normalized = _clean_concept_name(concept)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _clean_concept_name(value: str) -> str:
    value = _clean_space(value)
    value = re.sub(r"^(一下|一下子|这个|那个|关于)", "", value)
    value = re.sub(r"(怎么做|怎么理解|怎么办|这一块|这部分)$", "", value)
    return value.strip(" ：:，。！？；;,.!?")[:32]


def _infer_effective_strategy(user_text: str, assistant_text: str) -> str | None:
    text = f"{user_text} {assistant_text}"
    if any(marker in text for marker in ("\u751f\u6d3b", "\u4f8b\u5b50", "\u6bd4\u55bb", "\u573a\u666f")):
        return "life_examples"
    if any(marker in text for marker in ("\u4e00\u6b65", "\u6b65\u9aa4", "\u5148", "\u518d")):
        return "step_by_step"
    if any(marker in text for marker in ("\u516c\u5f0f", "\u5b9a\u4e49", "\u63a8\u5bfc")):
        return "formula_guided"
    if any(marker in text for marker in ("\u4f60\u8bd5\u8bd5", "\u60f3\u4e00\u60f3", "\u68c0\u67e5")):
        return "guided_socratic"
    return None


def _should_remember(content: str, importance: int) -> bool:
    if importance < MIN_MEMORY_IMPORTANCE:
        return False
    if len(content.strip()) < 4:
        return False
    return content.strip() not in {"好的", "嗯", "哈哈", "谢谢", "你好"}


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean_space(match.group(1))
    return None


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_interest_suffix(text: str) -> str:
    return re.sub(r"(感兴趣|有兴趣)$", "", text).strip()


def _looks_like_unknown_answer(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"什么", "啥", "哪一科", "哪个", "哪些"} or "什么" in stripped


def _looks_like_memory_question(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "记得",
            "记住",
            "叫什么",
            "我的名字",
            "我在学什么",
            "我学什么",
            "最近学什么",
            "最近在学什么",
            "学什么",
            "喜欢什么",
        )
    )




def _tokenize(text: str) -> list[str]:
    ascii_terms = re.findall(r"[A-Za-z0-9_]+", text.lower())
    cjk_terms = re.findall(r"[\u4e00-\u9fff]", text)
    return ascii_terms + cjk_terms


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS navi_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    input_type TEXT DEFAULT 'voice',
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_navi_messages_session ON navi_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_navi_messages_user ON navi_messages(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS navi_user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT UNIQUE NOT NULL,
    profile_data JSONB NOT NULL DEFAULT '{}',
    field_count INT DEFAULT 0,
    completeness FLOAT DEFAULT 0.0,
    last_updated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_navi_profile_user ON navi_user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_navi_profile_data ON navi_user_profiles USING gin(profile_data);

CREATE TABLE IF NOT EXISTS navi_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    importance INT DEFAULT 5,
    sensitivity TEXT DEFAULT 'normal',
    confidence FLOAT DEFAULT 0.75,
    tags TEXT[] DEFAULT '{}',
    evidence JSONB NOT NULL DEFAULT '{}',
    decay_score FLOAT DEFAULT 1.0,
    decay_category TEXT DEFAULT 'medium',
    source_agent TEXT,
    source_session_id TEXT,
    content_hash TEXT,
    usage_count INT DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ,
    deleted_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE navi_memories ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.75;
ALTER TABLE navi_memories ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}';
ALTER TABLE navi_memories ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '{}';
ALTER TABLE navi_memories ADD COLUMN IF NOT EXISTS source_session_id TEXT;
ALTER TABLE navi_memories ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE navi_memories ADD COLUMN IF NOT EXISTS usage_count INT DEFAULT 0;
ALTER TABLE navi_memories ADD COLUMN IF NOT EXISTS deleted_reason TEXT;
UPDATE navi_memories SET confidence = 0.75 WHERE confidence IS NULL;
UPDATE navi_memories SET tags = '{}' WHERE tags IS NULL;
UPDATE navi_memories SET evidence = '{}' WHERE evidence IS NULL;
UPDATE navi_memories SET usage_count = 0 WHERE usage_count IS NULL;
UPDATE navi_memories SET content_hash = encode(digest(lower(trim(content)), 'sha1'), 'hex') WHERE content_hash IS NULL;

CREATE INDEX IF NOT EXISTS idx_navi_memories_user_type ON navi_memories(user_id, memory_type) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_navi_memories_importance ON navi_memories(user_id, importance DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_navi_memories_active_expires ON navi_memories(user_id, expires_at) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_navi_memories_tags ON navi_memories USING gin(tags) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_navi_memories_user_hash_active ON navi_memories(user_id, content_hash) WHERE NOT is_deleted AND content_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_navi_memories_fts ON navi_memories USING gin(to_tsvector('simple', content)) WHERE NOT is_deleted;

CREATE TABLE IF NOT EXISTS navi_learning_concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    concept TEXT NOT NULL,
    mastery_score FLOAT DEFAULT 0.4,
    confidence FLOAT DEFAULT 0.5,
    exposure_count INT DEFAULT 0,
    weakness_count INT DEFAULT 0,
    correct_count INT DEFAULT 0,
    incorrect_count INT DEFAULT 0,
    last_event_at TIMESTAMPTZ,
    last_practiced_at TIMESTAMPTZ,
    effective_strategy TEXT,
    evidence JSONB NOT NULL DEFAULT '{}',
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, subject, concept)
);
ALTER TABLE navi_learning_concepts ADD COLUMN IF NOT EXISTS correct_count INT DEFAULT 0;
ALTER TABLE navi_learning_concepts ADD COLUMN IF NOT EXISTS incorrect_count INT DEFAULT 0;
ALTER TABLE navi_learning_concepts ADD COLUMN IF NOT EXISTS last_practiced_at TIMESTAMPTZ;
ALTER TABLE navi_learning_concepts ADD COLUMN IF NOT EXISTS effective_strategy TEXT;
CREATE INDEX IF NOT EXISTS idx_navi_learning_concepts_user ON navi_learning_concepts(user_id, updated_at DESC) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_navi_learning_concepts_subject ON navi_learning_concepts(user_id, subject, mastery_score) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_navi_learning_concepts_evidence ON navi_learning_concepts USING gin(evidence) WHERE NOT is_deleted;

"""
