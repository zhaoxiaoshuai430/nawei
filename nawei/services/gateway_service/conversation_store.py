"""Bounded in-memory conversation history for the Gateway service."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

from app_protocols import Message


@dataclass
class _SessionEntry:
    messages: list[Message] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = field(default_factory=time.monotonic)
    users: int = 0


class ConversationSession:
    """A locked view of one session's conversation history."""

    def __init__(self, entry: _SessionEntry, max_messages: int) -> None:
        self._entry = entry
        self._max_messages = max_messages

    @property
    def history(self) -> Sequence[Message]:
        return tuple(self._entry.messages)

    def append_turn(self, user_message: Message, assistant_message: Message) -> None:
        self._entry.messages.extend((user_message, assistant_message))
        if len(self._entry.messages) > self._max_messages:
            overflow = len(self._entry.messages) - self._max_messages
            if overflow % 2:
                overflow += 1
            del self._entry.messages[:overflow]


class InMemoryConversationStore:
    """Keep recent conversations in-process with per-session serialization."""

    def __init__(
        self,
        *,
        max_sessions: int = 1000,
        max_messages_per_session: int = 40,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if max_messages_per_session < 2:
            raise ValueError("max_messages_per_session must be at least 2")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0")
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, _SessionEntry] = {}
        self._index_lock = asyncio.Lock()

    @asynccontextmanager
    async def session(self, session_id: str) -> AsyncIterator[ConversationSession]:
        entry = await self._reserve_entry(session_id)
        await entry.lock.acquire()
        try:
            entry.last_access = time.monotonic()
            yield ConversationSession(entry, self.max_messages_per_session)
        finally:
            entry.last_access = time.monotonic()
            entry.lock.release()
            async with self._index_lock:
                entry.users -= 1

    async def clear(self) -> None:
        """Clear idle sessions, primarily for tests and administrative hooks."""

        async with self._index_lock:
            busy = [session_id for session_id, entry in self._sessions.items() if entry.users]
            if busy:
                raise RuntimeError("Cannot clear conversation history while sessions are active.")
            self._sessions.clear()

    async def _reserve_entry(self, session_id: str) -> _SessionEntry:
        async with self._index_lock:
            now = time.monotonic()
            self._prune_expired(now)
            entry = self._sessions.get(session_id)
            if entry is None:
                self._make_capacity()
                entry = _SessionEntry(last_access=now)
                self._sessions[session_id] = entry
            entry.users += 1
            entry.last_access = now
            return entry

    def _prune_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, entry in self._sessions.items()
            if entry.users == 0 and now - entry.last_access >= self.ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def _make_capacity(self) -> None:
        while len(self._sessions) >= self.max_sessions:
            idle = (
                (session_id, entry)
                for session_id, entry in self._sessions.items()
                if entry.users == 0
            )
            oldest = min(idle, key=lambda item: item[1].last_access, default=None)
            if oldest is None:
                return
            del self._sessions[oldest[0]]
