"""Bounded process-local concurrency control for Gateway requests."""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict


class ConcurrencyRejected(RuntimeError):
    """Raised when a request cannot enter the bounded concurrency queue."""

    def __init__(self, reason: str, retry_after_seconds: float) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_seconds = max(1, math.ceil(retry_after_seconds))


@dataclass(frozen=True)
class ConcurrencySnapshot:
    active: int
    waiting: int
    global_limit: int
    per_key_limit: int
    queue_limit: int
    per_key_queue_limit: int


class ConcurrencyLease:
    """Idempotent lease returned after a request obtains capacity."""

    def __init__(self, limiter: "KeyedConcurrencyLimiter", key: str) -> None:
        self._limiter = limiter
        self._key = key
        self._release_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "ConcurrencyLease":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.release()

    async def release(self) -> None:
        # Share one release operation; a disconnected caller must not cancel it.
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._limiter.release(self._key))
        await asyncio.shield(self._release_task)


class KeyedConcurrencyLimiter:
    """Atomically enforce global, per-key, and bounded waiting limits."""

    def __init__(
        self,
        *,
        global_limit: int,
        per_key_limit: int,
        queue_limit: int,
        per_key_queue_limit: int,
        wait_timeout_seconds: float,
    ) -> None:
        for name, value in {
            "global_limit": global_limit,
            "per_key_limit": per_key_limit,
            "queue_limit": queue_limit,
            "per_key_queue_limit": per_key_queue_limit,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be greater than 0")

        self.global_limit = global_limit
        self.per_key_limit = per_key_limit
        self.queue_limit = queue_limit
        self.per_key_queue_limit = per_key_queue_limit
        self.wait_timeout_seconds = wait_timeout_seconds
        self._condition = asyncio.Condition()
        self._active = 0
        self._waiting = 0
        self._active_by_key: DefaultDict[str, int] = defaultdict(int)
        self._waiting_by_key: DefaultDict[str, int] = defaultdict(int)

    async def acquire(self, key: str) -> ConcurrencyLease:
        normalized_key = key.strip() or "anonymous"
        async with self._condition:
            if self._can_start(normalized_key):
                self._start(normalized_key)
                return ConcurrencyLease(self, normalized_key)
            if self._waiting >= self.queue_limit:
                raise ConcurrencyRejected("Gateway concurrency queue is full.", self.wait_timeout_seconds)
            if self._waiting_by_key[normalized_key] >= self.per_key_queue_limit:
                raise ConcurrencyRejected("User concurrency queue is full.", self.wait_timeout_seconds)

            self._waiting += 1
            self._waiting_by_key[normalized_key] += 1
            try:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: self._can_start(normalized_key)),
                        timeout=self.wait_timeout_seconds,
                    )
                except TimeoutError as exc:
                    raise ConcurrencyRejected(
                        "Gateway concurrency queue wait timed out.",
                        self.wait_timeout_seconds,
                    ) from exc
                self._start(normalized_key)
                return ConcurrencyLease(self, normalized_key)
            finally:
                self._waiting -= 1
                self._decrement(self._waiting_by_key, normalized_key)

    async def release(self, key: str) -> None:
        async with self._condition:
            if self._active_by_key.get(key, 0) < 1 or self._active < 1:
                raise RuntimeError("Concurrency lease released without active capacity.")
            self._active -= 1
            self._decrement(self._active_by_key, key)
            self._condition.notify_all()

    def snapshot(self) -> ConcurrencySnapshot:
        return ConcurrencySnapshot(
            active=self._active,
            waiting=self._waiting,
            global_limit=self.global_limit,
            per_key_limit=self.per_key_limit,
            queue_limit=self.queue_limit,
            per_key_queue_limit=self.per_key_queue_limit,
        )

    def _can_start(self, key: str) -> bool:
        return self._active < self.global_limit and self._active_by_key[key] < self.per_key_limit

    def _start(self, key: str) -> None:
        self._active += 1
        self._active_by_key[key] += 1

    @staticmethod
    def _decrement(values: DefaultDict[str, int], key: str) -> None:
        values[key] -= 1
        if values[key] <= 0:
            values.pop(key, None)
