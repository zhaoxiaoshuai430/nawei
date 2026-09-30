"""Bounded asynchronous admission control for TTS generation."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass


class CapacityRejected(RuntimeError):
    def __init__(self, reason: str, retry_after_seconds: float) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_seconds = max(1, math.ceil(retry_after_seconds))


@dataclass(frozen=True)
class CapacitySnapshot:
    active: int
    waiting: int
    limit: int
    queue_limit: int


class CapacityLease:
    def __init__(self, capacity: "GenerationCapacity") -> None:
        self._capacity = capacity
        self._release_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "CapacityLease":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.release()

    async def release(self) -> None:
        # Share one release operation; a disconnected caller must not cancel it.
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._capacity.release())
        await asyncio.shield(self._release_task)


class GenerationCapacity:
    def __init__(self, *, limit: int, queue_limit: int, wait_timeout_seconds: float) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if queue_limit < 1:
            raise ValueError("queue_limit must be at least 1")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be greater than 0")
        self.limit = limit
        self.queue_limit = queue_limit
        self.wait_timeout_seconds = wait_timeout_seconds
        self._condition = asyncio.Condition()
        self._active = 0
        self._waiting = 0

    async def acquire(self) -> CapacityLease:
        async with self._condition:
            if self._active < self.limit:
                self._active += 1
                return CapacityLease(self)
            if self._waiting >= self.queue_limit:
                raise CapacityRejected("TTS generation queue is full.", self.wait_timeout_seconds)

            self._waiting += 1
            try:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(lambda: self._active < self.limit),
                        timeout=self.wait_timeout_seconds,
                    )
                except TimeoutError as exc:
                    raise CapacityRejected("TTS generation queue wait timed out.", self.wait_timeout_seconds) from exc
                self._active += 1
                return CapacityLease(self)
            finally:
                self._waiting -= 1

    async def release(self) -> None:
        async with self._condition:
            if self._active < 1:
                raise RuntimeError("TTS capacity released without an active generation.")
            self._active -= 1
            self._condition.notify(1)

    def snapshot(self) -> CapacitySnapshot:
        return CapacitySnapshot(
            active=self._active,
            waiting=self._waiting,
            limit=self.limit,
            queue_limit=self.queue_limit,
        )
