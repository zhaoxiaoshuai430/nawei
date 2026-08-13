"""Gateway bounded-concurrency behavior."""

from __future__ import annotations

import asyncio
import unittest

from services.gateway.concurrency import ConcurrencyRejected, KeyedConcurrencyLimiter


class GatewayConcurrencyLimiterTest(unittest.IsolatedAsyncioTestCase):
    async def test_different_users_can_run_in_parallel(self) -> None:
        limiter = self._limiter(global_limit=2, per_key_limit=1)

        first = await limiter.acquire("user-a")
        second = await limiter.acquire("user-b")
        self.assertEqual(limiter.snapshot().active, 2)

        await first.release()
        await second.release()
        self.assertEqual(limiter.snapshot().active, 0)

    async def test_same_user_waits_until_capacity_is_released(self) -> None:
        limiter = self._limiter(global_limit=2, per_key_limit=1)
        first = await limiter.acquire("user-a")
        waiting = asyncio.create_task(limiter.acquire("user-a"))
        await self._wait_for_waiters(limiter, 1)

        await first.release()
        second = await asyncio.wait_for(waiting, timeout=0.5)
        self.assertEqual(limiter.snapshot().active, 1)
        await second.release()

    async def test_global_queue_rejects_excess_waiters(self) -> None:
        limiter = self._limiter(global_limit=1, per_key_limit=1, queue_limit=1)
        first = await limiter.acquire("user-a")
        waiting = asyncio.create_task(limiter.acquire("user-b"))
        await self._wait_for_waiters(limiter, 1)

        with self.assertRaises(ConcurrencyRejected) as raised:
            await limiter.acquire("user-c")
        self.assertIn("queue is full", raised.exception.reason)

        await first.release()
        second = await asyncio.wait_for(waiting, timeout=0.5)
        await second.release()

    async def test_wait_timeout_releases_waiter_accounting(self) -> None:
        limiter = self._limiter(global_limit=1, per_key_limit=1, wait_timeout_seconds=0.02)
        first = await limiter.acquire("user-a")

        with self.assertRaises(ConcurrencyRejected) as raised:
            await limiter.acquire("user-b")
        self.assertIn("timed out", raised.exception.reason)
        self.assertEqual(limiter.snapshot().waiting, 0)

        await first.release()

    @staticmethod
    def _limiter(
        *,
        global_limit: int,
        per_key_limit: int,
        queue_limit: int = 4,
        wait_timeout_seconds: float = 0.5,
    ) -> KeyedConcurrencyLimiter:
        return KeyedConcurrencyLimiter(
            global_limit=global_limit,
            per_key_limit=per_key_limit,
            queue_limit=queue_limit,
            per_key_queue_limit=2,
            wait_timeout_seconds=wait_timeout_seconds,
        )

    async def _wait_for_waiters(self, limiter: KeyedConcurrencyLimiter, count: int) -> None:
        for _ in range(50):
            if limiter.snapshot().waiting == count:
                return
            await asyncio.sleep(0.001)
        self.fail(f"Expected {count} queued request(s).")


if __name__ == "__main__":
    unittest.main()
