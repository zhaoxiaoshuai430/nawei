"""ASR bounded inference-capacity tests."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from inference_capacity import CapacityRejected, InferenceCapacity  # noqa: E402


class ASRInferenceCapacityTest(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_allows_configured_parallelism(self) -> None:
        capacity = InferenceCapacity(limit=2, queue_limit=2, wait_timeout_seconds=0.5)
        first = await capacity.acquire()
        second = await capacity.acquire()
        self.assertEqual(capacity.snapshot().active, 2)
        await first.release()
        await second.release()

    async def test_queue_limit_and_timeout_are_enforced(self) -> None:
        capacity = InferenceCapacity(limit=1, queue_limit=1, wait_timeout_seconds=0.02)
        active = await capacity.acquire()
        waiting = asyncio.create_task(capacity.acquire())
        await self._wait_for_waiters(capacity, 1)

        with self.assertRaises(CapacityRejected):
            await capacity.acquire()
        with self.assertRaises(CapacityRejected):
            await waiting

        await active.release()
        self.assertEqual(capacity.snapshot().waiting, 0)

    async def _wait_for_waiters(self, capacity: InferenceCapacity, count: int) -> None:
        for _ in range(50):
            if capacity.snapshot().waiting == count:
                return
            await asyncio.sleep(0.001)
        self.fail(f"Expected {count} queued ASR request(s).")


if __name__ == "__main__":
    unittest.main()
