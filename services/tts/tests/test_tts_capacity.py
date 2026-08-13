"""TTS bounded generation-capacity tests."""

from __future__ import annotations

import asyncio
import unittest

from app.services.tts.capacity import CapacityRejected, GenerationCapacity


class TTSGenerationCapacityTest(unittest.IsolatedAsyncioTestCase):
    async def test_generation_capacity_queues_and_releases(self) -> None:
        capacity = GenerationCapacity(limit=1, queue_limit=2, wait_timeout_seconds=0.5)
        first = await capacity.acquire()
        waiting = asyncio.create_task(capacity.acquire())
        await self._wait_for_waiter(capacity)

        await first.release()
        second = await asyncio.wait_for(waiting, timeout=0.5)
        self.assertEqual(capacity.snapshot().active, 1)
        await second.release()

    async def test_generation_capacity_rejects_full_queue(self) -> None:
        capacity = GenerationCapacity(limit=1, queue_limit=1, wait_timeout_seconds=0.02)
        active = await capacity.acquire()
        waiting = asyncio.create_task(capacity.acquire())
        await self._wait_for_waiter(capacity)

        with self.assertRaisesRegex(CapacityRejected, "queue is full"):
            await capacity.acquire()
        with self.assertRaisesRegex(CapacityRejected, "timed out"):
            await waiting
        await active.release()

    async def _wait_for_waiter(self, capacity: GenerationCapacity) -> None:
        for _ in range(50):
            if capacity.snapshot().waiting == 1:
                return
            await asyncio.sleep(0.001)
        self.fail("Expected one queued TTS request.")


if __name__ == "__main__":
    unittest.main()
