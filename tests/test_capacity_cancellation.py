"""Capacity must recover after disconnects and cancellation during release."""

import asyncio

import pytest

from services.asr.src.inference_capacity import InferenceCapacity
from services.gateway.concurrency import KeyedConcurrencyLimiter
from services.tts.app.services.tts.capacity import GenerationCapacity


@pytest.fixture(params=["gateway", "asr", "tts"])
def capacity_factory(request):
    def create(limit=1):
        if request.param == "gateway":
            capacity = KeyedConcurrencyLimiter(
                global_limit=limit, per_key_limit=limit, queue_limit=4,
                per_key_queue_limit=4, wait_timeout_seconds=1,
            )
            return capacity, lambda: capacity.acquire("test-user")
        cls = InferenceCapacity if request.param == "asr" else GenerationCapacity
        capacity = cls(limit=limit, queue_limit=4, wait_timeout_seconds=1)
        return capacity, capacity.acquire
    return create


async def cancel(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelled_release_can_still_return_its_capacity(capacity_factory):
    capacity, acquire = capacity_factory()
    lease = await acquire()
    async with capacity._condition:
        releasing = asyncio.create_task(lease.release())
        await asyncio.sleep(0)
        await cancel(releasing)
        assert capacity.snapshot().active == 1

    await lease.release()
    assert capacity.snapshot().active == 0
    async with asyncio.timeout(1):
        replacement = await acquire()
        await replacement.release()
    assert capacity.snapshot().active == 0


async def test_release_finishes_without_a_retry_after_caller_is_cancelled(capacity_factory):
    capacity, acquire = capacity_factory()
    lease = await acquire()
    async with capacity._condition:
        releasing = asyncio.create_task(lease.release())
        await asyncio.sleep(0)
        await cancel(releasing)

    async with asyncio.timeout(1):
        while capacity.snapshot().active:
            await asyncio.sleep(0)
    replacement = await acquire()
    await replacement.release()


async def test_duplicate_release_never_returns_another_requests_slot(capacity_factory):
    capacity, acquire = capacity_factory(limit=2)
    first, second = await acquire(), await acquire()
    async with capacity._condition:
        callers = [asyncio.create_task(first.release()) for _ in range(3)]
        await asyncio.sleep(0)
        await cancel(callers[0])
    await asyncio.gather(*callers[1:])
    assert capacity.snapshot().active == 1
    await first.release()
    assert capacity.snapshot().active == 1
    await second.release()
    assert capacity.snapshot().active == 0


async def test_cancelled_waiter_does_not_consume_or_leak_capacity(capacity_factory):
    capacity, acquire = capacity_factory()
    active = await acquire()
    waiting = asyncio.create_task(acquire())
    try:
        async with asyncio.timeout(1):
            while capacity.snapshot().waiting != 1:
                await asyncio.sleep(0)
        await cancel(waiting)
        assert capacity.snapshot().waiting == 0
        assert capacity.snapshot().active == 1
    finally:
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await active.release()
    assert capacity.snapshot().active == 0
