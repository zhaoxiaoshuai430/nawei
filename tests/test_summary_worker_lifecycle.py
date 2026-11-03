"""Shutdown must wait for scheduler admission and lease cleanup, not just inference."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.llm.drain import LocalDrainState
from services.llm.summary.policy import SummarySettings
from services.llm.summary.worker import SummaryWorker


def repository():
    return SimpleNamespace(
        settings=SummarySettings(), memory=SimpleNamespace(initialize=AsyncMock()),
        discover=AsyncMock(), claim=AsyncMock(return_value=None), publish=AsyncMock(),
        release=AsyncMock(), fail=AsyncMock(),
    )


@pytest.mark.parametrize("stage", ["initialize", "discover", "claim"])
async def test_drain_waits_for_pending_scheduler_operation(tmp_path, monkeypatch, stage):
    monkeypatch.setenv("LLM_DRAIN_DIRECTORY", str(tmp_path))
    repo = repository()
    entered, release = asyncio.Event(), asyncio.Event()
    job = {"input_snapshot": {"turns": []}, "job_id": "test"}

    async def blocked(*args):
        entered.set()
        await release.wait()
        return job if stage == "claim" else None

    target = repo.memory if stage == "initialize" else repo
    setattr(target, stage, AsyncMock(side_effect=blocked))
    worker = SummaryWorker(repo)
    state = LocalDrainState(str(tmp_path))
    runtime = SimpleNamespace(summary_worker=worker, persistence_status=lambda: {"pending": 0, "failed": 0})
    worker.start()
    try:
        await asyncio.wait_for(entered.wait(), 1)
        (tmp_path / "drain.request").touch()
        assert not state.snapshot(runtime)["drained"]
        release.set()
        await asyncio.wait_for(asyncio.shield(worker.task), 2)
        assert state.snapshot(runtime)["drained"]
        if stage == "claim":
            repo.publish.assert_awaited_once()
            repo.release.assert_awaited_once_with(job)
        else:
            repo.claim.assert_not_awaited()
    finally:
        release.set()
        await worker.close()


async def test_inflight_includes_pending_lease_release():
    repo = repository()
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(job):
        entered.set()
        await release.wait()

    repo.release.side_effect = blocked
    worker = SummaryWorker(repo)
    task = asyncio.create_task(worker.process({"input_snapshot": {"turns": []}, "job_id": "test"}))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        worker.stopping = True
        assert worker.status()["summary_inflight"] == 1
        assert not worker.status()["summary_scheduler_stopped"]
    finally:
        release.set()
        await task
    assert worker.inflight == 0


async def test_finished_scheduler_can_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_DRAIN_DIRECTORY", str(tmp_path))
    repo = repository()
    worker = SummaryWorker(repo)
    worker.task = asyncio.create_task(asyncio.sleep(0))
    await worker.task
    previous = worker.task
    try:
        worker.start()
        assert worker.task is not previous
        assert not worker.status()["summary_scheduler_stopped"]
    finally:
        await worker.close()
