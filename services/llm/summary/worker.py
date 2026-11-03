"""Recoverable scheduler sharing a single PostgreSQL slot across LLM processes."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4
import httpx
from pydantic import ValidationError

from services.llm.agents.summary_agent import ConversationSummaryAgent
from services.llm.summary.repository import SummaryRepository, decode
from services.llm.summary.schema import SummaryCandidate

logger = logging.getLogger(__name__)


class SummaryWorker:
    def __init__(self, repository: SummaryRepository, agent: Any = None) -> None:
        self.repository = repository
        self.agent = agent or ConversationSummaryAgent(repository.settings)
        self.owner = f"{os.getpid()}:{uuid4()}"
        self.task: asyncio.Task[None] | None = None
        self.stopping = False
        self.inflight = 0
        self.failed = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.stopping = False
            self.task = asyncio.create_task(self.run())

    def status(self) -> dict[str, Any]:
        stopped = (self.task is None or self.task.done()) and self.inflight == 0
        return {"summary_inflight": self.inflight, "summary_scheduler_stopped": stopped,
                "summary_stop_requested": self.stopping,
                "failed": self.failed, "last_error": self.last_error}

    def _stop_requested(self) -> bool:
        directory = os.getenv("LLM_DRAIN_DIRECTORY", "")
        if directory and (Path(directory) / "drain.request").exists():
            self.stopping = True
        return self.stopping

    async def close(self) -> None:
        self.stopping = True
        if self.task:
            try:
                async with asyncio.timeout(self.repository.settings.timeout_seconds + 5):
                    await asyncio.shield(self.task)
            except TimeoutError:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def run(self) -> None:
        while not self.stopping:
            try:
                if self._stop_requested():
                    break
                await self.repository.memory.initialize()
                if self._stop_requested():
                    break
                await self.repository.discover()
                if self._stop_requested():
                    break
                job = await self.repository.claim(self.owner)
                if job:
                    # A claim already in flight may finish after drain begins.
                    # Finish that job and release its lease before reporting stopped.
                    await self.process(job)
                self.last_error = None
            except Exception as exc:
                self.last_error = type(exc).__name__
                logger.warning("Summary scheduler paused: %s", self.last_error)
            if not self.stopping:
                await asyncio.sleep(1)

    async def process(self, job: dict[str, Any]) -> None:
        self.inflight += 1
        inference: asyncio.Task[Any] | None = None

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(5)
                try:
                    owned = await self.repository.renew(job)
                except Exception:
                    owned = False
                if not owned:
                    if inference:
                        inference.cancel()
                    return

        async def generate() -> tuple[SummaryCandidate, str]:
            snapshot = decode(job["input_snapshot"])
            if not snapshot["turns"] and not snapshot.get("reference_turns"):
                return SummaryCandidate(limitations=["窗口内没有符合预算的完整轮次"]), "no-model"
            return await self.agent.generate(snapshot, str(job["job_id"]))

        beat = asyncio.create_task(heartbeat())
        try:
            async with asyncio.timeout(self.repository.settings.timeout_seconds):
                inference = asyncio.create_task(generate())
                candidate, model = await inference
                await self.repository.publish(job, candidate, model)
        except asyncio.CancelledError:
            # Leave the durable running job for lease recovery on process shutdown.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except Exception as exc:
            self.failed += 1
            if isinstance(exc, ValidationError):
                code = "invalid_schema"
            elif isinstance(exc, httpx.HTTPStatusError):
                code = "rate_limited" if exc.response.status_code == 429 else "upstream_http"
            elif isinstance(exc, ValueError):
                # Only fixed application codes may be persisted; never exception bodies from SDKs.
                safe_codes = {"summary_model_unconfigured", "summary_prompt_budget", "summary_input_safety",
                              "summary_input_privacy", "summary_output_privacy",
                              "summary_incomplete_output", "summary_output_budget", "summary_output_safety"}
                code = str(exc) if str(exc) in safe_codes else "invalid_evidence_or_output"
            else:
                code = type(exc).__name__
            await self.repository.fail(job, code)
        finally:
            try:
                beat.cancel()
                await asyncio.gather(beat, return_exceptions=True)
                if inference and not inference.done():
                    inference.cancel()
                    await asyncio.gather(inference, return_exceptions=True)
                # Keep an interrupted running task's slot until its lease expires.
                current = asyncio.current_task()
                if not (current is not None and current.cancelling()):
                    await self.repository.release(job)
            finally:
                self.inflight -= 1
