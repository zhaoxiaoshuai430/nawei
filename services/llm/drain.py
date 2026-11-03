"""Coordinate local Windows shutdown across LLM worker processes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from pathlib import Path
import time
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


logger = logging.getLogger(__name__)


class LocalDrainState:
    def __init__(self, directory: str | None = None) -> None:
        configured = directory if directory is not None else os.getenv("LLM_DRAIN_DIRECTORY", "")
        self.directory = Path(configured) if configured else None
        self.active = 0
        self.token = os.getenv("LLM_DRAIN_TOKEN", "")

    def requested(self) -> bool:
        return self.directory is not None and (self.directory / "drain.request").is_file()

    def snapshot(self, runtime: Any) -> dict[str, Any]:
        persistence = runtime.persistence_status()
        worker = getattr(runtime, "summary_worker", None)
        if worker is not None and self.requested():
            worker.stopping = True
        summary = worker.status() if worker else {"summary_inflight": 0, "summary_scheduler_stopped": True}
        return {
            "pid": os.getpid(), "updated_at": time.time(), "active": self.active,
            "persistence": persistence,
            "summary": summary,
            "drained": self.requested() and self.active == 0 and persistence["pending"] == 0 and persistence["failed"] == 0
                       and summary["summary_inflight"] == 0 and summary["summary_scheduler_stopped"],
        }

    def _write(self, body: dict[str, Any]) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{os.getpid()}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(body), encoding="utf-8")
        for attempt in range(5):
            try:
                temporary.replace(target)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (2 ** attempt))

    async def monitor(self, runtime: Any) -> None:
        if self.directory is None:
            return
        while True:
            try:
                await asyncio.to_thread(self._write, self.snapshot(runtime))
            except OSError:
                logger.warning("Unable to publish LLM drain status; retrying", exc_info=True)
            await asyncio.sleep(0.25)


class LocalDrainMiddleware:
    def __init__(self, app: ASGIApp, state: LocalDrainState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in {"/health", "/ready", "/api/v1/llm/runtime"}:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        background_write = (
            scope.get("path") == "/api/v1/llm" and scope.get("method") == "POST" and bool(self.state.token)
            and secrets.compare_digest(headers.get(b"x-nawei-drain-token", b""), self.state.token.encode())
        )
        if self.state.requested() and not background_write:
            await JSONResponse({"detail": "LLM is draining"}, status_code=503, headers={"Retry-After": "5"})(scope, receive, send)
            return
        self.state.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.state.active -= 1
