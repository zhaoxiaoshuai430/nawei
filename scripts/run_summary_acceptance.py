"""Run real online summary acceptance on a temporary local LLM instance.

python -m scripts.run_summary_acceptance
The existing service and .env are not changed. Safety checks stay configured.
The temporary HTTP server is shut down after the test, including failures.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import socket
from uuid import uuid4

from dotenv import load_dotenv
import uvicorn

from scripts.check_summary_live import check


async def run(port: int, report: Path) -> bool:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    os.environ.setdefault("DATABASE_URL", "postgresql://nawei:nawei@127.0.0.1:5432/nawei")
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
    os.environ.update(SUMMARY_ENABLED="true", SUMMARY_APPLY_TO_PROMPT="true", SUMMARY_TIMEOUT_SECONDS="90",
                      LLM_MEMORY_REQUIRED="true", LLM_WORKERS="1", LLM_PORT=str(port))
    os.environ["LLM_DRAIN_DIRECTORY"] = str(root / ".runtime" / ("summary-acceptance-" + uuid4().hex))
    server = uvicorn.Server(uvicorn.Config("services.llm.main:app", host="127.0.0.1", port=port,
                                          log_level="warning", access_log=False, timeout_graceful_shutdown=45))
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(60):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Temporary LLM did not start")
                await asyncio.sleep(.2)
        print(f"Temporary acceptance service ready on port {port}", flush=True)
        return await check(f"http://127.0.0.1:{port}", report, verify_online=True)
    finally:
        server.should_exit = True
        try:
            async with asyncio.timeout(45):
                await asyncio.shield(task)
        except TimeoutError:
            server.force_exit = True
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--report", type=Path, default=Path(".runtime/summary-online-acceptance.json"))
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.port, args.report)) else 1)
