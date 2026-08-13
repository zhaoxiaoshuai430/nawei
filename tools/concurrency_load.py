"""Run a bounded concurrent text-completion load check against Gateway."""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import time
import uuid
from collections import Counter
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RequestResult:
    status_code: int
    elapsed_seconds: float
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--requests-per-user", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--bearer-token", default="")
    parser.add_argument(
        "--reuse-user-session",
        action="store_true",
        help="Reuse one session per user to verify same-session serialization.",
    )
    return parser.parse_args()


async def run_load(args: argparse.Namespace) -> int:
    for name in ("users", "requests_per_user", "concurrency"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")

    client_limit = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout, limits=limits) as client:
        tasks = [
            asyncio.create_task(_send(client, client_limit, args, user_index, request_index))
            for user_index in range(args.users)
            for request_index in range(args.requests_per_user)
        ]
        results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started

    statuses = Counter(result.status_code for result in results)
    latencies = sorted(result.elapsed_seconds for result in results)
    errors = [result.error for result in results if result.error]
    succeeded = sum(count for status, count in statuses.items() if 200 <= status < 300)
    print(f"requests={len(results)} succeeded={succeeded} elapsed_s={elapsed:.3f}")
    print(f"throughput_rps={len(results) / elapsed:.3f} statuses={dict(sorted(statuses.items()))}")
    print(
        "latency_ms="
        f"p50:{statistics.median(latencies) * 1000:.1f} "
        f"p95:{_percentile(latencies, 0.95) * 1000:.1f} "
        f"max:{max(latencies) * 1000:.1f}"
    )
    if errors:
        print(f"errors={errors[:5]}")
    return 0 if succeeded == len(results) else 1


async def _send(
    client: httpx.AsyncClient,
    client_limit: asyncio.Semaphore,
    args: argparse.Namespace,
    user_index: int,
    request_index: int,
) -> RequestResult:
    user_id = f"load-user-{user_index}"
    session_suffix = "shared" if args.reuse_user_session else str(request_index)
    request_id = uuid.uuid4().hex
    headers = {"X-User-ID": user_id}
    if args.bearer_token:
        headers = {"Authorization": f"Bearer {args.bearer_token}"}
    payload = {
        "request_id": request_id,
        "trace_id": f"load-trace-{request_id}",
        "session_id": f"load-session-{user_index}-{session_suffix}",
        "user_id": user_id,
        "message": {"role": "user", "content": f"并发测试 {user_index}-{request_index}"},
        "stream": False,
        "output_audio": False,
    }
    async with client_limit:
        started = time.perf_counter()
        try:
            response = await client.post("/api/v1/text/completions", headers=headers, json=payload)
            elapsed = time.perf_counter() - started
            error = "" if response.is_success else response.text[:300]
            return RequestResult(response.status_code, elapsed, error)
        except httpx.HTTPError as exc:
            return RequestResult(0, time.perf_counter() - started, str(exc))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return values[index]


def main() -> None:
    raise SystemExit(asyncio.run(run_load(parse_args())))


if __name__ == "__main__":
    main()
