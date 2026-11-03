"""Live summary smoke test. Uses real service/model and a disposable synthetic user.

Run after migration with SUMMARY_ENABLED=true on the LLM service:
    python scripts/check_summary_live.py --url http://127.0.0.1:8002
Only the generated test user's memory is deleted. Never prints API credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import math
from typing import Any
from uuid import uuid4

import httpx
from dotenv import load_dotenv


def latency_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0}
    return {"samples": len(ordered), "p50_ms": ordered[math.ceil(.5*len(ordered))-1],
            "p95_ms": ordered[math.ceil(.95*len(ordered))-1],
            "scope": "HTTP response completion; not voice first-audio latency"}


async def check(url: str, report_path: Path, *, verify_online: bool = False) -> bool:
    user = "summary-smoke-" + uuid4().hex
    report: dict[str, Any] = {"user_id": user, "checks": [], "passed": False}
    memory_path = f"/api/v1/llm/memory/{user}"
    summary_path = memory_path + "/summary"
    touched = False
    headers = {"X-Internal-API-Key": os.getenv("LLM_INTERNAL_API_KEY", "")}

    def passed(name: str) -> None:
        report["checks"].append(name)
        print("PASS:", name, flush=True)

    async with httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, timeout=90) as client:
        try:
            response = await client.get("/health", timeout=10)
            response.raise_for_status()
            health = response.json()
            if health.get("status") != "ok" or health.get("provider") == "stub":
                raise RuntimeError("Service unhealthy or configured with stub model")
            if not health.get("runtime", {}).get("summary", {}).get("enabled"):
                raise RuntimeError("SUMMARY_ENABLED is not active on the running service")
            report["runtime_summary_config"] = health["runtime"]["summary"]
            if verify_online and not health["runtime"]["summary"].get("apply_to_prompt"):
                raise RuntimeError("Online acceptance requires global apply-to-prompt on the controlled instance")
            if health["runtime"]["memory"].get("postgres") != "ok":
                raise RuntimeError("PostgreSQL is not healthy")
            passed("real_service_and_summary_enabled")
            initial = await client.get(summary_path)
            initial.raise_for_status()
            if initial.json()["summary"] is not None:
                raise RuntimeError("Disposable user unexpectedly has summary data")
            passed("empty_initial_summary")

            messages = ["我正在学习初中数学分数加减法，希望讲解时分成小步骤，先给提示让我自己尝试。",
                        "我的目标是这周练习异分母分数加法，今天先练习一道，之后继续。"]
            for index, content in enumerate(messages):
                payload = {"request_id": uuid4().hex, "trace_id": uuid4().hex,
                           "session_id": f"{user}-{index}", "user_id": user,
                           "messages": [{"role": "user", "content": content}],
                           "generation_config": {"agent_enabled": True, "answer_mode": "fast"}}
                touched = True
                answer = await client.post("/api/v1/llm", json=payload)
                answer.raise_for_status()
                body = answer.json()
                metadata = body.get("metadata", {})
                report.setdefault("conversation_results", []).append({
                    "finish_reason": body.get("finish_reason"),
                    "persistence": metadata.get("persistence"),
                    "provider": metadata.get("provider"),
                    "fallback": metadata.get("fallback"),
                    "upstream_status_code": metadata.get("upstream_status_code"),
                })
                if body.get("finish_reason") != "stop" or not body.get("metadata", {}).get("persistence", {}).get("saved_turn"):
                    raise RuntimeError("Conversation did not finish with durable history")
                report.setdefault("answers", []).append(body["text"])
            passed("two_real_conversations_persisted_across_sessions")

            response = await client.post(summary_path + "/refresh")
            if response.status_code != 202:
                raise RuntimeError(f"Refresh returned HTTP {response.status_code}")
            job_id = response.json()["job_id"]
            passed("async_refresh_accepted")
            deadline = time.monotonic() + 360
            while time.monotonic() < deadline:
                job_response = await client.get(summary_path + "/jobs/" + job_id)
                job_response.raise_for_status()
                job = job_response.json()
                report["job"] = job
                if job["status"] == "succeeded":
                    break
                if job["status"] in {"failed", "obsolete", "cancelled"}:
                    raise RuntimeError(f"Summary job {job['status']}: {job.get('error_code')}")
                await asyncio.sleep(1)
            else:
                raise RuntimeError("Summary did not finish within 360 seconds")
            passed("real_summary_model_job_succeeded")
            response = await client.get(summary_path)
            response.raise_for_status()
            document = response.json()["summary"]
            report["summary"] = document
            if not document or document["coverage"]["actual_input_turn_count"] != 2:
                raise RuntimeError("Unexpected summary coverage")
            if not (document["topics"] or document["claims"]):
                raise RuntimeError("Model produced no useful topics or claims")
            passed("evidenced_summary_readable")
            if verify_online:
                async def online_case(name: str, mode: str, query: str, expected: str) -> None:
                    started = time.perf_counter()
                    response = await client.post("/api/v1/llm", json={
                        "request_id": uuid4().hex, "trace_id": uuid4().hex, "user_id": user,
                        "session_id": user + "-online-" + uuid4().hex,
                        "messages": [{"role": "user", "content": query}],
                        "generation_config": {"agent_enabled": True, "memory_mode": mode, "answer_mode": "fast"},
                    })
                    response.raise_for_status()
                    answer = response.json()
                    selection = answer.get("metadata", {}).get("runtime", {}).get("summary", {})
                    report.setdefault("online_results", []).append({"case": name, "query": query,
                        "mode": mode, "expected": expected, "selection": selection, "answer": answer.get("text"),
                        "elapsed_ms": round((time.perf_counter()-started)*1000, 2), "usage": answer.get("usage"),
                        "finish_reason": answer.get("finish_reason")})
                    if answer.get("finish_reason") != "stop" or selection.get("status") != expected:
                        raise RuntimeError(f"Online case {name}: expected {expected}, got {selection.get('status')}")
                    passed("online_" + name)

                response = await client.get(summary_path + "/settings")
                response.raise_for_status()
                if not response.json()["effective_apply_to_prompt"]:
                    raise RuntimeError("New user's summary was not automatically enabled; apply migration v2")
                passed("automatic_summary_for_new_user")
                for name, mode, query, expected in (
                    ("summary_enabled", "full", "数学分数复习怎么安排", "applied"),
                    ("lightweight", "lightweight", "数学分数复习怎么安排", "applied"),
                    ("correction", "lightweight", "数学不要再分步骤讲解，直接给出结论", "current_correction"),
                    ("memory_off", "off", "数学分数复习怎么安排", "memory_off"),
                ):
                    await online_case(name, mode, query, expected)
                rows = report["online_results"]
                report["online_http_latency"] = latency_summary([row["elapsed_ms"] for row in rows])
                report["online_usage"] = {
                    key: sum((row.get("usage") or {}).get(key, 0) or 0 for row in rows)
                    for key in ("input_tokens", "output_tokens")}
                report["quality_review"] = {"status": "pending_human_review",
                    "note": "Routing assertions passed; answer semantics, background token cost and voice latency are not certified."}
                passed("online_selection_correction_and_memory_off")
            response = await client.delete(summary_path)
            response.raise_for_status()
            if response.json()["summary"] is not None:
                raise RuntimeError("Summary remained after deletion")
            response = await client.post(summary_path + "/refresh")
            if response.status_code != 409:
                raise RuntimeError("Deleted history window was eligible for regeneration")
            passed("deletion_excludes_previous_window")
            report["passed"] = True
        except Exception as exc:
            # HTTP/SDK exception strings can embed secrets; store the type only.
            report["error_type"] = type(exc).__name__
            if isinstance(exc, RuntimeError):
                report["error"] = str(exc)
            print("FAILED:", report.get("error", report["error_type"]), flush=True)
        finally:
            if touched:
                try:
                    response = await client.delete(memory_path)
                    response.raise_for_status()
                    report["test_user_cleaned"] = True
                except Exception as exc:
                    report["cleanup_error_type"] = type(exc).__name__
                    report["passed"] = False
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", report_path, flush=True)
    return bool(report["passed"])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--report", type=Path, default=root / ".runtime" / "summary-live-test.json")
    parser.add_argument("--verify-online", action="store_true",
                        help="Requires global SUMMARY_APPLY_TO_PROMPT=true and migration v2")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(check(args.url, args.report, verify_online=args.verify_online)) else 1)
