"""Live Aliyun checks through local LLM HTTP/SSE boundaries; inference is stubbed."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app_protocols import LLMResponse, Usage
from services.llm import main
from services.llm.content_safety import REPLIES, enabled


async def run() -> None:
    if not enabled():
        raise RuntimeError("Enable Aliyun moderation in the backend environment first.")
    # These overrides exist only in this probe process. No databases or model API are used.
    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["LLM_AGENT_ENABLED"] = "false"
    os.environ["LLM_INTERNAL_API_KEY"] = "local-moderation-probe"
    insult = "\u4f60\u662f\u4e2a\u50bb\u903c\uff0c\u6eda\u86cb\u3002"
    benign = "\u4f60\u597d\uff0c\u8bf7\u4ecb\u7ecd\u4e00\u4e0b\u4e09\u89d2\u51fd\u6570\u3002"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://llm.test",
        headers={"X-Internal-API-Key": "local-moderation-probe"},
    ) as client:
        for name, text, stream in [("benign", benign, False), ("blocked_input", insult, False), ("blocked_sse_input", insult, True)]:
            payload = {"request_id": f"aliyun-{name}", "trace_id": "aliyun-live-probe",
                       "session_id": "aliyun-live-probe", "user_id": "aliyun-live-probe",
                       "messages": [{"role": "user", "content": text}]}
            response = await client.post("/api/v1/llm/stream" if stream else "/api/v1/llm", json=payload)
            assert response.status_code == 200, f"{name}: HTTP {response.status_code}"
            if stream:
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                answer = "".join(event["data"].get("text", "") for event in events if event["event"] == "message.delta")
                decision = events[-1]["data"]["safety"]["input"]["external_review"]
            else:
                result = response.json()
                answer = result["text"]
                decision = result["metadata"]["safety"]["input"]["external_review"]
                if name == "benign":
                    output = result["metadata"]["safety"]["output"]["external_review"]
                    assert output["action"] == "allow" and not output["failed"]
            assert not decision["failed"], f"{name}: review unavailable"
            assert decision["action"] == ("allow" if name == "benign" else "block")
            if name != "benign":
                assert answer == REPLIES["profanity"]
            print(json.dumps({"case": name, "passed": True, "action": decision["action"], "category": decision["category"]}), flush=True)

    reviewed = await main._finalize_agent_response(
        None, LLMResponse(request_id="aliyun-output-probe", text=insult, usage=Usage(input_tokens=0, output_tokens=0)),
        input_decision=main.safety_guard.audit_input(benign),
    )
    assert reviewed.text == REPLIES["profanity"] and reviewed.finish_reason == "safety"
    assert not reviewed.metadata["safety"]["output"]["external_review"]["failed"]
    print(json.dumps({"case": "blocked_output", "passed": True, "category": "profanity"}), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
