"""Live Gateway APIs, device lifecycle and recovery acceptance checks."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import secrets
import struct
import sys
import time
import uuid
import wave
from pathlib import Path

import av
import asyncpg
import httpx
from dotenv import dotenv_values
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.gateway.xiaozhi.opus_codec import OpusEncoder


class FullFlow:
    def __init__(self, output: Path, user_id: str | None = None):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.user = user_id or "full-flow-" + uuid.uuid4().hex[:12]
        self.config = {**dotenv_values(ROOT / ".env"), **dotenv_values(ROOT / ".env.robot"), **os.environ}
        self.token = os.getenv("NAWEI_SIMULATOR_USER_TOKEN")
        if self.token and not user_id:
            raise ValueError("Pass --user-id matching NAWEI_SIMULATOR_USER_TOKEN")
        self.headers = {"Authorization": "Bearer " + self.token} if self.token else {"X-User-ID": self.user}
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=180, trust_env=False)
        self.results: dict = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "mock_services": False,
                              "user_id": self.user, "checks": []}
        self.pcm = b""
        self.wav = b""

    async def check(self, name, operation):
        started = time.monotonic()
        try:
            details = await operation()
            result = {"name": name, "passed": True, "details": details}
        except Exception as error:
            result = {"name": name, "passed": False, "error_type": type(error).__name__}
            if isinstance(error, AssertionError):
                result["message"] = str(error)
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        self.results["checks"].append(result)
        self.results["passed"] = all(check["passed"] for check in self.results["checks"])
        (self.output / "checks.json").write_text(json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=True), flush=True)

    async def request(self, method, path, **kwargs):
        kwargs.setdefault("headers", self.headers)
        response = await self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    async def health(self):
        gateway = await self.request("GET", "/api/v1/status")
        assert gateway["status"] == "ready", "Gateway dependencies are not ready: " + json.dumps(gateway.get("services", {}))
        llm = await self.request("GET", "http://127.0.0.1:8002/health")
        knowledge = await self.request("GET", "http://127.0.0.1:8003/health")
        assert llm["provider"] != "stub", "LLM is using a stub provider"
        assert all(llm["runtime"]["memory"][name] == "ok" for name in ["postgres", "redis"])
        assert knowledge["ready"] and knowledge["qdrant"] == knowledge["neo4j"] == "ok"
        await self.request("GET", "/api/v1/devices")
        return {"gateway": gateway["status"], "llm_provider": llm["provider"], "memory": "postgres+redis",
                "knowledge": "qdrant+neo4j", "robot_url": self.config.get("NAWEI_PUBLIC_ROBOT_URL")}

    async def source_audio(self):
        audio = await self.request("POST", "/api/v1/audio/speech", json={
            "request_id": uuid.uuid4().hex, "trace_id": uuid.uuid4().hex, "session_id": self.user,
            "text": "一加一等于几？请只回答结果。", "voice": {"speaker_id": "liliya", "language": "Chinese"},
            "audio_config": {"format": "pcm16", "sample_rate": 16000, "channels": 1},
        })
        assert audio["format"] == "pcm16" and audio["sample_rate"] == 16000
        self.pcm = base64.b64decode(audio["audio_base64"], validate=True)
        peak = max(abs(sample[0]) for sample in struct.iter_unpack("<h", self.pcm))
        assert peak > 64, "TTS source audio is silent"
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(self.pcm)
        self.wav = buffer.getvalue()
        return {"duration_ms": len(self.pcm) // 32, "peak": peak}

    async def upload_asr(self, audio_format="wav"):
        assert self.wav, "No source audio"
        result = await self.request("POST", "/api/v1/audio/transcriptions/base64", json={
            "request_id": uuid.uuid4().hex, "trace_id": uuid.uuid4().hex, "session_id": self.user,
            "audio_base64": base64.b64encode(self.pcm if audio_format == "pcm16" else self.wav).decode(),
            "format": audio_format, "sample_rate": 16000,
            "channels": 1, "language": "zh", "metadata": {"use_itn": True},
        })
        assert result["text"], "Uploaded speech was not transcribed"
        return {"text": result["text"]}

    async def conversation_memory(self):
        await self.text_stream("我叫小明，我最喜欢蓝色，请记住。", "memory-store")
        recalled = await self.text_stream("我叫什么名字？我最喜欢什么颜色？", "memory-recall")
        assert "小明" in recalled["text"] and "蓝色" in recalled["text"], "Conversation memory was not recalled"
        key = self.config.get("LLM_INTERNAL_API_KEY")
        connection = await asyncpg.connect(
            self.config.get("DATABASE_URL") or "postgresql://nawei:nawei@127.0.0.1:5432/nawei", timeout=10,
        )
        started = time.monotonic()
        try:
            while True:
                memory = await self.request(
                    "GET", "http://127.0.0.1:8002/api/v1/llm/memory/" + self.user,
                    headers={"X-Internal-API-Key": key} if key else {},
                    params={"session_id": self.user, "query": "小明 蓝色", "limit": 50},
                )
                rows = await connection.fetch(
                    "SELECT role, content FROM navi_messages WHERE user_id=$1 AND session_id=$2 ORDER BY created_at ASC",
                    self.user, self.user,
                )
                complete = (len(rows) >= 4 and rows[-1]["role"] == "assistant"
                            and rows[-1]["content"] == recalled["text"].strip())
                if complete and [dict(row) for row in rows] == memory["history"]:
                    break
                assert time.monotonic() - started < 10, "Redis and PostgreSQL history did not converge within 10 seconds"
                await asyncio.sleep(0.2)
        finally:
            await connection.close()
        assert memory["health"]["postgres"] == memory["health"]["redis"] == "ok"
        assert memory["profile"]["nickname"] == "小明"
        return {"answer": recalled["text"], "ordered_messages": len(rows), "postgres": "ok", "redis": "ok",
                "persistence_wait_ms": round((time.monotonic() - started) * 1000)}

    async def text_stream(self, question="三角函数是什么？请简短回答。", artifact="text-stream"):
        events = []
        text = ""
        metadata = {}
        started = time.monotonic()
        first_text_ms = None
        async with self.client.stream("POST", "/api/v1/text/completions/stream", headers=self.headers, json={
            "request_id": uuid.uuid4().hex, "trace_id": uuid.uuid4().hex, "session_id": self.user,
            "user_id": self.user, "message": {"role": "user", "content": question},
            "output_audio": False,
        }) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = json.loads(line[5:])
                events.append(body["event"])
                assert body["event"] != "error", "Text stream returned an error event"
                if body["event"] == "message.delta":
                    if first_text_ms is None and body["data"].get("text"):
                        first_text_ms = round((time.monotonic() - started) * 1000)
                    text += body["data"].get("text", "")
                if body["event"] == "message.done":
                    metadata = body.get("data", {})
        assert text and "message.done" in events
        (self.output / (artifact + ".json")).write_text(json.dumps({"text": text, "done": metadata, "first_text_ms": first_text_ms}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"text": text, "events": len(events), "message_done": True, "first_text_ms": first_text_ms}

    async def response_policy(self):
        result = await self.text_stream("一加一等于几？请只回答结果。", "result-only")
        assert result["text"].strip(" 。.!！\n") in {"2", "二"}, "Result-only request included extra content"
        single = await self.text_stream("三角函数是什么？只用一句话回答，不要反问。", "single-sentence")
        body = single["text"].partition("参考来源")[0].strip()
        assert len(body) < 160 and body.count("。") <= 1 and "？" not in body and "?" not in body
        detailed = await self.text_stream("详细介绍三角函数。", "detailed-teaching")
        body = detailed["text"].partition("参考来源")[0].strip()
        full_text = detailed["text"].strip()
        assert len(body) > 100, "Detailed request was shortened"
        assert full_text.count("？") + full_text.count("?") == 1, "Teaching answer must contain one exploration question"
        invitation = next((line for line in full_text.splitlines() if "？" in line or "?" in line), "")
        assert "和" in invitation and not any(word in invitation for word in ("还是", "或者", "或是", "二选一"))
        return {"result_only": result, "single_sentence": single, "detailed_teaching": detailed}

    async def knowledge(self):
        key = self.config.get("KNOWLEDGE_INTERNAL_API_KEY")
        result = await self.request("POST", "http://127.0.0.1:8003/api/v1/knowledge/retrieve",
                                    headers={"X-Internal-API-Key": key} if key else {},
                                    json={"query": "三角函数", "knowledge_base_id": "default"})
        assert result["chunks"] and result["relations"], "No live vector or graph retrieval results"
        return {"chunks": len(result["chunks"]), "relations": len(result["relations"]), "metadata": result["metadata"]}

    async def realtime_voice(self):
        assert self.pcm, "No source audio"
        events = []
        transcript = ""
        answer = ""
        samples = 0
        peak = 0
        async with connect("ws://127.0.0.1:8000/api/v1/voice/realtime", max_size=16 * 1024 * 1024) as websocket:
            await websocket.send(json.dumps({"type": "start", "request_id": uuid.uuid4().hex,
                "trace_id": uuid.uuid4().hex, "session_id": self.user, "user_id": self.user,
                "access_token": self.token, "output_audio": True,
                "voice": {"speaker_id": "liliya", "language": "Chinese"},
                "output_audio_config": {"format": "wav", "sample_rate": 24000, "channels": 1}}))
            for offset in range(0, len(self.pcm), 1920):
                await websocket.send(self.pcm[offset:offset + 1920])
                await asyncio.sleep(0.06)
            await websocket.send(json.dumps({"type": "stop"}))
            async with asyncio.timeout(120):
                async for raw in websocket:
                    body = json.loads(raw)
                    kind = body["event"]
                    events.append(kind)
                    data = body.get("data", {})
                    assert kind != "error", "Browser voice returned an error event"
                    if kind == "asr.completed":
                        transcript = data["text"]
                    elif kind == "message.delta":
                        answer += data.get("text", "")
                    elif kind == "audio.delta":
                        with av.open(io.BytesIO(base64.b64decode(data["audio_base64"], validate=True))) as container:
                            for frame in container.decode(audio=0):
                                values = frame.to_ndarray()
                                samples += frame.samples
                                peak = max(peak, float(abs(values).max()))
        assert transcript and answer and samples > 0 and peak > 0
        assert "audio.done" in events and "message.done" in events
        return {"transcript": transcript, "answer": answer, "decoded_samples": samples, "peak": peak, "events": len(events)}

    async def lifecycle(self, restart=False):
        device = "02:" + ":".join(f"{value:02x}" for value in secrets.token_bytes(5))
        identity = {"Device-Id": device, "Client-Id": str(uuid.uuid4()), "X-Device-Key": secrets.token_hex(32)}
        path = "/api/v1/devices/" + device
        checks = []
        sockets = []
        bound = False

        async def open_device(headers, rate=24000):
            socket = await connect(configuration["url"], additional_headers=headers, open_timeout=10)
            sockets.append(socket)
            await socket.send(json.dumps({"type": "hello", "version": 1, "transport": "websocket",
                "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
                "output_audio_params": {"format": "opus", "sample_rate": rate, "channels": 1, "frame_duration": 60}}))
            hello = json.loads(await asyncio.wait_for(socket.recv(), 10))
            assert hello["type"] == "hello" and hello["audio_params"]["sample_rate"] == rate
            return socket, hello["session_id"]

        async def rejected(headers):
            try:
                async with connect(configuration["url"], additional_headers=headers, open_timeout=10):
                    raise AssertionError("Invalid device credentials were accepted")
            except InvalidStatus as error:
                assert error.response.status_code in {401, 403}

        try:
            initial = await self.request("POST", "/xiaozhi/ota/", headers=identity, json={})
            assert "websocket" not in initial
            pending = await self.client.post("/xiaozhi/ota/activate", headers=identity)
            assert pending.status_code == 202
            code = initial["activation"]["code"]
            await self.request("POST", "/api/v1/devices/bind", json={"code": code, "name": "Full flow temporary device"})
            bound = True
            repeated = await self.client.post("/api/v1/devices/bind", headers=self.headers, json={"code": code})
            assert repeated.status_code == 409
            assert (await self.client.post("/xiaozhi/ota/activate", headers=identity)).status_code == 200
            configuration = (await self.request("POST", "/xiaozhi/ota/", headers=identity))["websocket"]
            headers = {**identity, "Authorization": "Bearer " + configuration["token"], "Protocol-Version": "1"}
            assert (await self.client.get(path, headers={"X-User-ID": "other-" + self.user})).status_code == 404
            checks.append("activation_binding_ownership")
            first, first_session = await open_device(headers)
            if restart:
                process = await asyncio.create_subprocess_exec("powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(ROOT / "scripts/restart_robot_services.ps1"), "-Names", "gateway", "-TimeoutSeconds", "30", cwd=ROOT,
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                try:
                    await asyncio.wait_for(process.wait(), 45)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
                assert process.returncode == 0, "Gateway restart command failed"
                await asyncio.wait_for(first.wait_closed(), 10)
                for attempt in range(20):
                    try:
                        response = await self.client.get("/health", timeout=3)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
                assert (await self.request("POST", "/xiaozhi/ota/", headers=identity))["websocket"]["token"] == configuration["token"]
                second, session_id = await open_device(headers)
                assert session_id != first_session
                checks.append("gateway_restart_keeps_binding_and_credential")
            else:
                for override in [{"Authorization": "Bearer invalid"}, {"Client-Id": str(uuid.uuid4())}, {"Protocol-Version": "2"}]:
                    await rejected({**headers, **override})
                checks.append("reject_invalid_token_identity_version")
                second, session_id = await open_device(headers, 16000)
                await asyncio.wait_for(first.wait_closed(), 5)
                assert first_session != session_id
                checks.append("reconnect_replaces_connection_and_negotiates_16k")
                await second.send(json.dumps({"type": "listen", "state": "start", "mode": "manual"}))
                encoder = OpusEncoder(16000)
                for packet in encoder.feed(bytes(1920), 16000):
                    await second.send(packet)
                await second.send(json.dumps({"type": "abort"}))
                stopped = json.loads(await asyncio.wait_for(second.recv(), 5))
                assert stopped.get("state") == "stop"
                await second.send(json.dumps({"type": "ping"}))
                assert json.loads(await asyncio.wait_for(second.recv(), 5))["type"] == "pong"
                checks.append("abort_listening_then_ping")
                await self.request("PATCH", path, json={"enabled": False})
                await asyncio.wait_for(second.wait_closed(), 5)
                await rejected(headers)
                await self.request("PATCH", path, json={"enabled": True})
                second, _ = await open_device(headers)
                checks.append("disable_revokes_live_connection_and_enable_recovers")
            await self.request("DELETE", path + "/binding")
            bound = False
            await asyncio.wait_for(second.wait_closed(), 5)
            await rejected(headers)
            fresh = {**identity, "Client-Id": str(uuid.uuid4()), "X-Device-Key": secrets.token_hex(32)}
            assert "activation" in await self.request("POST", "/xiaozhi/ota/", headers=fresh)
            checks.append("unbind_revokes_live_and_cached_credential_allows_nvs_reset")
            return {"checks": checks}
        finally:
            for socket in sockets:
                await socket.close()
            if bound:
                await self.request("DELETE", path + "/binding")

    async def inspect_robot_memory(self, report_path):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["passed"] and len(report["turns"]) >= 2
        headers = {}
        if self.config.get("LLM_INTERNAL_API_KEY"):
            headers["X-Internal-API-Key"] = self.config["LLM_INTERNAL_API_KEY"]
        memory = await self.request("GET", "http://127.0.0.1:8002/api/v1/llm/memory/" + report["user_id"],
                                    headers=headers, params={"session_id": report["session_id"], "query": "小明 蓝色", "limit": 50})
        assert os.environ.get("DATABASE_URL"), "Initialize-NaweiLocalEnvironment before this PostgreSQL check"
        connection = await asyncpg.connect(os.environ["DATABASE_URL"], timeout=10)
        try:
            rows = await connection.fetch(
                "SELECT role, content FROM navi_messages WHERE user_id=$1 AND session_id=$2 ORDER BY created_at ASC",
                report["user_id"], report["session_id"],
            )
            memory["postgres_history"] = [dict(row) for row in rows]
        finally:
            await connection.close()
        (self.output / "memory.json").write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
        assert memory["health"]["postgres"] == memory["health"]["redis"] == "ok"
        assert [message["role"] for message in memory["history"]] == ["user", "assistant"] * len(report["turns"])
        assert memory["postgres_history"] == memory["history"], "Redis and PostgreSQL history differ"
        assert memory["profile"]["nickname"] == "小明"
        assert any("蓝色" in item["content"] for item in memory["memories"])
        assert "小明" in report["turns"][1]["answer"] and "蓝色" in report["turns"][1]["answer"]
        return {"ordered_messages": len(memory["history"]), "memories": len(memory["memories"]),
                "nickname": memory["profile"]["nickname"], "recall_answer": report["turns"][1]["answer"],
                "postgres": "ok", "redis": "ok"}

    async def run(self, restart_only=False, memory_report=None, response_policy_only=False, skip_robot=False):
        if skip_robot and (restart_only or memory_report):
            raise ValueError("Robot checks cannot be combined with --skip-robot")
        if self.token and not (skip_robot or response_policy_only):
            raise ValueError("Cross-user lifecycle checks require local authentication; use --skip-robot with JWT")
        self.results["robot_checks_skipped"] = skip_robot
        try:
            await self.check("health", self.health)
            if memory_report:
                await self.check("robot_memory", lambda: self.inspect_robot_memory(memory_report))
            elif response_policy_only:
                await self.check("response_policy", self.response_policy)
            elif restart_only:
                await self.check("gateway_restart", lambda: self.lifecycle(restart=True))
            else:
                if not skip_robot:
                    await self.check("device_lifecycle", self.lifecycle)
                await self.check("tts_source", self.source_audio)
                await self.check("audio_upload_asr", self.upload_asr)
                await self.check("pcm16_upload_asr", lambda: self.upload_asr("pcm16"))
                await self.check("text_sse_and_knowledge_question", self.text_stream)
                await self.check("graphrag_retrieval", self.knowledge)
                await self.check("gateway_voice_protocol", self.realtime_voice)
                if skip_robot:
                    await self.check("conversation_memory", self.conversation_memory)
            await self.check("final_health", self.health)
        finally:
            await self.client.aclose()
        return self.results["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/full-flow")
    parser.add_argument("--restart-only", action="store_true")
    parser.add_argument("--memory-report", type=Path, help="Inspect the three-turn nickname/color simulator scenario")
    parser.add_argument("--response-policy-only", action="store_true")
    parser.add_argument("--skip-robot", action="store_true", help="Skip cross-user device lifecycle checks")
    parser.add_argument("--user-id", help="User ID matching NAWEI_SIMULATOR_USER_TOKEN when using JWT")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(FullFlow(args.output, args.user_id).run(
        args.restart_only, args.memory_report, args.response_policy_only, args.skip_robot,
    )) else 1)
