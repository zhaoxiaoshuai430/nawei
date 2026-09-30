"""Warm local speech inference without creating a conversation or calling an LLM."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_protocols import ASRRequest


async def warm_voice(client: httpx.AsyncClient, config: dict[str, Any]) -> dict[str, Any]:
    request_id = "voice-warmup-" + uuid.uuid4().hex
    common = {"request_id": request_id, "trace_id": request_id, "session_id": request_id}
    timings = {}

    def headers(service: str) -> dict[str, str]:
        key = config.get(service + "_INTERNAL_API_KEY")
        return {"X-Internal-API-Key": key} if key else {}

    started = time.perf_counter()
    response = await client.post(
        config.get("GATEWAY_TTS_URL") or "http://127.0.0.1:19000/api/v1/audio/speech",
        headers=headers("TTS"),
        json={**common, "text": "\u4f60\u597d\uff0c\u8fd9\u662f\u8bed\u97f3\u670d\u52a1\u9884\u70ed\u3002",
              "voice": {"speaker_id": "liliya", "language": "Chinese"},
              "audio_config": {"format": "pcm16", "sample_rate": 16000, "channels": 1}},
    )
    response.raise_for_status()
    audio = response.json()
    if audio.get("format") != "pcm16" or audio.get("sample_rate") != 16000 or audio.get("channels", 1) != 1:
        raise ValueError("Warmup requires mono PCM16 at 16000 Hz")
    pcm = base64.b64decode(audio["audio_base64"], validate=True)
    if not pcm or len(pcm) % 2:
        raise ValueError("Warmup synthesis returned invalid audio")
    timings["tts_ms"] = round((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    response = await client.post(
        config.get("GATEWAY_ASR_URL") or "http://127.0.0.1:8001/api/v1/audio/transcriptions/base64",
        headers=headers("ASR"),
        json=ASRRequest(**common, audio_base64=audio["audio_base64"], format="pcm16", sample_rate=16000,
                        channels=1, language="zh", options={"use_itn": True}).model_dump(mode="json"),
    )
    response.raise_for_status()
    if not response.json().get("text", "").strip():
        raise ValueError("Warmup transcription returned no text")
    timings["asr_ms"] = round((time.perf_counter() - started) * 1000)

    response = await client.get(config.get("GATEWAY_ASR_HEALTH_URL") or "http://127.0.0.1:8001/health")
    response.raise_for_status()
    workers = int(response.json().get("streaming_concurrency", {}).get("limit", 0))
    if workers:
        barrier = asyncio.Barrier(workers)

        async def warm_stream(index: int) -> None:
            async with connect(config.get("GATEWAY_ASR_REALTIME_URL") or "ws://127.0.0.1:8001/api/v1/asr/realtime") as socket:
                await socket.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1,
                    "internal_api_key": config.get("ASR_INTERNAL_API_KEY") or ""}))
                if json.loads(await socket.recv()).get("type") != "ready":
                    raise ValueError("Realtime ASR is not ready")
                # Keep all sessions open so each worker receives one connection.
                await barrier.wait()
                for offset in range(0, len(pcm), 32000):
                    await socket.send(pcm[offset:offset + 32000])
                await socket.send(json.dumps({"type": "stop"}))
                transcribed = False
                async for raw in socket:
                    event = json.loads(raw)
                    if event.get("type") == "error":
                        raise ValueError("Realtime ASR warmup failed")
                    if event.get("type") == "transcript.final" and event.get("text"):
                        transcribed = True
                    if event.get("type") == "session.done":
                        if not transcribed:
                            raise ValueError("Realtime ASR warmup returned no text")
                        return
                raise ValueError("Realtime ASR warmup closed before completion")

        started = time.perf_counter()
        async with asyncio.TaskGroup() as group:
            for index in range(workers):
                group.create_task(warm_stream(index))
        timings["realtime_asr_ms"] = round((time.perf_counter() - started) * 1000)
    return {"ready": True, "streaming_workers": workers, "timings": timings}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/voice-warmup.json")
    args = parser.parse_args()
    config = {**dotenv_values(ROOT / ".env"), **dotenv_values(ROOT / ".env.robot"), **os.environ}
    try:
        async with asyncio.timeout(args.timeout):
            async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
                result = await warm_voice(client, config)
    except Exception as error:
        result = {"ready": False, "error_type": type(error).__name__}
        if isinstance(error, httpx.HTTPStatusError):
            result.update(status_code=error.response.status_code, endpoint=error.request.url.path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
