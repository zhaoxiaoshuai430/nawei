"""Compare real TTS chunk sizes with interleaved trials and playable PCM output."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import struct
import time
import uuid
import wave
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


async def measure(client: httpx.AsyncClient, url: str, key: str, text: str, size: int, path: Path) -> dict:
    identity = uuid.uuid4().hex
    payload = {
        "request_id": identity, "trace_id": identity, "session_id": "tts-speed-test",
        "text": text, "stream": True,
        "voice": {"speaker_id": "liliya", "language": "Chinese"},
        "audio_config": {"format": "pcm16", "sample_rate": 24000, "channels": 1},
        "metadata": {"chunk_size": size, "temperature": 0.1, "top_k": 1},
    }
    started = time.perf_counter()
    first_ms = None
    audio = bytearray()
    chunks = 0
    async with client.stream("POST", url, headers={"X-Internal-API-Key": key}, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = json.loads(line[6:])
            if body.get("type") == "error":
                raise RuntimeError("TTS stream returned an error")
            if body.get("type") != "chunk":
                continue
            pcm = base64.b64decode(body["audio_b64"], validate=True)
            if pcm and first_ms is None:
                first_ms = round((time.perf_counter() - started) * 1000)
            audio.extend(pcm)
            chunks += 1
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    peak = max((abs(sample[0]) for sample in struct.iter_unpack("<h", audio)), default=0)
    if first_ms is None or peak <= 64:
        raise ValueError("TTS returned empty or silent audio")
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        output.writeframes(audio)
    duration = len(audio) / 48000
    return {"chunk_size": size, "first_audio_ms": first_ms, "total_ms": elapsed_ms,
            "audio_seconds": round(duration, 3), "rtf": round(elapsed_ms / 1000 / duration, 3),
            "chunks": chunks, "peak": peak, "audio_path": str(path)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="\u6c34\u84b8\u6c14\u5347\u5230\u9ad8\u7a7a\u9047\u51b7\uff0c")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 4, 2])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/tts-speed")
    args = parser.parse_args()
    if not 1 <= args.rounds <= 20 or any(size < 1 or size > 25 for size in args.sizes):
        parser.error("Use 1-20 rounds and chunk sizes in 1-25")
    config = {**dotenv_values(ROOT / ".env"), **dotenv_values(ROOT / ".env.robot"), **os.environ}
    url = config.get("GATEWAY_TTS_STREAM_URL") or "http://127.0.0.1:19000/api/v1/audio/speech/stream"
    key = config.get("TTS_INTERNAL_API_KEY") or ""
    args.output.mkdir(parents=True, exist_ok=True)
    trials = []
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        for index in range(args.rounds):
            sizes = args.sizes[index % len(args.sizes):] + args.sizes[:index % len(args.sizes)]
            for size in sizes:
                record = await measure(client, url, key, args.text, size, args.output / f"round-{index + 1}-chunk-{size}.wav")
                trials.append(record)
                print(json.dumps(record), flush=True)
    summary = {str(size): {
        "median_first_audio_ms": statistics.median(item["first_audio_ms"] for item in trials if item["chunk_size"] == size),
        "median_rtf": statistics.median(item["rtf"] for item in trials if item["chunk_size"] == size),
    } for size in args.sizes}
    (args.output / "report.json").write_text(json.dumps({"text": args.text, "trials": trials, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
