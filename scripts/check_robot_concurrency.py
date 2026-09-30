"""Run two synchronized robot sessions against local services and record capacity."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import uuid
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_full_flow import FullFlow
from scripts.simulate_robot import simulate


async def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    flow = FullFlow(output)
    questions = [
        "\u4e00\u52a0\u4e00\u7b49\u4e8e\u51e0\uff1f\u8bf7\u53ea\u56de\u7b54\u7ed3\u679c\u3002",
        "\u4e09\u52a0\u4e09\u7b49\u4e8e\u51e0\uff1f\u8bf7\u53ea\u56de\u7b54\u7ed3\u679c\u3002",
    ]
    fixtures = []
    try:
        for index, question in enumerate(questions):
            audio = await flow.request("POST", "/api/v1/audio/speech", json={
                "request_id": uuid.uuid4().hex, "trace_id": uuid.uuid4().hex,
                "session_id": flow.user, "text": question,
                "voice": {"speaker_id": "liliya", "language": "Chinese"},
                "audio_config": {"format": "pcm16", "sample_rate": 16000, "channels": 1},
            })
            assert audio["format"] == "pcm16" and audio["sample_rate"] == 16000
            fixture = output / f"question-{index}.wav"
            with wave.open(str(fixture), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(16000)
                target.writeframes(base64.b64decode(audio["audio_base64"], validate=True))
            fixtures.append(fixture)
    finally:
        await flow.client.aclose()

    barrier = asyncio.Barrier(2)
    samples = []
    stop = asyncio.Event()
    began = time.monotonic()

    async def monitor():
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while not stop.is_set():
                sample = {"elapsed_ms": round((time.monotonic() - began) * 1000)}
                for name, url in [
                    ("asr", "http://127.0.0.1:8001/health"),
                    ("tts", "http://127.0.0.1:19000/api/v1/status"),
                ]:
                    try:
                        response = await client.get(url)
                        response.raise_for_status()
                        key = "streaming_concurrency" if name == "asr" else "concurrency"
                        sample[name] = response.json()[key]
                    except Exception as error:
                        sample[name + "_error"] = type(error).__name__
                samples.append(sample)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.05)
                except TimeoutError:
                    pass

    monitor_task = asyncio.create_task(monitor())
    try:
        results = await asyncio.gather(*[
            simulate(argparse.Namespace(
                url="http://127.0.0.1:8000", text=[question], voice="liliya",
                source_wav=fixtures[index], rounds=1, mode="manual", output_rate=24000,
                turn_timeout=180, report=output / f"robot-{index}.json", start_barrier=barrier,
            ))
            for index, question in enumerate(questions)
        ], return_exceptions=True)
        await asyncio.sleep(0.3)
    finally:
        stop.set()
        await monitor_task

    successful = [result for result in results if isinstance(result, dict) and result.get("passed")]
    report = {
        "mock_services": False, "clients": 2, "successful_clients": len(successful),
        "errors": [type(result).__name__ for result in results if isinstance(result, BaseException)],
        "samples": samples,
        "peaks": {
            name: {key: max((sample.get(name, {}).get(key, 0) for sample in samples), default=0)
                   for key in ("active", "waiting", "sessions")}
            for name in ("asr", "tts")
        },
        "final": samples[-1] if samples else {},
        "robots": successful,
    }
    report["passed"] = (
        len(successful) == 2
        and all(result.get("unbound") for result in successful)
        and report["peaks"]["asr"]["active"] == 2
        and all(not any(key.endswith("_error") for key in sample) for sample in samples)
        and all(report["final"].get(name, {}).get("active") == 0 for name in ("asr", "tts"))
        and all(report["final"].get(name, {}).get("waiting") == 0 for name in ("asr", "tts"))
        and report["final"].get("asr", {}).get("sessions") == 0
    )
    (output / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"samples", "robots"}}, ensure_ascii=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/robot-concurrency")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output))["passed"] else 1)
