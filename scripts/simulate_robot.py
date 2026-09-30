"""Exercise OTA, account binding, real Opus voice turns and revocation against a running Gateway."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import struct
import sys
import time
import uuid
import wave
from pathlib import Path

import httpx
from websockets.asyncio.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.gateway.xiaozhi.opus_codec import OpusDecoder, OpusEncoder


async def simulate(args: argparse.Namespace) -> dict:
    user = getattr(args, "user_id", None) or "robot-integration-" + uuid.uuid4().hex[:12]
    authorization = os.getenv("NAWEI_SIMULATOR_USER_TOKEN")
    user_headers = {"Authorization": "Bearer " + authorization} if authorization else {"X-User-ID": user}
    device = "02:" + ":".join(f"{value:02x}" for value in secrets.token_bytes(5))
    identity = {"Device-Id": device, "Client-Id": str(uuid.uuid4()), "X-Device-Key": secrets.token_hex(32)}
    mode = getattr(args, "mode", "manual")
    output_rate = getattr(args, "output_rate", 24000)
    result: dict = {"device_id": device, "user_id": user, "transport": "opus-v1", "mode": mode,
                    "mock_services": False, "source": "wav_fixture" if getattr(args, "source_wav", None) else "generated_tts", "turns": []}
    bound = False
    async with httpx.AsyncClient(base_url=args.url, timeout=180, trust_env=False) as client:
        try:
            started = time.perf_counter()
            response = await client.post("/xiaozhi/ota/", headers=identity, json={})
            response.raise_for_status()
            result["ota_ms"] = round((time.perf_counter() - started) * 1000)
            code = response.json()["activation"]["code"]
            response = await client.post("/api/v1/devices/bind", headers=user_headers,
                                         json={"code": code, "name": "Protocol integration simulator"})
            response.raise_for_status()
            bound = True
            response = await client.post("/xiaozhi/ota/", headers=identity, json={})
            response.raise_for_status()
            configuration = response.json()["websocket"]
            packets_by_text = {}
            for text in args.text:
                if getattr(args, "source_wav", None):
                    with wave.open(str(args.source_wav), "rb") as source:
                        assert source.getsampwidth() == 2 and source.getnchannels() == 1
                        pcm = source.readframes(source.getnframes())
                        audio = {"format": "pcm16", "channels": 1, "sample_rate": source.getframerate()}
                else:
                    response = await client.post("/api/v1/audio/speech", headers=user_headers, json={
                        "request_id": uuid.uuid4().hex, "trace_id": uuid.uuid4().hex, "session_id": user,
                        "text": text, "voice": {"speaker_id": args.voice, "language": "Chinese"},
                        "audio_config": {"format": "pcm16", "sample_rate": 24000, "channels": 1},
                    })
                    response.raise_for_status()
                    audio = response.json()
                    pcm = base64.b64decode(audio["audio_base64"], validate=True)
                if audio["format"] != "pcm16" or audio.get("channels", 1) != 1:
                    raise ValueError("Simulator source TTS must return mono PCM16")
                encoder = OpusEncoder(16000)
                if mode == "auto":
                    pcm += bytes(audio["sample_rate"] * 2)
                packets_by_text[text] = encoder.feed(pcm, audio["sample_rate"]) + encoder.finish()
            headers = {**identity, "Authorization": "Bearer " + configuration["token"], "Protocol-Version": "1"}
            started = time.perf_counter()
            async with connect(configuration["url"], additional_headers=headers, open_timeout=10, max_size=65536) as websocket:
                await websocket.send(json.dumps({"type": "hello", "version": 1, "transport": "websocket",
                                                "audio_params": {"format": "opus", "sample_rate": 16000,
                                                                 "channels": 1, "frame_duration": 60},
                                                "output_audio_params": {"format": "opus", "sample_rate": output_rate,
                                                                        "channels": 1, "frame_duration": 60}}))
                hello = json.loads(await asyncio.wait_for(websocket.recv(), 10))
                assert hello["type"] == "hello"
                result["hello_ms"] = round((time.perf_counter() - started) * 1000)
                barrier = getattr(args, "start_barrier", None)
                if barrier is not None:
                    await asyncio.wait_for(barrier.wait(), timeout=30)
                result["session_id"] = hello["session_id"]
                result["output_sample_rate"] = hello["audio_params"]["sample_rate"]
                assert result["output_sample_rate"] == output_rate
                for index in range(args.rounds):
                    text = args.text[index % len(args.text)]
                    started = time.perf_counter()
                    await websocket.send(json.dumps({"type": "listen", "state": "start", "mode": mode}))
                    for packet in packets_by_text[text]:
                        await websocket.send(packet)
                        await asyncio.sleep(0.06)
                    if mode == "manual":
                        await websocket.send(json.dumps({"type": "listen", "state": "stop"}))
                    stopped = time.perf_counter()
                    record: dict = {"turn": index + 1, "question": text, "audio_packets": 0,
                                    "decoded_bytes": 0, "audio_peak": 0}
                    decoder = OpusDecoder(result["output_sample_rate"])
                    speaking = False
                    async with asyncio.timeout(args.turn_timeout):
                        while True:
                            message = await websocket.recv()
                            elapsed = round((time.perf_counter() - stopped) * 1000)
                            if isinstance(message, bytes):
                                assert speaking, "Audio arrived outside tts/start and tts/stop"
                                record.setdefault("first_audio_ms", elapsed)
                                record["audio_packets"] += 1
                                decoded = decoder.decode(message)
                                record["decoded_bytes"] += len(decoded)
                                record["audio_peak"] = max(record["audio_peak"],
                                                           max((abs(sample[0]) for sample in struct.iter_unpack("<h", decoded)), default=0))
                                continue
                            body = json.loads(message)
                            if body["type"] == "ping":
                                await websocket.send(json.dumps({"type": "pong"}))
                            elif body["type"] == "error":
                                raise RuntimeError(body.get("message", "Gateway error"))
                            elif body["type"] == "stt" and body.get("final", True):
                                record["asr_ms"] = elapsed
                                record["transcript"] = body["text"]
                            elif body["type"] == "tts":
                                if body["state"] == "start":
                                    speaking = True
                                elif body["state"] == "sentence_start":
                                    record.setdefault("first_sentence_ms", elapsed)
                                    record["answer"] = record.get("answer", "") + body.get("text", "")
                                elif body["state"] == "stop":
                                    assert record["audio_packets"] > 0, "No answer audio"
                                    assert record["audio_peak"] > 64, "Answer audio is silent"
                                    assert record.get("transcript") and record.get("answer"), "Missing transcript or answer"
                                    break
                    record["total_ms"] = round((time.perf_counter() - started) * 1000)
                    result["turns"].append(record)
                    print(json.dumps(record, ensure_ascii=True), flush=True)
            result["passed"] = True
        except Exception as error:
            result["passed"] = False
            result["error_type"] = type(error).__name__
            raise
        finally:
            try:
                if bound:
                    response = await client.delete(f"/api/v1/devices/{device}/binding", headers=user_headers)
                    result["unbound"] = response.status_code == 204
                    response.raise_for_status()
            except Exception as error:
                result["passed"] = False
                result["cleanup_error_type"] = type(error).__name__
                raise
            finally:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", help="Use the JWT subject when testing an authenticated public deployment")
    parser.add_argument("--text", action="append")
    parser.add_argument("--voice", default="liliya")
    parser.add_argument("--source-wav", type=Path, help="Use a fixed mono PCM16 WAV without generating or warming TTS input")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--mode", choices=["manual", "auto"], default="manual")
    parser.add_argument("--output-rate", type=int, choices=[16000, 24000], default=24000)
    parser.add_argument("--turn-timeout", type=int, default=180)
    parser.add_argument("--report", type=Path, default=Path(".runtime/xiaozhi-live.json"))
    args = parser.parse_args()
    if not 1 <= args.rounds <= 100:
        parser.error("rounds must be between 1 and 100")
    args.text = args.text or ["你好，请用一句话介绍你自己。"]
    if args.source_wav and len(args.text) != 1:
        parser.error("source-wav requires exactly one reference text")
    asyncio.run(simulate(args))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
