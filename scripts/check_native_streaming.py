"""Verify partial text before input ends, using real ASR and robot voice services."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import secrets
import sys
import time
import uuid
from pathlib import Path

import av
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_full_flow import FullFlow
from services.gateway.xiaozhi.opus_codec import OpusDecoder, OpusEncoder


async def exercise(flow: FullFlow, kind: str) -> dict:
    robot = kind == "robot"
    direct = kind in {"asr", "legacy"}
    legacy = kind == "legacy"
    device = "02:" + ":".join(f"{value:02x}" for value in secrets.token_bytes(5))
    bound = False
    headers = {}
    path = "/api/v1/devices/" + device
    session = uuid.uuid4().hex
    try:
        if robot:
            identity = {"Device-Id": device, "Client-Id": str(uuid.uuid4()), "X-Device-Key": secrets.token_hex(32)}
            activation = await flow.request("POST", "/xiaozhi/ota/", headers=identity, json={})
            await flow.request("POST", "/api/v1/devices/bind", json={"code": activation["activation"]["code"], "name": "Streaming acceptance"})
            bound = True
            config = (await flow.request("POST", "/xiaozhi/ota/", headers=identity))["websocket"]
            headers = {**identity, "Authorization": "Bearer " + config["token"], "Protocol-Version": "1"}
            url = "ws://127.0.0.1:8000/xiaozhi/v1/"
            start = {"type": "hello", "version": 1, "transport": "websocket",
                     "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60}}
        elif direct:
            url = "ws://127.0.0.1:8001/api/v1/asr/realtime" + ("/sensevoice" if legacy else "")
            start = {"type": "start", "sample_rate": 16000, "channels": 1,
                     "internal_api_key": flow.config.get("ASR_INTERNAL_API_KEY", "")}
        else:
            url = "ws://127.0.0.1:8000/api/v1/voice/realtime"
            start = {"type": "start", "request_id": session, "trace_id": session, "session_id": session,
                     "user_id": flow.user, "access_token": flow.token, "output_audio": True,
                     "voice": {"speaker_id": "liliya", "language": "Chinese"},
                     "output_audio_config": {"format": "wav", "sample_rate": 24000, "channels": 1}}
        events = []
        transcript = ""
        answer = ""
        first_partial = None
        first_audio = None
        input_finished = None
        audio_samples = 0
        audio_peak = 0
        began = time.perf_counter()
        decoder = OpusDecoder(24000) if robot else None
        async with connect(url, additional_headers=headers, max_size=16 * 1024 * 1024) as socket:
            await socket.send(json.dumps(start))
            ready = json.loads(await asyncio.wait_for(socket.recv(), 15))
            if robot:
                assert ready["type"] == "hello"
                await socket.send(json.dumps({"type": "listen", "state": "start", "mode": "manual"}))
            elif direct:
                assert ready["type"] == "ready", ready
                assert ready["partial_results"] is not legacy
            else:
                assert ready["event"] == "ready", ready
                assert ready["data"]["partial_results"]

            async def send_audio():
                nonlocal input_finished
                if robot:
                    encoder = OpusEncoder(16000)
                    packets = encoder.feed(flow.pcm, 16000) + encoder.finish()
                else:
                    packets = [flow.pcm[offset:offset + 1920] for offset in range(0, len(flow.pcm), 1920)]
                for packet in packets:
                    await socket.send(packet)
                    await asyncio.sleep(0.06)
                input_finished = round((time.perf_counter() - began) * 1000)
                await socket.send(json.dumps({"type": "listen", "state": "stop"} if robot else {"type": "stop"}))

            sender = asyncio.create_task(send_audio())
            try:
                async with asyncio.timeout(120):
                    async for raw in socket:
                        elapsed = round((time.perf_counter() - began) * 1000)
                        if isinstance(raw, bytes):
                            assert decoder is not None
                            pcm = decoder.decode(raw)
                            import struct
                            audio_samples += len(pcm) // 2
                            audio_peak = max(audio_peak, max((abs(value[0]) for value in struct.iter_unpack("<h", pcm)), default=0))
                            if first_audio is None:
                                first_audio = elapsed
                            continue
                        body = json.loads(raw)
                        event = body.get("event", body.get("type"))
                        data = body.get("data", body)
                        assert event != "error", data
                        events.append({"type": event, "state": data.get("state"), "text": data.get("text"),
                                       "final": data.get("final"), "elapsed_ms": elapsed})
                        if event == "ping":
                            await socket.send(json.dumps({"type": "ping"}))
                        if event == "transcript.partial" or (event == "stt" and data.get("final") is False):
                            if first_partial is None:
                                first_partial = elapsed
                        if event == "transcript.final" or event == "asr.completed" or (event == "stt" and data.get("final") is not False):
                            transcript = data.get("text", "")
                        if event == "message.delta":
                            answer += data.get("text", "")
                        if event == "tts" and data.get("state") == "sentence_start":
                            answer += data.get("text", "")
                        if event == "audio.delta":
                            if first_audio is None:
                                first_audio = elapsed
                            with av.open(io.BytesIO(base64.b64decode(data["audio_base64"]))) as container:
                                for frame in container.decode(audio=0):
                                    audio_samples += frame.samples
                                    audio_peak = max(audio_peak, float(abs(frame.to_ndarray()).max()))
                        if (direct and event == "session.done") or (robot and event == "tts" and data.get("state") == "stop"):
                            break
                await sender
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
        assert transcript, "Missing final transcription"
        if not legacy:
            assert first_partial is not None and input_finished is not None and first_partial < input_finished, "No real partial before input ended"
        if not direct:
            assert answer and audio_samples > 0 and audio_peak > 0, "Voice response did not complete"
            if not robot:
                assert {"message.done", "audio.done"}.issubset({event["type"] for event in events})
        report = dict(first_partial_ms=first_partial, input_finished_ms=input_finished, first_audio_ms=first_audio,
                      transcript=transcript, answer=answer, audio_samples=audio_samples, audio_peak=audio_peak,
                      events=events, local_transport=True, physical_robot=False)
        (flow.output / f"{kind}-events.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {key: value for key, value in report.items() if key != "events"}
    finally:
        if bound:
            await flow.request("DELETE", path + "/binding")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", help="User ID matching NAWEI_SIMULATOR_USER_TOKEN when using JWT")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/native-streaming-20260910")
    args = parser.parse_args()
    flow = FullFlow(args.output, args.user_id)
    try:
        await flow.check("health", flow.health)
        await flow.check("tts_source", flow.source_audio)
        for kind in ("asr", "gateway", "robot", "legacy"):
            await flow.check(kind + "_streaming", lambda kind=kind: exercise(flow, kind))
        await flow.check("final_health", flow.health)
    finally:
        await flow.client.aclose()
    if not flow.results.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
