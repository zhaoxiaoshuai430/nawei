from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import aclosing
from typing import Any

from websockets.asyncio.client import connect

from app_protocols import AudioConfig, AudioFormat, EventType, Message, MessageRole, VoiceConfig

from .event_adapter import EventAdapter
from .opus_codec import OpusDecoder
from .session import DeviceSession
from services.gateway.voice_preview import VoicePreview

logger = logging.getLogger(__name__)


async def transcribe(session: DeviceSession, mode: str) -> str:
    url = os.getenv("GATEWAY_ASR_REALTIME_URL", "ws://127.0.0.1:8001/api/v1/asr/realtime")
    async with connect(url, open_timeout=10, max_size=1024 * 1024) as upstream:
        await upstream.send(json.dumps({
            "type": "start", "request_id": f"{session.session_id}-{session.turn_number}",
            "trace_id": session.session_id, "session_id": session.session_id,
            "sample_rate": 16000, "channels": 1, "language": "auto", "use_itn": True,
            "internal_api_key": os.getenv("ASR_INTERNAL_API_KEY", ""),
        }))
        decoder = OpusDecoder()

        async def forward() -> None:
            total = 0
            while True:
                packet = await session.audio.get()
                if packet is None:
                    await upstream.send(json.dumps({"type": "stop"}))
                    return
                pcm = decoder.decode(packet)
                total += len(pcm)
                if total > 16000 * 2 * 60:
                    raise ValueError("Maximum utterance duration exceeded")
                if pcm:
                    await upstream.send(pcm)

        async def collect() -> str:
            texts: list[str] = []
            async for raw in upstream:
                if not isinstance(raw, str):
                    continue
                data = json.loads(raw)
                kind = data.get("type")
                if kind == "error":
                    raise RuntimeError("ASR failed")
                if kind == "transcript.partial":
                    partial_text = " ".join([*texts, str(data.get("text") or "").strip()]).strip()
                    if session.preview is not None:
                        await session.preview.update(partial_text)
                    await session.writer.send({"type": "stt", "text": partial_text, "final": False,
                                               "segment_index": data.get("segment_index", 0),
                                               "session_id": session.session_id})
                if kind == "transcript.final" and data.get("text", "").strip():
                    texts.append(data["text"].strip())
                    if mode != "manual":
                        return " ".join(texts)
                if kind == "session.done":
                    return " ".join(texts)
            raise RuntimeError("ASR closed before final result")

        sender = asyncio.create_task(forward())
        reader = asyncio.create_task(collect())
        try:
            done, _ = await asyncio.wait({sender, reader}, return_when=asyncio.FIRST_COMPLETED)
            if sender in done:
                await sender
            return await reader
        finally:
            sender.cancel()
            reader.cancel()
            await asyncio.gather(sender, reader, return_exceptions=True)


async def run_turn(session: DeviceSession, mode: str) -> None:
    from services.gateway import main as gateway

    generation = session.writer.generation
    started = time.perf_counter()
    timings: dict[str, Any] = {"session_id": session.session_id, "turn": session.turn_number, "mode": mode, "status": "ok"}
    request = gateway.RealtimeVoiceRequest(
        request_id=f"{session.session_id}-{session.turn_number}", trace_id=session.session_id,
        session_id=session.session_id, user_id=session.device["user_id"],
        voice=VoiceConfig(speaker_id=session.device["voice"]),
        output_audio_config=AudioConfig(format=AudioFormat.PCM16, sample_rate=session.sample_rate, channels=1),
        metadata={"device_id": session.device["device_id"], "client_id": session.device["client_id"],
                  "device_role": session.device["role"], "turn_number": session.turn_number},
    )
    preview = session.preview = VoicePreview(request)
    try:
        async with asyncio.timeout(180):
            transcript = await transcribe(session, mode)
            asr_finished = time.perf_counter()
            timings["asr_from_listen_ms"] = round((asr_finished - started) * 1000)
            if session.input_stopped_at is not None:
                timings["asr_after_stop_ms"] = round((asr_finished - session.input_stopped_at) * 1000)
            session.listening = False
            if not transcript:
                await session.writer.send({"type": "tts", "state": "stop", "session_id": session.session_id}, generation)
                return
            await session.writer.send({"type": "stt", "text": transcript, "final": True,
                                       "session_id": session.session_id}, generation)
            request.voice_preview_token = await preview.finish(transcript)
            timings["voice_preview_wait_ms"] = round((time.perf_counter() - asr_finished) * 1000)
            timings["voice_preview_ready"] = bool(request.voice_preview_token)
            timings["voice_preview_attempts"] = preview.attempts
            lease = await gateway._acquire_gateway_capacity(request.user_id)
            async with lease:
                timings["capacity_after_asr_ms"] = round((time.perf_counter() - asr_finished) * 1000)
                adapter = EventAdapter(session)
                async with aclosing(gateway._stream_voice_response(
                    request, Message(role=MessageRole.USER, content=transcript)
                )) as events:
                    async for event in events:
                        elapsed = round((time.perf_counter() - asr_finished) * 1000)
                        if event.event == EventType.MESSAGE_DELTA and event.data.get("text"):
                            timings.setdefault("llm_first_text_after_asr_ms", elapsed)
                        elif event.event == EventType.AUDIO_STARTED:
                            timings.setdefault("tts_started_after_asr_ms", elapsed)
                        elif event.event == EventType.AUDIO_DELTA:
                            timings.setdefault("tts_first_chunk_after_asr_ms", elapsed)
                        elif event.event == EventType.MESSAGE_DONE:
                            runtime = event.data.get("runtime", {})
                            timings["voice_preview_used"] = bool(runtime.get("voice_preview_used")) if isinstance(runtime, dict) else False
                            for key in ("context_ms", "tools_ms", "memory_write_ms"):
                                value = runtime.get("timings", {}).get(key) if isinstance(runtime, dict) else None
                                if isinstance(value, (int, float)):
                                    timings["llm_" + key] = value
                        await adapter.event(event)
                await adapter.finish()
            await session.writer.flush()
    except asyncio.CancelledError:
        timings["status"] = "cancelled"
        raise
    except Exception as error:
        timings["status"] = "error"
        logger.warning("Robot voice turn failed session=%s turn=%d error_type=%s",
                       session.session_id, session.turn_number, type(error).__name__)
        session.writer.invalidate()
        await session.writer.send({"type": "error", "message": "Voice turn failed", "session_id": session.session_id})
        await session.writer.send({"type": "tts", "state": "stop", "session_id": session.session_id})
    finally:
        await preview.close()
        session.preview = None
        timings["total_from_listen_ms"] = round((time.perf_counter() - started) * 1000)
        logging.getLogger("uvicorn.error").info("Robot turn timing %s", json.dumps(timings, ensure_ascii=True))
        session.listening = False
