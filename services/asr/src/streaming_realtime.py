"""Bounded PCM WebSocket transport for the native online ASR model."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from realtime import PCM16VadSegmenter, RealtimeVadConfig, RealtimeVadEvent
from service_security import verify_asr_internal_api_key
from streaming_model import MODEL_ID, StreamingDecoder, StreamingTranscript

logger = logging.getLogger(__name__)


def parse_command(message: dict[str, Any]) -> dict[str, Any]:
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect()
    text = message.get("text")
    if not isinstance(text, str) or len(text.encode("utf-8")) > 4096:
        raise ValueError("Expected a JSON command of at most 4096 bytes.")
    command = json.loads(text)
    if not isinstance(command, dict):
        raise ValueError("Expected a JSON command object.")
    return command


async def serve_streaming(websocket: WebSocket, decoder: StreamingDecoder) -> None:
    await websocket.accept()
    tasks: list[asyncio.Task[Any]] = []
    send_lock = asyncio.Lock()

    async def send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    try:
        start = parse_command(await asyncio.wait_for(websocket.receive(), timeout=10))
        if start.get("type") != "start":
            raise ValueError("First message must be start.")
        verify_asr_internal_api_key(str(start.get("internal_api_key") or "") or None)
        if start.get("sample_rate", 16000) != 16000 or start.get("channels", 1) != 1:
            raise ValueError("Expected mono PCM16 at 16000 Hz.")
        if start.get("language", "auto") not in {"auto", "zh", "cn"}:
            raise ValueError("Online ASR supports Chinese; use the SenseVoice endpoint for other languages.")
        config = RealtimeVadConfig.from_env()
        if config.sample_rate != 16000:
            raise ValueError("Online ASR requires ASR_REALTIME_SAMPLE_RATE=16000.")
        max_seconds = float(os.getenv("ASR_REALTIME_MAX_SESSION_SECONDS", "3600"))
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Invalid maximum session duration.")
        segmenter = PCM16VadSegmenter(config, emit_chunks=True)
        transcript = StreamingTranscript(decoder)
        queue: asyncio.Queue[list[RealtimeVadEvent] | None] = asyncio.Queue(maxsize=16)

        async def receive() -> None:
            received = 0
            while True:
                message = await asyncio.wait_for(websocket.receive(), timeout=60)
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect()
                pcm = message.get("bytes")
                if pcm is not None:
                    if not pcm or len(pcm) > 32000:
                        raise ValueError("Audio packets must contain 1 to 32000 bytes.")
                    received += len(pcm)
                    if received > int(max_seconds * 32000):
                        raise ValueError("Maximum audio session duration exceeded.")
                    events = segmenter.feed(pcm)
                    if events:
                        await queue.put(events)
                    continue
                command = parse_command(message)
                kind = command.get("type")
                if kind == "ping":
                    await send({"type": "pong"})
                elif kind == "stop":
                    await queue.put(segmenter.flush())
                    await queue.put(None)
                    return
                elif kind == "cancel":
                    raise WebSocketDisconnect()
                else:
                    raise ValueError("Unknown command.")

        async def recognize() -> None:
            while True:
                events = await queue.get()
                if events is None:
                    await send({"type": "session.done", "segments": transcript.segment_index})
                    return
                for event in events:
                    if event.kind == "speech_started":
                        await send({"type": "speech.started", "segment_index": transcript.segment_index})
                    elif event.kind == "audio":
                        for partial in await transcript.feed(event.pcm):
                            await send(partial)
                    elif event.kind == "segment":
                        await send({"type": "speech.ended", "segment_index": transcript.segment_index})
                        await send(await transcript.finish())
                    elif event.kind == "discard":
                        transcript.reset()

        await send({
            "type": "ready", "sample_rate": 16000, "channels": 1, "format": "pcm16le",
            "partial_results": True, "keeps_decoder_state": True,
            "streaming_mode": "native-online", "model": MODEL_ID, "chunk_ms": 600,
        })
        reader = asyncio.create_task(receive())
        worker = asyncio.create_task(recognize())
        tasks = [reader, worker]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if reader in done:
            await worker
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        pass
    except HTTPException:
        await send({"type": "error", "code": "unauthorized", "message": "Invalid internal credential."})
        await websocket.close(code=4401)
    except (ValueError, TypeError, TimeoutError) as error:
        await send({"type": "error", "code": "invalid_request", "message": str(error)})
        await websocket.close(code=4400)
    except Exception:
        logger.exception("Online ASR session failed")
        try:
            await send({"type": "error", "code": "recognition_failed", "message": "Online ASR failed."})
            await websocket.close(code=1011)
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
