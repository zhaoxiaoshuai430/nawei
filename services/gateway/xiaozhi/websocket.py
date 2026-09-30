import asyncio
import json
import time
from typing import Any

import anyio

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from .auth import device_identity
from .config import Settings
from .device_registry import DeviceRegistry
from .device_writer import DeviceWriter
from .pipeline import run_turn
from .schemas import Hello
from .session import DeviceSession, SessionManager


async def receive_command(websocket: WebSocket) -> dict[str, Any]:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect()
    text = message.get("text")
    if text is None or len(text.encode()) > 4096:
        raise ValueError("Expected a bounded JSON command")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


async def serve(websocket: WebSocket, registry: DeviceRegistry, sessions: SessionManager, settings: Settings) -> None:
    session: DeviceSession | None = None
    tasks: list[asyncio.Task[Any]] = []
    close_code = 1000
    try:
        device, client = device_identity(websocket.headers)
        authorization = websocket.headers.get("authorization", "")
        if websocket.headers.get("protocol-version") != "1" or not authorization.startswith("Bearer nwdev_"):
            raise HTTPException(401, "Invalid protocol or credential")
        token = authorization[7:]
        record = await asyncio.to_thread(registry.authenticate, device, client, token)
        await websocket.accept()
        hello = Hello.model_validate(await asyncio.wait_for(receive_command(websocket), settings.hello_timeout))
        writer = DeviceWriter(websocket)
        session = DeviceSession(record, writer, hello.output_rate())
        await sessions.attach(session)
        tasks.append(asyncio.create_task(writer.run()))
        await writer.send({"type": "hello", "version": 1, "transport": "websocket", "session_id": session.session_id,
                           "audio_params": {"format": "opus", "sample_rate": session.sample_rate,
                                            "channels": 1, "frame_duration": 60},
                           "features": {"barge_in": True, "full_duplex_transport": True}})
        await asyncio.to_thread(registry.seen, device)

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(settings.heartbeat_interval)
                await asyncio.to_thread(registry.authenticate, device, client, token)
                await asyncio.to_thread(registry.seen, device)
                await writer.send({"type": "ping", "session_id": session.session_id})

        tasks.extend((asyncio.create_task(heartbeat()), asyncio.create_task(_receive(websocket, session, registry, client, token)),
                      asyncio.create_task(session.stop.wait())))
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except HTTPException:
        close_code = 4401
    except (ValueError, TimeoutError):
        close_code = 4400
    except WebSocketDisconnect:
        pass
    except Exception:
        close_code = 1011
    finally:
        for task in tasks:
            task.cancel()
        try:
            with anyio.move_on_after(10, shield=True):
                if session:
                    await session.cancel_turn()
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await websocket.close(code=close_code)
                except (RuntimeError, WebSocketDisconnect):
                    pass
        finally:
            if session:
                sessions.detach(session)


async def _receive(websocket: WebSocket, session: DeviceSession, registry: DeviceRegistry, client: str, token: str) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        packet = message.get("bytes")
        if packet is not None:
            if not packet or len(packet) > 1275:
                raise ValueError("Invalid Opus packet size")
            if session.listening:
                try:
                    session.audio.put_nowait(packet)
                except asyncio.QueueFull as exc:
                    raise ValueError("Audio queue full") from exc
            continue
        text = message.get("text", "")
        if len(text.encode()) > 4096:
            raise ValueError("Command too large")
        command = json.loads(text)
        if not isinstance(command, dict):
            raise ValueError("Invalid command")
        kind = command.get("type")
        if command.get("session_id", session.session_id) != session.session_id:
            raise ValueError("Session mismatch")
        if kind == "ping":
            await session.writer.send({"type": "pong", "session_id": session.session_id})
        elif kind == "abort":
            await session.cancel_turn()
            await session.writer.send({"type": "tts", "state": "stop", "session_id": session.session_id})
        elif kind == "listen":
            state = command.get("state")
            if state == "start":
                mode = command.get("mode", "auto")
                if mode not in {"auto", "manual", "realtime"}:
                    raise ValueError("Invalid listen mode")
                session.device = await asyncio.to_thread(registry.authenticate, session.device["device_id"], client, token)
                interrupted = session.turn is not None and not session.turn.done() and not session.writer.turn_complete
                await session.cancel_turn()
                if interrupted:
                    await session.writer.send({"type": "tts", "state": "stop", "reason": "barge_in",
                                               "session_id": session.session_id})
                session.turn_number += 1
                session.input_stopped_at = None
                session.listening = True
                session.turn = asyncio.create_task(run_turn(session, mode))
            elif state == "stop" and session.listening:
                session.input_stopped_at = time.perf_counter()
                session.listening = False
                try:
                    session.audio.put_nowait(None)
                except asyncio.QueueFull as exc:
                    raise ValueError("Audio queue full") from exc
