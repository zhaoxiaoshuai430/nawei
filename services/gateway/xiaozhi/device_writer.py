import asyncio
import time
from typing import Any

from fastapi import WebSocket


class DeviceWriter:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.queue: asyncio.Queue[tuple[int, dict[str, Any] | bytes]] = asyncio.Queue(maxsize=64)
        self.generation = 0
        self.stopped_generation: int | None = None

    @property
    def turn_complete(self) -> bool:
        return self.stopped_generation == self.generation

    def invalidate(self) -> None:
        self.generation += 1
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

    async def send(self, payload: dict[str, Any] | bytes, generation: int | None = None) -> None:
        version = self.generation if generation is None else generation
        if version == self.generation:
            await asyncio.wait_for(self.queue.put((version, payload)), timeout=5)

    async def flush(self) -> None:
        await asyncio.wait_for(self.queue.join(), timeout=10)

    async def run(self) -> None:
        next_audio = 0.0
        while True:
            generation, payload = await self.queue.get()
            try:
                if generation != self.generation:
                    continue
                if isinstance(payload, bytes):
                    await asyncio.sleep(max(0, next_audio - time.monotonic()))
                    if generation != self.generation:
                        continue
                    await asyncio.wait_for(self.websocket.send_bytes(payload), 5)
                    next_audio = time.monotonic() + 0.06
                else:
                    if payload.get("type") == "tts" and payload.get("state") == "stop":
                        await asyncio.sleep(max(0, next_audio - time.monotonic()))
                        if generation != self.generation:
                            continue
                        # A client can start its next turn before send_json returns.
                        # A barge-in stop belongs to the cancelled turn, not the new generation.
                        if payload.get("reason") != "barge_in":
                            self.stopped_generation = generation
                    await asyncio.wait_for(self.websocket.send_json(payload), 5)
                    if payload.get("type") == "tts" and payload.get("state") == "start":
                        next_audio = time.monotonic() + 0.06
            finally:
                self.queue.task_done()
