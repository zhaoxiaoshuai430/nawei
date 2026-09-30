import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from .device_writer import DeviceWriter
from services.gateway.voice_preview import VoicePreview


@dataclass
class DeviceSession:
    device: dict[str, Any]
    writer: DeviceWriter
    sample_rate: int
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    audio: asyncio.Queue[bytes | None] = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    turn: asyncio.Task[None] | None = None
    turn_number: int = 0
    listening: bool = False
    input_stopped_at: float | None = None
    preview: VoicePreview | None = None

    async def cancel_turn(self) -> None:
        self.listening = False
        self.writer.invalidate()
        if self.turn is not None:
            self.turn.cancel()
            await asyncio.gather(self.turn, return_exceptions=True)
            self.turn = None
        while not self.audio.empty():
            self.audio.get_nowait()


class SessionManager:
    def __init__(self) -> None:
        self.active: dict[str, DeviceSession] = {}

    async def attach(self, session: DeviceSession) -> None:
        device = session.device["device_id"]
        previous = self.active.get(device)
        self.active[device] = session
        if previous:
            previous.stop.set()
            await asyncio.wait_for(previous.finished.wait(), timeout=10)

    def detach(self, session: DeviceSession) -> None:
        device = session.device["device_id"]
        if self.active.get(device) is session:
            del self.active[device]
        session.finished.set()

    def disconnect(self, device: str) -> None:
        if session := self.active.get(device):
            session.stop.set()
