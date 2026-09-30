import base64
from typing import Any

from app_protocols import EventType, StreamEvent

from .opus_codec import OpusEncoder
from .session import DeviceSession


class EventAdapter:
    def __init__(self, session: DeviceSession):
        self.session = session
        self.encoder = OpusEncoder(session.sample_rate)
        self.generation = session.writer.generation
        self.started = False
        self.segment = -1

    async def send(self, payload: dict[str, Any] | bytes) -> None:
        if isinstance(payload, dict):
            payload = {"session_id": self.session.session_id, **payload}
        await self.session.writer.send(payload, self.generation)

    async def event(self, event: StreamEvent) -> None:
        if event.event == EventType.AUDIO_STARTED:
            self.started = True
            await self.send({"type": "tts", "state": "start"})
        elif event.event == EventType.AUDIO_DELTA:
            data = event.data
            if data.get("format") != "pcm16" or data.get("channels", 1) != 1:
                raise ValueError("TTS must return mono PCM16")
            if not self.started:
                self.started = True
                await self.send({"type": "tts", "state": "start"})
            if data.get("segment_index", 0) != self.segment:
                self.segment = data.get("segment_index", 0)
                await self.send({"type": "tts", "state": "sentence_start", "text": data.get("text", "")})
            pcm = base64.b64decode(data["audio_base64"], validate=True)
            for packet in self.encoder.feed(pcm, int(data["sample_rate"])):
                await self.send(packet)
        elif event.event == EventType.AUDIO_DONE:
            await self.finish()
        elif event.event == EventType.ERROR:
            raise RuntimeError("Voice service returned an error")

    async def finish(self) -> None:
        if self.started:
            for packet in self.encoder.finish():
                await self.send(packet)
            await self.send({"type": "tts", "state": "stop"})
            self.started = False
