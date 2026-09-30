from __future__ import annotations

import av
from av.audio.codeccontext import AudioCodecContext
from typing import cast


class OpusDecoder:
    def __init__(self, sample_rate: int = 16000):
        self.codec = cast(AudioCodecContext, av.CodecContext.create("opus", "r"))
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        self.sample_rate = sample_rate

    def decode(self, packet: bytes) -> bytes:
        if not packet or len(packet) > 1275:
            raise ValueError("Invalid Opus packet size")
        output = bytearray()
        frames = self.codec.decode(av.Packet(packet))
        if sum(frame.samples / frame.sample_rate for frame in frames) != 0.06:
            raise ValueError("Opus packet must contain 60ms of audio")
        for frame in frames:
            for converted in self.resampler.resample(frame):
                output.extend(bytes(converted.planes[0])[:converted.samples * 2])
        return bytes(output)


class OpusEncoder:
    def __init__(self, sample_rate: int = 24000):
        if sample_rate not in {16000, 24000}:
            raise ValueError("Unsupported sample rate")
        self.sample_rate = sample_rate
        self.samples = sample_rate * 60 // 1000
        self.frame_bytes = self.samples * 2
        self.codec = cast(AudioCodecContext, av.CodecContext.create("libopus", "w"))
        self.codec.sample_rate = sample_rate
        self.codec.layout = "mono"
        self.codec.format = "s16"
        self.codec.bit_rate = 32000
        self.codec.options = {"frame_duration": "60", "application": "voip"}
        self.codec.open()
        self.buffer = bytearray()
        self.resampler: av.AudioResampler | None = None
        self.input_rate: int | None = None

    def feed(self, pcm: bytes, input_rate: int) -> list[bytes]:
        if len(pcm) % 2 or input_rate not in {16000, 24000, 48000}:
            raise ValueError("Invalid PCM16 data")
        if self.input_rate not in {None, input_rate}:
            raise ValueError("TTS sample rate changed during a turn")
        self.input_rate = input_rate
        if input_rate != self.sample_rate:
            if self.resampler is None:
                self.resampler = av.AudioResampler(format="s16", layout="mono", rate=self.sample_rate)
            frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // 2)
            frame.sample_rate = input_rate
            frame.planes[0].update(pcm)
            for converted in self.resampler.resample(frame):
                self.buffer.extend(bytes(converted.planes[0])[:converted.samples * 2])
        else:
            self.buffer.extend(pcm)
        return self._packets()

    def finish(self) -> list[bytes]:
        if self.resampler is not None:
            for converted in self.resampler.resample(None):
                self.buffer.extend(bytes(converted.planes[0])[:converted.samples * 2])
        if len(self.buffer) % self.frame_bytes:
            self.buffer.extend(bytes(self.frame_bytes - len(self.buffer) % self.frame_bytes))
        return self._packets() + [bytes(packet) for packet in self.codec.encode(None)]

    def _packets(self) -> list[bytes]:
        packets: list[bytes] = []
        while len(self.buffer) >= self.frame_bytes:
            frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples)
            frame.sample_rate = self.sample_rate
            frame.planes[0].update(bytes(self.buffer[:self.frame_bytes]))
            del self.buffer[:self.frame_bytes]
            packets.extend(bytes(packet) for packet in self.codec.encode(frame))
        return packets
