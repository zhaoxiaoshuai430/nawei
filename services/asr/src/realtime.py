"""Realtime PCM segmentation helpers for the SenseVoice service."""

from __future__ import annotations

import array
import math
import os
import sys
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RealtimeVadConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    start_ms: int = 60
    end_silence_ms: int = 420
    pre_roll_ms: int = 200
    min_speech_ms: int = 180
    max_segment_seconds: float = 20.0
    rms_threshold: int = 500

    @classmethod
    def from_env(cls) -> "RealtimeVadConfig":
        return cls(
            sample_rate=int(os.getenv("ASR_REALTIME_SAMPLE_RATE", "16000")),
            frame_ms=int(os.getenv("ASR_REALTIME_FRAME_MS", "20")),
            start_ms=int(os.getenv("ASR_REALTIME_START_MS", "60")),
            end_silence_ms=int(os.getenv("ASR_REALTIME_END_SILENCE_MS", "420")),
            pre_roll_ms=int(os.getenv("ASR_REALTIME_PRE_ROLL_MS", "200")),
            min_speech_ms=int(os.getenv("ASR_REALTIME_MIN_SPEECH_MS", "180")),
            max_segment_seconds=float(os.getenv("ASR_REALTIME_MAX_SEGMENT_SECONDS", "20")),
            rms_threshold=int(os.getenv("ASR_REALTIME_RMS_THRESHOLD", "500")),
        )

    @property
    def frame_bytes(self) -> int:
        return self.sample_rate * self.frame_ms // 1000 * 2

    @property
    def start_frames(self) -> int:
        return max(1, math.ceil(self.start_ms / self.frame_ms))

    @property
    def end_silence_frames(self) -> int:
        return max(1, math.ceil(self.end_silence_ms / self.frame_ms))

    @property
    def pre_roll_frames(self) -> int:
        return max(1, math.ceil(self.pre_roll_ms / self.frame_ms))

    @property
    def min_segment_bytes(self) -> int:
        return self.sample_rate * self.min_speech_ms // 1000 * 2

    @property
    def max_segment_bytes(self) -> int:
        return int(self.sample_rate * self.max_segment_seconds * 2)


@dataclass(frozen=True)
class RealtimeVadEvent:
    kind: str
    pcm: bytes = b""


class PCM16VadSegmenter:
    """Split little-endian mono PCM16 into utterances using RMS energy."""

    def __init__(self, config: RealtimeVadConfig) -> None:
        if config.sample_rate <= 0 or config.frame_ms <= 0:
            raise ValueError("sample_rate and frame_ms must be positive.")
        if config.rms_threshold < 0:
            raise ValueError("rms_threshold cannot be negative.")
        self.config = config
        self._pending = bytearray()
        self._pre_roll: deque[bytes] = deque(maxlen=max(config.pre_roll_frames, config.start_frames))
        self._segment = bytearray()
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, pcm: bytes) -> list[RealtimeVadEvent]:
        if len(pcm) % 2:
            raise ValueError("PCM16 payload must contain an even number of bytes.")
        self._pending.extend(pcm)
        events: list[RealtimeVadEvent] = []
        frame_bytes = self.config.frame_bytes
        while len(self._pending) >= frame_bytes:
            frame = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            events.extend(self._process_frame(frame))
        return events

    def flush(self) -> list[RealtimeVadEvent]:
        if self._pending and self._in_speech:
            usable = len(self._pending) - (len(self._pending) % 2)
            self._segment.extend(self._pending[:usable])
        self._pending.clear()
        if not self._in_speech:
            self._reset_idle()
            return []
        return self._finish_segment()

    def _process_frame(self, frame: bytes) -> list[RealtimeVadEvent]:
        voiced = _pcm16_rms(frame) >= self.config.rms_threshold
        if not self._in_speech:
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + 1 if voiced else 0
            if self._speech_run < self.config.start_frames:
                return []
            self._in_speech = True
            self._segment = bytearray().join(self._pre_roll)
            self._pre_roll.clear()
            self._silence_run = 0
            return [RealtimeVadEvent("speech_started")]

        self._segment.extend(frame)
        self._silence_run = 0 if voiced else self._silence_run + 1
        if (
            self._silence_run >= self.config.end_silence_frames
            or len(self._segment) >= self.config.max_segment_bytes
        ):
            return self._finish_segment()
        return []

    def _finish_segment(self) -> list[RealtimeVadEvent]:
        segment = bytes(self._segment)
        self._segment.clear()
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._pre_roll.clear()
        if len(segment) < self.config.min_segment_bytes:
            return []
        return [RealtimeVadEvent("segment", segment)]

    def _reset_idle(self) -> None:
        self._pre_roll.clear()
        self._segment.clear()
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False


def _pcm16_rms(pcm: bytes) -> int:
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return int(math.sqrt(mean_square))
