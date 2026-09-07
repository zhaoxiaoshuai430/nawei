"""TTS streaming timing helpers."""
from __future__ import annotations


def stream_chunk_metrics(mode: str, timing: dict) -> tuple[float, float, float]:
    infer_ms = float(timing.get("prefill_ms", 0) + timing.get("decode_ms", 0))
    codec_ms = float(timing.get("codec_decode_ms", 0))
    prepare_ms = float(timing.get("prepare_ms", 0)) if mode == "voice_clone" else 0.0
    return prepare_ms, infer_ms, codec_ms


def real_time_factor(*, generation_ms: float, audio_seconds: float) -> float:
    """Return generation time divided by generated audio duration."""

    if generation_ms <= 0 or audio_seconds <= 0:
        return 0.0
    return (generation_ms / 1000.0) / audio_seconds
