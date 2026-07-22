"""Per-sentence TTS streaming for chat pipeline."""
from __future__ import annotations

import json
import time
from asyncio import Queue

from faster_qwen3_tts import FasterQwen3TTS

from app.config import MAX_TEXT_NEW_TOKENS
from app.utils.audio import concat_audio, to_wav_b64


def stream_chunk_metrics(mode: str, timing: dict) -> tuple[float, float, float]:
    infer_ms = float(timing.get("prefill_ms", 0) + timing.get("decode_ms", 0))
    codec_ms = float(timing.get("codec_decode_ms", 0))
    prepare_ms = float(timing.get("prepare_ms", 0)) if mode == "voice_clone" else 0.0
    return prepare_ms, infer_ms, codec_ms


def _emit_audio_chunks(
    *,
    loop,
    queue: Queue,
    chunks,
    sentence: str,
    metrics_mode: str,
    metrics: dict,
    t0: float,
) -> None:
    for i, (audio_chunk, sr, timing) in enumerate(chunks):
        voice_clone_ms, ttfa_ms, _ = stream_chunk_metrics(metrics_mode, timing)
        if metrics["voice_clone_ms"] == 0:
            metrics["voice_clone_ms"] = voice_clone_ms
        if metrics["ttfa_ms"] is None:
            metrics["ttfa_ms"] = ttfa_ms
        metrics["total_gen_ms"] += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)

        audio_chunk = concat_audio(audio_chunk)
        dur = len(audio_chunk) / sr
        metrics["total_audio_s"] += dur
        rtf = (
            metrics["total_audio_s"] / (metrics["total_gen_ms"] / 1000)
            if metrics["total_gen_ms"] > 0
            else 0.0
        )
        payload = {
            "type": "chunk",
            "audio_b64": to_wav_b64(audio_chunk, sr),
            "sample_rate": sr,
            "sentence": sentence if i == 0 else "",
            "ttfa_ms": round(metrics["ttfa_ms"] or 0),
            "voice_clone_ms": round(metrics["voice_clone_ms"]),
            "rtf": round(rtf, 3),
            "total_audio_s": round(metrics["total_audio_s"], 3),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
        }
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))


def emit_tts_stream(
    *,
    loop,
    queue: Queue,
    model: FasterQwen3TTS,
    sentence: str,
    language: str,
    tmp_path: str | None,
    ref_text: str,
    xvec_only: bool,
    chunk_size: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    metrics: dict,
    t0: float,
    voice_clone_prompt: dict | None = None,
) -> None:
    gen = model.generate_voice_clone_streaming(
        text=sentence,
        language=language,
        ref_audio=None if voice_clone_prompt is not None else tmp_path,
        ref_text=ref_text,
        xvec_only=xvec_only,
        chunk_size=chunk_size,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_new_tokens=MAX_TEXT_NEW_TOKENS,
        voice_clone_prompt=voice_clone_prompt,
    )
    first_audio = next(gen, None)
    chunks = []
    if first_audio is not None:
        chunks.append(first_audio)
    chunks.extend(gen)
    _emit_audio_chunks(
        loop=loop,
        queue=queue,
        chunks=chunks,
        sentence=sentence,
        metrics_mode="voice_clone",
        metrics=metrics,
        t0=t0,
    )


def emit_custom_voice_stream(
    *,
    loop,
    queue: Queue,
    model: FasterQwen3TTS,
    sentence: str,
    language: str,
    speaker: str,
    instruct: str,
    chunk_size: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    metrics: dict,
    t0: float,
) -> None:
    gen = model.generate_custom_voice_streaming(
        text=sentence,
        speaker=speaker,
        language=language,
        instruct=instruct,
        chunk_size=chunk_size,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_new_tokens=MAX_TEXT_NEW_TOKENS,
    )
    first_audio = next(gen, None)
    chunks = []
    if first_audio is not None:
        chunks.append(first_audio)
    chunks.extend(gen)
    _emit_audio_chunks(
        loop=loop,
        queue=queue,
        chunks=chunks,
        sentence=sentence,
        metrics_mode="custom",
        metrics=metrics,
        t0=t0,
    )
