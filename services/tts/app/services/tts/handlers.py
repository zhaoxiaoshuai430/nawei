"""TTS generation handlers for streaming and standard responses."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import AUDIO_TOO_LARGE_MSG, MAX_AUDIO_BYTES, MAX_TEXT_CHARS, MAX_TEXT_NEW_TOKENS
from app.services.tts import state
from app.services.tts import emotion_refs
from app.services.tts.capacity import CapacityRejected
from app.services.tts.presets import preset_refs
from app.services.tts.streaming import real_time_factor, stream_chunk_metrics
from app.utils.audio import concat_audio, encode_audio_b64
from app.utils.cache import get_cached_ref_path


logger = logging.getLogger(__name__)


@dataclass
class RefAudioContext:
    path: str | None
    is_cached: bool
    ref_text: str


async def run_blocking_until_complete(function, /, *args, **kwargs):
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await worker
        raise


async def _acquire_generation_capacity():
    try:
        return await state.generation_capacity.acquire()
    except CapacityRejected as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


def require_model_loaded() -> None:
    if not state.active_model_name or state.active_model_name not in state.model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")


def require_text_length(text: str) -> None:
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_CHARS} characters.",
        )


async def resolve_ref_audio(
    *,
    ref_preset: str,
    ref_audio: UploadFile | None,
    ref_text: str,
    required: bool = False,
) -> RefAudioContext:
    tmp_path: str | None = None
    tmp_is_cached = False

    if ref_preset and ref_preset in emotion_refs.emotion_refs:
        preset = emotion_refs.emotion_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_preset and ref_preset in preset_refs:
        preset = preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail=AUDIO_TOO_LARGE_MSG.format(size_mb=len(content) / 1024 / 1024),
            )
        tmp_path = get_cached_ref_path(content)
        tmp_is_cached = True

    if required and not tmp_path:
        raise HTTPException(status_code=400, detail="Reference audio required.")

    return RefAudioContext(path=tmp_path, is_cached=tmp_is_cached, ref_text=ref_text)


def cleanup_ref_path(path: str | None, is_cached: bool) -> None:
    if path and os.path.exists(path) and not is_cached:
        os.unlink(path)



async def generate_stream(
    *,
    text: str,
    language: str,
    mode: str,
    ref_text: str,
    speaker: str,
    instruct: str,
    xvec_only: bool,
    chunk_size: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    ref_preset: str,
    ref_audio: UploadFile | None,
    audio_format: str = "wav",
    sample_rate: int = 24000,
    channels: int = 1,
) -> StreamingResponse:
    require_model_loaded()
    require_text_length(text)
    logger.info("Starting streaming TTS generation for %s characters", len(text))

    ctx = await resolve_ref_audio(
        ref_preset=ref_preset,
        ref_audio=ref_audio,
        ref_text=ref_text,
        required=mode == "voice_clone",
    )
    tmp_path, tmp_is_cached, ref_text = ctx.path, ctx.is_cached, ctx.ref_text

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue(
        maxsize=int(os.environ.get("TTS_STREAM_QUEUE_SIZE", "8"))
    )
    stop_requested = threading.Event()
    consumer_closed = threading.Event()
    queue_put_timeout = float(os.environ.get("TTS_STREAM_QUEUE_PUT_TIMEOUT_SECONDS", "5"))

    def put_stream_message(message: str | None, *, required: bool = True) -> None:
        if stop_requested.is_set() and not required:
            return
        future = asyncio.run_coroutine_threadsafe(queue.put(message), loop)
        try:
            future.result(timeout=queue_put_timeout)
        except TimeoutError as exc:
            future.cancel()
            stop_requested.set()
            if required:
                raise RuntimeError("TTS stream consumer is too slow.") from exc

    def run_generation():
        try:
            model = state.model_cache.get(state.active_model_name)
            if model is None:
                raise RuntimeError("No model loaded. Please load a model first.")

            total_audio_s = 0.0
            voice_clone_ms = 0.0

            if mode == "voice_clone":
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=MAX_TEXT_NEW_TOKENS,
                )
            elif mode == "custom":
                if not speaker:
                    raise ValueError("Speaker ID is required for custom voice")
                gen = model.generate_custom_voice_streaming(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=MAX_TEXT_NEW_TOKENS,
                )
            else:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=instruct,
                    language=language,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=MAX_TEXT_NEW_TOKENS,
                )

            ttfa_ms = None
            total_gen_ms = 0.0
            t0 = time.perf_counter()

            first_audio = next(gen, None)
            if first_audio is not None and not stop_requested.is_set():
                audio_chunk, sr, timing = first_audio
                voice_clone_ms, ttfa_ms, _ = stream_chunk_metrics(mode, timing)
                total_gen_ms += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)

                audio_chunk = concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = real_time_factor(generation_ms=total_gen_ms, audio_seconds=total_audio_s)

                payload = {
                    "type": "chunk",
                    "audio_b64": encode_audio_b64(
                        audio_chunk,
                        sr,
                        output_format=audio_format,
                        target_sr=sample_rate,
                        channels=channels,
                    ),
                    "format": audio_format,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
                }
                put_stream_message(json.dumps(payload))

            for audio_chunk, sr, timing in gen:
                if stop_requested.is_set():
                    break
                total_gen_ms += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                audio_chunk = concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = real_time_factor(generation_ms=total_gen_ms, audio_seconds=total_audio_s)
                payload = {
                    "type": "chunk",
                    "audio_b64": encode_audio_b64(
                        audio_chunk,
                        sr,
                        output_format=audio_format,
                        target_sr=sample_rate,
                        channels=channels,
                    ),
                    "format": audio_format,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "ttfa_ms": round(ttfa_ms) if ttfa_ms is not None else 0,
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
                }
                put_stream_message(json.dumps(payload))

            if not stop_requested.is_set():
                rtf = real_time_factor(generation_ms=total_gen_ms, audio_seconds=total_audio_s)
                done_payload = {
                    "type": "done",
                    "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "total_ms": round((time.perf_counter() - t0) * 1000),
                }
                put_stream_message(json.dumps(done_payload))

        except Exception as exc:
            logger.exception("Streaming TTS generation failed")
            err = {"type": "error", "message": str(exc)}
            try:
                put_stream_message(json.dumps(err), required=False)
            except RuntimeError:
                pass
        finally:
            if not consumer_closed.is_set():
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            cleanup_ref_path(tmp_path, tmp_is_cached)

    async def sse():
        lease = None
        acquire_task = asyncio.create_task(state.generation_capacity.acquire())
        worker: threading.Thread | None = None
        try:
            await asyncio.sleep(0)
            if not acquire_task.done():
                position = max(1, state.generation_capacity.snapshot().waiting)
                yield f"data: {json.dumps({'type': 'queued', 'position': position})}\n\n"
            try:
                lease = await acquire_task
            except CapacityRejected as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': exc.reason, 'retry_after_seconds': exc.retry_after_seconds})}\n\n"
                return

            worker = threading.Thread(target=run_generation, daemon=True)
            worker.start()

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            consumer_closed.set()
            stop_requested.set()
            if not acquire_task.done():
                acquire_task.cancel()
            raise
        finally:
            stop_requested.set()
            if worker is not None and worker.is_alive():
                await run_blocking_until_complete(worker.join)
            if lease is not None:
                await lease.release()
            elif acquire_task.done() and not acquire_task.cancelled():
                try:
                    acquired_after_cancel = acquire_task.result()
                except CapacityRejected:
                    pass
                else:
                    await acquired_after_cancel.release()

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def generate_non_streaming(
    *,
    text: str,
    language: str,
    mode: str,
    ref_text: str,
    speaker: str,
    instruct: str,
    xvec_only: bool,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    ref_preset: str,
    ref_audio: UploadFile | None,
    audio_format: str = "wav",
    sample_rate: int = 24000,
    channels: int = 1,
) -> JSONResponse:
    require_model_loaded()
    require_text_length(text)
    logger.info("Starting non-streaming TTS generation for %s characters", len(text))

    ctx = await resolve_ref_audio(
        ref_preset=ref_preset,
        ref_audio=ref_audio,
        ref_text=ref_text,
        required=mode == "voice_clone",
    )
    tmp_path, tmp_is_cached, ref_text = ctx.path, ctx.is_cached, ctx.ref_text

    def run():
        model = state.model_cache.get(state.active_model_name)
        if model is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        t0 = time.perf_counter()
        if mode == "voice_clone":
            audio_list, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text,
                xvec_only=xvec_only,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=MAX_TEXT_NEW_TOKENS,
            )
        elif mode == "custom":
            if not speaker:
                raise ValueError("Speaker ID is required for custom voice")
            audio_list, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=MAX_TEXT_NEW_TOKENS,
            )
        else:
            audio_list, sr = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=MAX_TEXT_NEW_TOKENS,
            )
        elapsed = time.perf_counter() - t0
        audio = concat_audio(audio_list)
        dur = len(audio) / sr
        return audio, sr, elapsed, dur

    lease = await _acquire_generation_capacity()
    try:
        async with lease:
            audio, sr, elapsed, dur = await run_blocking_until_complete(run)
        rtf = real_time_factor(generation_ms=elapsed * 1000, audio_seconds=dur)
        audio_b64 = await run_blocking_until_complete(
            encode_audio_b64,
            audio,
            sr,
            output_format=audio_format,
            target_sr=sample_rate,
            channels=channels,
        )
        return JSONResponse({
            "audio_b64": audio_b64,
            "format": audio_format,
            "sample_rate": sample_rate,
            "channels": channels,
            "metrics": {
                "total_ms": round(elapsed * 1000),
                "audio_duration_s": round(dur, 3),
                "rtf": round(rtf, 3),
            },
        })
    finally:
        cleanup_ref_path(tmp_path, tmp_is_cached)
