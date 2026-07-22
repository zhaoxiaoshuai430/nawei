"""Gateway TTS handlers (prewarm, streaming/non-streaming generate)."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import AUDIO_TOO_LARGE_MSG, MAX_AUDIO_BYTES, MAX_TEXT_CHARS, MAX_TEXT_NEW_TOKENS
from app.services.tts import state
from app.services.tts import emotion_refs
from app.services.tts.presets import preset_refs
from app.services.tts.warmup import warm_voice_clone_cache
from app.services.tts.streaming import stream_chunk_metrics
from app.utils.audio import concat_audio, encode_audio_b64
from app.utils.cache import get_cached_ref_path


@dataclass
class RefAudioContext:
    path: str | None
    is_cached: bool
    ref_text: str


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


async def prewarm_clone(
    *,
    ref_text: str,
    xvec_only: bool,
    ref_preset: str,
    ref_audio: UploadFile | None,
) -> dict:
    require_model_loaded()
    if "Base" not in state.active_model_name:
        return {"status": "skipped", "reason": "not a Base clone model"}

    ctx = await resolve_ref_audio(
        ref_preset=ref_preset,
        ref_audio=ref_audio,
        ref_text=ref_text,
        required=True,
    )
    if not xvec_only and not ctx.ref_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Advanced (ICL) mode requires reference transcript.",
        )

    model = state.model_cache[state.active_model_name]

    def _do_warm():
        warm_voice_clone_cache(
            model,
            ref_audio=ctx.path,
            ref_text=ctx.ref_text,
            xvec_only=xvec_only,
        )

    try:
        await asyncio.to_thread(_do_warm)
    finally:
        cleanup_ref_path(ctx.path, ctx.is_cached)

    return {"status": "warmed", "xvec_only": xvec_only, "ref_preset": ref_preset or None}


async def prewarm_emotion_clone(
    *,
    emotion_tag: str = "",
    ref_id: str = "",
    xvec_only: bool = False,
) -> dict:
    """Prewarm an emotion-tagged reference clip for clone tone switching."""
    require_model_loaded()
    if "Base" not in state.active_model_name:
        return {"status": "skipped", "reason": "not a Base clone model"}
    if not emotion_refs.emotion_refs_enabled():
        raise HTTPException(status_code=400, detail="情绪参考库未就绪，请先放入 assets/emotion_refs/*.wav")

    resolved_id = (ref_id or "").strip()
    if not resolved_id:
        resolved_id = emotion_refs.resolve_asr_to_emotion_ref(emotion_tag) or ""
    if not resolved_id or resolved_id not in emotion_refs.emotion_refs:
        raise HTTPException(status_code=404, detail="未找到对应的情绪参考片段")

    preset = emotion_refs.emotion_refs[resolved_id]
    if not xvec_only and not preset["ref_text"].strip():
        raise HTTPException(status_code=400, detail="Advanced (ICL) mode requires reference transcript.")

    model = state.model_cache[state.active_model_name]

    def _do_warm():
        emotion_refs.prewarm_emotion_ref_sync(
            model,
            ref_id=resolved_id,
            xvec_only=xvec_only,
        )

    await asyncio.to_thread(_do_warm)
    return {
        "status": "warmed",
        "ref_id": resolved_id,
        "label": preset["label"],
        "tone": preset.get("tone", preset["label"]),
        "xvec_only": xvec_only,
    }


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
    print(f"Text Length：{len(text)}")

    ctx = await resolve_ref_audio(
        ref_preset=ref_preset,
        ref_audio=ref_audio,
        ref_text=ref_text,
        required=mode == "voice_clone",
    )
    tmp_path, tmp_is_cached, ref_text = ctx.path, ctx.is_cached, ctx.ref_text

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop_requested = threading.Event()

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
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

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
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            for audio_chunk, sr, timing in gen:
                if stop_requested.is_set():
                    break
                total_gen_ms += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                audio_chunk = concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
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
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            if not stop_requested.is_set():
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
                done_payload = {
                    "type": "done",
                    "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                    "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "total_ms": round((time.perf_counter() - t0) * 1000),
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(done_payload))

        except Exception as e:
            import traceback

            traceback.print_exc()
            err = {"type": "error", "message": str(e)}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            cleanup_ref_path(tmp_path, tmp_is_cached)

    async def sse():
        lock_acquired = False
        worker: threading.Thread | None = None
        state.generation_waiters += 1
        people_ahead = state.generation_waiters - 1 + (1 if state.generation_lock.locked() else 0)
        try:
            if people_ahead > 0:
                yield f"data: {json.dumps({'type': 'queued', 'position': people_ahead})}\n\n"

            await state.generation_lock.acquire()
            lock_acquired = True
            state.generation_waiters -= 1

            worker = threading.Thread(target=run_generation, daemon=True)
            worker.start()

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            stop_requested.set()
            raise
        finally:
            stop_requested.set()
            if worker is not None and worker.is_alive():
                await asyncio.to_thread(worker.join)
            if lock_acquired:
                state.generation_lock.release()
            else:
                state.generation_waiters -= 1

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
    print(f"Text Length：{len(text)}")

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

    lock_acquired = False
    try:
        await state.generation_lock.acquire()
        lock_acquired = True
        audio, sr, elapsed, dur = await asyncio.to_thread(run)
        rtf = dur / elapsed if elapsed > 0 else 0.0
        return JSONResponse({
            "audio_b64": encode_audio_b64(
                audio,
                sr,
                output_format=audio_format,
                target_sr=sample_rate,
                channels=channels,
            ),
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
        if lock_acquired:
            state.generation_lock.release()
        cleanup_ref_path(tmp_path, tmp_is_cached)
