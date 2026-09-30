#!/usr/bin/env python3
"""Standalone SenseVoice ASR service with HTTP and realtime PCM input."""
# ASR 主入口：音频解码/格式归一化 -> 容量控制 -> SenseVoice 推理 -> 标准转写结果。
# HTTP 接收完整音频；原生 WebSocket 接收 PCM16，在说话期间增量解码，由 VAD 确定段落边界。
# Paraformer 实时入口返回增量 partial；原 SenseVoice 分段入口和上传接口保持可用。
# 阅读顺序：_recognize_content_async -> transcribe_standard_base64 -> realtime_transcribe。
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, NoReturn, Optional

from pydantic import BaseModel, Field
try:  # Pydantic 2.x
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic 1.x
    ConfigDict = None  # type: ignore[assignment]
try:  # Pydantic 2.x
    from pydantic import model_validator
except ImportError:  # pragma: no cover - Pydantic 1.x
    model_validator = None  # type: ignore[assignment]
try:  # Pydantic 1.x
    from pydantic import root_validator
except ImportError:  # pragma: no cover - Pydantic 2.x
    root_validator = None  # type: ignore[assignment]
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent.parent
MODEL_ROOT = Path(os.environ.get("NAWEI_MODEL_ROOT", str(PROJECT_ROOT / "models"))).expanduser()
SENSEVOICE_DIR = ROOT / "sensevoice_runtime"
ASR_CACHE_DIR = ROOT / "hf_download"
MODELSCOPE_CACHE_DIR = ASR_CACHE_DIR / "modelscope"
HUGGINGFACE_CACHE_DIR = ASR_CACHE_DIR / "huggingface"
os.environ.setdefault("MODELSCOPE_CACHE", str(MODELSCOPE_CACHE_DIR))
os.environ.setdefault("MODELSCOPE_HOME", str(MODELSCOPE_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(HUGGINGFACE_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HUGGINGFACE_CACHE_DIR / "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", str(ASR_CACHE_DIR))

from runtime_tools import ensure_ffmpeg_on_path, is_native_pcm_wav, resolve_model_source

ensure_ffmpeg_on_path()

sys.path.insert(0, str(SENSEVOICE_DIR))

# 先导入 SenseVoice 模型定义以注册模型类
import model as _sensevoice_model  # noqa: F401

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from asr_protocol import MAX_AUDIO_BYTES as DEFAULT_MAX_AUDIO_BYTES
from asr_protocol import STREAMING_MODE, decode_base64_audio, read_upload_limited, streaming_capabilities
from inference_capacity import CapacityRejected, InferenceCapacity
from realtime import PCM16VadSegmenter, RealtimeVadConfig
from service_security import asr_cors_origins, require_asr_internal_api_key, verify_asr_internal_api_key
from streaming_model import ParaformerStreamingPool
from streaming_realtime import serve_streaming
from text_correction import correct_transcript
from tags import parse_sensevoice_raw

MODEL_ID = os.environ.get("ASR_MODEL_ID", "iic/SenseVoiceSmall")
VAD_MODEL_ID = os.environ.get("ASR_VAD_MODEL_ID", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
MODEL_SOURCE = resolve_model_source(
    MODEL_ID,
    ASR_CACHE_DIR,
    explicit_path=os.environ.get("ASR_MODEL_PATH", str(MODEL_ROOT / "asr" / "SenseVoiceSmall")),
)
VAD_MODEL_SOURCE = resolve_model_source(
    VAD_MODEL_ID,
    ASR_CACHE_DIR,
    explicit_path=os.environ.get("ASR_VAD_MODEL_PATH", str(MODEL_ROOT / "asr" / "fsmn-vad")),
)
DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TAG_RE = re.compile(r"<\|[^|]+\|>")
TARGET_SR = 16000
MAX_AUDIO_BYTES = int(os.environ.get("ASR_MAX_AUDIO_BYTES", str(DEFAULT_MAX_AUDIO_BYTES)))
MAX_AUDIO_FILES = int(os.environ.get("ASR_MAX_AUDIO_FILES", "4"))
MAX_AUDIO_BASE64_CHARS = ((MAX_AUDIO_BYTES + 2) // 3) * 4
logger = logging.getLogger(__name__)
INFERENCE_CAPACITY = InferenceCapacity(
    limit=int(os.environ.get("ASR_MAX_CONCURRENT_INFERENCES", "1")),
    queue_limit=int(os.environ.get("ASR_MAX_QUEUE_SIZE", "16")),
    wait_timeout_seconds=float(os.environ.get("ASR_QUEUE_TIMEOUT_SECONDS", "30")),
)

app = FastAPI(title="SenseVoice ASR", version="1.0.0")
cors_origins = asr_cors_origins()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Internal-API-Key", "X-Trace-ID"],
    )
ASR_API_DEPENDENCIES = [Depends(require_asr_internal_api_key)]

# 模型在模块初始化时加载并复用，不是每次请求重新加载；首次启动包含权重加载耗时。
# HTTP 与实时识别共用同一模型，实际并发由 INFERENCE_CAPACITY 控制。
print(f"Loading {MODEL_ID} from {MODEL_SOURCE} on {DEVICE} ...")
model = AutoModel(
    model=MODEL_SOURCE,
    trust_remote_code=False,
    vad_model=VAD_MODEL_SOURCE,
    vad_kwargs={"max_single_segment_time": 30000},
    device=DEVICE,
    disable_update=True,
    disable_pbar=True,
    check_latest=False,
)
print("SenseVoice model ready.")

REALTIME_BACKEND = os.getenv("ASR_REALTIME_BACKEND", "sensevoice").strip().lower()
if REALTIME_BACKEND not in {"sensevoice", "paraformer"}:
    raise ValueError("ASR_REALTIME_BACKEND must be sensevoice or paraformer.")
STREAMING_MODEL = (
    ParaformerStreamingPool(
        Path(os.getenv("ASR_STREAMING_MODEL_PATH", str(MODEL_ROOT / "asr" / "paraformer-online"))),
        device=os.getenv("ASR_STREAMING_DEVICE", "cpu"),
        threads=int(os.getenv("ASR_STREAMING_THREADS", "2")),
        workers=int(os.getenv("ASR_STREAMING_WORKERS", "1")),
    )
    if REALTIME_BACKEND == "paraformer" else None
)


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"Found system ffmpeg: {ffmpeg}")
        return ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Found imageio-ffmpeg: {ffmpeg_path}")
        return ffmpeg_path
    except ImportError:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")


def _prepare_wav_path(content: bytes, suffix: str) -> tuple[str, list[str]]:
    # 可直接使用的 PCM WAV 不额外转码；其余格式交给 FFmpeg 转为 16 kHz 单声道。
    # 返回路径及清理清单，成功时调用者负责回收；本函数失败时自行删除已创建的文件。
    suffix = (suffix or ".wav").lower()
    cleanup: list[str] = []
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as src:
            src.write(content)
            src_path = src.name
        cleanup.append(src_path)
        if is_native_pcm_wav(content):
            return src_path, cleanup
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
            dst_path = dst.name
        cleanup.append(dst_path)
        ffmpeg = _find_ffmpeg()
        cmd = [ffmpeg, "-y", "-i", src_path, "-ar", str(TARGET_SR), "-ac", "1", dst_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or b"")[-400:].decode("utf-8", errors="replace")
            logger.warning("Audio decode failed: %s", err)
            raise HTTPException(status_code=415, detail="Audio format could not be decoded.") from exc
        return dst_path, cleanup
    except BaseException:
        for path in cleanup:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        raise


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """Convert raw PCM data to WAV format."""
    # 原始 PCM 没有采样率/声道等文件头；这里仅补 WAV 容器，不执行重采样。
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bits // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


def _recognize_wav(wav_path: str, lang: str = "auto", use_itn: bool = True) -> dict[str, Any]:
    """对单个 wav 文件执行识别，返回和 /api/v1/asr 相同格式的结果"""
    res = model.generate(
        input=wav_path,
        cache={},
        language=lang,
        use_itn=use_itn,
        batch_size_s=60,
    )
    raw = res[0]["text"] if res else ""
    meta = parse_sensevoice_raw(raw)
    clean_text = rich_transcription_postprocess(raw) if raw else ""
    correction = correct_transcript(clean_text)
    return {
        **meta,
        "text": correction.text,
        "language": meta.get("language", ""),
        "raw_text": raw,
        "clean_text": correction.text,
        "corrections": list(correction.applied),
    }


def _decode_base64_audio(audio_base64: str) -> bytes:
    try:
        return decode_base64_audio(audio_base64, max_bytes=MAX_AUDIO_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413 if "exceeds" in str(exc) else 400, detail=str(exc)) from exc


async def _read_audio_upload(file: UploadFile) -> bytes:
    try:
        return await read_upload_limited(file, max_bytes=MAX_AUDIO_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413 if "exceeds" in str(exc) else 400, detail=str(exc)) from exc


def _raise_asr_failure(exc: Exception) -> NoReturn:
    logger.exception("ASR request failed", exc_info=exc)
    raise HTTPException(status_code=500, detail="ASR processing failed.") from exc


def _recognize_content(
    *,
    content: bytes,
    suffix: str,
    key: str,
    lang: str,
    use_itn: bool,
    include_raw: bool,
) -> dict[str, Any]:
    cleanup: list[str] = []
    try:
        wav_path, cleanup = _prepare_wav_path(content, suffix)
        res = model.generate(
            input=wav_path,
            cache={},
            language=lang or "auto",
            use_itn=use_itn,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        raw = res[0]["text"] if res else ""
        meta = parse_sensevoice_raw(raw)
        clean_text = rich_transcription_postprocess(raw) if raw else ""
        correction = correct_transcript(clean_text)
        item = {
            **meta,
            "key": key,
            "text": correction.text,
            "language": meta.get("language", ""),
            "corrections": list(correction.applied),
        }
        if include_raw:
            item["raw_text"] = raw
            item["clean_text"] = correction.text
        return item
    finally:
        for p in cleanup:
            if p and os.path.exists(p):
                os.unlink(p)


async def _run_inference_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    # 同步解码和模型推理移到线程，避免阻塞 FastAPI 的事件循环。
    # 取消协程不能停止已运行的线程，所以等待 worker 真正结束后再向外传播取消，
    # 使外层容量租约不会在模型仍占用资源时提前释放。
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await worker
        raise


async def _recognize_content_async(**kwargs: Any) -> dict[str, Any]:
    """Run blocking audio decoding and model inference without blocking FastAPI."""

    # 排队超限或等待超时返回 429，并提供 Retry-After；不无界积压 GPU 推理请求。
    try:
        lease = await INFERENCE_CAPACITY.acquire()
    except CapacityRejected as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    async with lease:
        return await _run_inference_thread(_recognize_content, **kwargs)


def _recognize_pcm_segment(pcm: bytes, *, lang: str, use_itn: bool, segment_index: int) -> dict[str, Any]:
    wav_bytes = _pcm_to_wav(pcm, sample_rate=TARGET_SR, channels=1, bits=16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(wav_bytes)
        audio_path = audio_file.name
    try:
        result = _recognize_wav(audio_path, lang=lang, use_itn=use_itn)
        result["key"] = f"segment-{segment_index}"
        result["segment_index"] = segment_index
        result["duration_ms"] = round(len(pcm) / (TARGET_SR * 2) * 1000)
        return result
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


async def _recognize_pcm_segment_async(
    pcm: bytes,
    *,
    lang: str,
    use_itn: bool,
    segment_index: int,
) -> dict[str, Any]:
    try:
        lease = await INFERENCE_CAPACITY.acquire()
    except CapacityRejected as exc:
        raise HTTPException(
            status_code=429,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    async with lease:
        return await _run_inference_thread(
            _recognize_pcm_segment,
            pcm,
            lang=lang,
            use_itn=use_itn,
            segment_index=segment_index,
        )


@app.get("/")
async def root():
    return {
        "service": "asr",
        "health": "/health",
        "realtime": "/api/v1/asr/realtime",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "streaming": streaming_capabilities(native=STREAMING_MODEL is not None),
        "concurrency": vars(INFERENCE_CAPACITY.snapshot()),
        "streaming_concurrency": STREAMING_MODEL.snapshot() if STREAMING_MODEL is not None else None,
    }


@app.get("/api/v1/capabilities")
async def capabilities() -> dict[str, object]:
    return {
        **streaming_capabilities(native=STREAMING_MODEL is not None),
        "concurrency": vars(INFERENCE_CAPACITY.snapshot()),
        "streaming_concurrency": STREAMING_MODEL.snapshot() if STREAMING_MODEL is not None else None,
    }


@app.websocket("/api/v1/asr/realtime")
async def realtime_transcribe(websocket: WebSocket) -> None:
    if STREAMING_MODEL is not None:
        with STREAMING_MODEL.session() as decoder:
            await serve_streaming(websocket, decoder)
    else:
        await sensevoice_realtime_transcribe(websocket)


@app.websocket("/api/v1/asr/realtime/sensevoice")
async def sensevoice_realtime_transcribe(websocket: WebSocket) -> None:
    """Accept mono 16 kHz PCM16 frames and emit VAD-finalized transcripts."""

    # 首条 JSON 声明音频格式并携带内部服务凭证，后续二进制帧才是音频。
    # 接收循环做 VAD，后台 worker 按序识别；发送锁避免控制事件和结果同时写连接。
    await websocket.accept()
    connection_open = True
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        nonlocal connection_open
        if not connection_open:
            return
        try:
            async with send_lock:
                await websocket.send_json(payload)
        except (RuntimeError, WebSocketDisconnect):
            connection_open = False

    try:
        start = await websocket.receive_json()
        if start.get("type") != "start":
            await send_json({"type": "error", "code": "invalid_start", "message": "First message must be start."})
            await websocket.close(code=4400)
            return
        try:
            verify_asr_internal_api_key(str(start.get("internal_api_key") or "") or None)
        except HTTPException as exc:
            await send_json({"type": "error", "code": "unauthorized", "message": str(exc.detail)})
            await websocket.close(code=4401)
            return

        config = RealtimeVadConfig.from_env()
        sample_rate = int(start.get("sample_rate") or config.sample_rate)
        channels = int(start.get("channels") or 1)
        if sample_rate != config.sample_rate or channels != 1:
            await send_json(
                {
                    "type": "error",
                    "code": "unsupported_audio_format",
                    "message": f"Expected mono PCM16 at {config.sample_rate} Hz.",
                }
            )
            await websocket.close(code=4400)
            return

        language = str(start.get("language") or "auto")
        use_itn = bool(start.get("use_itn", True))
        max_session_seconds = float(os.getenv("ASR_REALTIME_MAX_SESSION_SECONDS", "3600"))
        max_session_bytes = int(config.sample_rate * max_session_seconds * 2)
        # 一个队列元素是一段完整语音，而非一个网络包；有界队列限制待识别音频量。
        segment_queue: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue(
            maxsize=int(os.getenv("ASR_REALTIME_SEGMENT_QUEUE_SIZE", "4"))
        )
        segmenter = PCM16VadSegmenter(config)
        received_bytes = 0
        segment_index = 0

        async def recognize_segments() -> None:
            while True:
                queued = await segment_queue.get()
                if queued is None:
                    return
                current_index, pcm = queued
                await send_json({"type": "transcript.processing", "segment_index": current_index})
                try:
                    result = await _recognize_pcm_segment_async(
                        pcm,
                        lang=language,
                        use_itn=use_itn,
                        segment_index=current_index,
                    )
                except HTTPException as exc:
                    await send_json(
                        {
                            "type": "error",
                            "code": "recognition_failed",
                            "message": str(exc.detail),
                            "segment_index": current_index,
                        }
                    )
                    continue
                except Exception as exc:
                    logger.exception("Realtime ASR segment failed", exc_info=exc)
                    await send_json(
                        {
                            "type": "error",
                            "code": "recognition_failed",
                            "message": "Speech recognition failed.",
                            "segment_index": current_index,
                        }
                    )
                    continue
                await send_json({"type": "transcript.final", **result})

        worker = asyncio.create_task(recognize_segments())
        await send_json(
            {
                "type": "ready",
                "sample_rate": config.sample_rate,
                "channels": 1,
                "format": "pcm16le",
                "partial_results": False,
            }
        )

        stop_requested = False
        try:
            while not stop_requested:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    connection_open = False
                    break
                pcm = message.get("bytes")
                if pcm is not None:
                    received_bytes += len(pcm)
                    if received_bytes > max_session_bytes:
                        await send_json(
                            {
                                "type": "error",
                                "code": "session_too_long",
                                "message": f"Realtime session cannot exceed {max_session_seconds:g} seconds.",
                            }
                        )
                        break
                    try:
                        events = segmenter.feed(pcm)
                    except ValueError as exc:
                        await send_json({"type": "error", "code": "invalid_audio", "message": str(exc)})
                        continue
                    for event in events:
                        if event.kind == "speech_started":
                            await send_json({"type": "speech.started", "segment_index": segment_index})
                        elif event.kind == "segment":
                            await send_json({"type": "speech.ended", "segment_index": segment_index})
                            await segment_queue.put((segment_index, event.pcm))
                            segment_index += 1
                    continue

                text = message.get("text")
                if text is None:
                    continue
                try:
                    command = json.loads(text)
                except json.JSONDecodeError:
                    await send_json({"type": "error", "code": "invalid_command", "message": "Invalid JSON command."})
                    continue
                command_type = command.get("type")
                if command_type == "stop":
                    stop_requested = True
                elif command_type == "ping":
                    await send_json({"type": "pong"})
                else:
                    await send_json({"type": "error", "code": "invalid_command", "message": "Unknown command."})

            if connection_open:
                # stop 后排出 VAD 尚未结束的尾段，等待识别队列清空再发 session.done。
                # 客户端只有收到 session.done，才能确认最后一句也已完成处理。
                for event in segmenter.flush():
                    if event.kind == "segment":
                        await send_json({"type": "speech.ended", "segment_index": segment_index})
                        await segment_queue.put((segment_index, event.pcm))
                        segment_index += 1
                await segment_queue.put(None)
                await worker
                await send_json({"type": "session.done", "segments": segment_index})
                await websocket.close(code=1000)
            else:
                worker.cancel()
        finally:
            if not worker.done():
                worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
    except WebSocketDisconnect:
        connection_open = False


# ==================== 原有接口 ====================

@app.post("/api/v1/asr", dependencies=ASR_API_DEPENDENCIES)
async def transcribe(
    files: list[UploadFile] = File(..., description="Audio file(s): wav, mp3, webm"),
    keys: str = Form("", description="Comma-separated names for each audio"),
    lang: str = Form("auto", description="Language: auto, zh, en, yue, ja, ko"),
    use_itn: bool = Form(True, description="Inverse text normalization"),
    return_raw: bool = Form(False, description="Include raw SenseVoice output"),
) -> dict[str, Any]:
    if not files:
        return {"result": []}
    if len(files) > MAX_AUDIO_FILES:
        raise HTTPException(status_code=413, detail=f"At most {MAX_AUDIO_FILES} audio files are allowed per request.")

    key_list = keys.split(",") if keys else [f.filename or "audio" for f in files]

    out = []
    for i, file in enumerate(files):
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        content = await _read_audio_upload(file)
        try:
            item = await _recognize_content_async(
                content=content,
                suffix=suffix,
                key=key_list[i] if i < len(key_list) else file.filename or "audio",
                lang=lang,
                use_itn=use_itn,
                include_raw=return_raw,
            )
            out.append(item)
        except HTTPException:
            raise
        except Exception as exc:
            _raise_asr_failure(exc)

    return {"result": out}


class Base64Request(BaseModel):
    audio_base64: str = Field(..., min_length=1, max_length=MAX_AUDIO_BASE64_CHARS)
    filename: str = "audio.wav"
    lang: str = "auto"
    use_itn: bool = True
    return_raw: bool = False


class ProtocolModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid", use_enum_values=True)
    else:  # pragma: no cover - Pydantic 1.x
        class Config:
            extra = "forbid"
            use_enum_values = True


class ASRRequest(ProtocolModel):
    request_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = Field(None, max_length=MAX_AUDIO_BASE64_CHARS)
    language: str = "auto"
    return_timestamps: bool = False
    format: str = "wav"
    sample_rate: int = Field(16000, gt=0)
    channels: int = Field(1, ge=1)
    options: dict[str, Any] = Field(default_factory=dict)

    if model_validator is not None:
        @model_validator(mode="after")
        def validate_audio_source(self) -> "ASRRequest":
            _validate_single_audio_source(self.audio_url, self.audio_base64)
            return self
    elif root_validator is not None:  # pragma: no cover - Pydantic 1.x
        @root_validator
        def validate_audio_source(cls, values: dict[str, Any]) -> dict[str, Any]:
            _validate_single_audio_source(values.get("audio_url"), values.get("audio_base64"))
            return values


class ASRSegment(ProtocolModel):
    start_ms: int = Field(0, ge=0)
    end_ms: int = Field(0, ge=0)
    text: str


class ASREvent(ProtocolModel):
    tag: str
    label: Optional[str] = None
    emoji: Optional[str] = None


class ASRDetails(ProtocolModel):
    key: Optional[str] = None
    emotion: Optional[str] = None
    emotion_label: Optional[str] = None
    emotion_emoji: Optional[str] = None
    events: list[ASREvent] = Field(default_factory=list)
    emotion_hint: Optional[str] = None
    raw_text: Optional[str] = None
    clean_text: Optional[str] = None
    corrections: list[dict[str, str]] = Field(default_factory=list)


class ASRResponse(ProtocolModel):
    request_id: str = Field(..., min_length=1)
    text: str
    language: str = "auto"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    duration_ms: Optional[int] = Field(None, ge=0)
    segments: list[ASRSegment] = Field(default_factory=list)
    asr: ASRDetails = Field(default_factory=ASRDetails)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _validate_single_audio_source(audio_url: Optional[str], audio_base64: Optional[str]) -> None:
    has_url = bool(audio_url)
    has_base64 = bool(audio_base64)
    if has_url == has_base64:
        raise ValueError("Exactly one of audio_url or audio_base64 must be provided.")


def _asr_item_to_standard_response(request: ASRRequest, item: dict[str, Any]) -> ASRResponse:
    text = str(item.get("clean_text") or item.get("text") or "")
    return ASRResponse(
        request_id=request.request_id,
        text=text,
        language=str(item.get("language") or request.language),
        segments=[],
        asr=ASRDetails(
            key=item.get("key"),
            emotion=item.get("emotion"),
            emotion_label=item.get("emotion_label"),
            emotion_emoji=item.get("emotion_emoji"),
            events=item.get("events") or [],
            emotion_hint=item.get("emotion_hint"),
            raw_text=item.get("raw_text"),
            clean_text=item.get("clean_text") or text,
            corrections=item.get("corrections") or [],
        ),
        metadata={
            "provider": "sensevoice",
            "model": MODEL_ID,
            "device": DEVICE,
        },
    )


@app.post("/api/v1/asr/base64", dependencies=ASR_API_DEPENDENCIES)
async def transcribe_base64(request: Base64Request) -> dict[str, Any]:
    content = _decode_base64_audio(request.audio_base64)

    suffix = Path(request.filename).suffix or ".wav"
    try:
        item = await _recognize_content_async(
            content=content,
            suffix=suffix,
            key=request.filename,
            lang=request.lang,
            use_itn=request.use_itn,
            include_raw=request.return_raw,
        )
        return {"result": [item]}
    except HTTPException:
        raise
    except Exception as exc:
        _raise_asr_failure(exc)


@app.post(
    "/api/v1/asr/transcriptions",
    response_model=ASRResponse,
    dependencies=ASR_API_DEPENDENCIES,
)
@app.post(
    "/api/v1/audio/transcriptions/base64",
    response_model=ASRResponse,
    dependencies=ASR_API_DEPENDENCIES,
)
async def transcribe_standard_base64(request: ASRRequest) -> ASRResponse:
    # 标准 DTO 虽定义 audio_url，当前实现只支持 audio_base64，不能静默忽略 URL。
    if request.audio_url:
        raise HTTPException(status_code=400, detail="audio_url is not supported by this ASR service yet.")
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 is required.")

    filename = str(request.options.get("filename") or f"{request.request_id}.{request.format or 'wav'}")
    suffix = Path(filename).suffix or f".{request.format or 'wav'}"
    use_itn = bool(request.options.get("use_itn", True))
    content = _decode_base64_audio(request.audio_base64)

    try:
        if request.format.strip().lower() == "pcm16":
            # 每帧字节数 = 2 字节采样 * 声道数；不完整帧返回 400，防止音频错位。
            # 先按请求中的真实采样率封装 WAV，再交给公共解码流程处理。
            if len(content) % (2 * request.channels):
                raise HTTPException(status_code=400, detail="PCM16 audio must contain complete sample frames.")
            content = _pcm_to_wav(content, sample_rate=request.sample_rate, channels=request.channels)
            suffix = ".wav"
        item = await _recognize_content_async(
            content=content,
            suffix=suffix,
            key=filename,
            lang=request.language,
            use_itn=use_itn,
            include_raw=True,
        )
        return _asr_item_to_standard_response(request, item)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_asr_failure(exc)


# ==================== SSE 流式识别接口 ====================

class StreamRequest(BaseModel):
    """流式识别请求体（JSON格式）"""
    audio_base64: str = Field(..., min_length=1, max_length=MAX_AUDIO_BASE64_CHARS)
    filename: str = "audio.wav"
    lang: str = "auto"
    use_itn: bool = True
    is_final: bool = False


@app.post("/api/v1/asr/stream", dependencies=ASR_API_DEPENDENCIES)
async def stream_transcribe(request: StreamRequest):
    """
    独立音频块 SSE 识别接口（JSON格式）。
    
    每次 POST 独立识别一个可解码的音频块。SenseVoice 不保留连续解码器状态，
    因此该接口不承诺逐帧实时识别；连续实时场景应使用支持流式状态的 ASR 引擎。
    
    - audio_base64: Base64 编码的音频数据（支持任意格式：wav/mp3/webm/ogg）
    - filename: 文件名，用于推断格式（默认 audio.wav）
    - lang: 语言（auto/zh/en/yue/ja/ko）
    - use_itn: 是否启用逆文本归一化
    - is_final: 由客户端标记是否为最后一块，服务端仅原样返回该标记
    """
    content = _decode_base64_audio(request.audio_base64)

    async def _stream() -> AsyncGenerator[str, None]:
        suffix = Path(request.filename).suffix or ".wav"
        item = await _recognize_content_async(
            content=content,
            suffix=suffix,
            key="stream",
            lang=request.lang,
            use_itn=request.use_itn,
            include_raw=False,
        )
        result = {"result": [item], "is_final": request.is_final}
        yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-ASR-Streaming-Mode": STREAMING_MODE,
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ASR_PORT", "8001"))
    host = os.environ.get("ASR_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
