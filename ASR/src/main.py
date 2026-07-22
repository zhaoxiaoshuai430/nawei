#!/usr/bin/env python3
"""Standalone SenseVoice ASR HTTP Service with SSE Streaming."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
SENSEVOICE_DIR = ROOT / "SenseVoice"
ASR_CACHE_DIR = ROOT / "hf_download"
MODELSCOPE_CACHE_DIR = ASR_CACHE_DIR / "modelscope"
HUGGINGFACE_CACHE_DIR = ASR_CACHE_DIR / "huggingface"
os.environ.setdefault("MODELSCOPE_CACHE", str(MODELSCOPE_CACHE_DIR))
os.environ.setdefault("MODELSCOPE_HOME", str(MODELSCOPE_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(HUGGINGFACE_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HUGGINGFACE_CACHE_DIR / "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", str(ASR_CACHE_DIR))

sys.path.insert(0, str(SENSEVOICE_DIR))

# 先导入 SenseVoice 模型定义以注册模型类
import model as _sensevoice_model

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from tags import parse_sensevoice_raw

MODEL_ID = os.environ.get("ASR_MODEL_ID", "iic/SenseVoiceSmall")
DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TAG_RE = re.compile(r"<\|[^|]+\|>")
TARGET_SR = 16000
INFERENCE_LOCK = asyncio.Lock()

app = FastAPI(title="SenseVoice ASR", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading {MODEL_ID} on {DEVICE} ...")
model = AutoModel(
    model=MODEL_ID,
    trust_remote_code=True,
    remote_code=(SENSEVOICE_DIR / "model.py").as_posix() if SENSEVOICE_DIR.exists() else None,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device=DEVICE,
)
print("SenseVoice model ready.")


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
    suffix = (suffix or ".wav").lower()
    cleanup: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as src:
        src.write(content)
        src_path = src.name
    cleanup.append(src_path)
    dst_path = tempfile.mktemp(suffix=".wav")
    cleanup.append(dst_path)
    ffmpeg = _find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", src_path, "-ar", str(TARGET_SR), "-ac", "1", dst_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"")[-400:].decode("utf-8", errors="replace")
        raise HTTPException(status_code=415, detail=f"Audio decode failed: {err}") from exc
    return dst_path, cleanup


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """Convert raw PCM data to WAV format."""
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
    return {
        "text": rich_transcription_postprocess(raw) if raw else "",
        "language": meta.get("language", ""),
        **meta,
    }


def _decode_base64_audio(audio_base64: str) -> bytes:
    try:
        normalized = "".join(audio_base64.split())
        padding = len(normalized) % 4
        if padding:
            normalized += "=" * (4 - padding)
        content = base64.b64decode(normalized, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc
    if not content:
        raise HTTPException(status_code=400, detail="Decoded audio is empty.")
    return content


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
        item = {
            "key": key,
            "text": rich_transcription_postprocess(raw) if raw else "",
            "language": meta.get("language", ""),
            **meta,
        }
        if include_raw:
            item["raw_text"] = raw
            item["clean_text"] = TAG_RE.sub("", raw)
        return item
    finally:
        for p in cleanup:
            if p and os.path.exists(p):
                os.unlink(p)


async def _recognize_content_async(**kwargs: Any) -> dict[str, Any]:
    """Run blocking audio decoding and model inference without blocking FastAPI."""

    async with INFERENCE_LOCK:
        return await asyncio.to_thread(_recognize_content, **kwargs)


STATIC_DIR = ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(html_path))
    return {"status": "ok", "message": "ASR Service running. Visit /static/index.html for demo."}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


# ==================== 原有接口 ====================

@app.post("/api/v1/asr")
async def transcribe(
    files: list[UploadFile] = File(..., description="Audio file(s): wav, mp3, webm"),
    keys: str = Form("", description="Comma-separated names for each audio"),
    lang: str = Form("auto", description="Language: auto, zh, en, yue, ja, ko"),
    use_itn: bool = Form(True, description="Inverse text normalization"),
    return_raw: bool = Form(False, description="Include raw SenseVoice output"),
) -> dict[str, Any]:
    if not files:
        return {"result": []}

    key_list = keys.split(",") if keys else [f.filename or "audio" for f in files]

    out = []
    for i, file in enumerate(files):
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        content = await file.read()
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
            raise HTTPException(status_code=500, detail=f"ASR failed: {exc}") from exc

    return {"result": out}


class Base64Request(BaseModel):
    audio_base64: str = Field(..., min_length=1)
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
    audio_base64: Optional[str] = None
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
        ),
        metadata={
            "provider": "sensevoice",
            "model": MODEL_ID,
            "device": DEVICE,
        },
    )


@app.post("/api/v1/asr/base64")
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
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}") from exc


@app.post("/api/v1/asr/transcriptions", response_model=ASRResponse)
@app.post("/api/v1/audio/transcriptions/base64", response_model=ASRResponse)
async def transcribe_standard_base64(request: ASRRequest) -> ASRResponse:
    if request.audio_url:
        raise HTTPException(status_code=400, detail="audio_url is not supported by this ASR service yet.")
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 is required.")

    filename = str(request.options.get("filename") or f"{request.request_id}.{request.format or 'wav'}")
    suffix = Path(filename).suffix or f".{request.format or 'wav'}"
    use_itn = bool(request.options.get("use_itn", True))
    content = _decode_base64_audio(request.audio_base64)

    try:
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
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}") from exc


# ==================== SSE 流式识别接口 ====================

class StreamRequest(BaseModel):
    """流式识别请求体（JSON格式）"""
    audio_base64: str = Field(..., min_length=1)
    filename: str = "audio.wav"
    lang: str = "auto"
    use_itn: bool = True
    is_final: bool = False


@app.post("/api/v1/asr/stream")
async def stream_transcribe(request: StreamRequest):
    """
    SSE 流式识别接口（JSON格式）。
    
    客户端持续 POST 音频块，服务端通过 SSE 实时返回识别结果。
    
    - audio_base64: Base64 编码的音频数据（支持任意格式：wav/mp3/webm/ogg）
    - filename: 文件名，用于推断格式（默认 audio.wav）
    - lang: 语言（auto/zh/en/yue/ja/ko）
    - use_itn: 是否启用逆文本归一化
    - is_final: 是否为最后一块音频，设为 true 时处理剩余数据并结束
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
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ASR_PORT", "8001"))
    host = os.environ.get("ASR_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
