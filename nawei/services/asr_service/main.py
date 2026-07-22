"""ASR service stub aligned with the team ASR API document."""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field


MODEL_ID = "iic/SenseVoiceSmall"
DEVICE = "stub"

app = FastAPI(title="Navi ASR Service", version="0.1.0")


class ASRBase64Request(BaseModel):
    """Base64 audio request accepted by the ASR service."""

    audio_base64: str = Field(..., min_length=1)
    filename: str = "audio.wav"
    lang: str = "auto"
    use_itn: bool = True
    return_raw: bool = False


class ASREventItem(BaseModel):
    """Non-speech event returned by the ASR service."""

    tag: str
    label: str
    emoji: str = ""


class ASRResultItem(BaseModel):
    """Single ASR result item matching the document output."""

    key: str
    text: str
    language: str
    emotion: str
    emotion_label: str
    emotion_emoji: str = ""
    events: List[ASREventItem] = Field(default_factory=list)
    emotion_hint: str
    raw_text: Optional[str] = None
    clean_text: Optional[str] = None


class ASRDocumentResponse(BaseModel):
    """ASR document response wrapper."""

    result: List[ASRResultItem]


@app.get("/health")
async def health() -> dict[str, str]:
    """Return ASR service health."""

    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/api/v1/asr", response_model=ASRDocumentResponse)
async def transcribe_files(
    files: List[UploadFile] = File(...),
    keys: Optional[str] = Form(None),
    lang: str = Form("auto"),
    use_itn: bool = Form(True),
    return_raw: bool = Form(False),
) -> ASRDocumentResponse:
    """Transcribe uploaded audio files with placeholder results."""

    names = _resolve_keys(keys, files)
    results = [
        _build_stub_result(
            key=names[index],
            lang=lang,
            return_raw=return_raw,
            source_name=file.filename or names[index],
        )
        for index, file in enumerate(files)
    ]
    return ASRDocumentResponse(result=results)


@app.post("/api/v1/asr/base64", response_model=ASRDocumentResponse)
async def transcribe_base64(request: ASRBase64Request) -> ASRDocumentResponse:
    """Transcribe a base64 encoded audio payload with a placeholder result."""

    return ASRDocumentResponse(
        result=[
            _build_stub_result(
                key=request.filename,
                lang=request.lang,
                return_raw=request.return_raw,
                source_name=request.filename,
            )
        ]
    )


def _resolve_keys(keys: Optional[str], files: List[UploadFile]) -> list[str]:
    if not keys:
        return [file.filename or f"audio_{index}.wav" for index, file in enumerate(files)]
    parts = [part.strip() for part in keys.split(",") if part.strip()]
    return [
        parts[index] if index < len(parts) else files[index].filename or f"audio_{index}.wav"
        for index in range(len(files))
    ]


def _build_stub_result(key: str, lang: str, return_raw: bool, source_name: str) -> ASRResultItem:
    language = "中文" if lang in {"auto", "zh"} else lang
    text = f"{source_name} 的语音识别占位结果。"
    return ASRResultItem(
        key=key,
        text=text,
        language=language,
        emotion="NEUTRAL",
        emotion_label="平静",
        events=[],
        emotion_hint="情绪：平静",
        raw_text=text if return_raw else None,
        clean_text=text if return_raw else None,
    )
