"""Optional FastAPI router skeleton for the protocol layer.

The router is not imported or mounted by the current demo server. It is a
reference adapter for future Gateway/BFF adoption.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from .enums import ErrorCode, EventType, MessageRole
from .errors import ProtocolException
from .events import StreamEvent, sse_encode
from .orchestrator import ConversationOrchestrator
from .schemas import (
    ASRRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CreateSessionRequest,
    Message,
)

try:
    from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover - optional demo dependency
    APIRouter = None  # type: ignore[assignment]
    File = None  # type: ignore[assignment]
    Form = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    Query = None  # type: ignore[assignment]
    UploadFile = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]


def create_protocol_router(orchestrator: ConversationOrchestrator) -> Any:
    """Create an optional FastAPI router for protocol-first endpoints."""

    if APIRouter is None or HTTPException is None or StreamingResponse is None:
        raise RuntimeError("FastAPI is required to create the protocol router.")

    router = APIRouter(prefix="/api/v1", tags=["voice-ai-protocol"])

    @router.post("/sessions")
    async def create_session(request: CreateSessionRequest) -> dict[str, str]:
        """Create a placeholder session response."""

        return {
            "session_id": f"s_{request.user_id}",
            "created_at": "1970-01-01T00:00:00Z",
        }

    @router.post("/audio/transcriptions")
    async def transcribe_audio(
        session_id: str = Form(...),
        audio: UploadFile = File(...),
        language: str = Form("auto"),
        return_timestamps: bool = Form(False),
        x_request_id: str = Header(..., alias="X-Request-Id"),
        x_trace_id: str = Header(..., alias="X-Trace-Id"),
    ) -> Any:
        """Transcribe an uploaded audio file through the configured ASR provider.

        The current protocol layer does not own object storage. Gateway adopters
        should replace the upload URI with a durable object-storage URL.
        """

        try:
            return await orchestrator.transcribe(
                ASRRequest(
                    request_id=x_request_id,
                    trace_id=x_trace_id,
                    session_id=session_id,
                    audio_url=f"upload://{audio.filename or x_request_id}",
                    language=language,
                    return_timestamps=return_timestamps,
                    options={
                        "content_type": audio.content_type,
                        "filename": audio.filename,
                    },
                )
            )
        except ProtocolException as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail.dict()) from exc

    @router.post("/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run a non-streaming protocol chat completion."""

        if request.stream:
            raise HTTPException(status_code=400, detail="Use /chat/stream for streaming.")
        try:
            return await orchestrator.complete(request)
        except ProtocolException as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail.dict()) from exc

    @router.get("/chat/stream")
    async def stream_chat(
        session_id: str = Query(...),
        request_id: str = Query(...),
        trace_id: str = Query(...),
        message: str = Query(...),
        output_audio: bool = Query(False),
    ) -> Any:
        """Stream chat events with SSE from query-state input."""

        request = ChatCompletionRequest(
            request_id=request_id,
            trace_id=trace_id,
            session_id=session_id,
            message=Message(role=MessageRole.USER, content=message),
            stream=True,
            output_audio=output_audio,
        )

        async def stream_events() -> AsyncIterator[str]:
            try:
                async for event in orchestrator.stream(request):
                    yield sse_encode(event)
            except ProtocolException as exc:
                yield sse_encode(
                    StreamEvent(
                        event=EventType.ERROR,
                        request_id=request_id,
                        trace_id=trace_id,
                        session_id=session_id,
                        data=exc.detail.dict(),
                    )
                )

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    return router


def request_placeholder_event(message: str) -> StreamEvent:
    """Build a placeholder error event without importing FastAPI internals."""

    return StreamEvent(
        event=EventType.ERROR,
        request_id="placeholder",
        trace_id="placeholder",
        session_id="placeholder",
        data={
            "code": ErrorCode.INVALID_ARGUMENT,
            "message": message,
            "retryable": False,
        },
    )