"""Conversation orchestration contracts and a minimal composition skeleton."""

# 共享编排层只依赖 Provider 协议与 DTO，不导入具体模型，也不创建数据库连接。
# Gateway 注入 HTTPASRProvider/HTTPLLMProvider/HTTPTTSProvider，将这些步骤连接到独立服务。
# 本类处理普通文字/完整响应路径；低延迟的实时语音并发分段另见 gateway/main.py。

from __future__ import annotations

from typing import AsyncIterator, Iterable, Optional

from .enums import ErrorCode, EventType
from .errors import ProtocolErrorDetail, ProtocolException
from .events import StreamEvent
from .providers import ASRProvider, LLMProvider, TTSProvider
from .schemas import (
    ASRRequest,
    AudioTranscriptionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    LLMRequest,
    Message,
    TTSRequest,
)


class ConversationOrchestrator:
    """Lightweight ASR/LLM/TTS orchestration skeleton.

    Gateway injects HTTP providers into this shared composition layer.
    Persistent history and system prompts remain owned by the LLM service.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tts_provider: Optional[TTSProvider] = None,
        asr_provider: Optional[ASRProvider] = None,
    ) -> None:
        self.asr_provider = asr_provider
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider

    async def transcribe(self, request: ASRRequest) -> AudioTranscriptionResponse:
        """Run an audio transcription through the configured ASR provider."""

        if self.asr_provider is None:
            raise ProtocolException(
                ProtocolErrorDetail(
                    code=ErrorCode.MODEL_UNAVAILABLE,
                    message="ASR provider is not configured.",
                    request_id=request.request_id,
                    trace_id=request.trace_id,
                    retryable=True,
                )
            )

        response = await self.asr_provider.transcribe(request)
        return AudioTranscriptionResponse(
            request_id=response.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            text=response.text,
            language=response.language,
            confidence=response.confidence,
            duration_ms=response.duration_ms,
            segments=response.segments,
            asr=response.asr,
            metadata=response.metadata,
        )

    async def complete(
        self,
        request: ChatCompletionRequest,
        history: Iterable[Message] = (),
    ) -> ChatCompletionResponse:
        """Run a non-streaming chat turn through LLM and optional TTS providers."""

        # 先取得完整 LLM 回答，只有 output_audio 开启且有文字时才调用 TTS。
        # 最终响应同时携带文本、音频参数和上游元数据，客户端无需自行拼接服务响应。
        messages = self._build_messages(history, request.message)
        llm_response = await self.llm_provider.complete(
            LLMRequest(
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                user_id=request.user_id,
                messages=messages,
                stream=False,
                metadata=request.metadata,
            )
        )

        audio_url = None
        audio_base64 = None
        duration_ms = None
        audio_format = request.audio_config.format
        sample_rate = request.audio_config.sample_rate
        metadata = dict(llm_response.metadata)
        if request.output_audio and self.tts_provider is not None and llm_response.text:
            tts_response = await self.tts_provider.synthesize(
                TTSRequest(
                    request_id=request.request_id,
                    trace_id=request.trace_id,
                    session_id=request.session_id,
                    text=llm_response.text,
                    stream=False,
                    voice=request.voice,
                    audio_config=request.audio_config,
                )
            )
            audio_url = tts_response.audio_url
            audio_base64 = tts_response.audio_base64
            duration_ms = tts_response.duration_ms
            audio_format = tts_response.format
            sample_rate = tts_response.sample_rate
            metadata["tts"] = tts_response.metadata

        return ChatCompletionResponse(
            message_id=request.request_id,
            session_id=request.session_id,
            text=llm_response.text,
            audio_url=audio_url,
            audio_base64=audio_base64,
            duration_ms=duration_ms,
            format=audio_format,
            sample_rate=sample_rate,
            usage=llm_response.usage,
            metadata=metadata,
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
        history: Iterable[Message] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Run a streaming chat turn and optionally synthesize audio chunks."""

        # 事件顺序：MESSAGE_STARTED -> 文字增量 -> 可选音频事件 -> MESSAGE_DONE。
        # 此通用路径先累积完整文字再合成；它不同于网关实时语音的边生成边分句合成。
        yield StreamEvent(
            event=EventType.MESSAGE_STARTED,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
        )

        messages = self._build_messages(history, request.message)
        llm_request = LLMRequest(
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            user_id=request.user_id,
            messages=messages,
            stream=True,
            metadata=request.metadata,
        )

        buffer: list[str] = []
        message_done: StreamEvent | None = None
        async for event in self.llm_provider.stream_complete(llm_request):
            if event.event == EventType.MESSAGE_DELTA:
                text = str(event.data.get("text", ""))
                buffer.append(text)
                yield event
            elif event.event == EventType.MESSAGE_DONE:
                message_done = event
            else:
                yield event

        full_text = "".join(buffer).strip()
        if request.output_audio and self.tts_provider and full_text:
            yield StreamEvent(
                event=EventType.AUDIO_STARTED,
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
            )
            tts_request = TTSRequest(
                request_id=request.request_id,
                trace_id=request.trace_id,
                session_id=request.session_id,
                text=full_text,
                stream=True,
                voice=request.voice,
                audio_config=request.audio_config,
            )
            async for audio_event in self.tts_provider.stream_synthesize(tts_request):
                yield audio_event

        yield message_done or StreamEvent(
            event=EventType.MESSAGE_DONE,
            request_id=request.request_id,
            trace_id=request.trace_id,
            session_id=request.session_id,
            data={"message_id": request.request_id, "finish_reason": "stop"},
        )

    @staticmethod
    def _build_messages(history: Iterable[Message], current: Message) -> list[Message]:
        """Build messages while letting the standalone LLM own its system prompt."""

        # history 是调用方显式传入的历史，不在这里查询或保存；当前网关把记忆加载交给 LLM。
        messages = list(history)
        messages.append(current)
        return messages
