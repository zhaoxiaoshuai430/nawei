"""Debounced speculative voice answers; never sends text or audio to a client."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from app_protocols import Message, MessageRole

logger = logging.getLogger(__name__)
_slots = asyncio.BoundedSemaphore(2)


class VoicePreview:
    def __init__(self, request: Any) -> None:
        self.request = request
        self.text = ""
        self.task: asyncio.Task[str | None] | None = None
        self.started = False
        self.attempts = 0
        self.enabled = os.getenv("GATEWAY_VOICE_PREVIEW_ENABLED", "false").lower() in {"1", "true", "yes", "on"}

    async def update(self, text: str) -> None:
        text = text.strip()
        if not self.enabled or text == self.text:
            return
        await self.close()
        self.text = text
        self.started = False
        if len(text) < int(os.getenv("GATEWAY_VOICE_PREVIEW_MIN_CHARS", "6")):
            return
        if self.attempts >= int(os.getenv("GATEWAY_VOICE_PREVIEW_MAX_ATTEMPTS", "4")):
            return
        self.task = asyncio.create_task(self._generate(text))

    async def _generate(self, text: str) -> str | None:
        from services.gateway import main as gateway

        try:
            await asyncio.sleep(max(0, float(os.getenv("GATEWAY_VOICE_PREVIEW_STABLE_SECONDS", "0.25"))))
            # Speculation never waits for a slot or uses the TTS GPU.
            if _slots.locked():
                return None
            async with _slots:
                self.started = True
                self.attempts += 1
                async with asyncio.timeout(30):
                    decision = await gateway._evaluate_asr_keyword_filter(
                        self.request, text, gateway.KeywordFilterSettings.load(),
                    )
                    if decision.blocked:
                        return None
                    request = gateway._voice_llm_request(
                        self.request, Message(role=MessageRole.USER, content=text),
                    ).model_copy(update={"stream": False, "defer_persistence": True})
                    response = await gateway.orchestrator.llm_provider.complete(request)
                    token = response.metadata.get("voice_preview_token")
                    return str(token) if token else None
        except Exception as exc:
            logger.info("Voice preview skipped error_type=%s", type(exc).__name__)
            return None

    async def finish(self, transcript: str) -> str | None:
        if transcript.strip() != self.text or self.task is None or not self.started:
            await self.close()
            return None
        try:
            async with asyncio.timeout(max(0.01, float(os.getenv("GATEWAY_VOICE_PREVIEW_FINAL_WAIT_SECONDS", "8")))):
                return await self.task
        except TimeoutError:
            return None
        finally:
            await self.close()

    async def close(self) -> None:
        task, self.task = self.task, None
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
