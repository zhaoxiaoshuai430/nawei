"""Short-lived, single-use answers prepared before a voice turn is final."""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from app_protocols import LLMRequest, LLMResponse

if TYPE_CHECKING:
    from services.llm.agent_runtime import AgentTurn


def request_key(request: LLMRequest) -> dict[str, Any]:
    return {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "messages": [message.model_dump(exclude={"metadata": {"asr"}}) for message in request.messages],
        "metadata": request.metadata,
        "tools": request.tools,
        "generation_config": request.generation_config,
    }


class VoicePreviewCache:
    def __init__(self, limit: int = 64, ttl: float = 60) -> None:
        self.limit = limit
        self.ttl = ttl
        self.entries: OrderedDict[str, tuple[float, dict[str, Any], AgentTurn, LLMResponse]] = OrderedDict()

    def _expire(self) -> None:
        now = time.monotonic()
        for token, entry in list(self.entries.items()):
            if now - entry[0] >= self.ttl:
                del self.entries[token]

    def put(self, request: LLMRequest, turn: AgentTurn, response: LLMResponse) -> str:
        self._expire()
        while len(self.entries) >= self.limit:
            self.entries.popitem(last=False)
        token = uuid.uuid4().hex
        self.entries[token] = (time.monotonic(), request_key(request), turn, response)
        return token

    def take(self, request: LLMRequest) -> tuple[AgentTurn, LLMResponse] | None:
        self._expire()
        token = request.voice_preview_token or ""
        entry = self.entries.get(token)
        if entry is None or entry[1] != request_key(request):
            return None
        del self.entries[token]
        turn, response = entry[2:]
        turn.original_request = request
        turn.request = turn.request.model_copy(update={
            "request_id": request.request_id, "trace_id": request.trace_id,
            "defer_persistence": False, "voice_preview_token": None,
        })
        turn.metadata["voice_preview_used"] = True
        return turn, response.model_copy(update={"request_id": request.request_id})
