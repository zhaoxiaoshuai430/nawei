"""Shared TTS runtime state."""
from __future__ import annotations

import os
from collections import OrderedDict

from faster_qwen3_tts import FasterQwen3TTS

from app.services.tts.capacity import GenerationCapacity

model_cache: OrderedDict[str, FasterQwen3TTS] = OrderedDict()
model_cache_max: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
active_model_name: str | None = None
loading: bool = False
parakeet = None
generation_capacity = GenerationCapacity(
    limit=1,
    queue_limit=int(os.environ.get("TTS_MAX_QUEUE_SIZE", "8")),
    wait_timeout_seconds=float(os.environ.get("TTS_QUEUE_TIMEOUT_SECONDS", "120")),
)
