"""Shared TTS runtime state."""
from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

from faster_qwen3_tts import FasterQwen3TTS

if TYPE_CHECKING:
    from app.services.tts.staged import StagedCloneSession

model_cache: OrderedDict[str, FasterQwen3TTS] = OrderedDict()
model_cache_max: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
active_model_name: str | None = None
loading: bool = False
parakeet = None
generation_lock = asyncio.Lock()
generation_waiters: int = 0
staged_clone: StagedCloneSession | None = None
