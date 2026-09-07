"""Pure ASR request helpers that can be tested without loading the model."""

from __future__ import annotations

import base64
from typing import Protocol


STREAMING_MODE = "independent-chunk"
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class AsyncUpload(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


def decode_base64_audio(audio_base64: str, *, max_bytes: int = MAX_AUDIO_BYTES) -> bytes:
    """Decode a Base64 audio payload with whitespace and missing padding support."""

    try:
        normalized = "".join(audio_base64.split())
        if not normalized:
            raise ValueError("Decoded audio is empty.")
        max_encoded_chars = ((max_bytes + 2) // 3) * 4
        if len(normalized) > max_encoded_chars:
            raise ValueError(f"Audio exceeds the {max_bytes} byte upload limit.")
        normalized += "=" * (-len(normalized) % 4)
        content = base64.b64decode(normalized, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid base64 audio: {exc}") from exc
    if not content:
        raise ValueError("Decoded audio is empty.")
    if len(content) > max_bytes:
        raise ValueError(f"Audio exceeds the {max_bytes} byte upload limit.")
    return content


async def read_upload_limited(upload: AsyncUpload, *, max_bytes: int = MAX_AUDIO_BYTES) -> bytes:
    """Read at most one byte beyond the configured upload limit."""

    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"Audio exceeds the {max_bytes} byte upload limit.")
    if not content:
        raise ValueError("Uploaded audio is empty.")
    return content


def streaming_capabilities() -> dict[str, object]:
    """Describe the actual SenseVoice HTTP streaming semantics."""

    return {
        "realtime_streaming": False,
        "streaming_mode": STREAMING_MODE,
        "accepts_audio_chunks": True,
        "keeps_decoder_state": False,
        "realtime_input": True,
        "realtime_transport": "websocket-pcm16-vad",
        "partial_results": False,
    }
