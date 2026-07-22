"""Reference audio disk cache."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

_ref_cache: dict[str, str] = {}
_ref_cache_lock = threading.Lock()


def get_cached_ref_path(content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()
    with _ref_cache_lock:
        cached = _ref_cache.get(digest)
        if cached and os.path.exists(cached):
            return cached
        path = Path(tempfile.gettempdir()) / f"faster_qwen3_tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        _ref_cache[digest] = str(path)
        return str(path)
