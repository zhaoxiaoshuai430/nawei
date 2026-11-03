"""Reference audio disk cache."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

_ref_cache_lock = threading.Lock()


def _matches_content(path: Path, content: bytes) -> bool:
    try:
        with path.open("rb") as cached:
            if os.fstat(cached.fileno()).st_size != len(content):
                return False
            for offset in range(0, len(content), 65536):
                if cached.read(65536) != content[offset:offset + 65536]:
                    return False
            return cached.read(1) == b""
    except OSError:
        return False


def get_cached_ref_path(content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()
    path = Path(tempfile.gettempdir()) / f"faster_qwen3_tts_ref_{digest}.wav"
    with _ref_cache_lock:
        if _matches_content(path, content):
            return str(path)
        temporary: Path | None = None
        try:
            # Close the file before replacement on Windows. A unique sibling
            # file also prevents other processes from seeing a partial WAV.
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(content)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return str(path)
