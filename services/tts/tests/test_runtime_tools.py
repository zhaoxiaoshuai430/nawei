"""Runtime native-tool discovery tests for TTS."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.runtime_tools import ensure_sox_on_path


def test_exposes_configured_sox_on_path() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "sox.exe"
        source.write_bytes(b"sox")
        env = {"PATH": "", "NAWEI_SOX_EXECUTABLE": str(source)}

        with patch.dict(os.environ, env, clear=False):
            resolved = ensure_sox_on_path()
            assert resolved == source
            assert Path(shutil.which("sox") or "") == source
