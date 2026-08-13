"""Runtime native-tool discovery tests for ASR."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from runtime_tools import ensure_ffmpeg_on_path, resolve_model_source


class ASRRuntimeToolsTest(unittest.TestCase):
    def test_exposes_configured_ffmpeg_on_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source-ffmpeg.exe"
            source.write_bytes(b"ffmpeg")
            runtime_dir = temp_path / "runtime"
            env = {"PATH": "", "NAWEI_FFMPEG_EXECUTABLE": str(source)}

            with patch.dict(os.environ, env, clear=False):
                resolved = ensure_ffmpeg_on_path(runtime_dir)
                self.assertEqual(resolved, runtime_dir / "ffmpeg.exe")
                self.assertEqual(Path(shutil.which("ffmpeg") or ""), resolved)

    def test_prefers_complete_local_model_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            snapshot = cache_root / "modelscope" / "models" / "iic--demo" / "snapshots" / "master"
            snapshot.mkdir(parents=True)
            (snapshot / "config.yaml").write_text("model: Demo", encoding="utf-8")
            (snapshot / "model.pt").write_bytes(b"model")

            resolved = resolve_model_source("iic/demo", cache_root)

            self.assertEqual(resolved, str(snapshot.resolve()))

    def test_rejects_missing_explicit_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaises(FileNotFoundError):
            resolve_model_source("iic/demo", Path(temp_dir), explicit_path="missing-model")
