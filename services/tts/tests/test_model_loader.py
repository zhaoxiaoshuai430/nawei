"""Tests for TTS model loading compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.tts.model_loader import resolve_model_sample_rate


def test_reads_sample_rate_from_local_speech_tokenizer(tmp_path: Path) -> None:
    tokenizer = tmp_path / "speech_tokenizer"
    tokenizer.mkdir()
    (tokenizer / "preprocessor_config.json").write_text(
        json.dumps({"sampling_rate": 22050}),
        encoding="utf-8",
    )

    assert resolve_model_sample_rate(str(tmp_path)) == 22050


def test_uses_documented_default_for_remote_model(monkeypatch) -> None:
    monkeypatch.delenv("TTS_OUTPUT_SAMPLE_RATE", raising=False)

    assert resolve_model_sample_rate("Qwen/remote-model") == 24000
