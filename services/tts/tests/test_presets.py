from __future__ import annotations

from pathlib import Path

from app.services.tts import presets


def test_load_preset_refs_skips_empty_audio(monkeypatch, tmp_path: Path) -> None:
    valid_audio = tmp_path / "valid.wav"
    valid_audio.write_bytes(b"valid-audio")
    empty_audio = tmp_path / "empty.wav"
    empty_audio.touch()

    monkeypatch.setenv("TTS_SKIP_PRESET_FETCH", "1")
    monkeypatch.setattr(
        presets,
        "PRESET_REFS",
        [
            ("valid", valid_audio, "valid audio"),
            ("empty", empty_audio, "empty audio"),
        ],
    )
    monkeypatch.setattr(presets, "PRESET_TRANSCRIPTS", tmp_path / "missing.txt")
    monkeypatch.setattr(presets, "get_cached_ref_path", lambda content: f"cached-{len(content)}")

    presets.load_preset_refs()

    assert list(presets.preset_refs) == ["valid"]
    assert presets.preset_refs["valid"]["path"] == "cached-11"
