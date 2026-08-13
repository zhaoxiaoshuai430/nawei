"""Preset reference audio for TTS demo."""
from __future__ import annotations

import base64
import os
from pathlib import Path

from app.config import ASSET_DIR
from app.services.tts import emotion_refs
from app.services.tts.warmup import warm_voice_clone_cache
from app.utils.cache import get_cached_ref_path

PRESET_TRANSCRIPTS = ASSET_DIR / "samples" / "parity" / "icl_transcripts.txt"
PRESET_REFS = [
    ("ref_audio_1", ASSET_DIR / "ref_audio_11.wav", "女主持人"),
    ("ref_audio_5", ASSET_DIR / "ref_audio_15.wav", "男主持人"),
    ("ref_audio_2", ASSET_DIR / "ref_audio_12.wav", "女老者"),
    ("ref_audio_3", ASSET_DIR / "ref_audio_13.wav", "男老者"),
    ("ref_audio_4", ASSET_DIR / "ref_audio_14.wav", "儿童"),
]

_GITHUB_RAW = "https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main"
_PRESET_REMOTE = {
    "ref_audio_1": f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_2": f"{_GITHUB_RAW}/ref_audio_2.wav",
    "ref_audio_3": f"{_GITHUB_RAW}/ref_audio_3.wav",
    "ref_audio_4": f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_5": f"{_GITHUB_RAW}/ref_audio_2.wav",
}
_TRANSCRIPT_REMOTE = f"{_GITHUB_RAW}/samples/parity/icl_transcripts.txt"

preset_refs: dict[str, dict] = {}


def fetch_preset_assets() -> None:
    import urllib.request

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    if not PRESET_TRANSCRIPTS.exists():
        try:
            urllib.request.urlretrieve(_TRANSCRIPT_REMOTE, PRESET_TRANSCRIPTS)
        except Exception as e:
            print(f"Warning: could not fetch transcripts: {e}")
    for key, path, _ in PRESET_REFS:
        if (not path.is_file() or path.stat().st_size == 0) and key in _PRESET_REMOTE:
            try:
                urllib.request.urlretrieve(_PRESET_REMOTE[key], path)
                print(f"Downloaded {path.name}")
            except Exception as e:
                print(f"Warning: could not fetch {key}: {e}")


def _load_preset_transcripts() -> dict[str, str]:
    if not PRESET_TRANSCRIPTS.exists():
        return {}
    transcripts = {}
    for line in PRESET_TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


def load_preset_refs() -> None:
    global preset_refs
    if os.environ.get("TTS_SKIP_PRESET_FETCH", "").strip().lower() not in {"1", "true", "yes"}:
        fetch_preset_assets()
    transcripts = _load_preset_transcripts()
    preset_refs.clear()
    for key, path, label in PRESET_REFS:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        content = path.read_bytes()
        cached_path = get_cached_ref_path(content)
        preset_refs[key] = {
            "id": key,
            "label": label,
            "filename": path.name,
            "path": cached_path,
            "ref_text": transcripts.get(key, ""),
            "audio_b64": base64.b64encode(content).decode(),
        }


def prime_preset_voice_cache(model) -> None:
    if not preset_refs:
        return
    for preset in preset_refs.values():
        ref_path = preset["path"]
        ref_text = preset["ref_text"]
        for xvec_only in (True, False):
            try:
                warm_voice_clone_cache(
                    model,
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                )
            except Exception:
                continue


def resolve_ref_path(
    *,
    ref_preset: str,
    ref_audio_content: bytes | None,
    ref_text: str,
) -> tuple[str | None, bool, str]:
    """Return (path, is_cached, ref_text)."""
    if ref_preset and ref_preset in emotion_refs.emotion_refs:
        path, cached, resolved_text = emotion_refs.resolve_emotion_ref_path(
            ref_preset, ref_text=ref_text
        )
        if path:
            return path, cached, resolved_text
    if ref_preset and ref_preset in preset_refs:
        preset = preset_refs[ref_preset]
        if not ref_text:
            ref_text = preset["ref_text"]
        return preset["path"], True, ref_text
    if ref_audio_content:
        return get_cached_ref_path(ref_audio_content), True, ref_text
    return None, False, ref_text


def cleanup_ref_path(path: str | None, is_cached: bool) -> None:
    if path and os.path.exists(path) and not is_cached:
        os.unlink(path)
