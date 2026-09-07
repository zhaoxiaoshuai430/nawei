"""Emotion-tagged reference clips used by TTS generation."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from app.config import ASSET_DIR
from app.utils.cache import get_cached_ref_path

EMOTION_REFS_DIR = ASSET_DIR / "emotion_refs"
MANIFEST_PATH = EMOTION_REFS_DIR / "manifest.json"

emotion_refs: dict[str, dict[str, Any]] = {}
_enabled = False


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"enabled": True, "refs": []}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not read {MANIFEST_PATH}: {exc}")
        return {"enabled": True, "refs": []}


def load_emotion_refs() -> None:
    """Load available emotion reference clips from the local manifest."""
    global _enabled
    emotion_refs.clear()
    manifest = _load_manifest()
    _enabled = bool(manifest.get("enabled", True))

    EMOTION_REFS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in manifest.get("refs") or []:
        ref_id = (entry.get("id") or "").strip()
        audio_name = Path((entry.get("audio") or "").strip()).name
        if not ref_id or not audio_name:
            continue
        path = EMOTION_REFS_DIR / audio_name
        if not path.is_file():
            continue
        content = path.read_bytes()
        emotion_refs[ref_id] = {
            "id": ref_id,
            "label": entry.get("label") or ref_id,
            "tone": entry.get("tone") or entry.get("label") or ref_id,
            "filename": path.name,
            "path": get_cached_ref_path(content),
            "ref_text": (entry.get("ref_text") or "").strip(),
            "audio_b64": base64.b64encode(content).decode(),
            "available": True,
        }

    loaded = len(emotion_refs)
    if loaded:
        print(f"Emotion refs loaded: {loaded} clip(s) from {EMOTION_REFS_DIR}")
    elif _enabled:
        print(f"Emotion refs: no audio files yet in {EMOTION_REFS_DIR} (see README)")


def emotion_refs_enabled() -> bool:
    return _enabled and bool(emotion_refs)
