"""Voice-lab drama-style metadata used by the TTS status API."""
from __future__ import annotations

import yaml

from app.config import VOICE_LAB_DIR

EMOTIONS_YAML = VOICE_LAB_DIR / "configs" / "emotions.yaml"


def _load_emotions() -> dict:
    if not EMOTIONS_YAML.is_file():
        return {}
    with EMOTIONS_YAML.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("emotions") or {}


def list_drama_styles_for_chat() -> list[dict]:
    styles: list[dict] = []
    for key, emotion in _load_emotions().items():
        styles.append(
            {
                "id": key,
                "label": emotion.get("label") or key,
                "folder": emotion.get("folder") or key,
                "instruct": (emotion.get("instruct") or "").strip(),
            }
        )
    return styles
