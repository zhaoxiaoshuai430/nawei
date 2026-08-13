"""Resolve voice_lab fine-tuned checkpoints for Gateway loading."""
from __future__ import annotations

import os
import re
from pathlib import Path

from app.config import MODEL_ROOT, REPO_ROOT, VOICE_LAB_DIR, VOICE_LAB_SPEAKER

_EPOCH_RE = re.compile(r"checkpoint-epoch-(\d+)", re.I)


def voice_lab_model_id(speaker: str | None = None) -> str:
    name = (speaker or VOICE_LAB_SPEAKER).strip() or "nana"
    return f"voice_lab/{name}"


def is_voice_lab_model(model_id: str) -> bool:
    return (model_id or "").startswith("voice_lab/")


def is_custom_voice_capable(model_id: str | None) -> bool:
    """True for HF CustomVoice or a local voice_lab fine-tuned checkpoint."""
    if not model_id:
        return False
    if is_voice_lab_model(model_id):
        return True
    return "CustomVoice" in model_id


def voice_lab_speaker_from_model_id(model_id: str) -> str:
    if not is_voice_lab_model(model_id):
        return VOICE_LAB_SPEAKER
    return model_id.split("/", 1)[-1].strip() or VOICE_LAB_SPEAKER


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    m = _EPOCH_RE.search(path.name)
    epoch = int(m.group(1)) if m else -1
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (epoch, mtime)


def _output_roots_for_speaker(spk: str) -> list[Path]:
    roots = [
        MODEL_ROOT / "tts" / spk,
        VOICE_LAB_DIR / "speakers" / spk / "output",
        VOICE_LAB_DIR / "output" / spk,
    ]
    return [r for r in roots if r.is_dir()]


def list_trained_speakers() -> list[str]:
    """Return speaker ids that have at least one checkpoint."""
    found: set[str] = set()
    bundled_models = MODEL_ROOT / "tts"
    if bundled_models.is_dir():
        for child in bundled_models.iterdir():
            if child.is_dir() and find_latest_checkpoint(child.name):
                found.add(child.name)
    speakers_dir = VOICE_LAB_DIR / "speakers"
    if speakers_dir.is_dir():
        for child in speakers_dir.iterdir():
            if child.is_dir() and find_latest_checkpoint(child.name):
                found.add(child.name)
    legacy_out = VOICE_LAB_DIR / "output"
    if legacy_out.is_dir():
        for child in legacy_out.iterdir():
            if child.is_dir() and find_latest_checkpoint(child.name):
                found.add(child.name)
    return sorted(found)


def find_latest_checkpoint(speaker: str | None = None) -> Path | None:
    """Return newest checkpoint dir for a speaker."""
    spk = (speaker or VOICE_LAB_SPEAKER).strip() or "nana"
    explicit = (os.environ.get("VOICE_LAB_CHECKPOINT") or "").strip()
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p.resolve()
        if p.is_file():
            return p.parent.resolve()

    candidates: list[Path] = []
    for root in _output_roots_for_speaker(spk):
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if _EPOCH_RE.search(child.name) or (child / "config.json").exists():
                candidates.append(child)
        if (root / "config.json").exists():
            return root.resolve()

    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key).resolve()


def resolve_checkpoint_path(model_id: str | None = None) -> Path | None:
    if model_id and is_voice_lab_model(model_id):
        return find_latest_checkpoint(voice_lab_speaker_from_model_id(model_id))
    return find_latest_checkpoint()


def voice_lab_available(speaker: str | None = None) -> bool:
    return find_latest_checkpoint(speaker) is not None


def voice_lab_status() -> dict:
    speakers = list_trained_speakers()
    active = VOICE_LAB_SPEAKER if voice_lab_available(VOICE_LAB_SPEAKER) else (speakers[0] if speakers else None)
    ckpt = find_latest_checkpoint(active) if active else None
    model_id = voice_lab_model_id(active) if active else None
    return {
        "enabled": True,
        "speaker": active,
        "speakers": speakers,
        "model_id": model_id,
        "checkpoint": str(ckpt) if ckpt else None,
        "ready": ckpt is not None,
        "dir": str(VOICE_LAB_DIR.relative_to(REPO_ROOT)) if VOICE_LAB_DIR.exists() else "voice_lab",
    }
