"""Emotion-tagged reference clips for Base voice clone tone switching."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from app.config import ASSET_DIR, MAX_AUDIO_BYTES
from app.services.tts.warmup import warm_voice_clone_cache
from app.utils.audio import AudioFormatError, bytes_to_wav_bytes
from app.utils.cache import get_cached_ref_path

EMOTION_REFS_DIR = ASSET_DIR / "emotion_refs"
MANIFEST_PATH = EMOTION_REFS_DIR / "manifest.json"

DEFAULT_ASR_TO_REF: dict[str, str] = {
    "HAPPY": "emo_happy",
    "SAD": "emo_comfort",
    "ANGRY": "emo_calm",
    "NEUTRAL": "emo_neutral",
    "FEARFUL": "emo_comfort",
    "DISGUSTED": "emo_neutral",
    "SURPRISED": "emo_happy",
    "EMO_UNKNOWN": "emo_neutral",
}

ASR_EMOTION_LABELS: dict[str, str] = {
    "HAPPY": "开心",
    "SAD": "悲伤",
    "ANGRY": "生气",
    "NEUTRAL": "平静",
    "FEARFUL": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
    "EMO_UNKNOWN": "未知",
}

emotion_refs: dict[str, dict[str, Any]] = {}
_asr_to_ref: dict[str, str] = {}
_enabled = False


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"enabled": True, "refs": [], "asr_to_ref": DEFAULT_ASR_TO_REF}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read {MANIFEST_PATH}: {e}")
        return {"enabled": True, "refs": [], "asr_to_ref": DEFAULT_ASR_TO_REF}


def _write_manifest_ref_text(ref_id: str, ref_text: str) -> None:
    manifest = _load_manifest()
    for i, entry in enumerate(manifest.get("refs") or []):
        if (entry.get("id") or "").strip() == ref_id:
            manifest["refs"][i]["ref_text"] = ref_text
            MANIFEST_PATH.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return


def load_emotion_refs() -> None:
    """Load emotion reference bank from assets/emotion_refs/manifest.json."""
    global emotion_refs, _asr_to_ref, _enabled
    emotion_refs.clear()
    manifest = _load_manifest()
    _enabled = bool(manifest.get("enabled", True))
    raw_map = manifest.get("asr_to_ref") or {}
    _asr_to_ref = {**DEFAULT_ASR_TO_REF, **{k.upper(): v for k, v in raw_map.items()}}

    EMOTION_REFS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in manifest.get("refs") or []:
        ref_id = (entry.get("id") or "").strip()
        audio_name = (entry.get("audio") or "").strip()
        if not ref_id or not audio_name:
            continue
        path = EMOTION_REFS_DIR / audio_name
        if not path.exists():
            continue
        content = path.read_bytes()
        cached_path = get_cached_ref_path(content)
        emotion_refs[ref_id] = {
            "id": ref_id,
            "label": entry.get("label") or ref_id,
            "tone": entry.get("tone") or entry.get("label") or ref_id,
            "filename": path.name,
            "path": cached_path,
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


def list_emotion_refs() -> list[dict[str, Any]]:
    manifest = _load_manifest()
    slots: list[dict[str, Any]] = []
    for entry in manifest.get("refs") or []:
        ref_id = (entry.get("id") or "").strip()
        if not ref_id:
            continue
        loaded = emotion_refs.get(ref_id)
        slots.append(
            {
                "id": ref_id,
                "label": entry.get("label") or ref_id,
                "tone": entry.get("tone") or entry.get("label") or ref_id,
                "ref_text": (entry.get("ref_text") or "").strip(),
                "audio": entry.get("audio") or "",
                "available": loaded is not None,
            }
        )
    return slots


def list_asr_to_ref_display() -> list[dict[str, str]]:
    manifest = _load_manifest()
    raw_map = manifest.get("asr_to_ref") or {}
    merged = {**DEFAULT_ASR_TO_REF, **{k.upper(): v for k, v in raw_map.items()}}
    id_to_label = {e["id"]: e.get("label") or e["id"] for e in list_emotion_refs()}
    return [
        {
            "asr_tag": tag,
            "asr_label": ASR_EMOTION_LABELS.get(tag, tag),
            "ref_id": ref_id,
            "ref_label": id_to_label.get(ref_id, ref_id),
        }
        for tag, ref_id in merged.items()
    ]


def describe_asr_ref(emotion_tag: str | None) -> dict[str, str] | None:
    ref_id = resolve_asr_to_emotion_ref(emotion_tag)
    if not ref_id:
        return None
    preset = emotion_refs.get(ref_id)
    tag = (emotion_tag or "").strip().upper() or "NEUTRAL"
    return {
        "asr_tag": tag,
        "asr_label": ASR_EMOTION_LABELS.get(tag, tag),
        "ref_id": ref_id,
        "ref_label": preset["label"] if preset else ref_id,
        "tone": preset.get("tone", preset["label"]) if preset else ref_id,
    }


def validate_emotion_refs_for_chat() -> list[str]:
    errors: list[str] = []
    available = [s for s in list_emotion_refs() if s.get("available")]
    if not available:
        errors.append("请至少上传一段情绪参考音频")
        return errors
    for slot in available:
        if not (slot.get("ref_text") or "").strip():
            errors.append(f"「{slot['label']}」缺少参考转写（高级 ICL 必填）")
    return errors


def resolve_asr_to_emotion_ref(emotion_tag: str | None) -> str | None:
    tag = (emotion_tag or "").strip().upper() or "NEUTRAL"
    ref_id = _asr_to_ref.get(tag) or _asr_to_ref.get("NEUTRAL")
    if ref_id and ref_id in emotion_refs:
        return ref_id
    if "emo_neutral" in emotion_refs:
        return "emo_neutral"
    return next(iter(emotion_refs), None)


def resolve_emotion_ref_path(
    ref_id: str,
    *,
    ref_text: str = "",
) -> tuple[str | None, bool, str]:
    preset = emotion_refs.get(ref_id)
    if not preset:
        return None, False, ref_text
    if not ref_text:
        ref_text = preset["ref_text"]
    return preset["path"], True, ref_text


def get_emotion_ref_detail(ref_id: str) -> dict[str, Any] | None:
    preset = emotion_refs.get(ref_id)
    if not preset:
        slot = next((s for s in list_emotion_refs() if s["id"] == ref_id), None)
        if not slot:
            return None
        return {**slot, "audio_b64": None}
    return {
        "id": preset["id"],
        "label": preset["label"],
        "tone": preset.get("tone", preset["label"]),
        "filename": preset["filename"],
        "ref_text": preset["ref_text"],
        "audio_b64": preset.get("audio_b64"),
        "available": True,
    }


def clear_voice_clone_cache_for_ref(model, ref_id: str) -> None:
    preset = emotion_refs.get(ref_id)
    if preset is None or not hasattr(model, "_voice_prompt_cache"):
        return
    ref_path = str(preset["path"])
    for key in list(model._voice_prompt_cache.keys()):
        if key[0] == ref_path:
            model._voice_prompt_cache.pop(key, None)


def prewarm_emotion_ref_sync(
    model,
    *,
    ref_id: str,
    xvec_only: bool = False,
    clear_cache: bool = True,
) -> dict[str, str] | None:
    preset = emotion_refs.get(ref_id)
    if not preset:
        return None
    if clear_cache:
        clear_voice_clone_cache_for_ref(model, ref_id)
    warm_voice_clone_cache(
        model,
        ref_audio=preset["path"],
        ref_text=preset["ref_text"],
        xvec_only=xvec_only,
    )
    return {"id": ref_id, "label": preset["label"], "tone": preset.get("tone", preset["label"])}


def cleanup_emotion_ref_path(path: str | None, is_cached: bool) -> None:
    if path and os.path.exists(path) and not is_cached:
        os.unlink(path)


def _manifest_entry(ref_id: str) -> dict[str, Any] | None:
    manifest = _load_manifest()
    for entry in manifest.get("refs") or []:
        if (entry.get("id") or "").strip() == ref_id:
            return entry
    return None


def _slot_response(ref_id: str) -> dict[str, Any]:
    preset = emotion_refs.get(ref_id)
    entry = _manifest_entry(ref_id) or {}
    return {
        "id": ref_id,
        "label": preset["label"] if preset else entry.get("label", ref_id),
        "tone": preset.get("tone", entry.get("tone", ref_id)) if preset else entry.get("tone", ref_id),
        "ref_text": preset["ref_text"] if preset else (entry.get("ref_text") or "").strip(),
        "available": preset is not None,
        "audio_b64": preset.get("audio_b64") if preset else None,
    }


def save_emotion_ref_text(*, ref_id: str, ref_text: str) -> dict[str, Any]:
    ref_id = (ref_id or "").strip()
    if not _manifest_entry(ref_id):
        raise ValueError(f"未知情绪参考 id: {ref_id}")

    new_text = (ref_text or "").strip()
    if not new_text:
        raise ValueError("参考转写不能为空；请填写与音频一致的台词后点「保存转写」")

    _write_manifest_ref_text(ref_id, new_text)
    load_emotion_refs()
    return {
        "status": "text_saved",
        "ref_id": ref_id,
        "slot": _slot_response(ref_id),
        "emotion_refs": list_emotion_refs(),
        "emotion_refs_enabled": emotion_refs_enabled(),
    }


def save_emotion_ref_upload(
    *,
    ref_id: str,
    content: bytes,
    filename: str = "",
    ref_text: str | None = None,
) -> dict[str, Any]:
    """Save audio; ref_text=None keeps manifest transcript, non-empty updates it."""
    ref_id = (ref_id or "").strip()
    entry = _manifest_entry(ref_id)
    if not entry:
        raise ValueError(f"未知情绪参考 id: {ref_id}")

    if len(content) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"音频过大 ({len(content) / 1024 / 1024:.1f} MB)，请使用 1 分钟以内的短片段"
        )

    if ref_text is not None and not ref_text.strip():
        raise ValueError("参考转写不能为空；换音频时会保留原转写，改台词请点「保存转写」")

    audio_name = Path((entry.get("audio") or "").strip()).name
    if not audio_name or audio_name in (".", "..") or "/" in audio_name or "\\" in audio_name:
        raise ValueError("manifest 中 audio 文件名无效")

    suffix = Path(filename or audio_name).suffix or ".wav"
    try:
        wav_bytes = bytes_to_wav_bytes(content, suffix)
    except AudioFormatError as exc:
        raise ValueError(str(exc.detail)) from exc

    EMOTION_REFS_DIR.mkdir(parents=True, exist_ok=True)
    dest = EMOTION_REFS_DIR / audio_name
    dest.write_bytes(wav_bytes)

    if ref_text is not None:
        _write_manifest_ref_text(ref_id, ref_text.strip())

    load_emotion_refs()
    return {
        "status": "saved",
        "ref_id": ref_id,
        "slot": _slot_response(ref_id),
        "filename": audio_name,
        "emotion_refs": list_emotion_refs(),
        "emotion_refs_enabled": emotion_refs_enabled(),
    }
