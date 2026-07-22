"""CustomVoice(instruct) → Base clone staged pipeline for chat."""
from __future__ import annotations

import os
from dataclasses import dataclass

import soundfile as sf

from app.config import SESSIONS_DIR
from app.services.tts import state
from app.services.tts.engine import load_model_into_cache
from app.services.tts.emotion_instruct import resolve_tts_instruct
from app.utils.audio import concat_audio, to_wav_b64

DEFAULT_CV_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REF_TEXT = "别担心，我会一直陪在你身边的，有什么都可以跟我说。"


@dataclass
class StagedCloneSession:
    ref_path: str
    ref_text: str
    instruct: str
    speaker: str
    cv_model: str
    base_model: str


def staged_ready() -> bool:
    sc = state.staged_clone
    return sc is not None and os.path.isfile(sc.ref_path)


def _write_ref_wav(path: str, audio_list, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio = concat_audio(audio_list)
    sf.write(path, audio, sr, subtype="PCM_16")


def prepare_staged_clone_sync(
    *,
    speaker: str,
    ref_text: str,
    instruct: str,
    language: str = "Chinese",
    cv_model: str = DEFAULT_CV_MODEL,
    base_model: str = DEFAULT_BASE_MODEL,
) -> dict:
    """Generate CV reference, switch to Base, warm clone cache."""
    ref_text = (ref_text or DEFAULT_REF_TEXT).strip()
    speaker = speaker.strip().lower()
    instruct = (instruct or "").strip()
    if not speaker:
        raise ValueError("请选择说话人")
    if not ref_text:
        raise ValueError("参考台词不能为空")

    load_model_into_cache(cv_model)
    cv = state.model_cache[cv_model]
    audio_list, sr = cv.generate_custom_voice(
        text=ref_text,
        speaker=speaker,
        language=language,
        instruct=instruct or None,
    )

    ref_path = str(SESSIONS_DIR / "staged_clone_ref.wav")
    _write_ref_wav(ref_path, audio_list, sr)
    ref_b64 = to_wav_b64(concat_audio(audio_list), sr)

    load_model_into_cache(base_model)
    base = state.model_cache[base_model]
    from app.services.tts.warmup import warm_voice_clone_cache

    warm_voice_clone_cache(
        base,
        ref_audio=ref_path,
        ref_text=ref_text,
        xvec_only=False,
    )

    state.staged_clone = StagedCloneSession(
        ref_path=ref_path,
        ref_text=ref_text,
        instruct=instruct,
        speaker=speaker,
        cv_model=cv_model,
        base_model=base_model,
    )
    state.active_model_name = base_model

    return {
        "status": "ready",
        "ref_text": ref_text,
        "instruct": instruct,
        "speaker": speaker,
        "cv_model": cv_model,
        "base_model": base_model,
        "active_model": base_model,
        "audio_b64": ref_b64,
        "sample_rate": sr,
    }


def resolve_staged_instruct(*, emotion_tag: str | None, override: str) -> str:
    return resolve_tts_instruct(emotion_tag=emotion_tag, override=override)
