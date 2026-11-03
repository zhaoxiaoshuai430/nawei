"""Voice-clone cache warmup compatible with pip faster-qwen3-tts."""
from __future__ import annotations

from faster_qwen3_tts import FasterQwen3TTS


def warm_voice_clone_cache(
    model: FasterQwen3TTS,
    *,
    ref_audio: str,
    ref_text: str = "",
    xvec_only: bool = False,
) -> None:
    if hasattr(model, "warm_voice_clone_cache"):
        model.warm_voice_clone_cache(
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
        )
        return

    dummy = model.model._tokenize_texts([model.model._build_assistant_text("。")])[0]
    model._resolve_voice_clone_prompt_from_reference(
        input_ids=[dummy],
        ref_audio=ref_audio,
        ref_text=ref_text,
        xvec_only=xvec_only,
        append_silence=True,
    )
