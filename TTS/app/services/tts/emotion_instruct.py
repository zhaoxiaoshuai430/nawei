"""Map ASR emotion tags to CustomVoice instruct strings for chat TTS."""
from __future__ import annotations

DEFAULT_INSTRUCT = "用自然口语化的中文说，适合朋友聊天"

# Assistant reply tone given the user's detected emotion.
RESPONSE_INSTRUCT: dict[str, str] = {
    "HAPPY": "用轻松愉快、带一点笑意的语气说",
    "SAD": "用温柔安慰、共情的语气说，语速略慢",
    "ANGRY": "用冷静、耐心、安抚的语气说",
    "NEUTRAL": "用平和、自然的语气说",
    "FEARFUL": "用温和安抚的语气说，语速平稳，给人安全感",
    "DISGUSTED": "用克制、客观、不带情绪的语气说",
    "SURPRISED": "用略带惊讶后很快恢复自然的语气说",
    "EMO_UNKNOWN": DEFAULT_INSTRUCT,
}


def resolve_tts_instruct(*, emotion_tag: str | None = None, override: str = "") -> str:
    custom = (override or "").strip()
    if custom:
        return custom
    tag = (emotion_tag or "").strip().upper()
    return RESPONSE_INSTRUCT.get(tag, DEFAULT_INSTRUCT)
