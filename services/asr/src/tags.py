"""Parse SenseVoice rich tags (emotion, events, language) from raw ASR output."""
from __future__ import annotations

import re
from typing import Any

TAG_RE = re.compile(r"<\|([^|]+)\|>")

EMOTIONS: dict[str, tuple[str, str]] = {
    "HAPPY": ("开心", "😊"),
    "SAD": ("悲伤", "😔"),
    "ANGRY": ("生气", "😡"),
    "NEUTRAL": ("平静", ""),
    "FEARFUL": ("恐惧", "😰"),
    "DISGUSTED": ("厌恶", "🤢"),
    "SURPRISED": ("惊讶", "😮"),
    "EMO_UNKNOWN": ("未知", ""),
}

EVENTS: dict[str, tuple[str, str]] = {
    "Laughter": ("笑声", "😀"),
    "Applause": ("掌声", "👏"),
    "BGM": ("背景音乐", "🎼"),
    "Cry": ("哭声", "😭"),
    "Sneeze": ("喷嚏", "🤧"),
    "Cough": ("咳嗽", "🤧"),
    "Sing": ("唱歌", "🎤"),
}

LANG_LABELS = {
    "zh": "中文",
    "en": "英语",
    "yue": "粤语",
    "ja": "日语",
    "ko": "韩语",
    "nospeech": "无语音",
}


def parse_sensevoice_raw(raw: str) -> dict[str, Any]:
    tags = TAG_RE.findall(raw or "")
    emotion_tag: str | None = None
    emotion_label = ""
    emotion_emoji = ""
    events: list[dict[str, str]] = []
    language = ""

    for tag in tags:
        if tag in EMOTIONS:
            emotion_tag = tag
            emotion_label, emotion_emoji = EMOTIONS[tag]
        elif tag in EVENTS:
            label, emoji = EVENTS[tag]
            events.append({"tag": tag, "label": label, "emoji": emoji})
        elif tag in LANG_LABELS:
            language = LANG_LABELS[tag]

    hint_parts: list[str] = []
    if emotion_tag and emotion_tag not in ("NEUTRAL", "EMO_UNKNOWN"):
        hint_parts.append(f"情绪：{emotion_label}")
    elif emotion_tag == "NEUTRAL":
        hint_parts.append("情绪：平静")
    event_labels = [e["label"] for e in events if e["tag"] != "Speech"]
    if event_labels:
        hint_parts.append("伴随：" + "、".join(event_labels))

    return {
        "emotion": emotion_tag,
        "emotion_label": emotion_label,
        "emotion_emoji": emotion_emoji,
        "events": events,
        "language": language,
        "emotion_hint": "；".join(hint_parts),
    }
