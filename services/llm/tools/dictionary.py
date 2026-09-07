"""English dictionary tool backed by dictionaryapi.dev."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx


DICTIONARY_URL_TEMPLATE = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
DEFAULT_TIMEOUT_SECONDS = 8.0
DICT_MARKERS = ("什么意思", "单词", "英文", "英语", "怎么读", "definition", "define", "meaning")


@dataclass(frozen=True)
class DictionaryDefinition:
    part_of_speech: str
    definition: str
    example: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"part_of_speech": self.part_of_speech, "definition": self.definition, "example": self.example}


@dataclass(frozen=True)
class DictionaryResult:
    word: str
    phonetic: str | None = None
    audio_url: str | None = None
    definitions: list[DictionaryDefinition] = field(default_factory=list)
    source: str = "dictionaryapi.dev"

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "phonetic": self.phonetic,
            "audio_url": self.audio_url,
            "definitions": [definition.to_dict() for definition in self.definitions],
            "source": self.source,
        }


class DictionaryTool:
    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout_seconds = timeout_seconds or _read_float_env("DICTIONARY_TOOL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)

    def extract_word(self, text: str) -> str | None:
        if not any(marker.lower() in text.lower() for marker in DICT_MARKERS):
            return None
        patterns = [
            r"\b([A-Za-z][A-Za-z\-']{1,40})\b\s*(?:是什么意思|什么意思|怎么读|definition|meaning)?",
            r"(?:define|definition of|meaning of)\s+([A-Za-z][A-Za-z\-']{1,40})",
            r"(?:单词|英文|英语)\s*([A-Za-z][A-Za-z\-']{1,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                word = match.group(1).strip("'").lower()
                if word not in {"what", "define", "meaning", "word"}:
                    return word
        return None

    async def lookup_from_text(self, text: str) -> DictionaryResult | None:
        word = self.extract_word(text)
        if not word:
            return None
        return await self.lookup(word)

    async def lookup(self, word: str) -> DictionaryResult:
        word = word.strip().lower()
        if not re.fullmatch(r"[a-z][a-z\-']{1,40}", word):
            raise ValueError("Only English words are supported")
        request_url = DICTIONARY_URL_TEMPLATE.format(word=quote(word))
        attempt_timeout = min(self.timeout_seconds, 2.5)
        last_error: httpx.TransportError | None = None
        for _attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=attempt_timeout) as client:
                    response = await client.get(request_url)
                    response.raise_for_status()
                    body = response.json()
                break
            except httpx.TransportError as exc:
                last_error = exc
        else:
            assert last_error is not None
            raise last_error
        return _parse_dictionary_body(word=word, body=body)


def format_dictionary_system_prompt(result: DictionaryResult) -> str:
    lines = [
        "实时工具结果：英语词典\n",
        f"- 单词：{result.word}\n",
    ]
    if result.phonetic:
        lines.append(f"- 音标：{result.phonetic}\n")
    for index, definition in enumerate(result.definitions[:3], start=1):
        lines.append(f"- 释义 {index}（{definition.part_of_speech}）：{definition.definition}\n")
        if definition.example:
            lines.append(f"  例句：{definition.example}\n")
    lines.append("回答英语词汇问题时必须基于以上词典结果，解释要适合学生理解。")
    return "".join(lines)


def _parse_dictionary_body(*, word: str, body: Any) -> DictionaryResult:
    entries = body if isinstance(body, list) else []
    if not entries:
        raise ValueError(f"Dictionary result not found: {word}")
    entry = entries[0]
    phonetic = entry.get("phonetic")
    audio_url = None
    for phonetic_item in entry.get("phonetics") or []:
        if not phonetic and phonetic_item.get("text"):
            phonetic = phonetic_item.get("text")
        if phonetic_item.get("audio"):
            audio_url = phonetic_item.get("audio")
            break
    definitions: list[DictionaryDefinition] = []
    for meaning in entry.get("meanings") or []:
        part = str(meaning.get("partOfSpeech") or "")
        for item in meaning.get("definitions") or []:
            definition = str(item.get("definition") or "").strip()
            if definition:
                definitions.append(
                    DictionaryDefinition(
                        part_of_speech=part,
                        definition=definition,
                        example=item.get("example"),
                    )
                )
            if len(definitions) >= 5:
                break
        if len(definitions) >= 5:
            break
    return DictionaryResult(word=str(entry.get("word") or word), phonetic=phonetic, audio_url=audio_url, definitions=definitions)


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return float(raw)
