"""Conservative, configurable post-processing for finalized ASR transcripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


DEFAULT_CONTEXTUAL_CORRECTIONS = {
    "请指回答结果": "请只回答结果",
    "请知回答结果": "请只回答结果",
}


@dataclass(frozen=True)
class TranscriptCorrection:
    text: str
    applied: tuple[dict[str, str], ...] = ()


def correct_transcript(text: str) -> TranscriptCorrection:
    if not _read_bool_env("ASR_TEXT_CORRECTIONS_ENABLED", True):
        return TranscriptCorrection(text=text)

    corrected = text
    applied: list[dict[str, str]] = []
    for source, replacement in _load_corrections():
        if source not in corrected:
            continue
        occurrences = corrected.count(source)
        corrected = corrected.replace(source, replacement)
        applied.append({"source": source, "replacement": replacement, "occurrences": str(occurrences)})
    return TranscriptCorrection(text=corrected, applied=tuple(applied))


def _load_corrections() -> tuple[tuple[str, str], ...]:
    corrections = dict(DEFAULT_CONTEXTUAL_CORRECTIONS) if _read_bool_env(
        "ASR_TEXT_CORRECTIONS_INCLUDE_DEFAULTS", True
    ) else {}
    raw = os.getenv("ASR_TEXT_CORRECTIONS_JSON", "").strip()
    if raw:
        configured = json.loads(raw)
        if not isinstance(configured, dict):
            raise ValueError("ASR_TEXT_CORRECTIONS_JSON must be a JSON object.")
        if len(configured) > 256:
            raise ValueError("ASR_TEXT_CORRECTIONS_JSON contains too many entries.")
        for source, replacement in configured.items():
            if not isinstance(source, str) or not isinstance(replacement, str):
                raise ValueError("ASR text corrections must map strings to strings.")
            source = source.strip()
            replacement = replacement.strip()
            if not source or not replacement or len(source) > 128 or len(replacement) > 128:
                raise ValueError("ASR text correction entries must contain 1 to 128 characters.")
            corrections[source] = replacement
    return tuple(sorted(corrections.items(), key=lambda item: len(item[0]), reverse=True))


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
