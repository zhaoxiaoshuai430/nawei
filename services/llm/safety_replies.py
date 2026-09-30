"""Age-aware safety fallback replies shared by local and external review."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping


AGE_GROUPS = ("3_5", "6_8", "9_11", "12_14", "15_17", "adult", "unknown")
AGE_REPLY_DATA: dict[str, dict[str, str]] = json.loads(
    Path(__file__).with_name("safety_replies.json").read_text(encoding="utf-8")
)
REPLIES = {category: replies["unknown"] for category, replies in AGE_REPLY_DATA.items()}

_LOCAL_CATEGORY_ALIASES = {"danger": "violence"}
_AGE_TEXT_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\s*岁)?(?!\d)")
_GRADE_GROUPS = {
    "幼儿园": "3_5", "学前": "3_5",
    "一年级": "6_8", "1年级": "6_8", "二年级": "6_8", "2年级": "6_8",
    "三年级": "9_11", "3年级": "9_11", "四年级": "9_11", "4年级": "9_11",
    "五年级": "9_11", "5年级": "9_11", "六年级": "9_11", "6年级": "9_11",
    "初一": "12_14", "初1": "12_14", "初二": "12_14", "初2": "12_14",
    "初三": "15_17", "初3": "15_17", "高一": "15_17", "高1": "15_17",
    "高二": "15_17", "高2": "15_17", "高三": "15_17", "高3": "15_17",
}
_GROUP_REPRESENTATIVE_AGES = {
    "3_5": 4,
    "6_8": 7,
    "9_11": 10,
    "12_14": 13,
    "15_17": 16,
    "adult": 18,
}


def age_group_for(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age <= 5:
        return "3_5"
    if age <= 8:
        return "6_8"
    if age <= 11:
        return "9_11"
    if age <= 14:
        return "12_14"
    if age <= 17:
        return "15_17"
    return "adult"


def reply_for(category: str, age: int | None = None, *, age_group: str | None = None) -> str:
    normalized_category = _LOCAL_CATEGORY_ALIASES.get(category, category)
    replies = AGE_REPLY_DATA.get(normalized_category, AGE_REPLY_DATA["default"])
    group = age_group if age_group in AGE_GROUPS else age_group_for(age)
    return replies.get(group, replies["unknown"])


def resolve_age(
    metadata: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    *,
    today: date | None = None,
) -> int | None:
    for source in _candidate_sources(metadata, profile):
        for key in ("age", "user_age"):
            parsed = _parse_age(source.get(key))
            if parsed is not None:
                return parsed
        for key in ("birth_date", "birthday", "date_of_birth"):
            parsed = _age_from_birth_date(source.get(key), today=today)
            if parsed is not None:
                return parsed
    return None


def resolve_age_group(
    metadata: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    *,
    today: date | None = None,
) -> str:
    age = resolve_age(metadata, profile, today=today)
    if age is not None:
        return age_group_for(age)
    for source in _candidate_sources(metadata, profile):
        grade = str(source.get("grade") or "").strip()
        for marker, group in _GRADE_GROUPS.items():
            if marker in grade:
                return group
    return "unknown"


def resolve_safety_age(
    metadata: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    *,
    today: date | None = None,
) -> int | None:
    age = resolve_age(metadata, profile, today=today)
    if age is not None:
        return age
    return _GROUP_REPRESENTATIVE_AGES.get(resolve_age_group(metadata, profile, today=today))


def _candidate_sources(
    metadata: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for value in (metadata, profile):
        if not isinstance(value, Mapping):
            continue
        sources.append(value)
        for key in ("profile", "identity", "user_profile"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                sources.append(nested)
    return sources


def _parse_age(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        age = int(value)
    else:
        match = _AGE_TEXT_RE.fullmatch(str(value).strip())
        if not match:
            return None
        age = int(match.group(1))
    return age if 0 <= age <= 120 else None


def _age_from_birth_date(value: Any, *, today: date | None) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    parsed: date | None = None
    for parser in (
        lambda: date.fromisoformat(text[:10]),
        lambda: datetime.strptime(text, "%Y/%m/%d").date(),
        lambda: datetime.strptime(text, "%Y年%m月%d日").date(),
    ):
        try:
            parsed = parser()
            break
        except ValueError:
            continue
    if parsed is None:
        return None
    current = today or date.today()
    if parsed > current:
        return None
    age = current.year - parsed.year - ((current.month, current.day) < (parsed.month, parsed.day))
    return age if 0 <= age <= 120 else None


for _category, _replies in AGE_REPLY_DATA.items():
    missing = set(AGE_GROUPS) - set(_replies)
    if missing:
        raise ValueError(f"Safety category {_category} is missing age groups: {sorted(missing)}")
