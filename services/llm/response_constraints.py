"""Deterministic output constraints derived from Xiaowei's age response policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_CHINESE_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
_MARKDOWN_PREFIX_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s*|[-*+]\s+|\d+[.、]\s*)")
_SENTENCE_RE = re.compile(r"[^。！？!?]+[。！？!?]?", re.DOTALL)
_TERMINAL_RE = re.compile(r"[。！？!?]")
_QUESTION_MARKS = {"？", "?"}
_SAFETY_SCENES = {"crisis_support", "account_security"}
_DETAILED_TEACHING_SCENES = {
    "explain",
    "knowledge_exploration",
    "mistake_review",
    "remediation",
    "review",
}
_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?(?:参考来源|参考资料|参考文献|sources?|references?)\s*[:：]?\s*$"
)
_NO_FOLLOWUP_QUESTION_RE = re.compile(
    r"(?:不要|别|不用|无需)(?:再)?(?:反问|追问|提问|问我)"
)
_INVALID_EXPLORATION_CONNECTORS = ("还是", "或者", "或是", "以及", "二选一")
_DEFAULT_EXPLORATION_QUESTION = "你想继续看看这个问题背后的原理，和它在实际场景中怎样应用吗？"


@dataclass(frozen=True)
class AgeConstraintSpec:
    stage: str
    maximum_body_chars: int
    maximum_questions: int
    maximum_body_sentences: int | None = None
    maximum_sentence_chars: int | None = None
    maximum_body_paragraphs: int | None = None
    safety_exempt: bool = False


def age_constraint_spec(response_policy: Mapping[str, Any] | None) -> AgeConstraintSpec | None:
    policy = dict(response_policy or {})
    audience = policy.get("audience_profile")
    if not isinstance(audience, Mapping):
        return None
    stage = str(audience.get("age_stage") or "").strip()
    try:
        maximum_body_chars = int(audience.get("maximum_response_chars") or 0)
        maximum_questions = int(audience.get("maximum_questions") or 0)
    except (TypeError, ValueError):
        return None
    if stage not in {"S0", "S1", "S2", "S3", "S4"} or maximum_body_chars <= 0:
        return None
    stage_limits: dict[str, tuple[int | None, int | None, int | None]] = {
        "S0": (2, 25, 1),
        "S1": (4, 30, 2),
        "S2": (None, None, 2),
        "S3": (None, None, None),
        "S4": (None, None, None),
    }
    maximum_sentences, maximum_sentence_chars, maximum_paragraphs = stage_limits[stage]
    safety_exempt = (
        str(policy.get("scene") or "") in _SAFETY_SCENES
        or str(policy.get("care_focus") or "") == "immediate_safety"
    )
    return AgeConstraintSpec(
        stage=stage,
        maximum_body_chars=maximum_body_chars,
        maximum_questions=max(0, maximum_questions),
        maximum_body_sentences=maximum_sentences,
        maximum_sentence_chars=maximum_sentence_chars,
        maximum_body_paragraphs=maximum_paragraphs,
        safety_exempt=safety_exempt,
    )


def recommended_age_max_tokens(response_policy: Mapping[str, Any] | None) -> int | None:
    """Return a conservative model token budget with room for one exploration question."""

    spec = age_constraint_spec(response_policy)
    if spec is None or spec.safety_exempt:
        return None
    audience = dict((response_policy or {}).get("audience_profile") or {})
    try:
        target_chars = int(audience.get("generation_target_chars") or 0)
    except (TypeError, ValueError):
        return None
    if target_chars <= 0:
        return None
    return max(96, round(target_chars * 1.25) + 64)


def constrain_teaching_response(
    text: str,
    user_text: str,
    response_policy: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Enforce the single closing exploration question used by detailed teaching replies."""

    original = str(text or "").strip()
    policy = dict(response_policy or {})
    scene = str(policy.get("scene") or "").strip()
    response_depth = str(policy.get("response_depth") or "").strip()
    if scene not in _DETAILED_TEACHING_SCENES or response_depth != "detailed":
        return original, {"applied": False, "reason": "not_detailed_teaching"}

    body, references = _split_reference_section(original)
    suppress_question = bool(_NO_FOLLOWUP_QUESTION_RE.search(str(user_text or "")))
    if suppress_question:
        constrained_body = _replace_question_marks(body)
        appended = False
        preserved = False
    else:
        teaching_body, exploration = _split_body_and_exploration(body)
        if _valid_exploration_question(exploration):
            constrained_body = _join_body_and_exploration(
                _replace_question_marks(teaching_body),
                _keep_only_final_question_mark(exploration),
            )
            appended = False
            preserved = True
        else:
            constrained_body = _join_body_and_exploration(
                _replace_question_marks(body),
                _DEFAULT_EXPLORATION_QUESTION,
            )
            appended = True
            preserved = False

    constrained = _join_body_and_references(constrained_body, references)
    return constrained or original, {
        "applied": True,
        "changed": constrained != original,
        "scene": scene,
        "response_depth": response_depth,
        "question_suppressed": suppress_question,
        "exploration_appended": appended,
        "exploration_preserved": preserved,
        "final_body_question_count": sum(constrained_body.count(mark) for mark in _QUESTION_MARKS),
    }


def constrain_age_response(
    text: str,
    response_policy: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Return a plain-text response that satisfies deterministic age limits."""

    original = str(text or "").strip()
    spec = age_constraint_spec(response_policy)
    if spec is None:
        return original, {"applied": False, "reason": "no_supported_age_policy"}

    answer, references = _split_reference_section(original)
    body, exploration = _split_body_and_exploration(answer)
    before = _measure(body, answer)
    if spec.safety_exempt:
        return original, {
            "applied": False,
            "reason": "safety_exempt",
            "stage": spec.stage,
            **{f"original_{key}": value for key, value in before.items()},
        }

    constrained_body = _constrain_body(body, spec)
    constrained_body = _limit_question_marks(constrained_body, spec.maximum_questions)
    body_questions = sum(constrained_body.count(mark) for mark in _QUESTION_MARKS)
    remaining_questions = max(spec.maximum_questions - body_questions, 0)
    constrained_exploration = (
        _limit_question_marks(exploration, remaining_questions)
        if remaining_questions > 0
        else ""
    )
    constrained_answer = _join_body_and_exploration(constrained_body, constrained_exploration)
    combined = _join_body_and_references(constrained_answer, references)
    final_body, _ = _split_body_and_exploration(constrained_answer)
    after = _measure(final_body, constrained_answer)
    metadata = {
        "applied": True,
        "stage": spec.stage,
        "changed": combined != original,
        "maximum_body_chars": spec.maximum_body_chars,
        "maximum_questions": spec.maximum_questions,
        **{f"original_{key}": value for key, value in before.items()},
        **{f"final_{key}": value for key, value in after.items()},
    }
    return combined or original, metadata


class AgeResponseStreamLimiter:
    """Apply age limits to a live stream while emitting only complete sentences."""

    def __init__(self, response_policy: Mapping[str, Any] | None) -> None:
        self.spec = age_constraint_spec(response_policy)
        self.enabled = self.spec is not None and not self.spec.safety_exempt
        self.pending = ""
        self.body_chars = 0
        self.body_sentences = 0
        self.question_count = 0
        self.body_paragraphs = 0
        self.changed = False
        self.dropped_body = False

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        if not self.enabled:
            return [text]
        self.pending += text
        output: list[str] = []
        while True:
            match = _TERMINAL_RE.search(self.pending)
            if match is None:
                break
            end = match.end()
            sentence, self.pending = self.pending[:end], self.pending[end:]
            output.extend(self._accept(sentence))
        return output

    def finish(self) -> list[str]:
        if not self.enabled or not self.pending:
            return []
        pending, self.pending = self.pending, ""
        return self._accept(pending)

    def metadata(self) -> dict[str, Any]:
        if self.spec is None:
            return {"applied": False, "reason": "no_supported_age_policy"}
        if self.spec.safety_exempt:
            return {"applied": False, "reason": "safety_exempt", "stage": self.spec.stage}
        return {
            "applied": True,
            "stage": self.spec.stage,
            "changed": self.changed,
            "final_body_chinese_chars": self.body_chars,
            "final_question_count": self.question_count,
            "maximum_body_chars": self.spec.maximum_body_chars,
            "maximum_questions": self.spec.maximum_questions,
            "stream_sentence_buffered": True,
        }

    def _accept(self, sentence: str) -> list[str]:
        assert self.spec is not None
        cleaned = _clean_low_age_markdown(sentence) if self.spec.stage in {"S0", "S1"} else sentence
        is_question = cleaned.rstrip().endswith(("？", "?"))
        if is_question:
            if self.question_count >= self.spec.maximum_questions:
                self.changed = True
                return []
            self.question_count += 1
            separator = "\n\n" if self.body_sentences > 0 and self.question_count == 1 else ""
            return [f"{separator}{cleaned.lstrip()}"]

        if self.dropped_body:
            self.changed = True
            return []

        sentences = _split_long_sentence(cleaned, self.spec.maximum_sentence_chars)
        emitted: list[str] = []
        for item in sentences:
            if self.spec.maximum_body_sentences is not None and self.body_sentences >= self.spec.maximum_body_sentences:
                self.changed = True
                self.dropped_body = True
                break
            item = self._limit_paragraphs(item)
            item_chars = _chinese_count(item)
            remaining = self.spec.maximum_body_chars - self.body_chars
            if item_chars > remaining:
                limited = _truncate_chinese(item, remaining)
                if limited:
                    emitted.append(limited)
                    self.body_chars += _chinese_count(limited)
                    self.body_sentences += 1
                self.changed = True
                self.dropped_body = True
                break
            if item:
                emitted.append(item)
                self.body_chars += item_chars
                self.body_sentences += 1
        if "".join(emitted) != sentence:
            self.changed = True
        return emitted

    def _limit_paragraphs(self, text: str) -> str:
        assert self.spec is not None
        maximum = self.spec.maximum_body_paragraphs
        if maximum is None:
            return text
        pieces = re.split(r"(\n\s*\n)", text)
        output: list[str] = []
        for piece in pieces:
            if re.fullmatch(r"\n\s*\n", piece or ""):
                if self.body_paragraphs == 0:
                    self.body_paragraphs = 1
                if self.body_paragraphs >= maximum:
                    output.append("\n")
                    self.changed = True
                else:
                    output.append(piece)
                    self.body_paragraphs += 1
            else:
                output.append(piece)
        return "".join(output)


def _constrain_body(body: str, spec: AgeConstraintSpec) -> str:
    value = body.strip()
    if spec.stage in {"S0", "S1"}:
        value = _clean_low_age_markdown(value)
        sentences: list[str] = []
        for sentence in _sentences(value):
            sentences.extend(_split_long_sentence(sentence, spec.maximum_sentence_chars))
        if spec.maximum_body_sentences is not None:
            sentences = sentences[: spec.maximum_body_sentences]
        if spec.stage == "S0":
            value = "".join(sentence.strip() for sentence in sentences if sentence.strip())
        else:
            midpoint = min(2, len(sentences))
            first = "".join(sentences[:midpoint]).strip()
            second = "".join(sentences[midpoint:]).strip()
            value = "\n\n".join(part for part in (first, second) if part)
    elif spec.maximum_body_paragraphs is not None:
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", value) if item.strip()]
        if len(paragraphs) > spec.maximum_body_paragraphs:
            head = paragraphs[: spec.maximum_body_paragraphs - 1]
            tail = " ".join(paragraphs[spec.maximum_body_paragraphs - 1 :])
            paragraphs = [*head, tail]
        value = "\n\n".join(paragraphs)
    return _truncate_chinese(value, spec.maximum_body_chars)


def _split_body_and_exploration(text: str) -> tuple[str, str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text.strip()) if item.strip()]
    if len(paragraphs) >= 2 and paragraphs[-1].rstrip().endswith(("？", "?")):
        return "\n\n".join(paragraphs[:-1]), paragraphs[-1]
    return text.strip(), ""


def _join_body_and_exploration(body: str, exploration: str) -> str:
    return "\n\n".join(part.strip() for part in (body, exploration) if part.strip())


def _split_reference_section(text: str) -> tuple[str, str]:
    match = _REFERENCE_HEADING_RE.search(text)
    if match is None:
        return text.strip(), ""
    return text[: match.start()].rstrip(), text[match.start() :]


def _join_body_and_references(body: str, references: str) -> str:
    if not references:
        return body.strip()
    if not body.strip():
        return references
    return f"{body.strip()}\n\n{references}"


def _valid_exploration_question(text: str) -> bool:
    value = text.strip()
    return (
        bool(value)
        and value.endswith(("？", "?"))
        and "和" in value
        and not any(connector in value for connector in _INVALID_EXPLORATION_CONNECTORS)
    )


def _replace_question_marks(text: str) -> str:
    return "".join("。" if char in _QUESTION_MARKS else char for char in text).strip()


def _keep_only_final_question_mark(text: str) -> str:
    value = text.strip()
    if not value:
        return value
    return f"{_replace_question_marks(value[:-1]).rstrip('。')}？"


def _clean_low_age_markdown(text: str) -> str:
    return _MARKDOWN_PREFIX_RE.sub("", text).replace("**", "").strip()


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _split_long_sentence(sentence: str, maximum_chars: int | None) -> list[str]:
    if maximum_chars is None or _chinese_count(sentence) <= maximum_chars:
        return [sentence]
    pieces: list[str] = []
    pending = sentence.strip()
    while pending:
        if _chinese_count(pending) <= maximum_chars:
            pieces.append(pending)
            break
        split_index = _index_after_chinese_chars(pending, maximum_chars)
        prefix = pending[:split_index]
        boundary = max(prefix.rfind(marker) for marker in ("，", ",", "；", ";", "：", ":"))
        if boundary >= max(1, len(prefix) // 3):
            split_index = boundary + 1
        piece = pending[:split_index].strip().rstrip("，,；;：:")
        if piece and piece[-1] not in "。！？!?":
            piece += "。"
        pieces.append(piece)
        pending = pending[split_index:].strip()
    return pieces


def _truncate_chinese(text: str, maximum_chars: int) -> str:
    if maximum_chars <= 0:
        return ""
    if _chinese_count(text) <= maximum_chars:
        return text.strip()
    split_index = _index_after_chinese_chars(text, maximum_chars)
    prefix = text[:split_index].rstrip()
    last_terminal = max(prefix.rfind(marker) for marker in "。！？!?")
    if last_terminal >= len(prefix) // 2:
        prefix = prefix[: last_terminal + 1].rstrip()
    elif prefix and prefix[-1] not in "。！？!?":
        prefix = prefix.rstrip("，,；;：:") + "。"
    return prefix


def _index_after_chinese_chars(text: str, maximum_chars: int) -> int:
    count = 0
    for index, char in enumerate(text):
        if _CHINESE_CHAR_RE.fullmatch(char):
            count += 1
            if count >= maximum_chars:
                return index + 1
    return len(text)


def _limit_question_marks(text: str, maximum_questions: int) -> str:
    count = 0
    output: list[str] = []
    for char in text:
        if char in _QUESTION_MARKS:
            count += 1
            output.append(char if count <= maximum_questions else "。")
        else:
            output.append(char)
    return "".join(output).strip()


def _measure(body: str, full_text: str) -> dict[str, int]:
    sentences = _sentences(body)
    paragraphs = [item for item in re.split(r"\n\s*\n", body) if item.strip()]
    return {
        "body_chinese_chars": _chinese_count(body),
        "question_count": sum(full_text.count(mark) for mark in _QUESTION_MARKS),
        "body_sentence_count": len(sentences),
        "maximum_sentence_chinese_chars": max((_chinese_count(item) for item in sentences), default=0),
        "body_paragraph_count": len(paragraphs),
    }


def _chinese_count(text: str) -> int:
    return len(_CHINESE_CHAR_RE.findall(text))
