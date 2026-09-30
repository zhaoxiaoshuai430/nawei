"""Deterministic safety checks for LLM input, tool data, and output."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from services.llm.safety_replies import reply_for


LOGGER = logging.getLogger("nawei.llm.safety")

ACTION_ALLOW = "allow"
ACTION_MASK = "mask"
ACTION_BLOCK = "block"

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_PRIVACY_RULES: tuple[tuple[str, re.Pattern[str], Any], ...] = (
    (
        "privacy.api_key",
        re.compile(
            r"(?i)(\b(?:api[_ -]?key|access[_ -]?token|secret[_ -]?key|authorization)\b\s*[:=]?\s*)"
            r"([A-Za-z0-9_./+\-=]{8,})"
        ),
        lambda match: f"{match.group(1)}[已脱敏]",
    ),
    (
        "privacy.url_secret",
        re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret)=)([^&#\s]+)"),
        lambda match: f"{match.group(1)}[已脱敏]",
    ),
    (
        "privacy.password",
        re.compile(
            r"(?i)(((?:密码|口令)\s*[:：=]\s*|"
            r"(?:密码|口令)\s*(?:是|为)\s*(?=[^\s,，;；]{0,30}[A-Za-z0-9])|"
            r"(?:password|passwd|pwd)\s*[:=]\s*))"
            r"([^\s,，;；]{4,})"
        ),
        lambda match: f"{match.group(1)}[已脱敏]",
    ),
    (
        "privacy.id_card",
        re.compile(r"((?:身份证|身份证号|证件号)\s*(?:是|为)?\s*[:：]?\s*)(\d{6})\d{8}(\d{3}[0-9Xx])"),
        lambda match: f"{match.group(1)}{match.group(2)}********{match.group(3)}",
    ),
    (
        "privacy.phone",
        re.compile(r"((?:手机号|手机|电话|联系方式|联系)\s*(?:是|为)?\s*[:：]?\s*)(1[3-9]\d)(\d{4})(\d{4})"),
        lambda match: f"{match.group(1)}{match.group(2)}****{match.group(4)}",
    ),
    (
        "privacy.bank_card",
        re.compile(r"(?i)((?:银行卡|卡号|bank\s*card)\s*[:：]?\s*)(\d{4})[ -]?\d{8,11}[ -]?(\d{4})"),
        lambda match: f"{match.group(1)}{match.group(2)}********{match.group(3)}",
    ),
)

_PROMPT_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "prompt_injection.ignore",
        re.compile(
            r"(?i)(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|system)\s+"
            r"(?:instructions?|prompts?|messages?)"
        ),
    ),
    (
        "prompt_injection.override",
        re.compile(
            r"(?i)(?:system\s*prompt|developer\s*message)\s*[:：]|"
            r"(?:reveal|print|show)\s+(?:the\s+)?system\s+prompt"
        ),
    ),
    (
        "prompt_injection.zh",
        re.compile(r"(?:忽略|无视|忘记|覆盖).{0,8}(?:之前|以上|系统|开发者).{0,8}(?:指令|提示词|消息)"),
    ),
    (
        "prompt_injection.role",
        re.compile(r"(?:你现在是|从现在开始扮演|切换为).{0,16}(?:系统|管理员|开发者|无限制模式)"),
    ),
)

_HIGH_RISK_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "danger.explosive_instructions",
        re.compile(r"(?:如何|怎么|怎样|教我|教程|步骤|配方|方法).{0,18}(?:制作|制造|组装|合成).{0,12}(?:炸弹|爆炸物|爆炸装置|燃烧弹)"),
    ),
    (
        "danger.weapon_instructions",
        re.compile(r"(?:如何|怎么|怎样|教我|教程|步骤|图纸).{0,18}(?:制作|改装|组装).{0,12}(?:枪支|枪械|武器|消音器)"),
    ),
    (
        "cyber.intrusion_instructions",
        re.compile(r"(?:如何|怎么|教我|教程|步骤|脚本|代码).{0,18}(?:入侵|攻击|破解|绕过|盗取).{0,18}(?:网站|服务器|系统|账号|密码|验证码|安全验证)"),
    ),
    (
        "illegal.drug_instructions",
        re.compile(r"(?:如何|怎么|教我|教程|步骤|配方).{0,18}(?:制作|提炼|合成|种植).{0,12}(?:冰毒|海洛因|可卡因|毒品)"),
    ),
    (
        "self_harm.instructions",
        re.compile(r"(?:(?:如何|怎么)(?:自杀|结束生命|伤害自己)|(?:如何|怎么|哪种|什么方法).{0,18}(?:自杀|结束生命|伤害自己).{0,12}(?:成功|有效|方法|步骤))"),
    ),
    (
        "sexual.minor",
        re.compile(r"(?:索要|生成|提供|寻找|制作|描写).{0,12}(?:未成年人|儿童|小学生).{0,12}(?:色情|裸照|性行为|性内容)"),
    ),
)

_CRISIS_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("self_harm.crisis_signal", re.compile(r"(?:我|自己).{0,8}(?:不想活|想死|自杀|自残|结束生命|伤害自己)")),
    ("self_harm.immediate_action", re.compile(r"(?:我|自己).{0,8}(?:已经|正在|准备|打算).{0,8}(?:割腕|跳楼|服药|自杀|自残)")),
    ("abuse.immediate_danger", re.compile(r"(?:(?:有人|家里人|爸爸|妈妈|老师).{0,8}(?:正在|要|威胁).{0,6}(?:打我|伤害我|杀我)|我现在.{0,6}(?:不安全|被打|被威胁))")),
)

_BLOCKED_TOOL_VALUE = "[该字段已被本地安全规则移除]"
_INJECTION_REPLACEMENT = "[外部指令已移除]"


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    content: str
    categories: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    content_hash: str = ""
    external_review: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return self.action == ACTION_BLOCK

    @property
    def changed(self) -> bool:
        return self.action != ACTION_ALLOW

    def metadata(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "categories": list(self.categories),
            "rule_ids": list(self.rule_ids),
            "content_hash": self.content_hash,
            **({"external_review": self.external_review} if self.external_review is not None else {}),
        }


@dataclass(frozen=True)
class ToolSafetyDecision:
    action: str
    prompt: str | None
    result: dict[str, Any] | None
    error: str | None
    categories: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.action == ACTION_BLOCK

    def metadata(self) -> dict[str, Any]:
        return {"action": self.action, "categories": list(self.categories), "rule_ids": list(self.rule_ids)}


class LocalSafetyGuard:
    """Fast deterministic safety guard with no network or model calls."""

    def enabled(self, stage: str | None = None) -> bool:
        if not _read_bool_env("SAFETY_ENABLED", True):
            return False
        return not stage or _read_bool_env(f"SAFETY_{stage.upper()}_ENABLED", True)

    def audit_input(self, text: str, *, age: int | None = None) -> SafetyDecision:
        return self._audit_text(text, stage="input", sanitize_injection=True, age=age)

    def audit_output(self, text: str, *, age: int | None = None) -> SafetyDecision:
        return self._audit_text(text, stage="output", age=age)

    def audit_tool_result(
        self,
        tool_name: str,
        *,
        prompt: str | None,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> ToolSafetyDecision:
        if not self.enabled("tool_result"):
            return ToolSafetyDecision(ACTION_ALLOW, prompt, result, error)
        safe_result, blocked, categories, rule_ids = self._sanitize_value(result)
        safe_prompt = prompt
        if prompt:
            prompt_decision = self._audit_text(prompt, stage="tool_result", sanitize_injection=True)
            categories.update(prompt_decision.categories)
            rule_ids.update(prompt_decision.rule_ids)
            blocked = blocked or prompt_decision.blocked
            safe_prompt = (
                f"工具 {tool_name} 的部分结果未通过本地安全审核，不要推断或补全被移除的内容。"
                if prompt_decision.blocked
                else prompt_decision.content
            )
        safe_error = error
        if error:
            safe_error, privacy_rules = self._mask_privacy(self._clean_text(error)[:300])
            rule_ids.update(privacy_rules)
            if privacy_rules:
                categories.add("privacy")
        action = ACTION_BLOCK if blocked else ACTION_MASK if categories or rule_ids else ACTION_ALLOW
        return ToolSafetyDecision(
            action=action,
            prompt=safe_prompt,
            result=safe_result if isinstance(safe_result, dict) else None,
            error=safe_error,
            categories=tuple(sorted(categories)),
            rule_ids=tuple(sorted(rule_ids)),
        )

    def refusal_for(self, categories: tuple[str, ...], *, age: int | None = None) -> str:
        prefix = (categories[0] if categories else "default").split(".", 1)[0]
        return reply_for(prefix, age)

    def log(
        self,
        *,
        stage: str,
        decision: SafetyDecision | ToolSafetyDecision,
        request_id: str,
        tool_name: str | None = None,
    ) -> None:
        allowed = not decision.changed if isinstance(decision, SafetyDecision) else decision.action == ACTION_ALLOW
        if allowed or not _read_bool_env("SAFETY_AUDIT_LOG_ENABLED", True):
            return
        LOGGER.warning(
            "safety_event stage=%s request_id=%s tool=%s action=%s categories=%s rules=%s",
            stage,
            request_id,
            tool_name or "-",
            decision.action,
            ",".join(decision.categories),
            ",".join(decision.rule_ids),
        )

    def _audit_text(
        self,
        text: str,
        *,
        stage: str,
        sanitize_injection: bool = False,
        age: int | None = None,
    ) -> SafetyDecision:
        original = str(text or "")
        if not self.enabled(stage):
            return SafetyDecision(ACTION_ALLOW, original, content_hash=_content_hash(original))

        cleaned = self._clean_text(original)
        categories: set[str] = set()
        rule_ids: set[str] = set()
        if sanitize_injection:
            cleaned, injection_rules = self._remove_prompt_injection(cleaned)
            rule_ids.update(injection_rules)
            if injection_rules:
                categories.add("prompt_injection")

        masked, privacy_rules = self._mask_privacy(cleaned)
        rule_ids.update(privacy_rules)
        if privacy_rules:
            categories.add("privacy")

        risk_rules = self._match_high_risk(cleaned, stage=stage)
        if risk_rules:
            risk_categories = {rule_id.rsplit(".", 1)[0] for rule_id in risk_rules}
            return SafetyDecision(
                ACTION_BLOCK,
                self.refusal_for(tuple(sorted(risk_categories)), age=age),
                categories=tuple(sorted(risk_categories)),
                rule_ids=tuple(sorted(rule_ids | risk_rules)),
                content_hash=_content_hash(original),
            )

        has_hidden_controls = bool(_ZERO_WIDTH_RE.search(original) or _CONTROL_RE.search(original))
        if privacy_rules or rule_ids or has_hidden_controls:
            if has_hidden_controls:
                categories.add("text_normalization")
                rule_ids.add("text.hidden_controls")
            action = ACTION_MASK
            content = masked
        else:
            action = ACTION_ALLOW
            content = original
        return SafetyDecision(
            action,
            content,
            categories=tuple(sorted(categories)),
            rule_ids=tuple(sorted(rule_ids)),
            content_hash=_content_hash(original),
        )

    def _sanitize_value(self, value: Any, *, depth: int = 0) -> tuple[Any, bool, set[str], set[str]]:
        if depth > 8:
            return "[工具结果层级过深，已截断]", False, {"tool_limits"}, {"tool.max_depth"}
        if isinstance(value, str):
            decision = self._audit_text(value[:12000], stage="tool_result", sanitize_injection=True)
            if decision.blocked:
                return _BLOCKED_TOOL_VALUE, True, set(decision.categories), set(decision.rule_ids)
            return decision.content, False, set(decision.categories), set(decision.rule_ids)
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            blocked = False
            categories: set[str] = set()
            rule_ids: set[str] = set()
            for index, (key, item) in enumerate(value.items()):
                if index >= 100:
                    safe["_safety_truncated"] = True
                    categories.add("tool_limits")
                    rule_ids.add("tool.max_fields")
                    break
                safe_item, item_blocked, item_categories, item_rules = self._sanitize_value(item, depth=depth + 1)
                safe[str(key)[:200]] = safe_item
                blocked = blocked or item_blocked
                categories.update(item_categories)
                rule_ids.update(item_rules)
            return safe, blocked, categories, rule_ids
        if isinstance(value, (list, tuple)):
            safe_items: list[Any] = []
            list_blocked = False
            list_categories: set[str] = set()
            list_rule_ids: set[str] = set()
            for item in list(value)[:100]:
                safe_item, item_blocked, item_categories, item_rules = self._sanitize_value(item, depth=depth + 1)
                safe_items.append(safe_item)
                list_blocked = list_blocked or item_blocked
                list_categories.update(item_categories)
                list_rule_ids.update(item_rules)
            if len(value) > 100:
                list_categories.add("tool_limits")
                list_rule_ids.add("tool.max_items")
            return safe_items, list_blocked, list_categories, list_rule_ids
        if value is None or isinstance(value, (bool, int, float)):
            return value, False, set(), set()
        return str(value)[:1000], False, {"tool_limits"}, {"tool.unsupported_type"}

    @staticmethod
    def _clean_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = _ZERO_WIDTH_RE.sub("", normalized)
        return _CONTROL_RE.sub("", normalized)

    @staticmethod
    def _mask_privacy(text: str) -> tuple[str, set[str]]:
        if not _read_bool_env("SAFETY_PRIVACY_MASK_ENABLED", True):
            return text, set()
        masked = text
        matched: set[str] = set()
        for rule_id, pattern, replacement in _PRIVACY_RULES:
            masked, count = pattern.subn(replacement, masked)
            if count:
                matched.add(rule_id)
        return masked, matched

    @staticmethod
    def _remove_prompt_injection(text: str) -> tuple[str, set[str]]:
        cleaned = text
        matched: set[str] = set()
        for rule_id, pattern in _PROMPT_INJECTION_RULES:
            cleaned, count = pattern.subn(_INJECTION_REPLACEMENT, cleaned)
            if count:
                matched.add(rule_id)
        return cleaned, matched

    @staticmethod
    def _match_high_risk(text: str, *, stage: str) -> set[str]:
        matched = {rule_id for rule_id, pattern in _HIGH_RISK_RULES if pattern.search(text)}
        if stage == "input":
            matched.update(rule_id for rule_id, pattern in _CRISIS_RULES if pattern.search(text))
        return matched


class SafetyStreamBuffer:
    """Hold streamed text until enough context exists for deterministic auditing."""

    def __init__(
        self,
        guard: LocalSafetyGuard,
        max_chars: int | None = None,
        overlap_chars: int | None = None,
        age: int | None = None,
    ) -> None:
        self.guard = guard
        self.age = age
        self.max_chars = max(16, max_chars or _read_int_env("SAFETY_STREAM_BUFFER_CHARS", 24))
        requested_overlap = self.max_chars - 1 if overlap_chars is None else overlap_chars
        self.overlap_chars = min(max(8, requested_overlap), self.max_chars - 1)
        self.pending = ""
        self.blocked = False
        self.decision: SafetyDecision | None = None

    def feed(self, text: str) -> list[str]:
        if self.blocked or not text:
            return []
        self.pending += text
        emitted: list[str] = []
        while self.pending:
            if len(self.pending) >= self.max_chars:
                decision = self.guard.audit_output(self.pending, age=self.age)
                self._record(decision)
                if decision.blocked or decision.changed:
                    self.blocked = decision.blocked
                    self.pending = ""
                    return [decision.content]
                split_at = len(self.pending) - self.overlap_chars
                segment = self.pending[:split_at]
                self.pending = self.pending[split_at:]
            else:
                boundary = _sentence_boundary(self.pending)
                if boundary is None:
                    break
                segment = self.pending[:boundary]
                self.pending = self.pending[boundary:]
            decision = self.guard.audit_output(segment, age=self.age)
            self._record(decision)
            if decision.blocked:
                self.blocked = True
                self.pending = ""
                return [decision.content]
            emitted.append(decision.content)
            if self.pending and len(self.pending) < self.max_chars:
                break
        return emitted

    def finish(self) -> list[str]:
        if self.blocked or not self.pending:
            return []
        decision = self.guard.audit_output(self.pending, age=self.age)
        self.pending = ""
        self._record(decision)
        if decision.blocked:
            self.blocked = True
        return [decision.content]

    def _record(self, decision: SafetyDecision) -> None:
        if not decision.changed:
            return
        if self.decision is None:
            self.decision = decision
            return
        action = ACTION_BLOCK if self.decision.blocked or decision.blocked else ACTION_MASK
        self.decision = SafetyDecision(
            action=action,
            content=decision.content,
            categories=tuple(sorted(set(self.decision.categories) | set(decision.categories))),
            rule_ids=tuple(sorted(set(self.decision.rule_ids) | set(decision.rule_ids))),
            content_hash=decision.content_hash,
        )


def _sentence_boundary(text: str) -> int | None:
    for index, character in enumerate(text):
        if character in "。！？!?\n":
            return index + 1
    return None


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
