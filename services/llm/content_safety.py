"""Aliyun chat moderation and category-specific, server-controlled fallbacks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import replace
from typing import Any

from services.llm.safety import SafetyDecision
from services.llm.safety_replies import REPLIES as AGE_REPLIES, age_group_for, reply_for

logger = logging.getLogger(__name__)
REPLIES = AGE_REPLIES
# Exact provider labels and aliases only; unknown risk labels use the generic reply.
CATEGORY_LABELS = {
    "self_harm": {"self_harm", "suicide"},
    "medical": {"medical"},
    "violence": {"violence", "violent", "terrorism", "terrorism_explosion"},
    "illegal": {"contraband", "drug", "drugs", "gambling", "contraband_drug", "contraband_gamble"},
    "sexual": {"porn", "sexual", "sexual_content", "sexual_suggestive"},
    "sensitive": {"political", "politics", "political_content"},
    "privacy": {"privacy", "personal_info"},
    "profanity": {"profanity", "abuse"},
    "advertisement": {"ad", "advertisement", "spam"},
}
PRIORITY = tuple(CATEGORY_LABELS)
MAX_TEXT_CHARS = 20000
WINDOW_CHARS = 500
WINDOW_OVERLAP = 100


def enabled() -> bool:
    return os.getenv("ALIYUN_CONTENT_SAFETY_ENABLED", "false").strip().lower() in {"true", "1", "yes", "on"}


def fallback_for(labels: list[str], age: int | None = None) -> tuple[str, str]:
    categories = {category for category, aliases in CATEGORY_LABELS.items() if aliases.intersection(labels)}
    category = next((item for item in PRIORITY if item in categories), "default")
    return category, reply_for(category, age)


def parse_result(body: dict[str, Any]) -> dict[str, Any]:
    if str(body.get("Code")) != "200" or not isinstance(body.get("Data"), dict):
        raise ValueError("Invalid moderation response")
    data = body["Data"]
    labels = data.get("labels")
    if not isinstance(labels, str):
        raise ValueError("Missing moderation labels")
    labels = [item.strip().lower() for item in labels.split(",") if item.strip()]
    labels = [item for item in labels if item not in {"nonlabel", "normal"}]
    reason = data.get("reason") or {}
    if isinstance(reason, str):
        reason = json.loads(reason)
    if not isinstance(reason, dict):
        raise ValueError("Invalid moderation reason")
    risk = str(reason.get("riskLevel") or "").lower()
    if risk not in {"", "none", "low", "medium", "high"}:
        raise ValueError("Unknown moderation risk level")
    # Label-only responses are treated conservatively; low-risk findings are observed.
    blocked = risk in {"medium", "high"} or (bool(labels) and risk in {"", "none"})
    category, _ = fallback_for(labels)
    return {"blocked": blocked, "labels": labels, "risk_level": risk or "unspecified",
            "category": category if blocked else None, "request_id": body.get("RequestId")}


class AliyunContentSafety:
    def __init__(self) -> None:
        self._client: Any = None

    @staticmethod
    def unavailable(
        decision: SafetyDecision,
        *,
        stage: str,
        error_type: str,
        age: int | None = None,
    ) -> SafetyDecision:
        return replace(decision, action="block", content=reply_for("unavailable", age), external_review={
            "engine": "aliyun", "stage": stage, "action": "block", "failed": True,
            "category": "unavailable", "age_group": age_group_for(age), "error_type": error_type,
        })

    async def _request(self, text: str, timeout: float) -> dict[str, Any]:
        from alibabacloud_green20220302.client import Client
        from alibabacloud_green20220302.models import TextModerationRequest
        from alibabacloud_tea_openapi.models import Config
        from alibabacloud_tea_util.models import RuntimeOptions

        if self._client is None:
            key_id = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", "").strip()
            key_secret = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "").strip()
            if not key_id or not key_secret:
                raise ValueError("Missing moderation credentials")
            self._client = Client(Config(
                access_key_id=key_id, access_key_secret=key_secret,
                region_id=os.getenv("ALIYUN_GREEN_REGION_ID", "cn-shanghai"),
                endpoint=os.getenv("ALIYUN_GREEN_ENDPOINT", "green-cip.cn-shanghai.aliyuncs.com"),
                connect_timeout=int(timeout * 1000), read_timeout=int(timeout * 1000),
            ))
        request = TextModerationRequest(
            service="chat_detection", service_parameters=json.dumps({"content": text}, ensure_ascii=False),
        )
        response = await self._client.text_moderation_with_options_async(
            request, RuntimeOptions(autoretry=False, connect_timeout=int(timeout * 1000), read_timeout=int(timeout * 1000)),
        )
        if response.status_code != 200:
            raise ValueError("Moderation HTTP failure")
        return response.body.to_map()

    async def audit(
        self,
        decision: SafetyDecision,
        *,
        stage: str,
        request_id: str,
        age: int | None = None,
    ) -> SafetyDecision:
        if not enabled() or decision.blocked or not decision.content.strip():
            return decision
        try:
            timeout = float(os.getenv("ALIYUN_CONTENT_SAFETY_TIMEOUT_SECONDS", "12"))
            if not 0 < timeout <= 60 or len(decision.content) > MAX_TEXT_CHARS:
                raise ValueError("Moderation limit exceeded")
            findings = []
            # Review all windows before releasing any output; overlap preserves nearby context.
            async with asyncio.timeout(timeout):
                for start in range(0, len(decision.content), WINDOW_CHARS - WINDOW_OVERLAP):
                    findings.append(parse_result(await self._request(decision.content[start:start + WINDOW_CHARS], timeout)))
                    if start + WINDOW_CHARS >= len(decision.content):
                        break
            blocked = [item for item in findings if item["blocked"]]
            labels = sorted({label for item in blocked for label in item["labels"]})
            category, reply = fallback_for(labels, age)
            review = {"engine": "aliyun", "service": "chat_detection", "stage": stage,
                      "action": "block" if blocked else "allow", "failed": False,
                      "category": category if blocked else None, "age_group": age_group_for(age),
                      "findings": findings}
            logger.info("Content safety stage=%s request_id=%s action=%s category=%s", stage, request_id, review["action"], review["category"])
            return replace(decision, action="block" if blocked else decision.action,
                           content=reply if blocked else decision.content,
                           categories=tuple(sorted(set(decision.categories) | set(labels))), external_review=review)
        except Exception as exc:
            # SDK exceptions can contain request details; never log their text or payload.
            logger.warning("Content safety unavailable stage=%s request_id=%s error_type=%s", stage, request_id, type(exc).__name__)
            return self.unavailable(decision, stage=stage, error_type=type(exc).__name__, age=age)
