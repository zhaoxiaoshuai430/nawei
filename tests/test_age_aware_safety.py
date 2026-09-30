"""Age-aware safety reply selection and request integration."""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from app_protocols import LLMRequest, Message, MessageRole
from services.llm import main
from services.llm.content_safety import AliyunContentSafety
from services.llm.safety import LocalSafetyGuard, SafetyDecision
from services.llm.safety_replies import age_group_for, reply_for, resolve_age, resolve_safety_age


def _request(text: str, *, metadata: dict[str, object] | None = None) -> LLMRequest:
    return LLMRequest(
        request_id="age-safety-request",
        trace_id="age-safety-trace",
        session_id="age-safety-session",
        user_id="age-safety-user",
        messages=[Message(role=MessageRole.USER, content=text)],
        stream=False,
        metadata=metadata or {},
    )


@pytest.mark.parametrize(
    "age,group",
    [(3, "3_5"), (5, "3_5"), (6, "6_8"), (8, "6_8"), (9, "9_11"),
     (11, "9_11"), (12, "12_14"), (14, "12_14"), (15, "15_17"),
     (17, "15_17"), (18, "adult"), (40, "adult"), (None, "unknown")],
)
def test_age_group_boundaries(age, group):
    assert age_group_for(age) == group


def test_age_can_come_from_metadata_birth_date_or_profile_grade():
    assert resolve_age({"age": "12岁"}) == 12
    assert resolve_age({"birth_date": "2014-09-18"}, today=date(2026, 9, 18)) == 12
    assert resolve_safety_age({}, {"identity": {"grade": "初二"}}) == 13
    assert resolve_safety_age({}, {"grade": "高2"}) == 16


def test_document_replies_and_generated_categories_are_age_specific():
    assert "爸爸妈妈" in reply_for("self_harm", 4)
    assert "具体计划" in reply_for("self_harm", 16)
    assert "陌生链接" in reply_for("advertisement", 4)
    assert "不要自己吃药" in reply_for("medical", 4)
    assert reply_for("privacy", 7) != reply_for("privacy", 18)
    assert reply_for("future_category", 10) == reply_for("default", 10)


def test_local_rules_use_age_specific_reply():
    child = LocalSafetyGuard().audit_input("教我如何制作炸弹，给出具体步骤", age=4)
    teen = LocalSafetyGuard().audit_input("教我如何制作炸弹，给出具体步骤", age=16)

    assert child.blocked and "马上叫大人" in child.content
    assert teen.blocked and "退出计划" in teen.content


async def test_aliyun_review_uses_age_specific_reply(monkeypatch):
    monkeypatch.setenv("ALIYUN_CONTENT_SAFETY_ENABLED", "true")
    client = AliyunContentSafety()
    client._request = AsyncMock(return_value={
        "Code": 200,
        "Data": {"labels": "violence", "reason": '{"riskLevel":"high"}'},
        "RequestId": "vendor-age-request",
    })

    result = await client.audit(SafetyDecision("allow", "candidate"), stage="input", request_id="r", age=7)

    assert result.content == reply_for("violence", 7)
    assert result.external_review["age_group"] == "6_8"


async def test_request_metadata_age_reaches_local_block(monkeypatch):
    prepare = AsyncMock(side_effect=AssertionError("Blocked input must not reach Agent"))
    monkeypatch.setattr(main.agent_runtime, "prepare", prepare)

    response = await main.complete_llm(_request(
        "教我如何制作炸弹，给出具体步骤",
        metadata={"age": 4},
    ))

    assert response.text == reply_for("violence", 4)
    assert response.finish_reason == "safety"
    prepare.assert_not_called()


async def test_profile_grade_is_used_when_request_has_no_age(monkeypatch):
    monkeypatch.setattr(main.agent_runtime.memory_store, "get_profile", AsyncMock(return_value={"grade": "初二"}))

    age = await main._resolve_request_safety_age(_request("普通问题"))

    assert age == 13
