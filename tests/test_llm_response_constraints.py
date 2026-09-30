"""Tests for deterministic age-stage output constraints."""

from __future__ import annotations

import re

from services.llm.response_constraints import (
    AgeResponseStreamLimiter,
    constrain_age_response,
    constrain_teaching_response,
    recommended_age_max_tokens,
)
from services.llm import main as llm_main
from app_protocols import LLMRequest, Message, MessageRole


def policy(
    stage: str,
    *,
    maximum_chars: int,
    target_chars: int,
    maximum_questions: int,
    scene: str = "explain",
    care_focus: str = "learning_accuracy",
) -> dict[str, object]:
    return {
        "scene": scene,
        "care_focus": care_focus,
        "audience_profile": {
            "age_stage": stage,
            "maximum_response_chars": maximum_chars,
            "generation_target_chars": target_chars,
            "maximum_questions": maximum_questions,
        },
    }


def chinese_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", text))


def test_s0_removes_markdown_and_enforces_sentence_shape() -> None:
    response_policy = policy("S0", maximum_chars=80, target_chars=35, maximum_questions=1)
    text = (
        "1. 这是一个特别特别长的第一句话，因为它塞进去了很多低龄孩子不需要同时理解的内容。\n\n"
        "- 第二句话继续展开另一个原因。第三句话又补充了一个无关方向。\n\n"
        "你想继续看看它在生活里怎么用，和它还有什么变化吗？"
    )

    constrained, metadata = constrain_age_response(text, response_policy)
    body = constrained.rsplit("\n\n", 1)[0]
    sentences = [item for item in re.split(r"[。！？!?]+", body) if item.strip()]

    assert metadata["changed"]
    assert chinese_count(body) <= 80
    assert len(sentences) <= 2
    assert max(chinese_count(item) for item in sentences) <= 25
    assert len(re.split(r"\n\s*\n", body)) == 1
    assert not re.search(r"(?m)^\s*(?:[-*+]\s|\d+[.、])", body)
    assert constrained.count("？") + constrained.count("?") <= 1


def test_s1_limits_sentences_and_paragraphs() -> None:
    response_policy = policy("S1", maximum_chars=150, target_chars=105, maximum_questions=1)
    text = "。".join(["这一句包含一个清楚的原因和一个容易理解的例子" for _ in range(7)]) + "。"

    constrained, _ = constrain_age_response(text, response_policy)
    sentences = [item for item in re.split(r"[。！？!?]+", constrained) if item.strip()]

    assert len(sentences) <= 4
    assert max(chinese_count(item) for item in sentences) <= 30
    assert len(re.split(r"\n\s*\n", constrained)) <= 2


def test_s2_limits_body_to_two_paragraphs() -> None:
    response_policy = policy("S2", maximum_chars=250, target_chars=190, maximum_questions=2)
    text = "第一段。\n\n第二段。\n\n第三段。\n\n你想继续了解原因吗？"

    constrained, metadata = constrain_age_response(text, response_policy)
    body = constrained.rsplit("\n\n", 1)[0]

    assert metadata["changed"]
    assert len(re.split(r"\n\s*\n", body)) <= 2


def test_s4_truncates_at_complete_boundary() -> None:
    response_policy = policy("S4", maximum_chars=20, target_chars=12, maximum_questions=2)
    text = "这是第一句完整的话。这里是第二句而且明显会让正文超过规定的汉字上限。"

    constrained, metadata = constrain_age_response(text, response_policy)

    assert chinese_count(constrained) <= 20
    assert constrained.endswith("。")
    assert metadata["changed"]


def test_safety_response_is_exempt() -> None:
    response_policy = policy(
        "S0",
        maximum_chars=20,
        target_chars=10,
        maximum_questions=1,
        scene="crisis_support",
        care_focus="immediate_safety",
    )
    text = "这是一段必须保留完整求助步骤的安全回答，即使超过普通年龄字数也不能截断。"

    constrained, metadata = constrain_age_response(text, response_policy)

    assert constrained == text
    assert metadata["reason"] == "safety_exempt"
    assert recommended_age_max_tokens(response_policy) is None


def test_generation_budget_uses_age_target() -> None:
    response_policy = policy("S4", maximum_chars=500, target_chars=360, maximum_questions=2)

    assert recommended_age_max_tokens(response_policy) == 514


def test_generation_budget_caps_existing_request_limit() -> None:
    request = LLMRequest(
        request_id="age-budget",
        trace_id="age-budget",
        session_id="session",
        user_id="student",
        messages=[Message(role=MessageRole.USER, content="解释光合作用")],
        generation_config={"max_tokens": 900},
    )
    response_policy = policy("S2", maximum_chars=250, target_chars=190, maximum_questions=2)

    llm_main._apply_age_generation_budget(request, response_policy)

    assert request.generation_config["max_tokens"] == 302
    assert request.generation_config["age_generation_max_tokens"] == 302


def test_stream_limiter_emits_complete_sentences_within_budget() -> None:
    response_policy = policy("S0", maximum_chars=30, target_chars=20, maximum_questions=1)
    limiter = AgeResponseStreamLimiter(response_policy)
    output = limiter.feed("这是第一句，内容很清楚。第二句也能完整说明。")
    output.extend(limiter.feed("第三句应当被删除。你想继续看看吗？"))
    output.extend(limiter.finish())
    text = "".join(output)

    assert chinese_count(text.rsplit("？", 1)[0]) <= 30
    assert "第三句" not in text
    assert text.count("？") == 1
    assert limiter.metadata()["changed"]


def test_body_question_consumes_question_budget() -> None:
    response_policy = policy("S1", maximum_chars=150, target_chars=105, maximum_questions=1)
    text = (
        "同学不和你玩，心里会难受。可以先问：‘我能一起玩吗？’"
        "有时他们只是没注意到你。你也可以先找别人玩。\n\n"
        "你想继续看看怎样邀请同学吗？"
    )

    constrained, metadata = constrain_age_response(text, response_policy)

    assert constrained.count("？") == 1
    assert "继续看看" not in constrained
    assert metadata["changed"]


def detailed_teaching_policy() -> dict[str, object]:
    return {
        "scene": "explain",
        "response_depth": "detailed",
    }


def test_teaching_constraint_keeps_only_final_exploration_question() -> None:
    text = (
        "三角函数描述角与边长比例的关系。\n\n"
        "为什么说它是函数呢？因为相同角度对应固定的边长比。\n\n"
        "你想继续看看单位圆怎样表示任意角，和三角函数怎样描述周期变化吗？"
    )

    constrained, metadata = constrain_teaching_response(text, "详细介绍三角函数。", detailed_teaching_policy())

    assert constrained.count("？") + constrained.count("?") == 1
    assert "为什么说它是函数呢。" in constrained
    assert constrained.endswith("周期变化吗？")
    assert metadata["exploration_preserved"]


def test_teaching_constraint_appends_missing_exploration_question() -> None:
    text = "三角函数描述角与边长比例的关系。\n\n它还能描述周期变化。"

    constrained, metadata = constrain_teaching_response(text, "详细介绍三角函数。", detailed_teaching_policy())

    assert constrained.endswith("你想继续看看这个问题背后的原理，和它在实际场景中怎样应用吗？")
    assert constrained.count("？") + constrained.count("?") == 1
    assert metadata["exploration_appended"]


def test_teaching_constraint_respects_explicit_no_followup_request() -> None:
    text = "为什么说它是函数呢？\n\n你想继续看看单位圆，和周期变化吗？"

    constrained, metadata = constrain_teaching_response(
        text,
        "请详细解释，但不要反问也不要追问。",
        detailed_teaching_policy(),
    )

    assert "？" not in constrained and "?" not in constrained
    assert not constrained.endswith("应用吗？")
    assert metadata["question_suppressed"]


def test_teaching_constraint_preserves_reference_section() -> None:
    references = (
        "参考来源\n"
        "- 资料1：示例机构《示例资料》，路径：第一章，链接：https://example.com/doc?id=1"
    )
    text = f"三角函数描述角与边长比例的关系。\n\n{references}"

    constrained, _ = constrain_teaching_response(text, "详细介绍三角函数。", detailed_teaching_policy())
    age_policy = policy("S2", maximum_chars=30, target_chars=20, maximum_questions=1)
    constrained, _ = constrain_age_response(constrained, age_policy)

    assert constrained.partition("参考来源")[2] == text.partition("参考来源")[2]
    assert constrained.endswith(references)
