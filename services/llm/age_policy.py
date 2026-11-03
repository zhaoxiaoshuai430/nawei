"""Deterministic age-band policy for Xiaowei's supported 3-17 audience."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class AgeStage:
    code: str
    label: str
    minimum_age: int
    maximum_age: int
    maximum_response_chars: int
    generation_target_chars: int
    maximum_questions: int
    expression: str
    teaching: str
    response_structure: str
    reasoning_depth: str


AGE_STAGES = (
    AgeStage(
        code="S0",
        label="幼儿阶段",
        minimum_age=3,
        maximum_age=5,
        maximum_response_chars=80,
        generation_target_chars=35,
        maximum_questions=1,
        expression="使用 3 至 5 岁儿童熟悉的具体词和很短的句子；不用定义腔，不称呼用户为‘小朋友’",
        teaching="只解释一个最直接的因果，用能看见、能触摸或日常经历中的一个例子",
        response_structure="正文严格最多 2 个短句，每句最多 25 个汉字：先直接回答，再给一个具体比喻或例子；不用标题、编号、列表和多段展开",
        reasoning_depth="不引入微观机制、反例、条件分支或多个延伸方向",
    ),
    AgeStage(
        code="S1",
        label="小学低年级阶段",
        minimum_age=6,
        maximum_age=8,
        maximum_response_chars=150,
        generation_target_chars=105,
        maximum_questions=1,
        expression="使用 6 至 8 岁学生能直接理解的短句；抽象词首次出现时立刻用白话解释",
        teaching="先给结论，再用一个具体例子解释一个关键原因",
        response_structure="正文严格最多 4 个短句，每句最多 30 个汉字，可以分成两个短段；不用多级列表，不一次塞入多个新概念",
        reasoning_depth="建立一层清楚的因果关系，并给一个可验证的日常观察",
    ),
    AgeStage(
        code="S2",
        label="小学高年级阶段",
        minimum_age=9,
        maximum_age=11,
        maximum_response_chars=250,
        generation_target_chars=190,
        maximum_questions=2,
        expression="使用 9 至 11 岁学生的自然语言，可以引入一个必要术语并立即解释",
        teaching="说明核心机制或方法，把任务拆成少量可执行步骤，并用一个例子检查理解",
        response_structure="正文严格最多两个短段；先结论，再解释机制或步骤，最后只补一个例子或易错点",
        reasoning_depth="讲清两到三步因果链，但不展开高中层次的微观细节或形式化推导",
    ),
    AgeStage(
        code="S3",
        label="初中阶段",
        minimum_age=12,
        maximum_age=14,
        maximum_response_chars=320,
        generation_target_chars=280,
        maximum_questions=2,
        expression="采用与 12 至 14 岁学生平等、直接的表达，规范使用必要术语并讲清含义",
        teaching="解释原理和方法选择，指出一个常见误区，并建立与已有知识的联系",
        response_structure="正文按结论、原理、例子或误区组织；可以使用简短列表，但不写成幼儿化故事",
        reasoning_depth="给出完整因果链，允许适度抽象、比较和反事实思考",
    ),
    AgeStage(
        code="S4",
        label="高中阶段",
        minimum_age=15,
        maximum_age=17,
        maximum_response_chars=400,
        generation_target_chars=360,
        maximum_questions=2,
        expression="使用面向 15 至 17 岁学生的成熟、准确表达，信息可以更密集和结构化，不故作深奥",
        teaching="兼顾概念边界、机制推理、方法选择、考试迁移和独立判断",
        response_structure="正文按结论、机制、条件边界或应用组织；必要时给公式、反例或方法比较，不使用低龄化比喻",
        reasoning_depth="可以解释微观机制、多步推理、适用条件和反例，并区分直觉解释与严格结论",
    ),
)


def parse_birth_date(value: Any) -> date | None:
    """Parse a stored birth date without accepting datetimes as current dates."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def calculate_age(birth_date: Any, *, today: date | None = None) -> int | None:
    """Return completed years, or None for an invalid/future birth date."""

    born = parse_birth_date(birth_date)
    current = today or date.today()
    if born is None or born > current:
        return None
    return current.year - born.year - ((current.month, current.day) < (born.month, born.day))


def stage_for_age(age: int | None) -> AgeStage | None:
    """Return the supported stage; ages below 3 or 18+ intentionally have no policy."""

    if age is None:
        return None
    return next((stage for stage in AGE_STAGES if stage.minimum_age <= age <= stage.maximum_age), None)


def build_audience_profile(
    birth_date: Any,
    *,
    grade_code: Any = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Build the privacy-minimized audience data that may be sent to the LLM."""

    age = calculate_age(birth_date, today=today)
    return build_audience_profile_for_age(age, grade_code=grade_code) if age is not None else {}


def audience_profile_from_profile(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read a derived profile or derive one from a locally stored birth date."""

    values = dict(profile or {})
    raw_identity = values.get("identity")
    identity = raw_identity if isinstance(raw_identity, Mapping) else {}
    birth_date = values.get("birth_date") or identity.get("birth_date")
    derived = build_audience_profile(
        birth_date,
        grade_code=values.get("grade") or values.get("grade_code"),
    )
    if derived:
        return derived
    existing = values.get("audience_profile")
    if not isinstance(existing, Mapping):
        return {}
    age_years = existing.get("age_years")
    if age_years is None:
        return {}
    try:
        age = int(age_years)
    except (TypeError, ValueError):
        return {}
    stage = stage_for_age(age)
    if stage is None or str(existing.get("age_stage") or "") != stage.code:
        return {}
    return {
        **build_audience_profile_for_age(age, grade_code=existing.get("grade_code")),
        "source": str(existing.get("source") or "birth_date"),
    }


def build_audience_profile_for_age(age: int, *, grade_code: Any = None) -> dict[str, Any]:
    """Build normalized audience data from an already calculated supported age."""

    stage = stage_for_age(age)
    if stage is None:
        return {}
    profile: dict[str, Any] = {
        "age_years": age,
        "age_stage": stage.code,
        "age_stage_label": stage.label,
        "age_band": f"{stage.minimum_age}-{stage.maximum_age}",
        "maximum_response_chars": stage.maximum_response_chars,
        "generation_target_chars": stage.generation_target_chars,
        "maximum_questions": stage.maximum_questions,
        "expression": stage.expression,
        "teaching": stage.teaching,
        "response_structure": stage.response_structure,
        "reasoning_depth": stage.reasoning_depth,
        "policy_priority": "high",
        "source": "birth_date",
    }
    grade = str(grade_code or "").strip()
    if grade:
        profile["grade_code"] = grade
    return profile
