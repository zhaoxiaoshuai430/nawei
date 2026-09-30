"""Tests for deterministic 3-17 age adaptation in the standalone LLM service."""

from __future__ import annotations

import re
from datetime import date

from app_protocols.persona import format_response_policy
from services.llm.age_policy import (
    AGE_STAGES,
    audience_profile_from_profile,
    build_audience_profile,
    build_audience_profile_for_age,
    calculate_age,
    stage_for_age,
)
from services.llm.agents.tutor_graph import build_response_policy
from services.llm.memory import _normalize_education_profile


def test_completed_age_changes_on_birthday() -> None:
    birth_date = date(2015, 9, 10)

    assert calculate_age(birth_date, today=date(2026, 9, 9)) == 10
    assert calculate_age(birth_date, today=date(2026, 9, 10)) == 11


def test_s0_s4_boundaries_and_lengths() -> None:
    expected = {
        3: ("S0", 80, 1),
        5: ("S0", 80, 1),
        6: ("S1", 150, 1),
        8: ("S1", 150, 1),
        9: ("S2", 250, 2),
        11: ("S2", 250, 2),
        12: ("S3", 320, 2),
        14: ("S3", 320, 2),
        15: ("S4", 400, 2),
        17: ("S4", 400, 2),
    }

    assert len(AGE_STAGES) == 5
    assert stage_for_age(2) is None
    assert stage_for_age(18) is None
    for age, values in expected.items():
        stage = stage_for_age(age)
        assert stage is not None
        assert (stage.code, stage.maximum_response_chars, stage.maximum_questions) == values
        assert stage.generation_target_chars < stage.maximum_response_chars


def test_audience_profile_is_derived_without_exposing_birth_date() -> None:
    profile = build_audience_profile("2016-05-20", today=date(2026, 9, 9), grade_code="primary_4")

    assert profile["age_years"] == 10
    assert profile["age_stage"] == "S2"
    assert profile["grade_code"] == "primary_4"
    assert "birth_date" not in profile
    assert audience_profile_from_profile({"birth_date": "2016-05-20"})["source"] == "birth_date"


def test_memory_profile_derives_age_policy_without_persisting_it() -> None:
    source = {"birth_date": "2016-05-20", "grade": "四年级"}

    normalized = _normalize_education_profile(source)

    assert normalized["audience_profile"]["age_stage"] == "S2"
    assert "audience_profile" not in source


def test_response_policy_combines_persona_dialect_and_age_constraints() -> None:
    policy = build_response_policy(
        query="这次温柔一点，使用北京话解释光合作用",
        profile={"birth_date": "2016-05-20"},
        intent={"category": "explain"},
    )
    prompt = format_response_policy(policy)

    assert policy["persona_mode"] == "gentle"
    assert policy["dialect_mode"] == "beijing"
    assert policy["audience_profile"]["age_stage"] == "S2"
    assert "温柔小维" in prompt
    assert "北京话文字（硬约束）" in prompt
    assert "不得超过 250" in prompt
    assert "这是硬上限" in prompt


def test_all_stages_have_distinct_content_guidance() -> None:
    profiles = [build_audience_profile_for_age(age) for age in (4, 7, 10, 13, 16)]

    assert [item["age_stage"] for item in profiles] == ["S0", "S1", "S2", "S3", "S4"]
    assert len({item["response_structure"] for item in profiles}) == 5
    assert len({item["reasoning_depth"] for item in profiles}) == 5
    assert re.search(r"最多 2 个短句", profiles[0]["response_structure"])
