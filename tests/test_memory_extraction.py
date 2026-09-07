"""Regression tests for deterministic LLM memory extraction."""

import pytest

from services.llm.memory import MemoryStore, _extract_memory_records, _extract_profile_changes


def test_explicit_learning_preference_is_extracted() -> None:
    changes = _extract_profile_changes("请记住：我喜欢用类比和分步骤的方式学习。")

    assert changes["preferences"]["topics_enjoy"] == ["用类比和分步骤的方式学习"]


def test_possessive_preference_reference_is_not_extracted() -> None:
    text = "请按照我喜欢的学习方式解释。"

    assert "preferences" not in _extract_profile_changes(text)
    assert not any(record["memory_type"] == "preference" for record in _extract_memory_records(text))


def test_preference_recall_question_is_not_extracted() -> None:
    text = "我刚才说我喜欢哪种学习方式？"

    assert "preferences" not in _extract_profile_changes(text)
    assert not any(record["memory_type"] == "preference" for record in _extract_memory_records(text))


@pytest.mark.asyncio
async def test_empty_rule_extraction_does_not_write_profile() -> None:
    store = MemoryStore(enabled=False)
    writes = 0

    async def record_update_profile(**kwargs: object) -> dict[str, object]:
        nonlocal writes
        writes += 1
        return {}

    store.update_profile = record_update_profile  # type: ignore[method-assign]

    await store.extract_and_store(
        user_id="casual-user",
        session_id="casual-session",
        user_text="hello",
        use_llm=False,
    )

    assert writes == 0
