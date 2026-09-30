"""Regression tests for deterministic LLM memory extraction."""

import httpx
import pytest

from services.llm.memory import MemoryStore, _extract_memory_records, _extract_profile_changes


@pytest.mark.asyncio
async def test_llm_extractor_reuses_injected_http_client_without_owning_it() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"text": '{"profile": {}, "memories": [], "concepts": []}'})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    store = MemoryStore(enabled=False, extractor_client=client)
    store._should_run_llm_extractor = lambda _: True  # type: ignore[method-assign]

    first = await store.extract_with_llm(user_id="student", session_id="one", user_text="remember blue")
    second = await store.extract_with_llm(user_id="student", session_id="two", user_text="remember green")
    await store.close()

    assert first is not None
    assert second is not None
    assert len(requests) == 2
    assert not client.is_closed
    await client.aclose()


@pytest.mark.asyncio
async def test_memory_store_closes_owned_extractor_client() -> None:
    store = MemoryStore(enabled=False)
    client = store._extractor_http_client()

    await store.close()

    assert client is not None
    assert client.is_closed


def test_memory_fallback_bounds_users_and_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_FALLBACK_MAX_USERS", "2")
    monkeypatch.setenv("MEMORY_FALLBACK_MAX_SESSIONS", "2")
    store = MemoryStore(enabled=False)

    store.cache_message(user_id="one", session_id="one", role="user", content="first")
    store.cache_message(user_id="two", session_id="two", role="user", content="second")
    store.cache_message(user_id="three", session_id="three", role="user", content="third")

    assert list(store._fallback_user_order) == ["two", "three"]
    assert len(store._memory_messages) == 2
    assert store._history_key("one", "one") not in store._memory_messages


@pytest.mark.asyncio
async def test_memory_fallback_bounds_items_per_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_FALLBACK_MAX_ITEMS_PER_USER", "2")
    store = MemoryStore(enabled=False)

    for content in ("Exam on Monday", "Prefers visual explanations", "Weak at fractions"):
        assert await store.remember(
            user_id="student",
            session_id="session",
            content=content,
            importance=10,
        ) is not None

    assert [item["content"] for item in store._memory_items["student"]] == [
        "Prefers visual explanations",
        "Weak at fractions",
    ]


@pytest.mark.asyncio
async def test_clear_user_removes_all_fallback_sessions() -> None:
    store = MemoryStore(enabled=False)
    store.cache_message(user_id="student", session_id="one", role="user", content="first")
    store.cache_message(user_id="student", session_id="two", role="assistant", content="second")

    await store.clear_user(user_id="student")

    history_prefix = store._history_key("student", "").rsplit(":", 1)[0] + ":"
    assert not any(key.startswith(history_prefix) for key in store._memory_messages)
    assert "student" not in store._fallback_user_order


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


@pytest.mark.parametrize("text, nickname", [
    ("我叫小明我最喜欢蓝色请记住", "小明"),
    ("我叫小明，我最喜欢蓝色，请记住。", "小明"),
    ("我的名字是欧阳小明我最喜欢蓝色请你记住", "欧阳小明"),
    ("我叫Alice我最喜欢蓝色帮我记住", "Alice"),
])
def test_voice_profile_separates_name_preference_and_instruction(text, nickname):
    changes = _extract_profile_changes(text)
    assert changes["nickname"] == nickname
    assert changes["preferences"]["topics_enjoy"] == ["蓝色"]
    records = _extract_memory_records(text)
    assert any(record["memory_type"] == "preference" and record["content"] == "我最喜欢蓝色" for record in records)
    assert any(record["memory_type"] == "profile_fact" and record["content"].endswith(nickname) for record in records)
    assert all("记住" not in record["content"] for record in records)


def test_short_name_declaration_creates_memory():
    assert any(record["content"] == "我叫小明" for record in _extract_memory_records("我叫小明"))


def test_voice_likes_and_dislikes_are_both_preserved():
    preferences = _extract_profile_changes("我最喜欢蓝色我不喜欢红色请记住")["preferences"]
    assert preferences == {"topics_enjoy": ["蓝色"], "topics_avoid": ["红色"]}


def test_favorite_recall_question_does_not_change_preferences():
    text = "我最喜欢什么颜色"
    assert "preferences" not in _extract_profile_changes(text)
    assert not any(record["memory_type"] == "preference" for record in _extract_memory_records(text))


@pytest.mark.parametrize(
    ("text", "field", "mode"),
    [
        ("以后温柔一点", "assistant_persona", "gentle"),
        ("默认说话幽默一点", "assistant_persona", "humorous"),
        ("默认使用北京话", "assistant_dialect", "beijing"),
        ("以后使用普通话", "assistant_dialect", "mandarin"),
    ],
)
def test_persistent_persona_and_dialect_preferences_are_extracted(text: str, field: str, mode: str) -> None:
    assert _extract_profile_changes(text)[field] == {"mode": mode}


def test_per_turn_persona_request_is_not_persisted() -> None:
    changes = _extract_profile_changes("这次温柔一点，使用北京话回答")

    assert "assistant_persona" not in changes
    assert "assistant_dialect" not in changes
