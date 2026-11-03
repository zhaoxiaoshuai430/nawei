"""Regressions for the 150-turn report: recall, task queries and stale state."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from services.llm.summary.context import read_context, selection_skip_reason
from services.llm.summary.policy import SummarySettings
from services.llm.summary.repository import SummaryRepository, normalize
from services.llm.summary.schema import SummaryCandidate
from services.llm.summary.snapshot import pack
from tests.test_summary_understanding import claim, multi_turn, repo


@pytest.mark.parametrize("query", [
    "按你记得的列出我现在没完成的任务，以及已完成和已取消的事项，别把它们混在一起。",
    "哪些任务已完成？", "海报已经完成了吗？", "我已经完成了哪些任务？",
])
def test_task_status_question_is_not_a_correction(query):
    assert selection_skip_reason(query, history=[], direct_tool=False, followup=False) is None


@pytest.mark.parametrize("query", [
    "数学任务已经完成", "海报已经完成了，接下来做什么？", "我已经完成海报，列出其他任务。",
    "不要把已经完成的海报又当待办", "纠正一下：任务还没完成", "我不再喜欢长篇讲解",
])
def test_real_update_still_overrides_history(query):
    assert selection_skip_reason(query, history=[], direct_tool=False, followup=False) == "current_correction"


async def test_task_inventory_includes_named_open_and_terminal_tasks_before_general_preferences():
    snapshot = multi_turn()
    items = [claim(snapshot, "open_task", origin="user_explicit", subject=subject, content=content, progress=progress)
             for subject, content, progress in [
                 ("错题本封面", "错题本封面没开始", "open"),
                 ("海报", "海报已交给老师", "resolved"),
                 ("露营", "露营取消", "cancelled")]]
    items.append(claim(snapshot, "communication_preference", scope="general", content="喜欢温和简短的回答"))
    body, _ = normalize(SummaryCandidate(claims=items), snapshot)
    result = await read_context(repo(body), "owner", "列出未完成、已完成和取消的任务", "full")
    assert result and len(result.entries) == 4
    assert "错题本封面" in " ".join(result.entries[:3])
    assert any(text.startswith("已完成") for text in result.entries[:3])
    assert any(text.startswith("已取消") for text in result.entries[:3])
    assert result.entries[-1] == "喜欢温和简短的回答"


@pytest.mark.parametrize("new_first", [True, False])
@pytest.mark.parametrize("separate_outcome", [True, False])
async def test_new_terminal_state_never_reopens_in_projection_or_prompt(new_first, separate_outcome):
    snapshot = multi_turn()
    old = claim(snapshot, "open_task", progress="open", content="海报待完成", subject="海报",
                evidence_message_ids=[snapshot["turns"][0]["user_message_id"]])
    new = claim(snapshot, "open_task", progress="resolved", content="海报已经交付", subject="海报",
                evidence_message_ids=[snapshot["turns"][-1]["user_message_id"]])
    candidate = SummaryCandidate(claims=[new, old] if new_first else [old, new])
    if separate_outcome:
        candidate = SummaryCandidate(claims=[old], task_outcomes=[new])
    body, _ = normalize(candidate, snapshot)
    body["summary_revision"] = 1
    conn = SimpleNamespace(fetchval=AsyncMock(return_value=True), execute=AsyncMock())
    repository = SummaryRepository(SimpleNamespace(), SummarySettings(enabled=True))
    await repository._promote_durable_memories(conn, {"user_id": "owner"}, uuid4(), body, snapshot)
    assert conn.execute.await_count == 2
    assert "summary_claim_retracted" in conn.execute.call_args_list[0].args[0]
    assert "summary_plan_not_current" in conn.execute.call_args_list[1].args[0]
    selected = await read_context(repo(body), "owner", "海报进度", "full")
    assert selected and selected.entries == ("已解决或完成，避免重复安排：海报已经交付",)


@pytest.mark.parametrize("progress", ["open", "partial", "changed", None])
def test_task_outcome_rejects_unfinished_state(progress):
    snapshot = multi_turn()
    with pytest.raises(ValueError, match="completed or cancelled"):
        normalize(SummaryCandidate(task_outcomes=[claim(snapshot, "open_task", progress=progress)]), snapshot)


async def test_explicit_reopening_supersedes_an_older_task_outcome():
    snapshot = multi_turn()
    done = claim(snapshot, "open_task", subject="海报", content="海报已完成", progress="resolved",
                 evidence_message_ids=[snapshot["turns"][0]["user_message_id"]])
    reopened = claim(snapshot, "open_task", subject="海报", content="海报需要修改标题", progress="open",
                     evidence_message_ids=[snapshot["turns"][-1]["user_message_id"]])
    body, _ = normalize(SummaryCandidate(claims=[reopened], task_outcomes=[done]), snapshot)
    selected = await read_context(repo(body), "owner", "海报进展", "full")
    assert selected and selected.entries == ("海报需要修改标题",)


async def test_recovered_temporary_state_supersedes_earlier_fatigue():
    snapshot = multi_turn()
    old = claim(snapshot, "temporary_state", scope="general", subject="精神状态", content="疲劳需要放慢节奏",
                evidence_message_ids=[snapshot["turns"][0]["user_message_id"]])
    new = claim(snapshot, "temporary_state", scope="general", subject="精神状态", content="已经恢复正常节奏",
                evidence_message_ids=[snapshot["turns"][-1]["user_message_id"]])
    body, _ = normalize(SummaryCandidate(temporary_states=[new, old]), snapshot)
    selected = await read_context(repo(body), "owner", "开始复习", "full")
    assert selected and len(selected.entries) == 1
    assert "恢复正常" in selected.entries[0] and "疲劳" not in selected.entries[0]


async def test_personal_facts_survive_normalization_selection_projection_and_prior_snapshot():
    from datetime import datetime

    snapshot = multi_turn()
    snapshot["turns"][0]["user_text"] = "我是林晓，我家的猫叫豆包。"
    from services.llm.summary.repository import digest
    snapshot["turns"][0]["source_hash"] = digest([snapshot["turns"][0]["user_text"], snapshot["turns"][0]["assistant_text"]])
    facts = [dict(kind="personal_fact", origin="user_explicit", status="supported", subject=subject, content=content,
                  evidence_message_ids=[snapshot["turns"][0]["user_message_id"]])
             for subject, content in [("用户姓名", "用户姓名是林晓"), ("宠物名称", "家里的猫叫豆包")]]
    body, _ = normalize(SummaryCandidate(personal_facts=facts), snapshot)
    body["summary_revision"] = 1
    selected = await read_context(repo(body), "owner", "我是谁，家里的猫叫什么？", "full")
    assert selected and set(selected.entries) == {fact["content"] for fact in facts}
    assert await read_context(repo(body), "owner", "化学原子结构", "full") is None
    conn = SimpleNamespace(fetchval=AsyncMock(side_effect=[True, None, None]), execute=AsyncMock())
    repository = SummaryRepository(SimpleNamespace(), SummarySettings(enabled=True))
    await repository._promote_durable_memories(conn, {"user_id": "owner"}, uuid4(), body, snapshot)
    assert conn.execute.await_count == 3
    assert all(call.args[2] == "profile_fact" for call in conn.execute.call_args_list[:2])
    rows = [{**turn, "completed_at": datetime.fromisoformat(turn["completed_at"])} for turn in snapshot["turns"]]
    packed = pack(rows[-1:], body, rows, snapshot, 20000)
    assert len(packed["previous_summary"]["personal_facts"]) == 2
    assert packed["reference_turns"][0]["user_text"] == snapshot["turns"][0]["user_text"]
    changed = deepcopy(rows)
    changed[0]["user_text"] = "已修改原文"
    assert pack(rows[-1:], body, changed, snapshot, 20000)["previous_summary"] is None


async def test_fresh_facts_rank_ahead_of_general_profile_on_personal_recall():
    snapshot = multi_turn()
    facts = [dict(kind="personal_fact", subject=subject, content=content, origin="user_explicit", status="supported",
                  evidence_message_ids=[snapshot["turns"][-1]["user_message_id"]])
             for subject, content in [("用户姓名", "用户叫林晓"), ("宠物名称", "用户的猫叫豆包")]]
    general = [claim(snapshot, "communication_preference", subject=f"通用偏好{index}", scope="general",
                     content=f"用户表达的通用交流偏好内容{index}") for index in range(4)]
    body, _ = normalize(SummaryCandidate(personal_facts=facts, claims=general), snapshot)
    context = await read_context(repo(body), "owner", "我是谁，家里的猫叫什么？", "full")
    assert context and set(context.entries[:2]) == {"用户叫林晓", "用户的猫叫豆包"}


async def test_task_inventory_budget_preserves_active_tasks_and_terminal_coverage():
    snapshot = multi_turn()
    active = [claim(snapshot, "open_task", subject=subject, content=content, progress=progress)
              for subject, content, progress in [("错题本封面", "错题本封面没开始", "open"),
                                                 ("班会发言稿", "班会发言稿只写了提纲", "partial")]]
    outcomes = [claim(snapshot, "open_task", subject=f"历史任务{index}", content="用户已完成任务并提交成果。" * 10,
                      progress="resolved") for index in range(12)]
    body, _ = normalize(SummaryCandidate(claims=active, task_outcomes=outcomes), snapshot)
    selected = await read_context(repo(body), "owner", "列出未完成和已完成的任务", "full")
    assert selected and len(selected.entries) == 14
    assert all("历史任务" not in text for text in selected.entries[:2])
    assert all(text.startswith("已完成：") for text in selected.entries[2:])


@pytest.mark.parametrize("attack", ["assistant", "observed", "foreign"])
def test_personal_facts_require_explicit_owned_user_sources(attack):
    snapshot = multi_turn()
    item = dict(kind="personal_fact", subject="用户姓名", content="用户姓名是林晓", origin="user_explicit",
                status="supported", evidence_message_ids=[snapshot["turns"][0]["user_message_id"]])
    if attack == "assistant":
        item["evidence_message_ids"] = [snapshot["turns"][0]["assistant_message_id"]]
    elif attack == "foreign":
        item["evidence_message_ids"] = [str(uuid4())]
    else:
        item["origin"] = "observed"
    with pytest.raises((ValueError, ValidationError)):
        normalize(SummaryCandidate(personal_facts=[item]), snapshot)


async def test_persisted_teaching_preferences_are_read_from_repository_result():
    snapshot = multi_turn()
    repository = repo({})
    repository.read.return_value["teaching_preferences"] = [{
        "preference_key": "example-first", "subject": "数学", "content": "数学先举一个例子再让用户复述",
        "status": "supported", "observed_at": snapshot["turns"][-1]["completed_at"], "expires_at": None,
    }]
    selected = await read_context(repository, "owner", "数学练习", "full")
    assert selected and selected.claim_ids == ("example-first",)
    assert "先举一个例子" in selected.entries[0]
