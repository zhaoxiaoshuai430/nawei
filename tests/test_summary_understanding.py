"""Document requirements: evidence thresholds, longitudinal comparison and online use."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.llm.summary.context import read_context, render_context
from services.llm.summary.policy import SummarySettings, token_bound
from services.llm.summary.repository import normalize
from services.llm.summary.schema import SummaryCandidate
from services.llm.summary.snapshot import pack
from services.llm.summary.worker import SummaryWorker
from tests.test_conversation_summary import sample


def multi_turn():
    turns = []
    for index in range(3):
        snapshot, _ = sample()
        turn = snapshot["turns"][0]
        turn.update(session_id=f"session-{index % 2}", completed_seq=index + 1,
                    completed_at=(datetime.now(timezone.utc) - timedelta(minutes=3-index)).isoformat())
        turns.append(turn)
    return {"turns": turns}


def claim(snapshot, kind, **fields):
    return {"kind": kind, "subject": "数学", "content": "数学分步骤讲解后能独立完成", "origin": "observed",
            "status": "supported", "evidence_message_ids": [t["user_message_id"] for t in snapshot["turns"]], **fields}


def repo(body):
    return SimpleNamespace(settings=SummarySettings(enabled=True, apply_to_prompt=True, prompt_max_tokens=2000),
        read=AsyncMock(return_value={"summary": {**body, "summary_revision": 1}, "enabled": True,
                                    "apply_to_prompt": True, "memory_epoch": 0}))


def test_stable_behavior_requires_multiple_user_turns_and_sessions():
    snapshot = multi_turn()
    candidate = SummaryCandidate(profile_change_proposals=[claim(snapshot, "profile_change_proposal")])
    body, _ = normalize(candidate, snapshot)
    assert len(body["stable_profile"]) == 1
    assert normalize(candidate, snapshot, promote=False)[0]["stable_profile"] == []
    for turn in snapshot["turns"]:
        turn["session_id"] = "one-session"
    assert normalize(candidate, snapshot)[0]["stable_profile"] == []
    # Assistant evidence must not satisfy a repeated-user-behavior threshold.
    snapshot = multi_turn()
    item = claim(snapshot, "profile_change_proposal", evidence_message_ids=[snapshot["turns"][0]["user_message_id"]] +
                 [t["assistant_message_id"] for t in snapshot["turns"]])
    assert normalize(SummaryCandidate(profile_change_proposals=[item]), snapshot)[0]["stable_profile"] == []


@pytest.mark.parametrize("change", [{"status": "tentative"}, {"origin": "inferred", "status": "tentative"},
                                     {"counterevidence_message_ids": "self"}])
def test_ambiguous_profiles_never_promoted(change):
    snapshot = multi_turn()
    if change.get("counterevidence_message_ids") == "self":
        change = {"counterevidence_message_ids": [snapshot["turns"][-1]["user_message_id"]]}
    body, _ = normalize(SummaryCandidate(profile_change_proposals=[claim(snapshot, "profile_change_proposal", **change)]), snapshot)
    assert not body["stable_profile"]


def test_learning_trend_requires_chronological_user_performance():
    snapshot = multi_turn()
    item = claim(snapshot, "learning_observation", trend="independent")
    assert normalize(SummaryCandidate(learning_observations=[item]), snapshot)[0]["learning_observations"][0]["trend"] == "independent"
    item["evidence_message_ids"] = [snapshot["turns"][0]["user_message_id"], snapshot["turns"][1]["assistant_message_id"]]
    with pytest.raises(ValueError, match="longitudinal"):
        normalize(SummaryCandidate(learning_observations=[item]), snapshot)


def test_teaching_effectiveness_cannot_use_future_assistant_explanations():
    snapshot = multi_turn()
    item = claim(snapshot, "teaching_strategy_observation", evidence_message_ids=[
        snapshot["turns"][0]["user_message_id"], snapshot["turns"][1]["assistant_message_id"]])
    with pytest.raises(ValueError, match="feedback after"):
        normalize(SummaryCandidate(teaching_strategy_observations=[item]), snapshot)
    item["evidence_message_ids"].append(snapshot["turns"][2]["user_message_id"])
    assert normalize(SummaryCandidate(teaching_strategy_observations=[item]), snapshot)[0]["teaching_strategy_observations"]


@pytest.mark.parametrize("text", ["验证码是839201", "密码:abc123", "银行卡号是6222021234567890123", "身份证号110101199001011234",
                                  "用户智商很低", "用户很懒惰", "用户患有抑郁症", "用户MBTI为INTJ"])
def test_sensitive_data_and_diagnostic_labels_rejected(text):
    snapshot = multi_turn()
    with pytest.raises(ValueError, match="private data"):
        normalize(SummaryCandidate(temporary_states=[claim(snapshot, "temporary_state", content=text)]), snapshot)


async def test_strategies_states_trends_and_profiles_reach_main_agent():
    snapshot = multi_turn()
    body, _ = normalize(SummaryCandidate(
        teaching_strategy_observations=[claim(snapshot, "teaching_strategy_observation", content="数学先举例再解释效果好")],
        temporary_states=[claim(snapshot, "temporary_state", content="今天疲惫，适合少量步骤", scope="general")],
        learning_observations=[claim(snapshot, "learning_observation", content="数学由提示逐渐转为独立完成", trend="independent")],
        profile_change_proposals=[claim(snapshot, "profile_change_proposal", content="数学习惯先尝试再看提示")]), snapshot)
    context = await read_context(repo(body), "owner", "数学练习", "full")
    assert context and len(context.entries) == 4
    assert "历史教学反馈" in render_context(context) and "近期临时状态" in render_context(context)
    assert token_bound(render_context(context)) <= 2000
    body["temporary_states"][0]["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    context = await read_context(repo(body), "owner", "数学练习", "full")
    assert context and len(context.entries) == 3


async def test_resolved_tasks_are_not_resumed_and_supersede_open_tasks():
    snapshot = multi_turn()
    old = claim(snapshot, "open_task", progress="open", evidence_message_ids=[snapshot["turns"][0]["user_message_id"]])
    new = claim(snapshot, "open_task", progress="resolved", evidence_message_ids=[snapshot["turns"][-1]["user_message_id"]])
    body, _ = normalize(SummaryCandidate(claims=[old, new]), snapshot)
    assert await read_context(repo(body), "owner", "接着上次讲", "full", followup=True) is None
    context = await read_context(repo(body), "owner", "数学练习", "full")
    assert context and len(context.entries) == 1 and context.entries[0].startswith("已解决")


async def test_repeated_observed_general_preference_applies_beyond_subject():
    snapshot = multi_turn()
    body, _ = normalize(SummaryCandidate(claims=[claim(snapshot, "communication_preference", scope="general",
                                                        content="喜欢先举例再说明")]), snapshot)
    context = await read_context(repo(body), "owner", "讲解化学反应", "full")
    assert context and context.entries == ("喜欢先举例再说明",)


def test_prior_summary_requires_intact_raw_evidence_and_shares_budget():
    snapshot = multi_turn()
    previous, _ = normalize(SummaryCandidate(claims=[claim(snapshot, "open_task", progress="partial")]), snapshot)
    previous["summary_revision"] = 1
    rows = [{**t, "completed_at": datetime.fromisoformat(t["completed_at"])} for t in snapshot["turns"]]
    packed = pack(rows[-1:], previous, rows, snapshot, 20000)
    assert packed["previous_summary"]["claims"][0]["progress"] == "partial"
    assert len(packed["reference_turns"]) == 2
    assert token_bound(__import__("json").dumps(packed, ensure_ascii=False)) < 20000
    damaged = deepcopy(rows)
    damaged[0]["user_text"] = "内容被修改"
    assert pack(rows[-1:], previous, damaged, snapshot, 20000)["previous_summary"] is None
    assert pack(rows[-1:], previous, rows[1:], snapshot, 20000)["previous_summary"] is None
    tiny = pack(rows[-1:], previous, rows, snapshot, 1000)
    assert tiny["previous_summary"] is None


def test_secret_turn_omitted_before_job_snapshot_is_stored():
    snapshot = multi_turn()
    rows = [{**t, "completed_at": datetime.fromisoformat(t["completed_at"])} for t in snapshot["turns"]]
    rows[0]["user_text"] = "验证码是839201"
    packed = pack(rows, None, [], {}, 20000)
    assert packed["sensitive_turn_count"] == 1 and len(packed["turns"]) == 2
    assert "839201" not in str(packed)


async def test_reference_only_snapshot_still_preserves_previous_understanding():
    snapshot = multi_turn()
    snapshot["reference_turns"], snapshot["turns"] = snapshot["turns"], []
    repository = SimpleNamespace(settings=SummarySettings(), fail=AsyncMock(), release=AsyncMock(), publish=AsyncMock())
    agent = SimpleNamespace(generate=AsyncMock(return_value=(SummaryCandidate(), "test")))
    await SummaryWorker(repository, agent).process({"input_snapshot": snapshot, "job_id": "test"})
    agent.generate.assert_awaited_once()
    repository.publish.assert_awaited_once()
