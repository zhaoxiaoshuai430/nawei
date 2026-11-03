"""Fixed adversarial scenarios for online summary selection and prompt integration."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app_protocols import LLMRequest, Message, MessageRole
from services.knowledge.client import KnowledgeClient
from services.llm.agent_runtime import LLMAgentRuntime
from services.llm.memory import MemoryStore
from services.llm.summary.context import read_context, render_context
from services.llm.summary.policy import SummarySettings, token_bound
from services.llm.summary.repository import normalize
from services.llm.summary.schema import SummaryCandidate
from tests.test_conversation_summary import sample


def published():
    snapshot, raw = sample()
    body, _ = normalize(SummaryCandidate.model_validate(raw), snapshot)
    body["summary_revision"] = 3
    return {"summary": body, "enabled": True, "apply_to_prompt": True, "memory_epoch": 2}


@pytest.mark.parametrize("query,options,reason", [
    ("数学不要再分步骤", {}, "current_correction"),
    ("数学任务已经完成", {}, "current_correction"),
    ("数学练习", {"history": [{"role": "user", "content": "我不再喜欢分步骤讲解"}]}, "recent_correction"),
    ("继续", {"followup": True, "history": [{"role": "assistant", "content": "我们来讨论化学"}]}, "recent_history_priority"),
    ("今天数学考试几点", {"direct_tool": True}, "direct_tool"),
])
async def test_current_control_and_history_skip_before_database(query, options, reason):
    repo = SimpleNamespace(settings=SummarySettings(enabled=True, apply_to_prompt=True), read=AsyncMock())
    diagnostics = {}
    assert await read_context(repo, "owner", query, "full", diagnostics=diagnostics, **options) is None
    assert diagnostics["status"] == reason
    repo.read.assert_not_called()


async def test_online_baseline_switches_and_explicit_current_topic():
    result = published()
    repo = SimpleNamespace(settings=SummarySettings(enabled=True, apply_to_prompt=True), read=AsyncMock(return_value=result))
    report = {}
    selected = await read_context(repo, "owner", "数学分数复习", "lightweight", expected_epoch=2, diagnostics=report)
    assert selected and selected.claim_ids == tuple(report["claim_ids"])
    assert report["status"] == "applied" and report["selected_count"] == 1
    assert report["prompt_token_upper_bound"] == token_bound(render_context(selected)) <= 600
    assert selected.claim_ids[0] not in render_context(selected)
    assert await read_context(repo, "owner", "化学原子结构", "full") is None
    result["apply_to_prompt"] = False
    assert await read_context(repo, "owner", "数学分数复习", "full", diagnostics=report) is None
    assert report["status"] == "user_disabled"
    result["apply_to_prompt"] = True
    assert await read_context(repo, "owner", "数学分数复习", "full", expected_epoch=1, diagnostics=report) is None
    assert report["status"] == "epoch_changed"


@pytest.mark.parametrize("mutation", ["expired", "tentative", "contradicted", "inferred", "observed_preference", "counterevidence"])
async def test_uncertain_or_retracted_claim_never_used(mutation):
    result = published()
    claim = result["summary"]["claims"][0]
    if mutation == "expired":
        claim["expires_at"] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    elif mutation in {"tentative", "contradicted"}:
        claim["status"] = mutation
    elif mutation in {"inferred", "observed_preference"}:
        claim["origin"] = "inferred" if mutation == "inferred" else "observed"
    else:
        claim["counterevidence_message_ids"] = ["source"]
    repo = SimpleNamespace(settings=SummarySettings(enabled=True, apply_to_prompt=True), read=AsyncMock(return_value=result))
    assert await read_context(repo, "owner", "数学分数复习", "full") is None


async def test_ranking_budget_dedup_and_latin_word_boundaries():
    result = published()
    first = result["summary"]["claims"][0]
    unrelated = deepcopy(first)
    unrelated.update(subject="history", content="Study history", claim_id="history")
    result["summary"]["claims"] = [unrelated, first, deepcopy(first)]
    repo = SimpleNamespace(settings=SummarySettings(enabled=True, apply_to_prompt=True), read=AsyncMock(return_value=result))
    assert await read_context(repo, "owner", "chemistry", "full") is None
    selected = await read_context(repo, "owner", "数学分数", "full")
    assert selected and len(selected.entries) == 1
    assert token_bound(render_context(selected)) <= repo.settings.prompt_max_tokens


async def test_cross_session_continue_uses_open_task_only_without_local_history():
    result = published()
    task = deepcopy(result["summary"]["claims"][0])
    task.update(kind="open_task", content="继续练习异分母分数加法", claim_id="task")
    result["summary"]["claims"].append(task)
    repo = SimpleNamespace(settings=SummarySettings(enabled=True, apply_to_prompt=True), read=AsyncMock(return_value=result))
    selected = await read_context(repo, "owner", "继续", "lightweight", followup=True)
    assert selected and selected.entries == ("继续练习异分母分数加法",)
    assert await read_context(repo, "owner", "继续", "lightweight", followup=True,
                              history=[{"role": "assistant", "content": "我们来讨论化学"}]) is None


@pytest.mark.parametrize("memory_mode", ["full", "lightweight", "off"])
async def test_runtime_uses_single_typed_context_and_reports_usage(monkeypatch, memory_mode):
    monkeypatch.setenv("SUMMARY_ENABLED", "true")
    monkeypatch.setenv("SUMMARY_APPLY_TO_PROMPT", "true")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("KNOWLEDGE_ENABLED", "false")
    runtime = LLMAgentRuntime(memory_store=MemoryStore(enabled=False), knowledge_client=KnowledgeClient())
    runtime.summary_repository.state = AsyncMock(return_value={"memory_epoch": 2})
    runtime.summary_repository.previous_turn = AsyncMock(return_value=None)
    runtime.summary_repository.read = AsyncMock(return_value=published())
    request = LLMRequest(request_id="summary-online", trace_id="trace", user_id="owner", session_id="new-session",
        messages=[Message(role=MessageRole.USER, content="数学分数复习怎么安排")],
        generation_config={"memory_mode": memory_mode})
    try:
        turn = await runtime.prepare(request)
        prompt = turn.request.messages[0].content
        if memory_mode == "off":
            assert turn.context.summary_context is None
            assert "阶段历史数据" not in prompt
            runtime.summary_repository.read.assert_not_called()
        else:
            assert turn.context.summary_context is not None
            assert prompt.count("阶段历史数据") == 1
            assert turn.metadata["summary"]["status"] == "applied"
            assert turn.metadata["summary"]["selected_count"] == 1
    finally:
        await runtime.close()
