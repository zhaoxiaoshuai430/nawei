from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from app_protocols import LLMResponse, Usage
from scripts import run_light_health_current_comparison as comparison
from scripts.prepare_light_health_graph_comparison import REPLACEMENTS, build_question_set
from services.knowledge.schemas import KnowledgeChunkHit, KnowledgeContext, KnowledgeGraphRelation


@pytest.fixture
def case():
    return {
        "index": 1, "question": comparison.QUESTIONS[0][1], "category": comparison.QUESTIONS[0][0],
        "vector_context": KnowledgeContext(chunks=[KnowledgeChunkHit(
            chunk_id="book-light", document_id="book", title="光与健康", text="VECTOR_SENTINEL",
            source_name="光与健康", score=0.9,
        )]).model_dump(),
        "graph_context": KnowledgeContext(relations=[KnowledgeGraphRelation(
            source_id="light", source_name="GRAPH_SENTINEL", relation_type="AFFECTS",
            target_id="rhythm", target_name="昼夜节律",
        )]).model_dump(),
    }


@pytest.fixture
def environment(monkeypatch):
    for flag in comparison.TOOL_FLAGS:
        monkeypatch.setenv(flag, "false")
    monkeypatch.setenv("LLM_TOOL_SELECTION_MODE", "native")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    monkeypatch.setenv("KNOWLEDGE_MAX_CONTEXT_CHARS", "6000")


@pytest.mark.parametrize("mode", comparison.MODES)
def test_source_selection(case, mode):
    context = comparison.select_context(case, mode)
    assert bool(context.chunks) == (mode in {"both", "vector_only"})
    assert bool(context.relations) == (mode in {"both", "graph_only"})
    assert comparison.select_context(case, mode) is not context
    with pytest.raises(ValueError):
        comparison.select_context(case, "bad")


@pytest.mark.asyncio
async def test_current_pipeline_concurrent_isolation(case, environment, monkeypatch):
    runtime = comparison.ComparisonRuntime()
    monkeypatch.setattr(comparison.llm_main, "agent_runtime", runtime)
    monkeypatch.setattr(comparison.llm_main, "safety_guard", runtime.safety_guard)
    backend = AsyncMock(return_value=LLMResponse(
        request_id="test", text="真实推理在此测试中替换为固定文本。", finish_reason="stop",
        usage=Usage(input_tokens=100, output_tokens=20),
        metadata={"provider": "deepseek", "model": "test-model"},
    ))
    monkeypatch.setattr(comparison.llm_main, "complete_deepseek", backend)
    try:
        plans = await comparison.prepare_plans(runtime, {"cases": [case]}, "test")

        async def one(mode):
            active = {"case": case, "mode": mode, "retrieval_calls": 0}
            token = comparison.ACTIVE_CASE.set(active)
            try:
                response = await comparison.llm_main.complete_llm(comparison.request_for(case, mode, "test"))
                plan = plans[f"q01_{mode}"]
                record = {**plan, "response": response.model_dump(mode="json")}
                comparison.validate_record(record, {"model": "test-model"}, plan)
                prompt = active["prepared_request"]["messages"][0]["content"]
                assert ("VECTOR_SENTINEL" in prompt) == (mode in {"both", "vector_only"})
                assert ("GRAPH_SENTINEL" in prompt) == (mode in {"both", "graph_only"})
                assert "标准模式" in prompt
                assert "没有年龄硬约束时不设置固定长度" in prompt
                if mode in {"both", "vector_only"}:
                    assert "知识库引用规则" in prompt and "参考来源" in prompt
                config = active["prepared_request"]["generation_config"]
                assert config["agent_enabled"] is True
                assert config["temperature"] == 0.32
                assert "max_tokens" not in config
                assert active["evidence_audit"] == plan["evidence_audit"]
                agent_audit = response.metadata["runtime"]["comparison_agent_audit"]
                assert all(agent_audit[field] is True for field in (
                    "agent_enabled", "context_prepared", "strategy_prepared", "system_prompt_generated",
                ))
                assert agent_audit["knowledge_mode"] == mode
                damaged = deepcopy(record)
                damaged["response"]["metadata"]["runtime"]["comparison_agent_audit"]["context_prepared"] = False
                with pytest.raises(ValueError, match="Agent execution"):
                    comparison.validate_record(damaged, {"model": "test-model"}, plan)
                damaged = deepcopy(record)
                damaged["request"]["generation_config"]["agent_enabled"] = False
                with pytest.raises(ValueError, match="bypassed"):
                    comparison.validate_record(damaged, {"model": "test-model"}, plan)
                damaged = deepcopy(record)
                damaged["response"]["finish_reason"] = "length"
                with pytest.raises(ValueError, match="incomplete"):
                    comparison.validate_record(damaged, {"model": "test-model"}, plan)
                damaged = deepcopy(record)
                damaged["response"]["metadata"]["fallback"] = True
                with pytest.raises(ValueError, match="Fallback"):
                    comparison.validate_record(damaged, {"model": "test-model"}, plan)
                damaged = deepcopy(record)
                damaged["prepared_request"]["messages"][0]["content"] += "changed"
                with pytest.raises(ValueError, match="request changed"):
                    comparison.validate_record(damaged, {"model": "test-model"}, plan)
            finally:
                comparison.ACTIVE_CASE.reset(token)

        await asyncio.gather(*(one(mode) for mode in comparison.MODES))
        assert backend.await_count == 4
    finally:
        await runtime.close()


def test_original_questions_must_match():
    evidence = {"cases": [{"index": index, "category": category, "question": question}
                          for index, (category, question) in enumerate(comparison.QUESTIONS, 1)]}
    comparison.validate_evidence(evidence)
    evidence["cases"][0]["question"] += "changed"
    with pytest.raises(ValueError, match="question changed"):
        comparison.validate_evidence(evidence)


def test_citation_flags_do_not_rewrite_text():
    text = "正文[资料1]\n参考来源\n- 资料1：光与健康，链接：光与健康\n- 资料2：链接：https://example.org"
    flags = comparison.citation_flags(text)
    assert flags["has_reference_section"]
    assert flags["inline_source_markers"]
    assert flags["non_url_links"] == ["光与健康"]


def test_resume_fingerprint_detects_evidence_change():
    plans = {"q01_neither": {"prepared_request": {}, "knowledge_context": {}, "evidence_audit": {}}}
    previous = comparison.plan_fingerprint(plans)
    plans["q01_neither"]["knowledge_context"]["chunks"] = ["changed"]
    assert comparison.plan_fingerprint(plans) != previous


def test_graph_revision_retains_twenty_original_questions():
    question_set = build_question_set()
    assert len(question_set["questions"]) == 30
    assert question_set["replaced_count"] == 10
    assert sum(not item["replaced"] for item in question_set["questions"]) == 20
    evidence = {"cases": question_set["questions"]}
    comparison.validate_evidence(evidence, question_set)
    with pytest.raises(ValueError, match="question changed"):
        comparison.validate_evidence(evidence)
    for item in question_set["questions"]:
        assert item["replaced"] == (item["index"] in REPLACEMENTS)
        if item["index"] >= 26:
            assert not item["replaced"]
        request = comparison.request_for(item, "both", "revision-test")
        assert request.messages[0].content == item["question"]
        assert len(request.messages) == 1
        assert request.metadata == {"knowledge_base_id": "default"}


@pytest.mark.parametrize("corruption", ["index", "duplicate", "original", "marker", "count", "evidence"])
def test_graph_revision_rejects_mismatched_dataset(corruption):
    question_set = build_question_set()
    evidence = {"cases": deepcopy(question_set["questions"])}
    if corruption == "index":
        question_set["questions"][1]["index"] = 1
    elif corruption == "duplicate":
        question_set["questions"][1]["question"] = question_set["questions"][0]["question"]
    elif corruption == "original":
        question_set["questions"][0]["original_question"] = "wrong"
    elif corruption == "marker":
        question_set["questions"][0]["replaced"] = True
    elif corruption == "count":
        question_set["replaced_count"] = 9
    else:
        evidence["cases"][2]["question"] = comparison.QUESTIONS[2][1]
    with pytest.raises(ValueError):
        comparison.validate_evidence(evidence, question_set)
