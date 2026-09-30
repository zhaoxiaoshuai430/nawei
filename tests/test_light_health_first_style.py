from copy import deepcopy

import pytest

from scripts import run_light_health_comparison as original
from scripts.run_light_health_first_style import validate_inputs, validate_record
from services.knowledge.schemas import KnowledgeChunkHit, KnowledgeContext, KnowledgeGraphRelation


def dataset():
    cases = []
    for index in range(1, 31):
        question = f"问题{index}"
        vector = KnowledgeContext(query=question, chunks=[KnowledgeChunkHit(
            chunk_id="chunk", document_id="book", title="光与健康", text="VECTOR", score=0.9,
        )])
        graph = KnowledgeContext(query=question, relations=[KnowledgeGraphRelation(
            source_id="parent", source_name="GRAPH", relation_type="CONTAINS",
            target_id="child", target_name="分类",
        )])
        vector_prompt = vector.render_prompt(max_chars=7000)
        graph_prompt = graph.render_prompt(max_chars=5000)
        cases.append({"index": index, "question": question, "category": "分类",
                      "vector_context": vector.model_dump(), "graph_context": graph.model_dump(),
                      "prompts": {"neither": "", "both": vector_prompt + "\n\n" + graph_prompt,
                                  "vector_only": vector_prompt, "graph_only": graph_prompt}})
    return {"cases": cases}, {"questions": [
        {name: case[name] for name in ("index", "question", "category")} for case in cases
    ]}


def test_original_policy_remains_exact():
    assert original.GENERATION == {"agent_enabled": False, "temperature": 0, "top_p": 1, "max_tokens": 900}
    assert "正文约200至350个汉字，最多450字" in original.SYSTEM_PROMPT
    assert "结尾用一行‘依据：’" in original.SYSTEM_PROMPT


def test_original_evidence_budgets_and_revision_validation():
    evidence, question_set = dataset()
    validate_inputs(evidence, question_set)
    damaged = deepcopy(evidence)
    damaged["cases"][0]["prompts"]["graph_only"] += "VECTOR"
    with pytest.raises(ValueError, match="prompts"):
        validate_inputs(damaged, question_set)
    damaged = deepcopy(evidence)
    damaged["cases"][0]["question"] = "wrong"
    with pytest.raises(ValueError, match="selected questions"):
        validate_inputs(damaged, question_set)
    damaged = deepcopy(evidence)
    damaged["cases"][0]["graph_context"]["chunks"] = damaged["cases"][0]["vector_context"]["chunks"]
    with pytest.raises(ValueError, match="Mixed"):
        validate_inputs(damaged, question_set)


@pytest.mark.parametrize("mode", original.MODES)
def test_first_style_response_validation(mode):
    evidence, _ = dataset()
    case = evidence["cases"][0]
    context = case["prompts"][mode]
    record = {"index": 1, "mode": mode, "question": case["question"],
              "request": {"request_id": "test", "stream": False, "tools": [],
                          "generation_config": original.GENERATION,
                          "messages": [{"role": "system", "content": original.SYSTEM_PROMPT + ("\n\n" + context if context else "")},
                                       {"role": "user", "content": case["question"]}]},
              "response": {"request_id": "test", "text": "正文。\n依据：资料1。", "finish_reason": "stop",
                           "metadata": {"provider": "deepseek", "model": "model",
                                        "runtime": {"agent": {"enabled": False}}}}}
    validate_record(record, case, mode, {"model": "model"})
    damaged = deepcopy(record)
    damaged["response"]["metadata"]["runtime"]["agent"]["enabled"] = True
    with pytest.raises(ValueError, match="bypassed"):
        validate_record(damaged, case, mode, {"model": "model"})
    damaged = deepcopy(record)
    damaged["response"]["metadata"]["fallback"] = True
    with pytest.raises(ValueError, match="Fallback"):
        validate_record(damaged, case, mode, {"model": "model"})


def test_format_audit_does_not_enforce_or_rewrite_answers():
    assert original.answer_format_audit("光照ABC123。\n依据：资料1。") == {"has_basis": True, "body_han_chars": 2}
    assert original.answer_format_audit("光照") == {"has_basis": False, "body_han_chars": 2}
