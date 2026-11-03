"""Knowledge prompt budgets must not separate evidence from its attribution."""

import pytest

from services.knowledge.schemas import KnowledgeChunkHit, KnowledgeContext, KnowledgeGraphRelation


def chunk(**changes):
    values = dict(chunk_id="one", document_id="doc", title="数据库", text="正文。" * 300,
                  source_name="公开教材", source_url="https://example.test/chapter?section=1",
                  source_locator="第二章", page=2)
    return KnowledgeChunkHit(**(values | changes))


def test_long_evidence_preserves_complete_attribution_and_marks_excerpt():
    context = KnowledgeContext(chunks=[chunk()])
    prompt = context.render_prompt(max_chars=300)
    assert len(prompt) <= 300
    assert "[资料1]" in prompt
    assert "[正文截断]" in prompt
    assert "来源：公开教材，第2页；定位：第二章；链接：https://example.test/chapter?section=1；相关度：0.000" in prompt
    assert context.chunks[0].text == "正文。" * 300


@pytest.mark.parametrize("budget", [-1, 0, 1, 50])
def test_no_evidence_returns_empty_prompt_with_tiny_budget(budget):
    assert KnowledgeContext(chunks=[chunk()]).render_prompt(max_chars=budget) == ""


def test_unfittable_attribution_is_skipped_without_renumbering_sources():
    context = KnowledgeContext(chunks=[chunk(source_name="过长来源" * 300), chunk(text="短正文")])
    prompt = context.render_prompt(max_chars=300)
    assert "[资料1]" not in prompt
    assert "[资料2] 短正文" in prompt
    assert "https://example.test/chapter?section=1" in prompt


@pytest.mark.parametrize("grouped", [False, True])
def test_graph_relations_are_not_cut_mid_fact(grouped):
    relation = KnowledgeGraphRelation(source_id="a", source_name="数据库", relation_type="PART_OF",
                                      target_id="b", target_name="计算机科学")
    context = (KnowledgeContext(relation_groups={"taxonomy": [relation]}) if grouped
               else KnowledgeContext(relations=[relation]))
    complete = context.render_prompt()
    assert "数据库 --PART_OF--> 计算机科学" in complete
    assert context.render_prompt(max_chars=len(complete) - 1) == ""


def test_budget_boundaries_preserve_attribution_for_every_emitted_chunk():
    context = KnowledgeContext(chunks=[chunk(text="第一段"), chunk(text="第二段" * 80)])
    complete = context.render_prompt()
    for budget in range(len(complete) + 2):
        prompt = context.render_prompt(max_chars=budget)
        assert len(prompt) <= budget
        assert prompt.count("[资料") == prompt.count("；相关度：0.000")
        assert prompt.count("[资料") == prompt.count("https://example.test/chapter?section=1")
    assert context.render_prompt(max_chars=len(complete)) == complete
