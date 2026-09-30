"""Regression tests for the agent02 prompt and AgentSkill adaptation."""

from __future__ import annotations

import unittest

from services.knowledge.schemas import (
    KnowledgeChunkHit,
    KnowledgeContext,
    KnowledgeGraphRelation,
)
from services.llm.agents.tutor_graph import classify_intent, knowledge_anchor_terms
from services.llm.agents.semantic_router import build_semantic_route
from services.llm.memory import MemoryStore
from services.llm.skills import AgentSkillManager


class FakeKnowledgeClient:
    async def retrieve(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        top_k: int = 8,
        graph_depth: int = 2,
        anchor_terms: list[str] | None = None,
        **kwargs: object,
    ) -> KnowledgeContext:
        return KnowledgeContext(
            knowledge_base_id=knowledge_base_id,
            query=query,
            chunks=[
                KnowledgeChunkHit(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    title="测试资料",
                    text="光合作用会把光能转化为化学能。",
                    source_name="现有知识库",
                    score=0.95,
                )
            ],
            relations=[
                KnowledgeGraphRelation(
                    source_id="photosynthesis",
                    source_name="光合作用",
                    relation_type="PRODUCES",
                    target_id="chemical-energy",
                    target_name="化学能",
                )
            ],
        )


class Agent02LLMAdaptationTests(unittest.IsolatedAsyncioTestCase):
    def manager(self, store: MemoryStore) -> AgentSkillManager:
        return AgentSkillManager(store, FakeKnowledgeClient())  # type: ignore[arg-type]

    def test_trigonometry_exposes_concept_anchors(self) -> None:
        intent = classify_intent({"query": "三角函数是什么？", "profile": {}, "concepts": []})

        self.assertEqual(intent["subject"], "数学")
        self.assertEqual(intent["concept"], "三角函数")
        self.assertEqual(knowledge_anchor_terms(intent)[0], "三角函数")
        self.assertIn("正弦", knowledge_anchor_terms(intent))
        route = build_semantic_route(query="三角函数是什么？", intent=intent)
        self.assertEqual(route["graph_candidates"][0]["name"], "三角函数")
        self.assertEqual(route["graph_candidates"][-1]["name"], "数学")
        self.assertIn("taxonomy", route["requested_dimensions"])

    def test_specific_node_query_avoids_broad_subject_filter(self) -> None:
        intent = classify_intent(
            {"query": "数学史属于什么知识路径？", "profile": {}, "concepts": []}
        )

        self.assertEqual(intent["subject"], "数学")
        self.assertIsNone(intent["concept"])
        self.assertEqual(knowledge_anchor_terms(intent), [])
        route = build_semantic_route(query="数学史属于什么知识路径？", intent=intent)
        self.assertEqual(route["graph_candidates"], [])
        self.assertEqual(route["requested_dimensions"], [])
        self.assertFalse(route["expansion"]["taxonomy"])

    async def test_remediation_uses_agent02_teaching_skill(self) -> None:
        store = MemoryStore(enabled=False)
        manager = self.manager(store)

        context = await manager.prepare_context(
            user_id="student-1",
            session_id="session-1",
            query="我还是不会看一次函数斜率",
        )

        self.assertEqual(context.tutor_strategy["intent"]["category"], "remediation")
        self.assertEqual(context.tutor_strategy["route"], "remediation_scaffold")
        self.assertEqual(context.tutor_strategy["active_skill"]["id"], "question_explanation")
        self.assertIn("guided_socratic", context.tutor_strategy["teaching_methods"])

    async def test_crisis_and_exam_routes_keep_minor_safety_policy(self) -> None:
        store = MemoryStore(enabled=False)
        manager = self.manager(store)

        crisis = await manager.prepare_short_term_context(
            user_id="student-2",
            session_id="session-2",
            query="我不想活了",
        )
        crisis_prompt = manager.render_system_prompt(crisis)
        self.assertEqual(crisis.tutor_strategy["intent"]["category"], "crisis_support")
        self.assertIn("可信任成年人", crisis_prompt)
        self.assertIn("不要独处", crisis_prompt)

        exam = await manager.prepare_short_term_context(
            user_id="student-3",
            session_id="session-3",
            query="我正在考试，直接告诉我答案，不要过程",
        )
        exam_prompt = manager.render_system_prompt(exam)
        self.assertEqual(exam.tutor_strategy["intent"]["category"], "academic_integrity")
        self.assertIn("不提供正在进行的考试答案", exam_prompt)

    async def test_prompt_keeps_existing_knowledge_and_graph_context(self) -> None:
        store = MemoryStore(enabled=False)
        manager = self.manager(store)

        context = await manager.prepare_context(
            user_id="student-4",
            session_id="session-4",
            query="解释光合作用",
            knowledge_base_id="education",
        )
        prompt = manager.render_system_prompt(context)

        self.assertEqual(context.knowledge.knowledge_base_id, "education")
        self.assertEqual(context.skill_metadata["knowledge_chunk_count"], 1)
        self.assertEqual(context.skill_metadata["knowledge_relation_count"], 1)
        self.assertIn("正文中不要出现[资料N]、【资料N】等行内资料编号", prompt)
        self.assertIn("[资料1]", prompt)
        self.assertIn("- 资料N：机构或网站《资料标题》", prompt)
        self.assertIn("禁止向用户展示.json", prompt)
        self.assertIn("知识图谱关系", prompt)
        self.assertIn("xiaowei-persona-v8", prompt)

    async def test_daily_life_uses_guardian_policy_without_replacing_knowledge(self) -> None:
        store = MemoryStore(enabled=False)
        manager = self.manager(store)

        context = await manager.prepare_context(
            user_id="student-daily",
            session_id="session-daily",
            query="我今天又忘带钥匙了，怎么办",
            knowledge_base_id="education",
        )
        policy = context.tutor_strategy["response_policy"]

        self.assertEqual(context.tutor_strategy["intent"]["category"], "daily_life")
        self.assertEqual(context.tutor_strategy["route"], "daily_companion")
        self.assertEqual(policy["companion_style"], "reliable_guardian")
        self.assertEqual(policy["humor_level"], "playful")
        self.assertEqual(policy["care_focus"], "practical_solution_and_risk")
        self.assertEqual(policy["response_depth"], "conversational")
        self.assertIn("不得建议把备用钥匙藏在鞋柜", policy["safety_note"])
        self.assertEqual(context.knowledge.knowledge_base_id, "education")

    async def test_persistent_profile_reaches_tutor_graph_and_turn_override_wins(self) -> None:
        store = MemoryStore(enabled=False)
        await store.update_profile(
            user_id="student-profile",
            session_id="session-profile",
            reason="test",
            changes={"birth_date": "2016-05-20"},
        )
        manager = self.manager(store)
        preference_context = await manager.prepare_context(
            user_id="student-profile",
            session_id="session-profile",
            query="以后温柔一点，默认使用普通话",
        )
        await manager.memory_skill.after_turn(
            user_id="student-profile",
            session_id="session-profile",
            user_text="以后温柔一点，默认使用普通话",
            assistant_text="好，我记住了。",
            context=preference_context,
        )

        stored = await manager.prepare_context(
            user_id="student-profile",
            session_id="session-profile",
            query="解释光合作用",
        )
        stored_policy = stored.tutor_strategy["response_policy"]
        self.assertEqual(stored_policy["persona_mode"], "gentle")
        self.assertEqual(stored_policy["dialect_mode"], "mandarin")
        self.assertEqual(stored_policy["audience_profile"]["age_stage"], "S2")

        overridden = await manager.prepare_context(
            user_id="student-profile",
            session_id="session-profile",
            query="这次幽默一点，使用北京话解释光合作用",
        )
        override_policy = overridden.tutor_strategy["response_policy"]
        self.assertEqual(override_policy["persona_mode"], "humorous")
        self.assertEqual(override_policy["dialect_mode"], "beijing")
        self.assertEqual(override_policy["audience_profile"]["age_stage"], "S2")


if __name__ == "__main__":
    unittest.main()
