"""Regression tests for the agent02 05-no-app Agent upgrade."""

from __future__ import annotations

import asyncio
import unittest

from app_protocols.persona import XIAOWEI_PERSONA_VERSION
from services.knowledge.schemas import KnowledgeChunkHit, KnowledgeContext, KnowledgeGraphRelation
from services.llm.agents.tutor_graph import choose_route, classify_intent
from services.llm.memory import MemoryStore
from services.llm.skills import AgentSkillManager
from services.llm.tools import CalculatorTool, UnitConverterTool


class RecordingKnowledgeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        **kwargs: object,
    ) -> KnowledgeContext:
        self.calls += 1
        return KnowledgeContext(
            knowledge_base_id=knowledge_base_id,
            query=query,
            chunks=[
                KnowledgeChunkHit(
                    chunk_id="chunk-glass",
                    document_id="doc-physics",
                    title="热胀冷缩",
                    text="玻璃受热不均会产生热应力。",
                    source_name="物理知识库",
                    score=0.96,
                )
            ],
            relations=[
                KnowledgeGraphRelation(
                    source_id="thermal-stress",
                    source_name="热应力",
                    relation_type="MAY_CAUSE",
                    target_id="glass-fracture",
                    target_name="玻璃破裂",
                )
            ],
        )


class Agent0205UpgradeTests(unittest.IsolatedAsyncioTestCase):
    def test_persona_and_scene_routes_match_05(self) -> None:
        self.assertEqual(XIAOWEI_PERSONA_VERSION, "xiaowei-persona-v6")
        self.assertEqual(classify_intent({"query": "好的"})["category"], "followup")
        self.assertEqual(
            classify_intent({"query": "验证码能不能发给同学"})["category"],
            "account_security",
        )
        knowledge_intent = classify_intent({"query": "为什么玻璃杯倒入热水后可能炸裂"})
        self.assertEqual(knowledge_intent["category"], "knowledge_exploration")
        self.assertEqual(
            choose_route(intent=knowledge_intent, memories=[], concepts=[]),
            "knowledge_weaving",
        )
        self.assertEqual(
            classify_intent({"query": "2026年世界杯最新赛程"})["category"],
            "realtime",
        )

    async def test_knowledge_weaving_keeps_existing_graphrag_context(self) -> None:
        knowledge = RecordingKnowledgeClient()
        manager = AgentSkillManager(MemoryStore(enabled=False), knowledge)  # type: ignore[arg-type]

        context = await manager.prepare_context(
            user_id="student-05",
            session_id="knowledge-05",
            query="为什么玻璃杯倒入热水后可能炸裂",
            knowledge_base_id="education",
        )
        prompt = manager.render_system_prompt(context)

        self.assertEqual(context.tutor_strategy["active_skill"]["id"], "knowledge_weaving")
        self.assertEqual(knowledge.calls, 1)
        self.assertEqual(context.skill_metadata["knowledge_chunk_count"], 1)
        self.assertEqual(context.skill_metadata["knowledge_relation_count"], 1)
        self.assertFalse(context.skill_metadata["knowledge_retrieval_skipped"])
        self.assertIn("优先使用本轮知识图谱关系和参考资料", prompt)
        self.assertIn("热应力 --MAY_CAUSE--> 玻璃破裂", prompt)
        self.assertNotIn("当前尚未接入正式知识图谱", prompt)

    async def test_non_tutoring_scenes_skip_graphrag_retrieval(self) -> None:
        for query, category in (
            ("你好", "casual"),
            ("我今天又忘带钥匙了，怎么办", "daily_life"),
            ("2026年世界杯最新赛程", "realtime"),
        ):
            with self.subTest(query=query):
                knowledge = RecordingKnowledgeClient()
                manager = AgentSkillManager(MemoryStore(enabled=False), knowledge)  # type: ignore[arg-type]

                context = await manager.prepare_context(
                    user_id="latency-user",
                    session_id="latency-session",
                    query=query,
                )

                self.assertEqual(context.tutor_strategy["intent"]["category"], category)
                self.assertEqual(knowledge.calls, 0)
                self.assertEqual(context.skill_metadata["knowledge_chunk_count"], 0)
                self.assertTrue(context.skill_metadata["knowledge_retrieval_skipped"])
                self.assertEqual(
                    context.skill_metadata["knowledge_retrieval_skip_reason"],
                    f"intent:{category}",
                )

    async def test_fallback_skill_manager_skips_unneeded_graphrag(self) -> None:
        knowledge = RecordingKnowledgeClient()
        manager = AgentSkillManager(MemoryStore(enabled=False), knowledge)  # type: ignore[arg-type]
        manager.tutor_graph = None

        context = await manager.prepare_context(
            user_id="fallback-user",
            session_id="fallback-session",
            query="你好",
        )

        self.assertEqual(knowledge.calls, 0)
        self.assertTrue(context.skill_metadata["knowledge_retrieval_skipped"])
        self.assertEqual(
            context.skill_metadata["knowledge_retrieval_skip_reason"],
            "intent:casual",
        )

    async def test_followup_uses_local_history_without_requerying_knowledge(self) -> None:
        store = MemoryStore(enabled=False)
        store.cache_message(
            user_id="student-05",
            session_id="followup-05",
            role="user",
            content="为什么玻璃杯倒入热水后可能炸裂",
        )
        store.cache_message(
            user_id="student-05",
            session_id="followup-05",
            role="assistant",
            content="受热不均会产生热应力。你想继续看看玻璃材质的差异，和预热为什么能降低风险吗？",
        )
        knowledge = RecordingKnowledgeClient()
        manager = AgentSkillManager(store, knowledge)  # type: ignore[arg-type]

        context = await manager.prepare_short_term_context(
            user_id="student-05",
            session_id="followup-05",
            query="好的",
            knowledge_base_id="education",
        )
        prompt = manager.render_system_prompt(context)

        self.assertEqual(context.tutor_strategy["intent"]["category"], "followup")
        self.assertTrue(context.skill_metadata["tutoring_strategy_skipped"])
        self.assertEqual(knowledge.calls, 0)
        self.assertIn("受热不均会产生热应力", prompt)
        self.assertIn("本轮承接指令（最高优先级）", prompt)

    def test_native_math_and_unit_text_are_supported(self) -> None:
        calculator = CalculatorTool()
        square_root = calculator.calculate_from_text("144的平方根是多少")
        multiplication = calculator.calculate_from_text("13乘以27等于多少")
        conversion = UnitConverterTool().convert_from_text("3公里是几米")

        self.assertIsNotNone(square_root)
        self.assertEqual(square_root.result_text, "12")  # type: ignore[union-attr]
        self.assertIsNotNone(multiplication)
        self.assertEqual(multiplication.result_text, "351")  # type: ignore[union-attr]
        self.assertIsNotNone(conversion)
        self.assertEqual(conversion.result_text, "3000")  # type: ignore[union-attr]

    async def test_after_turn_skills_persist_in_parallel(self) -> None:
        manager = AgentSkillManager(MemoryStore(enabled=False))

        class BlockingSkill:
            def __init__(self, result: dict[str, object]) -> None:
                self.result = result
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def after_turn(self, **kwargs: object) -> dict[str, object]:
                self.started.set()
                await self.release.wait()
                return self.result

        memory_skill = BlockingSkill({"saved_turn": True})
        concept_skill = BlockingSkill({"updated_concepts": []})
        tutor_skill = BlockingSkill({"style": "concise"})
        manager.memory_skill = memory_skill  # type: ignore[assignment]
        manager.concept_skill = concept_skill  # type: ignore[assignment]
        manager.tutor_skill = tutor_skill  # type: ignore[assignment]

        persistence = asyncio.create_task(
            manager.after_turn(
                request_id="parallel",
                trace_id="parallel",
                session_id="parallel",
                user_id="parallel",
                user_text="你好",
                assistant_text="你好呀",
                context=None,
            )
        )
        await asyncio.wait_for(
            asyncio.gather(
                memory_skill.started.wait(),
                concept_skill.started.wait(),
                tutor_skill.started.wait(),
            ),
            timeout=0.5,
        )
        memory_skill.release.set()
        concept_skill.release.set()
        tutor_skill.release.set()

        result = await asyncio.wait_for(persistence, timeout=0.5)

        self.assertEqual(result["memory"], {"saved_turn": True})
        self.assertEqual(result["concept"], {"updated_concepts": []})
        self.assertEqual(result["tutor"], {"style": "concise"})

    async def test_casual_after_turn_skips_memory_and_concept_extraction(self) -> None:
        store = MemoryStore(enabled=False)
        manager = AgentSkillManager(store)
        captured: dict[str, object] = {}

        async def record_add_turn(**kwargs: object) -> None:
            captured.update(kwargs)

        store.add_turn = record_add_turn  # type: ignore[method-assign]
        context = await manager.prepare_context(
            user_id="casual-user",
            session_id="casual-session",
            query="你好",
        )

        result = await manager.after_turn(
            request_id="casual",
            trace_id="casual",
            session_id="casual-session",
            user_id="casual-user",
            user_text="你好",
            assistant_text="你好呀",
            context=context,
            history_already_saved=True,
        )

        self.assertFalse(captured["extract_memory"])
        self.assertTrue(result["memory"]["memory_extraction_skipped"])
        self.assertTrue(result["concept"]["concept_tracking_skipped"])


if __name__ == "__main__":
    unittest.main()
