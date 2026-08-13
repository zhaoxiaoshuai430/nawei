"""Tests for LLM-owned persistent memory and AgentSkill helpers."""

from __future__ import annotations

import unittest

from app_protocols import LLMRequest, Message, MessageRole
from services.llm.agent_runtime import LLMAgentRuntime
from services.llm.memory import (
    MemoryStore,
    _extract_concept_signals,
    _extract_memory_candidates,
    _extract_memory_records,
    _extract_profile_changes,
    _jsonb_to_dict,
    _merge_memory_records,
    _parse_llm_memory_extraction,
)
from services.llm.skills import AgentSkillManager


class MemoryServiceTest(unittest.TestCase):
    def test_jsonb_to_dict_accepts_asyncpg_string_json(self) -> None:
        self.assertEqual(_jsonb_to_dict('{"nickname":"小明"}'), {"nickname": "小明"})

    def test_jsonb_to_dict_accepts_dict(self) -> None:
        self.assertEqual(_jsonb_to_dict({"nickname": "小明"}), {"nickname": "小明"})

    def test_extracts_profile_and_memory_from_student_statement(self) -> None:
        text = "我叫小明，我最近在学二次函数，我喜欢数学例题"

        changes = _extract_profile_changes(text)
        memories = _extract_memory_candidates(text)

        self.assertEqual(changes["nickname"], "小明")
        self.assertEqual(changes["current_focus"], "正在学习二次函数")
        self.assertEqual(changes["preferences"]["topics_enjoy"], ["数学例题"])
        self.assertTrue(any("我叫小明" in content for content, _ in memories))
        self.assertTrue(any("二次函数" in content for content, _ in memories))

    def test_memory_question_does_not_overwrite_focus(self) -> None:
        self.assertEqual(_extract_profile_changes("你记得我在学什么吗？"), {})
        self.assertEqual(_extract_profile_changes("你还记得我叫什么吗？"), {})
        self.assertEqual(_extract_memory_candidates("你记得我在学什么吗？"), [])
        self.assertEqual(_extract_memory_candidates("你还记得我叫什么、最近在学什么吗？"), [])



    def test_extracts_grade_subject_and_weakness(self) -> None:
        text = "\u6211\u53eb\u963f\u6770\uff0c\u521d\u4e8c\uff0c\u6700\u8fd1\u5728\u5b66\u4e00\u6b21\u51fd\u6570\u56fe\u50cf\uff0c\u6211\u603b\u662f\u4e0d\u4f1a\u770b\u659c\u7387\uff0c\u6211\u559c\u6b22\u7528\u751f\u6d3b\u4f8b\u5b50\u8bb2\u3002"

        changes = _extract_profile_changes(text)
        records = _extract_memory_records(text)

        self.assertEqual(changes["nickname"], "\u963f\u6770")
        self.assertEqual(changes["grade"], "\u521d\u4e8c")
        self.assertEqual(changes["subjects"], ["\u6570\u5b66"])
        self.assertTrue(any(record["memory_type"] == "weakness" and "\u659c\u7387" in record["content"] for record in records))

    def test_merge_memory_records_deduplicates_similar_weaknesses(self) -> None:
        merged = _merge_memory_records(
            [
                {
                    "content": "\u6211\u603b\u662f\u4e0d\u4f1a\u770b\u659c\u7387",
                    "memory_type": "weakness",
                    "importance": 8,
                    "confidence": 0.85,
                    "tags": ["weakness"],
                },
                {
                    "content": "\u6211\u53eb\u963f\u6770",
                    "memory_type": "profile_fact",
                    "importance": 8,
                    "confidence": 0.85,
                    "tags": ["profile_fact"],
                }
            ],
            [
                {
                    "content": "\u4e0d\u4f1a\u770b\u4e00\u6b21\u51fd\u6570\u659c\u7387",
                    "memory_type": "weakness",
                    "importance": 8,
                    "confidence": 1.0,
                    "tags": ["math", "weakness"],
                    "extractor": "llm",
                },
                {
                    "content": "\u5b66\u751f\u6635\u79f0: \u963f\u6770",
                    "memory_type": "profile_fact",
                    "importance": 8,
                    "confidence": 1.0,
                    "tags": ["identity", "nickname"],
                    "extractor": "llm",
                }
            ],
        )

        self.assertEqual(len(merged), 2)
        weakness = next(record for record in merged if record["memory_type"] == "weakness")
        profile = next(record for record in merged if record["memory_type"] == "profile_fact")
        self.assertEqual(weakness["content"], "\u4e0d\u4f1a\u770b\u4e00\u6b21\u51fd\u6570\u659c\u7387")
        self.assertEqual(weakness["confidence"], 1.0)
        self.assertIn("math", weakness["tags"])
        self.assertEqual(profile["content"], "\u5b66\u751f\u6635\u79f0: \u963f\u6770")
        self.assertIn("identity", profile["tags"])

    def test_parse_llm_memory_extraction_sanitizes_records(self) -> None:
        raw = """
        ```json
        {
          "profile": {"nickname": "\u5c0f\u660e", "current_focus": "\u6b63\u5728\u5b66\u4e60\u4e8c\u6b21\u51fd\u6570"},
          "memories": [
            {"content":"\u6211\u6700\u8fd1\u5728\u5b66\u4e8c\u6b21\u51fd\u6570", "memory_type":"learning_state", "importance": 8, "confidence": 0.9, "tags":["math"]},
            {"content":"\u4f60\u8fd8\u8bb0\u5f97\u6211\u53eb\u4ec0\u4e48\u5417", "memory_type":"profile_fact", "importance": 8, "confidence": 0.9}
          ]
        }
        ```
        """

        parsed = _parse_llm_memory_extraction(raw)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["profile"]["nickname"], "\u5c0f\u660e")
        self.assertEqual(len(parsed["memories"]), 1)
        self.assertEqual(parsed["memories"][0]["memory_type"], "learning_state")
        self.assertIn("math", parsed["memories"][0]["tags"])

    def test_extracts_concept_weakness_signal(self) -> None:
        signals = _extract_concept_signals("\u6211\u603b\u662f\u4e0d\u4f1a\u770b\u4e00\u6b21\u51fd\u6570\u659c\u7387")

        self.assertTrue(signals)
        self.assertEqual(signals[0]["subject"], "\u6570\u5b66")
        self.assertEqual(signals[0]["event_type"], "weakness")
        self.assertTrue(any("\u659c\u7387" in signal["concept"] or "\u4e00\u6b21\u51fd\u6570" in signal["concept"] for signal in signals))


class MemoryStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_agent_runtime_selects_and_executes_calculator_tool(self) -> None:
        runtime = LLMAgentRuntime(MemoryStore(enabled=False))
        turn = await runtime.prepare(
            LLMRequest(
                request_id="tool-request",
                trace_id="tool-trace",
                session_id="tool-session",
                user_id="tool-user",
                messages=[Message(role=MessageRole.USER, content="请计算 2*(3+4)")],
                stream=False,
            )
        )

        self.assertIn("calculator", turn.metadata["tools"]["selected_tools"])
        self.assertEqual(turn.metadata["tools"]["results"]["calculator"]["result"], 14.0)
        self.assertIn("本轮可靠工具结果", turn.request.messages[0].content)

    async def test_in_memory_manual_remember_inspect_and_clear(self) -> None:
        store = MemoryStore(enabled=False)

        await store.remember(
            user_id="u1",
            session_id="s1",
            content="学生最近在学习二次函数，顶点式理解比较薄弱。",
            importance=7,
        )
        await store.update_profile(
            user_id="u1",
            session_id="s1",
            changes={"current_focus": "正在学习二次函数"},
            reason="test",
        )
        snapshot = await store.inspect(user_id="u1", session_id="s1", query="二次函数", limit=5)

        self.assertEqual(snapshot["profile"]["current_focus"], "正在学习二次函数")
        self.assertEqual(len(snapshot["memories"]), 1)
        self.assertIn("二次函数", snapshot["memories"][0]["content"])

        await store.clear_user(user_id="u1", session_id="s1")
        snapshot = await store.inspect(user_id="u1", session_id="s1", query="二次函数", limit=5)
        self.assertEqual(snapshot["profile"], {})
        self.assertEqual(snapshot["memories"], [])

    async def test_in_memory_governed_remember_merges_and_forgets_one_item(self) -> None:
        store = MemoryStore(enabled=False)

        first_id = await store.remember(
            user_id="u2",
            session_id="s2",
            content="\u6211\u6700\u8fd1\u5728\u5b66\u51fd\u6570\u56fe\u50cf",
            memory_type="learning_state",
            importance=6,
            confidence=0.7,
            tags=["math"],
        )
        second_id = await store.remember(
            user_id="u2",
            session_id="s2",
            content="\u6211\u6700\u8fd1\u5728\u5b66\u51fd\u6570\u56fe\u50cf",
            memory_type="learning_state",
            importance=8,
            confidence=0.9,
            tags=["weakness"],
        )

        self.assertEqual(first_id, second_id)
        snapshot = await store.inspect(user_id="u2", session_id="s2", query="\u51fd\u6570", limit=5)
        self.assertEqual(snapshot["stats"]["active_memories"], 1)
        memory = snapshot["memories"][0]
        self.assertEqual(memory["importance"], 8)
        self.assertEqual(memory["confidence"], 0.9)
        self.assertIn("math", memory["tags"])
        self.assertIn("weakness", memory["tags"])
        self.assertGreaterEqual(memory["usage_count"], 1)

        deleted = await store.forget_memory(user_id="u2", memory_id=str(first_id), reason="wrong")
        self.assertTrue(deleted)
        snapshot = await store.inspect(user_id="u2", session_id="s2", query="\u51fd\u6570", limit=5)
        self.assertEqual(snapshot["stats"]["active_memories"], 0)
        self.assertEqual(snapshot["memories"], [])

    async def test_sensitive_memory_is_marked(self) -> None:
        store = MemoryStore(enabled=False)

        memory_id = await store.remember(
            user_id="u3",
            session_id="s3",
            content="\u6211\u7684\u624b\u673a\u53f7\u662f123456",
            importance=8,
            sensitivity="normal",
        )

        self.assertIsNotNone(memory_id)
        snapshot = await store.inspect(user_id="u3", session_id="s3", query="\u624b\u673a\u53f7", limit=5)
        self.assertEqual(snapshot["memories"][0]["sensitivity"], "sensitive")

    async def test_concept_mastery_updates_from_turn(self) -> None:
        store = MemoryStore(enabled=False)

        updated = await store.update_concepts_from_turn(
            user_id="u4",
            session_id="s4",
            user_text="\u6211\u603b\u662f\u4e0d\u4f1a\u770b\u4e00\u6b21\u51fd\u6570\u659c\u7387",
            assistant_text="\u6211\u4eec\u7528\u722c\u5761\u7684\u4f8b\u5b50\u6765\u8bb2\u3002",
        )
        concepts = await store.list_concepts(user_id="u4", query="\u659c\u7387", limit=5)

        self.assertTrue(updated)
        self.assertTrue(concepts)
        self.assertGreaterEqual(concepts[0].weakness_count, 1)
        self.assertLess(concepts[0].mastery_score, 0.5)

    async def test_agent_skill_manager_renders_personalized_prompt(self) -> None:
        store = MemoryStore(enabled=False)
        manager = AgentSkillManager(store)

        await store.update_profile(
            user_id="u5",
            session_id="s5",
            changes={"preferences": {"topics_enjoy": ["\u751f\u6d3b\u4f8b\u5b50"]}},
            reason="test",
        )
        await store.update_concepts_from_turn(
            user_id="u5",
            session_id="s5",
            user_text="\u6211\u4e0d\u4f1a\u770b\u659c\u7387",
        )

        context = await manager.prepare_context(user_id="u5", session_id="s5", query="\u659c\u7387")
        prompt = manager.render_system_prompt(context)
        await manager.after_turn(
            request_id="r5",
            trace_id="t5",
            session_id="s5",
            user_id="u5",
            user_text="\u6211\u4e0d\u4f1a\u770b\u659c\u7387",
            assistant_text="\u7528\u751f\u6d3b\u4f8b\u5b50\u8bb2\u659c\u7387\u3002",
            context=context,
        )

        self.assertEqual(context.tutor_strategy["style"], "life_examples")
        self.assertIn("\u77e5\u8bc6\u70b9\u638c\u63e1\u72b6\u6001", prompt)


if __name__ == "__main__":
    unittest.main()
