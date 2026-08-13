"""Deterministic AgentSkill manager for education personalization."""

from __future__ import annotations

import json
import os
from typing import Any

from services.llm.memory import DEFAULT_PROMPT_VERSION, LearningConcept, MemoryContext, MemoryHit, MemoryStore, _current_date_context
from services.llm.agents import TutorGraph
from services.llm.skills.base import AgentSkillContext


class MemorySkill:
    """Load existing profile, history, and long-term memories."""

    name = "memory"

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    async def prepare(self, *, user_id: str, session_id: str, query: str) -> MemoryContext:
        return await self.memory_store.build_context(user_id=user_id, session_id=session_id, query=query)

    async def after_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        context: AgentSkillContext | None,
    ) -> dict[str, Any]:
        await self.memory_store.add_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
        )
        return {"saved_turn": True}


class ConceptSkill:
    """Track knowledge-point mastery and weakness state."""

    name = "concept"

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    async def prepare(self, *, user_id: str, session_id: str, query: str) -> list[LearningConcept]:
        return await self.memory_store.list_concepts(user_id=user_id, query=query, limit=8)

    async def after_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        context: AgentSkillContext | None,
    ) -> dict[str, Any]:
        updated = await self.memory_store.update_concepts_from_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
        )
        return {"updated_concepts": [item.id for item in updated]}


class TutorSkill:
    """Choose a deterministic teaching strategy from profile, memory, and concepts."""

    name = "tutor"

    def prepare(
        self,
        *,
        profile: dict[str, Any],
        memories: list[MemoryHit],
        concepts: list[LearningConcept],
        query: str,
    ) -> dict[str, Any]:
        preferences = profile.get("preferences") if isinstance(profile.get("preferences"), dict) else {}
        enjoy = " ".join(str(item) for item in preferences.get("topics_enjoy", []) if item)
        avoid = " ".join(str(item) for item in preferences.get("topics_avoid", []) if item)
        weakness_memories = [item for item in memories if item.memory_type == "weakness"]
        weak_concepts = [item for item in concepts if item.weakness_count > 0 or item.mastery_score < 0.35]

        style = "step_by_step"
        if any(marker in enjoy for marker in ("生活", "例子", "案例")):
            style = "life_examples"
        elif weakness_memories or weak_concepts:
            style = "guided_socratic"
        elif any(marker in avoid for marker in ("公式", "抽象")):
            style = "concrete_first"

        constraints = [
            "回答保持中文口语化，适合语音播放",
            "先解释思路，再给短例子",
            "不要直接堆公式，必要时用一步一步引导",
        ]
        if style == "life_examples":
            constraints.append("优先使用生活类比或具体场景")
        if style == "guided_socratic":
            constraints.append("遇到薄弱点先拆小步骤，并用一个检查理解的问题收尾")
        if style == "concrete_first":
            constraints.append("先讲具体例子，再抽象成规则")
        if query and any(marker in query for marker in ("不会", "不懂", "卡住")):
            constraints.append("用户当前表达了困难，先降低难度，不要责备")

        return {
            "style": style,
            "constraints": constraints,
            "weak_concepts": [f"{item.subject}:{item.concept}" for item in weak_concepts[:5]],
            "source": "deterministic_tutor_skill",
        }

    async def after_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        context: AgentSkillContext | None,
    ) -> dict[str, Any]:
        return {"style": (context.tutor_strategy.get("style") if context else None)}


class AgentSkillManager:
    """Prepare and persist AgentSkill context inside the LLM boundary."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store
        self.memory_skill = MemorySkill(memory_store)
        self.concept_skill = ConceptSkill(memory_store)
        self.tutor_skill = TutorSkill()
        self.agent_mode = os.getenv("LLM_AGENT_MODE", os.getenv("TUTOR_AGENT_MODE", "langgraph")).strip().lower()
        self.tutor_graph = (
            TutorGraph(memory_skill=self.memory_skill, concept_skill=self.concept_skill, tutor_skill=self.tutor_skill)
            if self.agent_mode in {"langgraph", "agent", "tutor_graph"}
            else None
        )

    async def prepare_context(self, *, user_id: str, session_id: str, query: str) -> AgentSkillContext:
        if self.tutor_graph is not None:
            return await self.tutor_graph.prepare(user_id=user_id, session_id=session_id, query=query)

        memory_context = await self.memory_skill.prepare(user_id=user_id, session_id=session_id, query=query)
        concepts = await self.concept_skill.prepare(user_id=user_id, session_id=session_id, query=query)
        strategy = self.tutor_skill.prepare(
            profile=memory_context.profile,
            memories=memory_context.memories,
            concepts=concepts,
            query=query,
        )
        metadata = {
            "skills": ["memory", "profile", "concept", "tutor"],
            "agent": "AgentSkillManager",
            "agent_mode": "skills",
            "langgraph_enabled": False,
            "langchain_enabled": False,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "user_id": user_id,
            "profile_fields": len(memory_context.profile),
            "history_count": len(memory_context.history),
            "memory_count": len(memory_context.memories),
            "concept_count": len(concepts),
            "tutor_style": strategy.get("style"),
        }
        return AgentSkillContext(
            memory=memory_context,
            concepts=concepts,
            tutor_strategy=strategy,
            skill_metadata=metadata,
        )

    async def after_turn(
        self,
        *,
        request_id: str,
        trace_id: str,
        session_id: str,
        user_id: str,
        user_text: str,
        assistant_text: str,
        context: AgentSkillContext | None,
        latency_ms: int | None = None,
        timings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_result = await self.memory_skill.after_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            context=context,
        )

        concept_result = await self.concept_skill.after_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            context=context,
        )

        tutor_result = await self.tutor_skill.after_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            context=context,
        )
        return {
            "memory": memory_result,
            "concept": concept_result,
            "tutor": tutor_result,
        }

    def render_system_prompt(self, context: AgentSkillContext) -> str:
        profile_text = json.dumps(context.memory.profile, ensure_ascii=False)[:180] if context.memory.profile else "无"
        history_text = "\n".join(
            f"{_role_name(item['role'])}: {str(item['content'])[:60]}"
            for item in context.memory.history[-3:]
        ) or "无"
        memory_text = "\n".join(
            f"- {item.content[:50]}"
            for item in context.memory.memories[:3]
        ) or "无"
        concept_text = "\n".join(
            f"- {item.subject}:{item.concept} 掌握{item.mastery_score:.2f}"
            for item in context.concepts[:3]
        ) or "无"
        strategy_text = "; ".join(str(item)[:40] for item in context.tutor_strategy.get("constraints", [])[:2]) or "清晰、简短"
        intent = context.tutor_strategy.get("intent") if isinstance(context.tutor_strategy.get("intent"), dict) else {}
        route = context.tutor_strategy.get("route") or "concept_explanation"
        intent_text = (
            f"类别={intent.get('category', 'explain')}；"
            f"学科={intent.get('subject', 'unknown')}；"
            f"路由={route}"
        )

        return (
            f"prompt_version: {DEFAULT_PROMPT_VERSION}\n"
            "你是小维，AI 教育语音助手。用简体中文回答，适合朗读，不提 DeepSeek。\n"
            f"当前日期:{_current_date_context()}。涉及今年、最新、赛事、新闻或可能变化的信息时，必须优先依据实时工具结果；没有实时工具结果时要明确说明不确定。\n"
            f"用户画像:{profile_text}\n"
            f"近期对话:{history_text}\n"
            f"相关长期记忆:{memory_text}\n"
            f"知识点掌握状态:{concept_text}\n"
            f"本轮 Agent 判断:{intent_text}\n"
            f"教学策略:{context.tutor_strategy.get('style', 'step_by_step')}; {strategy_text}"
        )


def _role_name(role: str) -> str:
    if role == "assistant":
        return "小维"
    if role == "system":
        return "系统"
    return "学生"
