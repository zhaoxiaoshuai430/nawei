"""Deterministic AgentSkill manager for education personalization."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from app_protocols.persona import format_response_policy, format_xiaowei_system_prompt
from services.knowledge.client import KnowledgeClient
from services.knowledge.schemas import GraphCandidate, KnowledgeContext, KnowledgeExpansionOptions
from services.llm.memory import DEFAULT_PROMPT_VERSION, LearningConcept, MemoryContext, MemoryHit, MemoryStore, _current_date_context
from services.llm.agents import TutorGraph
from services.llm.agents.tutor_graph import (
    build_skipped_knowledge_context,
    build_non_tutoring_strategy,
    build_response_policy,
    choose_route,
    classify_intent,
    intent_constraints,
    is_contextual_followup,
    knowledge_anchor_terms,
    should_retrieve_knowledge,
    should_skip_tutoring,
)
from services.llm.agents.semantic_router import build_semantic_route
from services.llm.skills.base import AgentSkillContext
from services.llm.skills.teaching import (
    TeachingSkillRegistry,
    format_active_skill_prompt,
    select_teaching_methods,
)


def _context_intent(context: AgentSkillContext | None) -> dict[str, Any]:
    if context is None:
        return {}
    value = context.tutor_strategy.get("intent")
    return value if isinstance(value, dict) else {}


def _context_intent_category(context: AgentSkillContext | None) -> str:
    return str(_context_intent(context).get("category") or "unknown")


class MemorySkill:
    """Load existing profile, history, and long-term memories."""

    name = "memory"

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    async def prepare(self, *, user_id: str, session_id: str, query: str) -> MemoryContext:
        return await self.memory_store.build_context(user_id=user_id, session_id=session_id, query=query)

    async def prepare_short_term(self, *, user_id: str, session_id: str, query: str) -> MemoryContext:
        followup = is_contextual_followup(query)
        return await self.memory_store.build_short_term_context(
            user_id=user_id,
            session_id=session_id,
            query=query,
            include_profile=not followup,
            prefer_local_history=followup,
        )

    async def after_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        context: AgentSkillContext | None,
        history_already_saved: bool = False,
    ) -> dict[str, Any]:
        category = _context_intent_category(context)
        extract_memory = category not in {
            "crisis_support",
            "academic_integrity",
            "account_security",
            "followup",
            "identity",
            "casual",
            "realtime",
        }
        await self.memory_store.add_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            persist_messages=not history_already_saved,
            extract_memory=extract_memory,
        )
        return {
            "saved_turn": True,
            "memory_extraction_skipped": not extract_memory,
            "skip_reason": f"intent:{category}" if not extract_memory else None,
        }


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
        intent = _context_intent(context)
        if intent and not bool(intent.get("needs_concept_tracking", False)):
            return {
                "updated_concepts": [],
                "concept_tracking_skipped": True,
                "skip_reason": f"intent:{intent.get('category') or 'unknown'}",
            }
        updated = await self.memory_store.update_concepts_from_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
        )
        return {"updated_concepts": [item.id for item in updated]}


class KnowledgeSkill:
    """Retrieve public knowledge chunks and graph relations."""

    name = "knowledge"

    def __init__(self, client: KnowledgeClient) -> None:
        self.client = client

    async def prepare(
        self,
        *,
        knowledge_base_id: str,
        query: str,
        anchor_terms: list[str] | None = None,
        graph_candidates: list[GraphCandidate] | None = None,
        expansion: KnowledgeExpansionOptions | None = None,
    ) -> KnowledgeContext:
        return await self.client.retrieve(
            knowledge_base_id=knowledge_base_id,
            query=query,
            top_k=int(os.getenv("KNOWLEDGE_CHUNK_TOP_K", "8")),
            graph_depth=int(os.getenv("KNOWLEDGE_GRAPH_MAX_DEPTH", "2")),
            anchor_terms=anchor_terms,
            graph_candidates=graph_candidates,
            expansion=expansion,
        )


class TutorSkill:
    """Choose a teaching style, primary Skill, and composable teaching methods."""

    name = "tutor"

    def __init__(self, registry: TeachingSkillRegistry | None = None) -> None:
        self.registry = registry or TeachingSkillRegistry()

    def prepare(
        self,
        *,
        profile: dict[str, Any],
        memories: list[MemoryHit],
        concepts: list[LearningConcept],
        query: str,
        intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preferences_value = profile.get("preferences")
        preferences: dict[str, Any] = preferences_value if isinstance(preferences_value, dict) else {}
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

        active_skill = self.registry.select(
            query=query,
            intent=intent,
            memories=memories,
            concepts=concepts,
        )
        teaching_methods = select_teaching_methods(
            query=query,
            intent=intent,
            memories=memories,
            concepts=concepts,
        )
        if "guided_socratic" in teaching_methods:
            constraints.append("启发式引导每次只推进一个关键问题，提示逐级变明确")

        return {
            "style": style,
            "constraints": constraints,
            "weak_concepts": [f"{item.subject}:{item.concept}" for item in weak_concepts[:5]],
            "active_skill": active_skill,
            "teaching_methods": teaching_methods,
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

    def __init__(self, memory_store: MemoryStore, knowledge_client: KnowledgeClient | None = None) -> None:
        self.memory_store = memory_store
        self.knowledge_client = knowledge_client or KnowledgeClient()
        self.memory_skill = MemorySkill(memory_store)
        self.concept_skill = ConceptSkill(memory_store)
        self.knowledge_skill = KnowledgeSkill(self.knowledge_client)
        self.tutor_skill = TutorSkill()
        self.agent_mode = os.getenv("LLM_AGENT_MODE", os.getenv("TUTOR_AGENT_MODE", "langgraph")).strip().lower()
        self.tutor_graph = (
            TutorGraph(
                memory_skill=self.memory_skill,
                concept_skill=self.concept_skill,
                knowledge_skill=self.knowledge_skill,
                tutor_skill=self.tutor_skill,
            )
            if self.agent_mode in {"langgraph", "agent", "tutor_graph"}
            else None
        )

    async def prepare_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        knowledge_base_id: str = "default",
    ) -> AgentSkillContext:
        if self.tutor_graph is not None:
            return await self.tutor_graph.prepare(
                user_id=user_id,
                session_id=session_id,
                query=query,
                knowledge_base_id=knowledge_base_id,
            )

        intent = classify_intent({"query": query, "profile": {}, "concepts": []})
        retrieve_knowledge = should_retrieve_knowledge(intent)
        semantic_route = build_semantic_route(query=query, intent=intent)
        graph_candidates = [
            GraphCandidate(**item) for item in semantic_route.get("graph_candidates") or []
        ]
        expansion = KnowledgeExpansionOptions(**(semantic_route.get("expansion") or {}))
        memory_task = self.memory_skill.prepare(user_id=user_id, session_id=session_id, query=query)
        concept_task = (
            self.concept_skill.prepare(user_id=user_id, session_id=session_id, query=query)
            if intent.get("needs_concept_tracking")
            else None
        )
        knowledge_task = (
            self.knowledge_skill.prepare(
                knowledge_base_id=knowledge_base_id,
                query=query,
                anchor_terms=knowledge_anchor_terms(intent),
                graph_candidates=graph_candidates,
                expansion=expansion,
            )
            if retrieve_knowledge
            else None
        )
        if concept_task is not None and knowledge_task is not None:
            memory_context, concepts, knowledge_context = await asyncio.gather(
                memory_task,
                concept_task,
                knowledge_task,
            )
        elif concept_task is not None:
            memory_context, concepts = await asyncio.gather(memory_task, concept_task)
            knowledge_context = build_skipped_knowledge_context(
                knowledge_base_id=knowledge_base_id,
                query=query,
                intent=intent,
            )
        elif knowledge_task is not None:
            memory_context, knowledge_context = await asyncio.gather(
                memory_task,
                knowledge_task,
            )
            concepts = []
        else:
            memory_context = await memory_task
            concepts = []
            knowledge_context = build_skipped_knowledge_context(
                knowledge_base_id=knowledge_base_id,
                query=query,
                intent=intent,
            )
        return self._assemble_context(
            user_id=user_id,
            query=query,
            memory_context=memory_context,
            concepts=concepts,
            knowledge_context=knowledge_context,
            agent_mode="skills",
        )

    async def prepare_short_term_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        knowledge_base_id: str = "default",
    ) -> AgentSkillContext:
        if self.tutor_graph is not None:
            return await self.tutor_graph.prepare_short_term(
                user_id=user_id,
                session_id=session_id,
                query=query,
                knowledge_base_id=knowledge_base_id,
            )

        intent = classify_intent({"query": query, "profile": {}, "concepts": []})
        memory_task = asyncio.create_task(
            self.memory_skill.prepare_short_term(user_id=user_id, session_id=session_id, query=query)
        )
        if not should_retrieve_knowledge(intent):
            memory_context = await memory_task
            knowledge_context = build_skipped_knowledge_context(
                knowledge_base_id=knowledge_base_id,
                query=query,
                intent=intent,
            )
        else:
            semantic_route = build_semantic_route(query=query, intent=intent)
            graph_candidates = [
                GraphCandidate(**item) for item in semantic_route.get("graph_candidates") or []
            ]
            expansion = KnowledgeExpansionOptions(**(semantic_route.get("expansion") or {}))
            memory_context, knowledge_context = await asyncio.gather(
                memory_task,
                self.knowledge_skill.prepare(
                    knowledge_base_id=knowledge_base_id,
                    query=query,
                    anchor_terms=knowledge_anchor_terms(intent),
                    graph_candidates=graph_candidates,
                    expansion=expansion,
                ),
            )
        return self._assemble_context(
            user_id=user_id,
            query=query,
            memory_context=memory_context,
            concepts=[],
            knowledge_context=knowledge_context,
            agent_mode="short_term",
        )

    async def prepare_conversation_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        knowledge_base_id: str,
        memory_mode: str,
        memory_timeout: float,
    ) -> AgentSkillContext:
        intent = classify_intent({"query": query, "profile": {}, "concepts": []})
        semantic_route = build_semantic_route(query=query, intent=intent)
        memory_status = {"mode": memory_mode, "timed_out": False, "error": None}

        async def load_memory() -> tuple[MemoryContext, list[LearningConcept]]:
            empty = MemoryContext(user_id=user_id, profile={}, history=[], memories=[])
            if memory_mode == "off":
                return empty, []

            async def prepare_memory() -> tuple[MemoryContext, list[LearningConcept]]:
                if memory_mode == "lightweight":
                    context = await self.memory_skill.prepare_short_term(
                        user_id=user_id, session_id=session_id, query=query,
                    )
                    return context, []
                await self.memory_skill.memory_store.initialize()
                context = await self.memory_skill.prepare(
                    user_id=user_id, session_id=session_id, query=query,
                )
                concepts = await self.concept_skill.prepare(
                    user_id=user_id, session_id=session_id, query=query,
                ) if intent.get("needs_concept_tracking") else []
                return context, concepts

            try:
                return await asyncio.wait_for(prepare_memory(), timeout=memory_timeout)
            except asyncio.TimeoutError:
                memory_status["timed_out"] = True
            except Exception as exc:
                memory_status["error"] = type(exc).__name__
            return empty, []

        async def load_knowledge() -> KnowledgeContext:
            if not should_retrieve_knowledge(intent):
                return build_skipped_knowledge_context(
                    knowledge_base_id=knowledge_base_id, query=query, intent=intent,
                )
            return await self.knowledge_skill.prepare(
                knowledge_base_id=knowledge_base_id,
                query=query,
                anchor_terms=knowledge_anchor_terms(intent),
                graph_candidates=[GraphCandidate(**item) for item in semantic_route.get("graph_candidates") or []],
                expansion=KnowledgeExpansionOptions(**(semantic_route.get("expansion") or {})),
            )

        (memory_context, concepts), knowledge_context = await asyncio.gather(load_memory(), load_knowledge())
        context_mode = "short_term" if memory_mode == "lightweight" else memory_mode
        if self.tutor_graph is not None:
            context = await self.tutor_graph._prepare(
                user_id=user_id, session_id=session_id, query=query,
                knowledge_base_id=knowledge_base_id, context_mode=context_mode,
                prepared_context={
                    "memory_context": memory_context, "concepts": concepts,
                    "knowledge_context": knowledge_context, "intent": intent,
                    "semantic_route": semantic_route,
                },
            )
        else:
            context = self._assemble_context(
                user_id=user_id, query=query, memory_context=memory_context,
                concepts=concepts, knowledge_context=knowledge_context, agent_mode=context_mode,
            )
        context.skill_metadata["memory_preparation"] = memory_status
        if memory_mode == "off":
            context.skill_metadata["skills"] = [
                skill for skill in context.skill_metadata.get("skills", [])
                if skill not in {"memory", "short_term_memory", "profile", "concept"}
            ]
        return context

    async def prepare_stateless_context(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        knowledge_base_id: str = "default",
    ) -> AgentSkillContext:
        """Select the turn's Skill without reading memory or knowledge storage."""

        memory_context = MemoryContext(user_id=user_id, profile={}, history=[], memories=[])
        knowledge_context = KnowledgeContext(knowledge_base_id=knowledge_base_id, query=query)
        return self._assemble_context(
            user_id=user_id,
            query=query,
            memory_context=memory_context,
            concepts=[],
            knowledge_context=knowledge_context,
            agent_mode="stateless",
        )

    def _assemble_context(
        self,
        *,
        user_id: str,
        query: str,
        memory_context: MemoryContext,
        concepts: list[LearningConcept],
        knowledge_context: KnowledgeContext,
        agent_mode: str,
    ) -> AgentSkillContext:
        intent = classify_intent(
            {"query": query, "profile": memory_context.profile, "concepts": concepts}
        )
        response_policy = build_response_policy(
            query=query,
            profile=memory_context.profile,
            intent=intent,
            history_count=len(memory_context.history),
        )
        tutoring_strategy_skipped = should_skip_tutoring(intent)
        if tutoring_strategy_skipped:
            strategy = build_non_tutoring_strategy(
                intent=intent,
                memories=memory_context.memories,
                concepts=concepts,
                response_policy=response_policy,
            )
        else:
            strategy = self.tutor_skill.prepare(
                profile=memory_context.profile,
                memories=memory_context.memories,
                concepts=concepts,
                query=query,
                intent=intent,
            )
            strategy = {
                **strategy,
                "intent": intent,
                "route": choose_route(intent=intent, memories=memory_context.memories, concepts=concepts),
                "response_policy": response_policy,
                "source": "agent_skill_manager",
            }
            strategy["constraints"] = _dedupe_strings(
                intent_constraints(intent) + list(strategy.get("constraints") or [])
            )

        active_skill_id = _active_skill_id(strategy)
        skills = ["memory", "profile", "concept", "knowledge"]
        if agent_mode == "short_term":
            skills = ["short_term_memory", "profile", "knowledge"]
        elif agent_mode == "stateless":
            skills = []
        if not tutoring_strategy_skipped:
            skills.append("tutor")
        if active_skill_id:
            skills.append(active_skill_id)
        metadata = {
            "skills": skills,
            "agent": "AgentSkillManager",
            "agent_mode": agent_mode,
            "context_mode": agent_mode,
            "langgraph_enabled": False,
            "langchain_enabled": False,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "user_id": user_id,
            "profile_fields": len(memory_context.profile),
            "history_count": len(memory_context.history),
            "memory_count": len(memory_context.memories),
            "concept_count": len(concepts),
            "knowledge_chunk_count": len(knowledge_context.chunks),
            "knowledge_relation_count": len(knowledge_context.relations),
            "knowledge_retrieval_skipped": bool(
                knowledge_context.metadata.get("retrieval_skipped")
            ),
            "knowledge_retrieval_skip_reason": knowledge_context.metadata.get("skip_reason"),
            "tutor_style": strategy.get("style"),
            "active_teaching_skill": active_skill_id,
            "teaching_methods": list(strategy.get("teaching_methods") or []),
            "intent": intent,
            "route": strategy.get("route"),
            "response_policy": response_policy,
            "tutoring_strategy_skipped": tutoring_strategy_skipped,
        }
        return AgentSkillContext(
            memory=memory_context,
            concepts=concepts,
            knowledge=knowledge_context,
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
        history_already_saved: bool = False,
    ) -> dict[str, Any]:
        memory_result, concept_result, tutor_result = await asyncio.gather(
            self.memory_skill.after_turn(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                assistant_text=assistant_text,
                context=context,
                history_already_saved=history_already_saved,
            ),
            self.concept_skill.after_turn(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                assistant_text=assistant_text,
                context=context,
            ),
            self.tutor_skill.after_turn(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                assistant_text=assistant_text,
                context=context,
            ),
        )
        return {
            "memory": memory_result,
            "concept": concept_result,
            "tutor": tutor_result,
        }

    def render_system_prompt(self, context: AgentSkillContext) -> str:
        profile_text = json.dumps(context.memory.profile, ensure_ascii=False)[:180] if context.memory.profile else "无"
        intent_value = context.tutor_strategy.get("intent")
        intent: dict[str, Any] = intent_value if isinstance(intent_value, dict) else {}
        preserve_followup = intent.get("category") == "followup"
        history_text = _format_recent_history(context.memory.history, preserve_followup=preserve_followup)
        memory_text = "\n".join(
            f"- {item.content[:50]}"
            for item in context.memory.memories[:3]
        ) or "无"
        concept_text = "\n".join(
            f"- {item.subject}:{item.concept} 掌握{item.mastery_score:.2f}"
            for item in context.concepts[:3]
        ) or "无"
        strategy_text = "; ".join(str(item) for item in context.tutor_strategy.get("constraints", [])[:2]) or "清晰、简短"
        route = context.tutor_strategy.get("route") or "concept_explanation"
        intent_text = (
            f"类别={intent.get('category', 'explain')}；"
            f"学科={intent.get('subject', 'unknown')}；"
            f"路由={route}"
        )
        response_policy = format_response_policy(context.tutor_strategy.get("response_policy"))
        active_skill_prompt = format_active_skill_prompt(
            context.tutor_strategy.get("active_skill"),
            context.tutor_strategy.get("teaching_methods"),
            subject=str(intent.get("subject") or "unknown"),
            response_policy=context.tutor_strategy.get("response_policy"),
        )
        followup_directive = ""
        if preserve_followup:
            followup_directive = (
                "\n本轮承接指令（最高优先级）：当前输入是对上方最近一条小维消息末尾提问的肯定或选择，"
                "不是一个新的独立问题。直接展开该提问邀请继续了解的内容；如果有两个方向且用户未指定，"
                "先深入第一个方向，再自然衔接第二个。禁止重新解释上一轮已经回答过的主体，禁止再次询问用户想聊什么。"
                "若回答末尾继续引导，只提一个探索式问题；两个延伸方向用‘和’连接，禁止使用‘还是’‘或者’‘二选一’。"
            )

        prompt = (
            f"prompt_version: {DEFAULT_PROMPT_VERSION}\n"
            f"{format_xiaowei_system_prompt(_current_date_context())}\n"
            f"用户画像:{profile_text}\n"
            f"近期对话:{history_text}\n"
            f"相关长期记忆:{memory_text}（需要时调用 memory_search）\n"
            f"知识点掌握状态:{concept_text}（需要时调用 concept_lookup）\n"
            f"本轮 Agent 判断:{intent_text}\n"
            f"教学策略:{context.tutor_strategy.get('style', 'step_by_step')}; {strategy_text}\n"
            f"{active_skill_prompt}\n"
            f"{response_policy}"
            f"{followup_directive}"
        )
        knowledge_text = context.knowledge.render_prompt(
            max_chars=int(os.getenv("KNOWLEDGE_MAX_CONTEXT_CHARS", "6000"))
        )
        if not knowledge_text:
            return prompt
        citation_rules = (
            "\n知识库引用规则：只要下方参考资料中出现任何[资料N]，就必须优先依据相关资料回答，"
            "正文中不要出现[资料N]、【资料N】等行内资料编号；"
            "结尾必须列出“参考来源”，逐条写出实际使用的资料编号、机构或网站、资料标题、"
            "可读的章节路径和链接，固定格式为“- 资料N：机构或网站《资料标题》，路径：具体章节，链接：URL”；"
            "禁止向用户展示.json、.xls、.xlsx、.csv、.md、.html等内部文件名，"
            "也禁止展示section:xxx、row-xxx、工作表!第N行等内部定位标识。"
            "不得引用未出现在参考资料中的来源，也不得把知识图谱推断写成原文直接陈述。"
        )
        return f"{prompt}{citation_rules}\n\n{knowledge_text}"

def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _role_name(role: str) -> str:
    if role == "assistant":
        return "小维"
    if role == "system":
        return "系统"
    return "学生"


def _format_recent_history(history: list[dict[str, str]], *, preserve_followup: bool) -> str:
    recent = _latest_completed_turn(history) if preserve_followup else history[-3:]
    lines: list[str] = []
    for item in recent:
        content = str(item.get("content") or "").strip()
        if preserve_followup and item.get("role") == "assistant" and len(content) > 520:
            content = f"{content[:100]}……[中间省略]……{content[-400:]}"
        elif preserve_followup:
            content = content[:520]
        else:
            content = content[:60]
        if content:
            lines.append(f"{_role_name(str(item.get('role') or 'user'))}: {content}")
    return "\n".join(lines) or "无"


def _latest_completed_turn(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only the latest user question and its assistant answer."""

    assistant_index = next(
        (index for index in range(len(history) - 1, -1, -1) if history[index].get("role") == "assistant"),
        None,
    )
    if assistant_index is None:
        return history[-2:]
    user_index = next(
        (index for index in range(assistant_index - 1, -1, -1) if history[index].get("role") == "user"),
        None,
    )
    if user_index is None:
        return [history[assistant_index]]
    return [history[user_index], history[assistant_index]]


def _active_skill_id(strategy: dict[str, Any]) -> str | None:
    active_skill = strategy.get("active_skill")
    if not isinstance(active_skill, dict):
        return None
    return str(active_skill.get("id") or "").strip() or None
