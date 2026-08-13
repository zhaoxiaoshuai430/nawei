"""LangGraph-based tutor agent for education personalization.

The graph prepares the context that will be injected into the LLM prompt. It
does not own ASR, TTS, WebSocket streaming, or the final model call.
"""

from __future__ import annotations

import asyncio
from typing import Any, NotRequired, TypedDict

from services.llm.memory import DEFAULT_PROMPT_VERSION, LearningConcept, MemoryContext, MemoryHit
from services.llm.skills.base import AgentSkillContext

try:  # pragma: no cover - depends on runtime optional package
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover - fallback keeps local tests lightweight
    RunnableLambda = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on runtime optional package
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback keeps lightweight LLM installs usable
    END = "__end__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


class TutorGraphState(TypedDict):
    """State passed between tutor-agent graph nodes."""

    user_id: str
    session_id: str
    query: str
    memory_context: NotRequired[MemoryContext]
    concepts: NotRequired[list[LearningConcept]]
    intent: NotRequired[dict[str, Any]]
    tutor_strategy: NotRequired[dict[str, Any]]
    context: NotRequired[AgentSkillContext]
    metadata: NotRequired[dict[str, Any]]


class TutorGraph:
    """Prepare personalized tutoring context with LangGraph/LangChain primitives."""

    def __init__(self, *, memory_skill: Any, concept_skill: Any, tutor_skill: Any) -> None:
        self.memory_skill = memory_skill
        self.concept_skill = concept_skill
        self.tutor_skill = tutor_skill
        self.uses_langgraph = StateGraph is not None
        self.uses_langchain = RunnableLambda is not None
        self._intent_chain = RunnableLambda(_classify_intent) if RunnableLambda is not None else None
        self._graph = self._compile_graph() if StateGraph is not None else None

    async def prepare(self, *, user_id: str, session_id: str, query: str) -> AgentSkillContext:
        """Run the graph and return an AgentSkillContext for prompt rendering."""

        state: TutorGraphState = {"user_id": user_id, "session_id": session_id, "query": query}
        if self._graph is None:
            state = await self._run_fallback(state)
        else:
            state = await self._graph.ainvoke(state)
        return state["context"]

    def _compile_graph(self) -> Any:
        graph = StateGraph(TutorGraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("plan_tutoring_strategy", self._plan_tutoring_strategy)
        graph.add_node("assemble_context", self._assemble_context)
        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "classify_intent")
        graph.add_edge("classify_intent", "plan_tutoring_strategy")
        graph.add_edge("plan_tutoring_strategy", "assemble_context")
        graph.add_edge("assemble_context", END)
        return graph.compile()

    async def _run_fallback(self, state: TutorGraphState) -> TutorGraphState:
        state = await self._load_context(state)
        state = await self._classify_intent(state)
        state = await self._plan_tutoring_strategy(state)
        return await self._assemble_context(state)

    async def _load_context(self, state: TutorGraphState) -> TutorGraphState:
        await self.memory_skill.memory_store.initialize()
        memory_task = self.memory_skill.prepare(
            user_id=state["user_id"],
            session_id=state["session_id"],
            query=state["query"],
        )
        concept_task = self.concept_skill.prepare(
            user_id=state["user_id"],
            session_id=state["session_id"],
            query=state["query"],
        )
        memory_context, concepts = await asyncio.gather(memory_task, concept_task)
        return {**state, "memory_context": memory_context, "concepts": concepts}

    async def _classify_intent(self, state: TutorGraphState) -> TutorGraphState:
        if self._intent_chain is not None:
            intent = await self._intent_chain.ainvoke(
                {
                    "query": state["query"],
                    "profile": state["memory_context"].profile,
                    "concepts": state.get("concepts", []),
                }
            )
        else:
            intent = _classify_intent(
                {
                    "query": state["query"],
                    "profile": state["memory_context"].profile,
                    "concepts": state.get("concepts", []),
                }
            )
        return {**state, "intent": intent}

    async def _plan_tutoring_strategy(self, state: TutorGraphState) -> TutorGraphState:
        memory_context = state["memory_context"]
        concepts = state.get("concepts", [])
        strategy = self.tutor_skill.prepare(
            profile=memory_context.profile,
            memories=memory_context.memories,
            concepts=concepts,
            query=state["query"],
        )
        intent = state.get("intent", {})
        strategy = {
            **strategy,
            "intent": intent,
            "route": _choose_route(intent=intent, memories=memory_context.memories, concepts=concepts),
            "source": "langgraph_tutor_agent",
        }
        strategy["constraints"] = _dedupe_strings(
            list(strategy.get("constraints") or []) + _intent_constraints(intent)
        )
        return {**state, "tutor_strategy": strategy}

    async def _assemble_context(self, state: TutorGraphState) -> TutorGraphState:
        memory_context = state["memory_context"]
        concepts = state.get("concepts", [])
        strategy = state.get("tutor_strategy", {})
        intent = state.get("intent", {})
        metadata = {
            "skills": ["memory", "profile", "concept", "tutor"],
            "agent": "TutorGraph",
            "agent_mode": "langgraph" if self.uses_langgraph else "fallback_graph",
            "langgraph_enabled": self.uses_langgraph,
            "langchain_enabled": self.uses_langchain,
            "graph_nodes": ["load_context", "classify_intent", "plan_tutoring_strategy", "assemble_context"],
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "user_id": state["user_id"],
            "profile_fields": len(memory_context.profile),
            "history_count": len(memory_context.history),
            "memory_count": len(memory_context.memories),
            "concept_count": len(concepts),
            "tutor_style": strategy.get("style"),
            "intent": intent,
            "route": strategy.get("route"),
        }
        context = AgentSkillContext(
            memory=memory_context,
            concepts=concepts,
            tutor_strategy=strategy,
            skill_metadata=metadata,
        )
        return {**state, "context": context, "metadata": metadata}


def _classify_intent(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    weak_markers = ("不会", "不懂", "卡住", "搞不清", "听不懂", "看不懂", "错了")
    practice_markers = ("出题", "练习", "测试", "考我", "做题", "题目")
    review_markers = ("复习", "总结", "回顾", "记得", "上次")
    identity_markers = ("你是谁", "叫什么", "你叫什么")

    if any(marker in query for marker in identity_markers):
        category = "identity"
    elif any(marker in query for marker in practice_markers):
        category = "practice"
    elif any(marker in query for marker in review_markers):
        category = "review"
    elif any(marker in query for marker in weak_markers):
        category = "remediation"
    else:
        category = "explain"

    subject = _infer_subject(query)
    return {
        "category": category,
        "subject": subject,
        "needs_memory": category in {"review", "remediation", "explain", "practice"},
        "needs_concept_tracking": subject != "unknown" or category in {"practice", "remediation"},
        "confidence": 0.82 if query else 0.25,
    }


def _infer_subject(query: str) -> str:
    subject_markers = {
        "数学": ("数学", "函数", "斜率", "方程", "几何", "代数", "概率", "导数"),
        "英语": ("英语", "单词", "语法", "作文", "阅读理解", "听力"),
        "语文": ("语文", "作文", "阅读", "文言文", "古诗"),
        "物理": ("物理", "力", "电路", "速度", "压强", "浮力"),
        "化学": ("化学", "方程式", "溶液", "元素", "反应"),
    }
    for subject, markers in subject_markers.items():
        if any(marker in query for marker in markers):
            return subject
    return "unknown"


def _choose_route(*, intent: dict[str, Any], memories: list[MemoryHit], concepts: list[LearningConcept]) -> str:
    category = str(intent.get("category") or "explain")
    has_weakness = any(item.memory_type == "weakness" for item in memories) or any(
        item.weakness_count > 0 or item.mastery_score < 0.35 for item in concepts
    )
    if category == "identity":
        return "identity_guardrail"
    if category == "practice":
        return "exercise_generation"
    if category == "review":
        return "memory_review"
    if category == "remediation" or has_weakness:
        return "remediation_scaffold"
    return "concept_explanation"


def _intent_constraints(intent: dict[str, Any]) -> list[str]:
    category = str(intent.get("category") or "explain")
    if category == "identity":
        return ["如果学生问你是谁，只回答你是小维，是他的 AI 教育语音助手"]
    if category == "practice":
        return ["先给 1 道短题或小练习，再根据学生回答继续讲"]
    if category == "review":
        return ["优先结合近期对话和长期记忆，说明你记得的学习状态"]
    if category == "remediation":
        return ["先承认卡点，再把概念拆成更小步骤"]
    return ["围绕当前问题解释，不主动扩展太多无关知识"]


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
