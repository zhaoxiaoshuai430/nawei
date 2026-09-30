"""LangGraph-based tutor agent for education personalization.

The graph prepares the context that will be injected into the LLM prompt. It
does not own ASR, TTS, WebSocket streaming, or the final model call.
"""

# TutorGraph 是“回答前的上下文规划图”，不是另一个对外服务，也不是最终模型调用器。
# 五个节点依次加载资料、判断意图、选择回答风格、规划教学策略并汇总 AgentSkillContext。
# 这里决定如何使用 Skills；具体提示词渲染在 skills/manager.py，推理在 llm/main.py。

from __future__ import annotations

import asyncio
import re
from typing import Any, NotRequired, TypedDict

from app_protocols.persona import normalize_assistant_dialect_mode, normalize_assistant_persona_mode
from services.knowledge.schemas import GraphCandidate, KnowledgeContext, KnowledgeExpansionOptions
from services.llm.age_policy import audience_profile_from_profile
from services.llm.agents.semantic_router import build_semantic_route
from services.llm.memory import DEFAULT_PROMPT_VERSION, LearningConcept, MemoryContext, MemoryHit
from services.llm.skills.base import AgentSkillContext
from services.llm.skills.knowledge_weaving import is_knowledge_weaving_query

try:  # pragma: no cover - depends on runtime optional package
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover - fallback keeps local tests lightweight
    RunnableLambda = None  # type: ignore[misc,assignment]

try:  # pragma: no cover - depends on runtime optional package
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - fallback keeps Gateway usable
    END = "__end__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[misc,assignment]

class PreparedTutorContext(TypedDict, total=False):
    memory_context: MemoryContext
    concepts: list[LearningConcept]
    knowledge_context: KnowledgeContext
    intent: dict[str, Any]
    semantic_route: dict[str, Any]


class TutorGraphState(PreparedTutorContext):
    """State passed between tutor-agent graph nodes."""

    # 各节点通过返回新字典增补状态；NotRequired 字段由后续节点逐步产生。
    # user_id/session_id 控制个性化范围，knowledge_base_id 控制公共知识检索范围。
    user_id: str
    session_id: str
    query: str
    knowledge_base_id: str
    context_mode: NotRequired[str]
    response_policy: NotRequired[dict[str, Any]]
    tutor_strategy: NotRequired[dict[str, Any]]
    context: NotRequired[AgentSkillContext]
    metadata: NotRequired[dict[str, Any]]


_KNOWLEDGE_CONCEPTS: tuple[dict[str, Any], ...] = (
    {
        "subject": "数学",
        "concept": "三角函数",
        "aliases": ("三角函数", "三角比", "正弦", "余弦", "正切", "sin", "cos", "tan"),
    },
)


class TutorGraph:
    """Prepare personalized tutoring context with LangGraph/LangChain primitives."""

    def __init__(self, *, memory_skill: Any, concept_skill: Any, knowledge_skill: Any, tutor_skill: Any) -> None:
        self.memory_skill = memory_skill
        self.concept_skill = concept_skill
        self.knowledge_skill = knowledge_skill
        self.tutor_skill = tutor_skill
        self.uses_langgraph = StateGraph is not None
        self.uses_langchain = RunnableLambda is not None
        self._intent_chain = RunnableLambda(classify_intent) if RunnableLambda is not None else None
        self._graph = self._compile_graph() if StateGraph is not None else None

    async def prepare(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        knowledge_base_id: str = "default",
    ) -> AgentSkillContext:
        """Run the graph and return an AgentSkillContext for prompt rendering."""

        return await self._prepare(
            user_id=user_id,
            session_id=session_id,
            query=query,
            knowledge_base_id=knowledge_base_id,
            context_mode="full",
        )

    async def prepare_short_term(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        knowledge_base_id: str = "default",
    ) -> AgentSkillContext:
        """Run the graph with profile, recent history, and current knowledge only."""

        return await self._prepare(
            user_id=user_id,
            session_id=session_id,
            query=query,
            knowledge_base_id=knowledge_base_id,
            context_mode="short_term",
        )

    async def _prepare(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        knowledge_base_id: str,
        context_mode: str,
        prepared_context: PreparedTutorContext | None = None,
    ) -> AgentSkillContext:
        state: TutorGraphState = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "knowledge_base_id": knowledge_base_id,
            "context_mode": context_mode,
            **(prepared_context or {}),
        }
        if self._graph is None:
            state = await self._run_fallback(state)
        else:
            state = await self._graph.ainvoke(state)
        return state["context"]

    def _compile_graph(self) -> Any:
        # 当前拓扑是固定顺序的准备流水线，而不是无限循环的自主 Agent。
        # 未安装可选 LangGraph 依赖时，_run_fallback 以相同顺序直接调用这些节点。
        graph = StateGraph(TutorGraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("plan_response_style", self._plan_response_style)
        graph.add_node("plan_tutoring_strategy", self._plan_tutoring_strategy)
        graph.add_node("assemble_context", self._assemble_context)
        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "classify_intent")
        graph.add_edge("classify_intent", "plan_response_style")
        graph.add_edge("plan_response_style", "plan_tutoring_strategy")
        graph.add_edge("plan_tutoring_strategy", "assemble_context")
        graph.add_edge("assemble_context", END)
        return graph.compile()

    async def _run_fallback(self, state: TutorGraphState) -> TutorGraphState:
        state = await self._load_context(state)
        state = await self._classify_intent(state)
        state = await self._plan_response_style(state)
        state = await self._plan_tutoring_strategy(state)
        return await self._assemble_context(state)

    async def _load_context(self, state: TutorGraphState) -> TutorGraphState:
        # 已由 SkillManager 准备的记忆/知识直接复用，避免同一轮重复检索。
        # 先做轻量意图判断，确定是否需要知识、概念锚点和图谱候选实体。
        if "memory_context" in state and "knowledge_context" in state:
            return state
        short_term = state.get("context_mode") == "short_term"
        local_followup = short_term and is_contextual_followup(state["query"])
        if not local_followup:
            await self.memory_skill.memory_store.initialize()

        intent = classify_intent({"query": state["query"], "profile": {}, "concepts": []})
        retrieve_knowledge = should_retrieve_knowledge(intent)
        anchor_terms = knowledge_anchor_terms(intent)
        semantic_route = build_semantic_route(query=state["query"], intent=intent)
        graph_candidates = [
            GraphCandidate(**item) for item in semantic_route.get("graph_candidates") or []
        ]
        expansion = KnowledgeExpansionOptions(**(semantic_route.get("expansion") or {}))

        if short_term:
            # 短期模式只装载画像与近期历史；上下文追问可跳过新知识检索，延续当前话题。
            # 非追问且需要知识时，让记忆读取和知识请求并发，减少串行等待。
            memory_task = asyncio.create_task(
                self.memory_skill.prepare_short_term(
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                    query=state["query"],
                )
            )
            if local_followup or not retrieve_knowledge:
                memory_context = await memory_task
                knowledge_context = build_skipped_knowledge_context(
                    knowledge_base_id=state["knowledge_base_id"],
                    query=state["query"],
                    intent=intent,
                )
            else:
                knowledge_task = asyncio.create_task(
                    self.knowledge_skill.prepare(
                        knowledge_base_id=state["knowledge_base_id"],
                        query=state["query"],
                        anchor_terms=anchor_terms,
                        graph_candidates=graph_candidates,
                        expansion=expansion,
                    )
                )
                memory_context, knowledge_context = await asyncio.gather(memory_task, knowledge_task)
            return {
                **state,
                "memory_context": memory_context,
                "concepts": [],
                "knowledge_context": knowledge_context,
                "intent": intent,
                "semantic_route": semantic_route,
            }

        memory_task = self.memory_skill.prepare(
            user_id=state["user_id"],
            session_id=state["session_id"],
            query=state["query"],
        )
        concept_task = (
            self.concept_skill.prepare(
                user_id=state["user_id"],
                session_id=state["session_id"],
                query=state["query"],
            )
            if intent.get("needs_concept_tracking")
            else None
        )
        context_tasks = [memory_task]
        if concept_task is not None:
            context_tasks.append(concept_task)
        if retrieve_knowledge:
            context_tasks.append(
                self.knowledge_skill.prepare(
                    knowledge_base_id=state["knowledge_base_id"],
                    query=state["query"],
                    anchor_terms=anchor_terms,
                    graph_candidates=graph_candidates,
                    expansion=expansion,
                )
            )
        context_results = await asyncio.gather(*context_tasks)
        memory_context = context_results[0]
        concepts = context_results[1] if concept_task is not None else []
        knowledge_context = (
            context_results[-1]
            if retrieve_knowledge
            else build_skipped_knowledge_context(
                knowledge_base_id=state["knowledge_base_id"],
                query=state["query"],
                intent=intent,
            )
        )
        return {
            **state,
            "memory_context": memory_context,
            "concepts": concepts,
            "knowledge_context": knowledge_context,
            "intent": intent,
            "semantic_route": semantic_route,
        }

    async def _classify_intent(self, state: TutorGraphState) -> TutorGraphState:
        if state.get("intent"):
            return state
        if self._intent_chain is not None:
            intent = await self._intent_chain.ainvoke(
                {
                    "query": state["query"],
                    "profile": state["memory_context"].profile,
                    "concepts": state.get("concepts", []),
                }
            )
        else:
            intent = classify_intent(
                {
                    "query": state["query"],
                    "profile": state["memory_context"].profile,
                    "concepts": state.get("concepts", []),
                }
            )
        return {**state, "intent": intent}

    async def _plan_response_style(self, state: TutorGraphState) -> TutorGraphState:
        # 回答长度、语气与互动限制由问题、画像和已有对话共同决定，不在网关拼提示词。
        memory_context = state["memory_context"]
        policy = build_response_policy(
            query=state["query"],
            profile=memory_context.profile,
            intent=state.get("intent", {}),
            history_count=len(memory_context.history),
        )
        return {**state, "response_policy": policy}

    async def _plan_tutoring_strategy(self, state: TutorGraphState) -> TutorGraphState:
        # 日常/非教学意图允许走非教学策略，避免把所有聊天都强行变成课程讲解。
        memory_context = state["memory_context"]
        concepts = state.get("concepts", [])
        intent = state.get("intent", {})
        if should_skip_tutoring(intent):
            strategy = build_non_tutoring_strategy(
                intent=intent,
                memories=memory_context.memories,
                concepts=concepts,
                response_policy=state.get("response_policy", {}),
            )
            return {**state, "tutor_strategy": strategy}

        strategy = self.tutor_skill.prepare(
            profile=memory_context.profile,
            memories=memory_context.memories,
            concepts=concepts,
            query=state["query"],
            intent=intent,
        )
        strategy = {
            **strategy,
            "intent": intent,
            "route": choose_route(intent=intent, memories=memory_context.memories, concepts=concepts),
            "response_policy": state.get("response_policy", {}),
            "source": "langgraph_tutor_agent",
        }
        strategy["constraints"] = _dedupe_strings(
            intent_constraints(intent) + list(strategy.get("constraints") or [])
        )
        return {**state, "tutor_strategy": strategy}

    async def _assemble_context(self, state: TutorGraphState) -> TutorGraphState:
        # 将记忆、学习概念、检索证据和策略封装为同一上下文，同时记录可诊断的路由元数据。
        # metadata 用于解释本轮为何采用某策略，不是新的用户消息或模型回答。
        memory_context = state["memory_context"]
        concepts = state.get("concepts", [])
        knowledge_context = state.get(
            "knowledge_context",
            KnowledgeContext(knowledge_base_id=state["knowledge_base_id"], query=state["query"]),
        )
        strategy = state.get("tutor_strategy", {})
        intent = state.get("intent", {})
        active_skill_value = strategy.get("active_skill")
        active_skill: dict[str, Any] = (
            active_skill_value if isinstance(active_skill_value, dict) else {}
        )
        active_skill_id = str(active_skill.get("id") or "").strip()
        short_term = state.get("context_mode") == "short_term"
        tutoring_strategy_skipped = should_skip_tutoring(intent)
        skills = ["short_term_memory", "profile", "knowledge"] if short_term else ["memory", "profile", "concept", "knowledge"]
        if not tutoring_strategy_skipped:
            skills.append("tutor")
        if active_skill_id:
            skills.append(active_skill_id)
        graph_nodes = ["load_context", "classify_intent", "plan_response_style"]
        if not tutoring_strategy_skipped:
            graph_nodes.append("plan_tutoring_strategy")
        graph_nodes.append("assemble_context")
        metadata = {
            "skills": skills,
            "agent": "TutorGraph",
            "agent_mode": "langgraph" if self.uses_langgraph else "fallback_graph",
            "context_mode": state.get("context_mode", "full"),
            "langgraph_enabled": self.uses_langgraph,
            "langchain_enabled": self.uses_langchain,
            "graph_nodes": graph_nodes,
            "prompt_version": DEFAULT_PROMPT_VERSION,
            "user_id": state["user_id"],
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
            "active_teaching_skill": active_skill_id or None,
            "teaching_methods": list(strategy.get("teaching_methods") or []),
            "intent": intent,
            "semantic_route": state.get("semantic_route", {}),
            "route": strategy.get("route"),
            "response_policy": strategy.get("response_policy", {}),
            "tutoring_strategy_skipped": tutoring_strategy_skipped,
        }
        context = AgentSkillContext(
            memory=memory_context,
            concepts=concepts,
            knowledge=knowledge_context,
            tutor_strategy=strategy,
            skill_metadata=metadata,
        )
        return {**state, "context": context, "metadata": metadata}


def classify_intent(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    subject = _infer_subject(query)
    concept = _infer_knowledge_concept(query, subject)
    weak_markers = ("不会", "不懂", "卡住", "搞不清", "听不懂", "看不懂", "错了")
    practice_markers = ("出题", "练习", "测试", "考我", "做题", "题目")
    mistake_markers = ("错题", "错因", "订正", "改错", "做错", "答错", "错在哪", "为什么错", "错误原因")
    review_markers = ("复习", "总结", "回顾", "记得", "上次")
    identity_markers = ("你是谁", "叫什么", "你叫什么")
    casual_markers = ("你好", "早上好", "晚上好", "谢谢", "聊聊天", "笑话", "故事")
    support_markers = ("心情不好", "难受", "焦虑", "压力大", "不想学", "崩溃")
    preference_markers = ("以后回答", "以后说话", "回答简短", "回答详细", "语气轻松", "生动有趣", "有趣一点", "直接一点", "多用生活例子", "一步一步讲")
    planning_markers = ("计划", "安排", "怎么学", "学习路线", "这周做什么")
    has_weak_signal = any(marker in query for marker in weak_markers if marker != "不会") or (
        "不会" in query and "会不会" not in query
    )

    if _looks_like_crisis_signal(query):
        category = "crisis_support"
    elif _looks_like_academic_integrity_request(query):
        category = "academic_integrity"
    elif _looks_like_account_security_request(query):
        category = "account_security"
    elif is_contextual_followup(query):
        category = "followup"
    elif any(marker in query for marker in identity_markers):
        category = "identity"
    elif any(marker in query for marker in preference_markers):
        category = "preference"
    elif any(marker in query for marker in support_markers):
        category = "support"
    elif any(marker in query for marker in casual_markers):
        category = "casual"
    elif any(marker in query for marker in mistake_markers):
        category = "mistake_review"
    elif any(marker in query for marker in practice_markers):
        category = "practice"
    elif any(marker in query for marker in review_markers):
        category = "review"
    elif any(marker in query for marker in planning_markers):
        category = "planning"
    elif has_weak_signal:
        category = "remediation"
    elif is_knowledge_weaving_query(query, subject=subject):
        category = "knowledge_exploration"
    elif _looks_like_daily_life(query):
        category = "daily_life"
    elif _looks_like_realtime_request(query):
        category = "realtime"
    else:
        category = "explain"

    return {
        "category": category,
        "subject": subject,
        "concept": concept["concept"] if concept else None,
        "concept_aliases": list(concept["aliases"]) if concept else [],
        "needs_memory": category in {"followup", "review", "mistake_review", "remediation", "explain", "practice", "planning"},
        "needs_concept_tracking": subject != "unknown" or category in {"practice", "mistake_review", "remediation"},
        "confidence": 0.82 if query else 0.25,
    }


def is_contextual_followup(query: str) -> bool:
    """Recognize short replies whose meaning depends on the previous assistant turn."""

    compact = re.sub(r"[\s，。！？!?、,.]+", "", str(query or "").strip())
    if 0 < len(compact) <= 24:
        step = r"第[一二三四五六七八九十百两0-9]{1,4}(?:步|点|个步骤)"
        prefix = r"(?:请|麻烦)?(?:你)?"
        if re.fullmatch(
            prefix + rf"(?:把)?{step}(?:再)?(?:讲清楚|解释一下|展开讲讲|展开讲|详细讲讲|讲一下|怎么做|为什么(?:这样做)?|没听懂|是什么意思)?",
            compact,
        ) or re.fullmatch(prefix + rf"(?:再)?(?:解释一下|讲一下|讲讲|详细讲讲|展开讲讲){step}", compact):
            return True
    if not compact or len(compact) > 12:
        return False
    exact_replies = {
        "想",
        "想听",
        "想看",
        "想了解",
        "好",
        "好的",
        "可以",
        "行",
        "没问题",
        "继续",
        "接着",
        "接着讲",
        "展开讲",
        "展开讲讲",
        "讲讲",
        "讲讲吧",
        "都想",
        "两个都想",
        "都讲讲",
        "第一个",
        "第二个",
        "先讲第一个",
        "先讲第二个",
        "前者",
        "后者",
        "这个",
        "那个",
    }
    return compact in exact_replies


def knowledge_anchor_terms(intent: dict[str, Any]) -> list[str]:
    concept = str(intent.get("concept") or "").strip()
    aliases = [str(item).strip() for item in intent.get("concept_aliases") or []]
    return _dedupe_strings([concept, *aliases])[:20]


def _infer_knowledge_concept(query: str, subject: str) -> dict[str, Any] | None:
    normalized = query.strip().lower()
    for definition in _KNOWLEDGE_CONCEPTS:
        if subject not in {"unknown", definition["subject"]}:
            continue
        aliases = tuple(str(item).lower() for item in definition["aliases"])
        for alias in aliases:
            if alias.isascii():
                if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", normalized):
                    return definition
            elif alias in normalized:
                return definition
    return None


def _infer_subject(query: str) -> str:
    explicit_subjects = ("数学", "英语", "语文", "物理", "化学", "生物", "历史", "地理", "道德与法治")
    for subject in explicit_subjects:
        if subject in query:
            return subject

    subject_markers = {
        "数学": ("数学", "函数", "斜率", "方程", "几何", "代数", "概率", "导数"),
        "英语": ("英文", "单词", "语法", "阅读理解", "听力"),
        "语文": ("语文", "作文", "阅读", "文言文", "古诗"),
        "物理": ("物理", "力学", "受力", "重力", "摩擦力", "电路", "速度", "压强", "浮力", "导热", "热量", "金属温度", "摸铁", "木头", "材料"),
        "化学": ("化学", "方程式", "溶液", "元素", "反应", "食盐", "无机盐", "潮解", "结晶"),
        "生物": (
            "细胞", "遗传", "基因", "光合作用", "呼吸作用", "生态系统", "DNA", "植物学", "果实", "种子",
            "西红柿", "番茄", "黄瓜", "茄子", "辣椒", "水果", "蔬菜", "植物", "绿植",
        ),
        "历史": (
            "朱元璋", "秦始皇", "汉武帝", "唐太宗", "明朝", "清朝", "唐朝", "宋朝", "元朝", "秦朝", "汉朝",
            "朝代", "皇帝", "起义", "变法", "战役", "革命", "历史事件",
        ),
        "地理": ("气候", "天气", "雷阵雨", "雷暴", "积雨云", "地形", "经纬度", "季风", "洋流", "板块", "区域地理", "人口分布"),
        "医学": ("医学", "心慌", "心率", "睡眠不足", "昼夜节律", "交感神经", "注意力下降"),
        "法学": ("合同", "口头承诺", "口头约定", "法律效力", "举证", "民法"),
        "道德与法治": ("宪法", "法律", "公民权利", "公民义务", "法治", "道德规范"),
        "艺术": ("艺术", "美学", "构图", "对称", "视觉平衡", "审美"),
        "经济": ("经济", "商品涨价", "凡勃伦", "消费", "价格", "需求"),
        "心理": ("心理", "倾诉", "自我披露", "黑暗环境", "社会评价", "安全感"),
    }
    for subject, markers in subject_markers.items():
        if any(marker in query for marker in markers):
            return subject
    return "unknown"


def choose_route(*, intent: dict[str, Any], memories: list[MemoryHit], concepts: list[LearningConcept]) -> str:
    category = str(intent.get("category") or "explain")
    has_weakness = any(item.memory_type == "weakness" for item in memories) or any(
        item.weakness_count > 0 or item.mastery_score < 0.35 for item in concepts
    )
    if category == "crisis_support":
        return "minor_crisis_escalation"
    if category == "academic_integrity":
        return "academic_integrity_guardrail"
    if category == "account_security":
        return "account_security_guardrail"
    if category == "followup":
        return "contextual_followup"
    if category == "identity":
        return "identity_guardrail"
    if category == "preference":
        return "preference_update"
    if category == "casual":
        return "casual_conversation"
    if category == "daily_life":
        return "daily_companion"
    if category == "knowledge_exploration":
        return "knowledge_weaving"
    if category == "support":
        return "emotional_support"
    if category == "planning":
        return "action_plan"
    if category == "realtime":
        return "realtime_answer"
    if category == "mistake_review":
        return "mistake_review"
    if category == "practice":
        return "exercise_generation"
    if category == "review":
        return "memory_review"
    if category == "remediation" or has_weakness:
        return "remediation_scaffold"
    return "concept_explanation"


def intent_constraints(intent: dict[str, Any]) -> list[str]:
    category = str(intent.get("category") or "explain")
    if category == "crisis_support":
        return [
            "停止普通教学和延伸追问，优先确认现实安全；建议立即告诉身边可信任成年人、不要独处并远离危险物品；已经受伤或准备马上行动时联系当地急救或报警服务"
        ]
    if category == "academic_integrity":
        return [
            "不提供正在进行的考试答案，不代写可直接提交的受评作业；简短说明边界后转为概念解释、解题提示、学生草稿反馈或考后讲解"
        ]
    if category == "account_security":
        return ["不要向任何人发送密码、验证码或支付凭据；说明风险，并给出保护账号的直接做法"]
    if category == "followup":
        return [
            "这是对上一轮小维提问的省略式承接。必须根据近期对话中最后一条小维消息确定用户同意或选择的具体方向，不要重新回答上一轮原问题；用户只作肯定且上一轮给出两个方向时，先自然展开第一个方向，再在合适时衔接第二个；近期对话没有可承接内容时，简短追问用户想了解什么，严禁猜测"
        ]
    if category == "identity":
        return ["如果学生问你是谁，只回答你是小维，是他的 AI 教育语音助手"]
    if category == "preference":
        return ["用生动有趣的方式确认理解了哪些表达偏好，可简要展示今后的表达感觉，但不虚构用户问过的问题"]
    if category == "casual":
        return ["像熟悉的交流对象一样自然回应，不要强行把闲聊转成课程"]
    if category == "daily_life":
        return ["先理解具体处境，再给最关键的少量办法；自然幽默一句，并提醒一个真正相关的风险或下一步，但不写成长篇教程，不说教"]
    if category == "knowledge_exploration":
        return [
            "先用一句自然结论回答当前知识问题，再详细讲清关键概念和因果关系；分类不唯一时标明不同语境和适用边界"
        ]
    if category == "support":
        return ["先回应用户具体感受，不空泛安慰；再给一个负担较小的下一步"]
    if category == "planning":
        return ["结合用户当前目标给出有优先级、可执行的安排"]
    if category == "realtime":
        return ["优先使用实时工具事实，先直接回答结果，再补充必要背景"]
    if category == "mistake_review":
        return ["先依据题干和学生原答案定位错因；证据不足时先补齐信息，不把错误简单归为粗心"]
    if category == "practice":
        return ["先给 1 道短题或小练习，再根据学生回答继续讲"]
    if category == "review":
        return ["优先结合近期对话和长期记忆，说明你记得的学习状态"]
    if category == "remediation":
        return ["先承认卡点，再把概念拆成更小步骤"]
    return ["围绕当前问题解释，不主动扩展太多无关知识"]


def build_response_policy(
    *,
    query: str,
    profile: dict[str, Any],
    intent: dict[str, Any],
    history_count: int = 0,
) -> dict[str, Any]:
    """Build deterministic expression controls for the final LLM response."""

    category = str(intent.get("category") or "explain")
    interaction_style = profile.get("interaction_style")
    if not isinstance(interaction_style, dict):
        interaction_style = {}
    preferences = profile.get("preferences")
    if not isinstance(preferences, dict):
        preferences = {}
    audience_profile = audience_profile_from_profile(profile)

    emotion = _infer_emotion(query)
    urgent = any(marker in query for marker in ("快点", "赶时间", "马上", "来不及", "急用"))
    explicit_brief = any(marker in query for marker in ("简短", "简洁", "一句话", "短一点"))
    explicit_detailed = any(marker in query for marker in ("详细", "深入", "展开讲", "讲透"))
    stored_depth = str(interaction_style.get("response_length") or "standard").strip().lower()
    if category == "preference":
        response_depth = "standard"
    elif urgent or explicit_brief:
        response_depth = "brief"
    elif explicit_detailed:
        response_depth = "detailed"
    elif stored_depth in {"brief", "standard", "detailed"}:
        response_depth = stored_depth
    else:
        response_depth = "standard"
    if (
        category in {"casual", "daily_life"}
        and not urgent
        and not explicit_brief
        and not explicit_detailed
        and stored_depth not in {"brief", "detailed"}
    ):
        response_depth = "conversational"

    scene_settings = {
        "crisis_support": ("calm_safety_first", "safety_then_trusted_adult"),
        "academic_integrity": ("firm_supportive", "boundary_then_learning_help"),
        "account_security": ("calm_safety_first", "safety_then_practical_action"),
        "followup": ("warm_professional", "continue_previous_direction"),
        "identity": ("natural_concise", "direct_answer"),
        "preference": ("lively_engaging", "acknowledge_preference"),
        "casual": ("relaxed_natural", "conversational"),
        "daily_life": ("warm_lively", "understand_then_practical_help"),
        "knowledge_exploration": ("warm_professional", "explain_then_connect"),
        "support": ("calm_empathetic", "acknowledge_then_small_step"),
        "practice": ("encouraging_interactive", "exercise_then_feedback"),
        "mistake_review": ("patient_analytical", "evidence_then_correction"),
        "review": ("familiar_reflective", "recall_then_summarize"),
        "planning": ("practical_direct", "prioritized_actions"),
        "remediation": ("patient_nonjudgmental", "small_steps_then_check"),
        "realtime": ("clear_confident", "answer_then_context"),
        "explain": ("warm_direct", "answer_then_explain"),
    }
    tone, interaction_mode = scene_settings.get(category, scene_settings["explain"])
    stored_tone = str(interaction_style.get("tone") or "").strip().lower()
    preferred_tones = {
        "warm": "warm_direct",
        "direct": "practical_direct",
        "light": "relaxed_lively",
    }
    if emotion == "neutral" and category not in {"identity", "support"} and stored_tone in preferred_tones:
        tone = preferred_tones[stored_tone]
    if urgent:
        tone, interaction_mode = "calm_direct", "action_first"
    elif emotion == "frustrated":
        tone = "patient_nonjudgmental"
    elif emotion == "anxious":
        tone = "calm_steady"
    elif emotion == "positive" and category != "realtime":
        tone = "warm_lively"

    expression_profiles = {
        "crisis_support": ("safety_guardian", "none", "immediate_safety"),
        "account_security": ("safety_guardian", "none", "immediate_safety"),
        "followup": ("professional_tutor", "light", "contextual_continuity"),
        "support": ("steady_companion", "none", "emotion_and_small_step"),
        "casual": ("familiar_guardian", "lively", "natural_connection"),
        "daily_life": ("reliable_guardian", "lively", "practical_solution_and_risk"),
        "knowledge_exploration": ("professional_tutor", "light", "knowledge_structure_and_transfer"),
        "planning": ("practical_guide", "none", "prioritized_action"),
        "realtime": ("practical_guide", "none", "answer_quality"),
        "preference": ("familiar_guardian", "light", "natural_connection"),
    }
    companion_style, humor_level, care_focus = expression_profiles.get(
        category,
        ("professional_tutor", "none", "learning_accuracy"),
    )
    persona_mode = _explicit_persona_mode(query) or normalize_assistant_persona_mode(profile)
    dialect_mode = _explicit_dialect_mode(query) or normalize_assistant_dialect_mode(profile)
    learning_categories = {
        "followup", "knowledge_exploration", "practice", "mistake_review", "review", "remediation", "explain",
    }
    if persona_mode == "gentle":
        humor_level = "none"
        if category in {"casual", "daily_life", "identity", "preference"}:
            companion_style = "gentle_guardian"
        elif category in learning_categories:
            companion_style = "gentle_mentor"
    elif category in {"casual", "daily_life", "identity", "preference"}:
        companion_style = "reliable_guardian" if category == "daily_life" else "familiar_guardian"
        humor_level = "playful"
    elif category in learning_categories:
        companion_style = "witty_tutor"
        humor_level = "light"

    if category == "crisis_support" or _looks_like_immediate_risk(query):
        companion_style, humor_level, care_focus = "safety_guardian", "none", "immediate_safety"
    elif _looks_like_serious_topic(query):
        companion_style, humor_level, care_focus = "steady_companion", "none", "practical_solution_and_risk"
    elif urgent or emotion in {"frustrated", "anxious", "low"}:
        humor_level = "none"
        if category in {"casual", "daily_life"}:
            companion_style, care_focus = "steady_companion", "emotion_and_small_step"

    preferred = str(interaction_style.get("preferred_explanation") or "").strip()
    if not preferred:
        enjoy = " ".join(str(item) for item in preferences.get("topics_enjoy", []) if item)
        if any(marker in enjoy for marker in ("生活", "例子", "案例")):
            preferred = "life_examples"
        elif any(marker in enjoy for marker in ("步骤", "一步一步")):
            preferred = "step_by_step"

    return {
        "scene": category,
        "emotion": emotion,
        "persona_mode": persona_mode,
        "dialect_mode": dialect_mode,
        "tone": tone,
        "response_depth": response_depth,
        "interaction_mode": interaction_mode,
        "companion_style": companion_style,
        "humor_level": humor_level,
        "care_focus": care_focus,
        "safety_note": _daily_safety_note(query) if category == "daily_life" else "",
        "preferred_explanation": preferred,
        "avoid": str(interaction_style.get("avoid") or "").strip(),
        "familiarity": "returning" if history_count >= 2 else "new",
        "audience": _minor_audience(profile),
        "audience_profile": audience_profile,
    }


def _explicit_persona_mode(query: str) -> str | None:
    normalized = re.sub(r"\s+", "", query)
    context_markers = ("这次", "这一轮", "现在", "接下来", "以后", "今后", "从现在开始", "说话", "回答", "切换", "改成", "设为")
    if not any(marker in normalized for marker in context_markers) and normalized not in {"温柔一点", "幽默一点", "有趣一点"}:
        return None
    if any(marker in normalized for marker in ("温柔", "柔和")):
        return "gentle"
    if any(marker in normalized for marker in ("幽默", "有趣")):
        return "humorous"
    return None


def _explicit_dialect_mode(query: str) -> str | None:
    normalized = re.sub(r"\s+", "", query)
    context_markers = ("这次", "这一轮", "现在", "接下来", "以后", "今后", "从现在开始", "说", "回答", "切换", "改成", "设为", "使用")
    if not any(marker in normalized for marker in context_markers) and normalized not in {"普通话", "北京话", "北京方言"}:
        return None
    if any(marker in normalized for marker in ("北京话", "北京方言")):
        return "beijing"
    if "普通话" in normalized:
        return "mandarin"
    return None


def _looks_like_crisis_signal(query: str) -> bool:
    patterns = (
        r"(?:我|自己).{0,8}(?:不想活|想死|自杀|自残|结束生命|伤害自己)",
        r"(?:活着|生活).{0,8}(?:没意思|没有意义|不如死)",
        r"(?:我|自己).{0,8}(?:已经|正在|准备|打算).{0,8}(?:割腕|跳楼|服药|自杀|自残)",
    )
    return any(re.search(pattern, query) for pattern in patterns)


def _looks_like_academic_integrity_request(query: str) -> bool:
    explicit_misconduct = ("帮我作弊", "替我考试", "帮我代考", "代写作业", "替我写作业", "代写论文", "直接替我完成")
    if any(marker in query for marker in explicit_misconduct):
        return True
    assessment = any(marker in query for marker in ("正在考试", "考试中", "正在测验", "测验中", "在线考试", "闭卷考试", "监考"))
    asks_for_answer = any(marker in query for marker in ("答案", "代答", "直接告诉我", "帮我做完", "不要过程"))
    return assessment and asks_for_answer


def _looks_like_account_security_request(query: str) -> bool:
    credentials = ("密码", "验证码", "支付密码", "银行卡号", "私钥", "密钥")
    disclosure = ("发给", "告诉", "分享", "转发", "给朋友", "给同学", "给别人", "能不能给", "应该给")
    return any(marker in query for marker in credentials) and any(marker in query for marker in disclosure)


def _looks_like_realtime_request(query: str) -> bool:
    realtime_topics = (
        "新闻", "冠军", "比赛结果", "比分", "赛程", "排名", "世界杯", "奥运会", "诺贝尔", "获奖", "发布会",
    )
    temporal_markers = ("今年", "最新", "现在", "今天", "目前", "刚刚", "近期", "本届")
    has_year = re.search(r"(?:19|20)\d{2}\s*年?", query) is not None
    return any(topic in query for topic in realtime_topics) and (
        has_year or any(marker in query for marker in temporal_markers) or "新闻" in query
    )


_NON_TUTORING_CATEGORIES = {
    "crisis_support",
    "academic_integrity",
    "account_security",
    "followup",
    "identity",
    "preference",
    "casual",
    "daily_life",
    "support",
    "planning",
    "realtime",
}


def should_skip_tutoring(intent: dict[str, Any] | None) -> bool:
    """Return whether this scene should bypass teaching workflow selection."""

    return str((intent or {}).get("category") or "explain") in _NON_TUTORING_CATEGORIES


def should_retrieve_knowledge(intent: dict[str, Any] | None) -> bool:
    """Return whether this scene benefits from public GraphRAG context."""

    return not should_skip_tutoring(intent)


def build_skipped_knowledge_context(
    *,
    knowledge_base_id: str,
    query: str,
    intent: dict[str, Any] | None,
) -> KnowledgeContext:
    """Build an observable empty context when GraphRAG is unnecessary."""

    category = str((intent or {}).get("category") or "explain")
    return KnowledgeContext(
        knowledge_base_id=knowledge_base_id,
        query=query,
        metadata={
            "retrieval_skipped": True,
            "skip_reason": f"intent:{category}",
        },
    )


def build_non_tutoring_strategy(
    *,
    intent: dict[str, Any],
    memories: list[MemoryHit],
    concepts: list[LearningConcept],
    response_policy: dict[str, Any],
) -> dict[str, Any]:
    """Build the lightweight strategy used by chat, safety, and utility scenes."""

    category = str(intent.get("category") or "explain")
    return {
        "style": "natural_conversation" if category == "casual" else "scene_policy",
        "constraints": intent_constraints(intent),
        "active_skill": None,
        "teaching_methods": [],
        "intent": intent,
        "route": choose_route(intent=intent, memories=memories, concepts=concepts),
        "response_policy": response_policy,
        "source": "deterministic_scene_router",
    }


def _looks_like_daily_life(query: str) -> bool:
    markers = (
        "忘带",
        "忘记带",
        "迟到",
        "睡不着",
        "熬夜",
        "起不来",
        "吃什么",
        "吃饭",
        "钥匙",
        "门锁",
        "穿什么",
        "衣服",
        "收拾房间",
        "东西丢了",
        "找不到东西",
        "零花钱",
        "怎么拒绝",
        "怎么道歉",
        "跟朋友",
        "和朋友",
        "跟同学",
        "和同学",
        "跟家人",
        "和家人",
        "吵架",
        "无聊",
        "出去玩",
        "回家",
    )
    return any(marker in query for marker in markers)


def _looks_like_serious_topic(query: str) -> bool:
    markers = (
        "受伤",
        "流血",
        "生病",
        "发烧",
        "医院",
        "危险",
        "打架",
        "被欺负",
        "霸凌",
        "威胁",
        "隐私",
        "密码泄露",
    )
    return any(marker in query for marker in markers)


def _looks_like_immediate_risk(query: str) -> bool:
    markers = ("受伤", "流血", "危险", "打架", "被欺负", "霸凌", "威胁")
    return any(marker in query for marker in markers)


def _daily_safety_note(query: str) -> str:
    if any(marker in query for marker in ("钥匙", "门锁", "开锁", "进不了门")):
        return (
            "不得建议把备用钥匙藏在鞋柜、脚垫、花盆、门框、手机壳等位置；"
            "只建议由可信任家人或成年人保管，或联系正规物业、正规开锁服务"
        )
    if any(marker in query for marker in ("密码", "验证码", "账号", "转账", "付款", "银行卡")):
        return "不得索要或建议分享密码、验证码和支付凭据；涉及转账或陌生人要求付款时先核实身份"
    return ""


def _minor_audience(profile: dict[str, Any]) -> str:
    audience = audience_profile_from_profile(profile)
    grade = str(
        profile.get("grade")
        or profile.get("grade_code")
        or profile.get("grade_level")
        or profile.get("school_grade")
        or ""
    ).strip()
    if audience:
        grade_note = f"，已知年级={grade}" if grade else ""
        return (
            f"{audience['age_years']} 岁学生，年龄阶段={audience['age_stage']}"
            f"（{audience['age_band']} 岁）{grade_note}；按年龄调整词汇、抽象程度、例子和信息量"
        )
    if grade:
        return f"3 至 17 岁学生，年龄未知，已知年级={grade}；按该年级使用中性易懂表达"
    return "年龄未设置、年级未知；不套用分龄字数限制，先用中性易懂表达，只有任务难度确实依赖年龄或年级时才追问"


def _infer_emotion(query: str) -> str:
    if any(marker in query for marker in ("烦", "崩溃", "受不了", "怎么又", "太难了", "不想学")):
        return "frustrated"
    if any(marker in query for marker in ("焦虑", "紧张", "害怕", "来不及", "压力大", "担心")):
        return "anxious"
    if any(marker in query for marker in ("明白了", "会了", "太好了", "开心", "搞定了")):
        return "positive"
    if any(marker in query for marker in ("难受", "伤心", "低落", "没精神")):
        return "low"
    return "neutral"


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
