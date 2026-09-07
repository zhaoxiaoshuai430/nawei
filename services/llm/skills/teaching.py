"""Reusable teaching workflows selected by the Tutor Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from services.llm.memory import LearningConcept, MemoryHit
from services.llm.skills.knowledge_weaving import (
    build_knowledge_weaving_skill,
    format_knowledge_weaving_prompt,
)


_SUBJECT_CONTEXT_FRAMES = {
    "历史": "时代背景 -> 关键原因 -> 发展过程 -> 结果 -> 后续影响",
    "数学": "必要前置知识 -> 当前概念或方法 -> 与相近概念的联系 -> 典型应用 -> 后续知识",
    "物理": "现象与条件 -> 核心规律 -> 因果机制 -> 实验或应用 -> 适用边界",
    "化学": "物质与条件 -> 反应或微观原理 -> 现象证据 -> 应用 -> 易混关系",
    "生物": "结构或现象 -> 生理机制 -> 影响因素 -> 系统联系 -> 现实应用",
    "语文": "文本或创作背景 -> 关键内容证据 -> 主题与手法 -> 同类联系 -> 迁移理解",
    "英语": "使用语境 -> 核心含义或规则 -> 形式与例句 -> 易混表达 -> 迁移使用",
    "地理": "空间位置与背景 -> 形成条件 -> 分布或变化规律 -> 人地影响 -> 区域联系",
    "道德与法治": "现实情境 -> 核心概念或规则 -> 权利义务关系 -> 社会影响 -> 行动边界",
}

_TEACHING_RESPONSE_CONTRACT = (
    "\n教学回答统一规则（本轮最终格式约束，优先于人格中的通用收尾和各 Skill 的建议标题）："
    "第一段只写一句自然的核心结论，让用户先知道答案。不得出现‘一句话总结’‘总结’‘详细解释’等元话语或标题。"
    "随后根据问题和学科详细展开，优先讲清关键概念、因果关系、步骤、例子或易错点，不为凑结构堆砌内容。"
    "正文全部使用陈述句，禁止设问、反问和带问号的标题；唯一的问句只能放在回答末尾。"
    "最后单独一段且只写一个探索式问句，在同一句中提出两个与当前知识点相关的新探索方向，并用‘和’连接。"
    "它的作用是引出用户还没有问到的相关内容，不是巩固、测验或检查理解。"
    "不得要求用户解释、判断、归纳、复述或做题，不使用‘你能解释、你能判断、你能说说、你来试试’等考查式句型。"
    "正确句法参考：‘你想继续看看“导热”背后更深的原理（比如分子层面热量怎样传递），和它在保温杯、锅具这些日常用品里怎样被利用吗？’。"
    "只能借鉴‘你想继续看看方向A，和方向B吗’的句法，不能照抄示例内容。"
    "两个方向从更深机制、前置或下位概念、相关概念、历史演变、跨学科联系、现实影响、技术或生活应用中动态选择；"
    "优先选择真正值得继续展开的两个不同方向，不要求固定为深层原理和生活应用。"
    "结尾段只能出现一个疑问语气和一个问号，两个任务之间唯一允许的连接词是‘和’；严禁使用冒号列出两问，严禁写成‘问题A？问题B？’。"
    "结尾禁止出现‘还是’‘或者’‘或是’‘以及’‘二选一’，不得再追加第二个问题。问号后立即结束回答，不要继续解释或回答这两个延伸点。"
    "输出前逐字自检：全文问号数必须等于1；最后一段必须含‘和’；最后一段中‘还是、或者、或是、以及’的出现次数必须等于0。不满足就先重写，不能带着错误格式输出。"
)


@dataclass(frozen=True)
class TeachingSkillSpec:
    """Prompt contract for one teaching workflow."""

    skill_id: str
    name: str
    purpose: str
    workflow: tuple[str, ...]
    output_sections: tuple[str, ...]
    quality_checks: tuple[str, ...]
    recommended_tools: tuple[str, ...] = ()
    unavailable_capabilities: tuple[str, ...] = ()

    def as_dict(self, *, reason: str) -> dict[str, Any]:
        return {
            "id": self.skill_id,
            "name": self.name,
            "purpose": self.purpose,
            "activation_reason": reason,
            "workflow": list(self.workflow),
            "output_sections": list(self.output_sections),
            "quality_checks": list(self.quality_checks),
            "recommended_tools": list(self.recommended_tools),
            "unavailable_capabilities": list(self.unavailable_capabilities),
        }


QUESTION_EXPLANATION = TeachingSkillSpec(
    skill_id="question_explanation",
    name="AI 讲题",
    purpose="先定位卡点，再用学生能听懂的语言拆解，最后确认能否迁移到同类题。",
    workflow=(
        "确认完整题意，区分已知条件、目标和学生已做到的位置",
        "定位卡点属于概念、审题、方法选择、计算还是表达",
        "沿一条主线分步讲解，每一步说明为什么这样做",
        "指出一个最相关的易错点",
        "补充两个值得继续探索的相关方向，但不在本轮展开",
    ),
    output_sections=("题意与卡点", "关键思路", "分步讲解", "易错提醒", "关联延展"),
    quality_checks=(
        "不能只给最终答案",
        "语言和难度要匹配学生年级与当前状态",
        "题干缺少关键条件时先澄清，不能自行补条件",
        "数学步骤使用可读纯文本，不向用户输出 JSON、Python 运算符或 LaTeX 命令",
    ),
    recommended_tools=("concept_lookup",),
    unavailable_capabilities=("当前没有正式教材 RAG，教材版本相关事实不能仅凭 Skill 保证准确",),
)


MISTAKE_REVIEW = TeachingSkillSpec(
    skill_id="mistake_review",
    name="错题复盘",
    purpose="从订正答案升级为识别错因、错误模式和下一次复习动作。",
    workflow=(
        "核对题干、学生原答案、正确答案和原思路是否齐全",
        "依据证据判断是概念、审题、方法、计算、步骤还是表达错误",
        "给出正确思路，并对照原答案指出偏离发生在哪一步",
        "提炼一句可复用的订正规则",
        "安排重做、同类题和间隔复习动作",
    ),
    output_sections=("错因证据", "正确思路", "订正规则", "同类练习", "复习安排"),
    quality_checks=(
        "错因必须来自题目和原答案，不能一律归为粗心",
        "缺少原答案时只能讲解，不能断言学生的错因",
        "复习动作要具体、少量且可执行",
    ),
    recommended_tools=("history_search", "memory_search", "concept_lookup"),
    unavailable_capabilities=("当前没有独立错题表，无法保证自动找回完整题干和原答案",),
)


QUICK_PRACTICE = TeachingSkillSpec(
    skill_id="quick_practice",
    name="快速巩固",
    purpose="围绕一个小目标生成 10 到 15 分钟的短训练，并留下反馈与复习动作。",
    workflow=(
        "确认年级、知识点、可用时间和难度；信息不足时只追问最关键的一项",
        "先用少量代表题判断概念、计算、审题或步骤问题",
        "按基础、迁移、应用组织少量题目，先让学生尝试",
        "根据学生答案给出错因反馈，而不只是公布答案",
        "留下隔天复测或升级难度的下一步",
    ),
    output_sections=("今日目标", "短练题组", "答案与讲解", "错因标签", "下次复习"),
    quality_checks=(
        "题量应能在 10 到 15 分钟完成",
        "一次只围绕一个主要知识点",
        "没有学生答案时不能声称已经完成诊断或判分",
        "题目和答案使用普通中文与可读算式，不输出内部结构化数据",
    ),
    recommended_tools=("concept_lookup", "memory_search"),
    unavailable_capabilities=("当前没有确定性练习判分工具，开放题评价仍由最终 LLM 完成",),
)


class TeachingSkillRegistry:
    """Select one primary teaching workflow while allowing composable methods."""

    def __init__(self, skills: Iterable[TeachingSkillSpec] | None = None) -> None:
        configured = tuple(skills or (QUESTION_EXPLANATION, MISTAKE_REVIEW, QUICK_PRACTICE))
        self._skills = {item.skill_id: item for item in configured}

    @property
    def skill_ids(self) -> list[str]:
        return list(self._skills)

    def select(
        self,
        *,
        query: str,
        intent: dict[str, Any] | None,
        memories: list[MemoryHit],
        concepts: list[LearningConcept],
    ) -> dict[str, Any] | None:
        normalized = query.strip()
        category = str((intent or {}).get("category") or "explain")
        subject = str((intent or {}).get("subject") or "unknown")

        mistake_markers = ("错题", "错因", "订正", "改错", "做错", "答错", "错在哪", "为什么错", "错误原因")
        explanation_markers = ("讲题", "讲解", "怎么做", "怎么解", "不会", "不懂", "卡住", "为什么", "思路")

        knowledge_skill = build_knowledge_weaving_skill(normalized, intent=intent)
        if knowledge_skill is not None:
            return knowledge_skill
        if category in {"crisis_support", "academic_integrity", "account_security", "identity", "preference", "casual", "daily_life", "support", "planning", "realtime"}:
            return None
        if category == "mistake_review" or any(marker in normalized for marker in mistake_markers):
            return self._skills["mistake_review"].as_dict(reason="检测到错题订正或错因分析需求")
        if category == "practice":
            return self._skills["quick_practice"].as_dict(reason="检测到出题、练习或测试需求")
        if category == "remediation" or subject != "unknown" or any(marker in normalized for marker in explanation_markers):
            reason = "检测到学习卡点" if category == "remediation" else "检测到题目或知识讲解需求"
            return self._skills["question_explanation"].as_dict(reason=reason)
        if memories or concepts:
            return None
        return None


def select_teaching_methods(
    *,
    query: str,
    intent: dict[str, Any] | None,
    memories: list[MemoryHit],
    concepts: list[LearningConcept],
) -> list[str]:
    """Return methods that can be layered on top of a primary workflow."""

    category = str((intent or {}).get("category") or "explain")
    subject = str((intent or {}).get("subject") or "unknown")
    has_weakness = any(item.memory_type == "weakness" for item in memories) or any(
        item.weakness_count > 0 or item.mastery_score < 0.35 for item in concepts
    )
    asks_for_guidance = any(marker in query for marker in ("不要直接给答案", "提示我", "引导我", "问我"))
    contextual_markers = (
        "介绍", "讲讲", "讲一下", "解释", "为什么", "是什么", "原理", "原因", "背景", "过程", "影响", "关系",
    )
    methods: list[str] = []
    if category in {"remediation", "mistake_review"} or has_weakness or asks_for_guidance:
        methods.append("guided_socratic")
    if category in {"explain", "remediation"} and (
        subject != "unknown" or any(marker in query for marker in contextual_markers)
    ):
        methods.append("contextual_chain")
    return methods


def format_active_skill_prompt(
    active_skill: dict[str, Any] | None,
    methods: list[str] | None = None,
    *,
    subject: str = "unknown",
) -> str:
    """Render a compact, actionable Skill block for the final LLM."""

    if not active_skill:
        return "本轮教学 Skill:无，按当前场景自然回答。"

    workflow = " -> ".join(str(item) for item in active_skill.get("workflow", []))
    output_sections = "、".join(str(item) for item in active_skill.get("output_sections", []))
    checks = "；".join(str(item) for item in active_skill.get("quality_checks", []))
    tools = "、".join(str(item) for item in active_skill.get("recommended_tools", [])) or "无"
    limitations = "；".join(str(item) for item in active_skill.get("unavailable_capabilities", [])) or "无"
    method_text = "、".join(methods or []) or "默认"
    method_instruction = ""
    if "guided_socratic" in (methods or []):
        method_instruction = (
            "\n苏格拉底引导规则:每次只推进一个关键问题，提示逐级变明确；"
            "不要连续追问造成挫败；用户明确要快速核对时直接给必要答案。"
        )
    if "contextual_chain" in (methods or []):
        frame = _SUBJECT_CONTEXT_FRAMES.get(subject, "必要背景或前置 -> 当前核心 -> 关键联系 -> 后续影响或应用")
        subject_guard = ""
        if subject == "历史":
            subject_guard = (
                "历史回答要区分可确认史实与后世解释；存在争议或多重原因时使用审慎表述，"
                "不能把复杂历史结果确定地归因于单一人物或单一事件。"
            )
        method_instruction += (
            f"\n知识脉络讲解规则:当前学科={subject}；内部按“{frame}”梳理后再回答。"
            "先直接回答用户当前问题，再自然补充最相关的前置、横向联系或后续影响，总共只选择2到3条，"
            "不要机械展示内部步骤标签，不要为了完整而堆砌无关知识。"
            "脉络方法优先于主 Skill 的建议标题；人物或概念介绍不要强行套用“题意与卡点、分步讲解”等讲题标题。"
            "结合已有掌握状态决定哪些前置需要解释；没有可靠依据的关系要说明不确定，不能编造教材出处或知识关系。"
            f"{subject_guard}"
        )
    if str(active_skill.get("id") or "") == "knowledge_weaving":
        method_instruction += format_knowledge_weaving_prompt(active_skill)
    return (
        f"本轮教学 Skill:{active_skill.get('id')}（{active_skill.get('name')}）；"
        f"启用原因:{active_skill.get('activation_reason')}\n"
        f"教学目标:{active_skill.get('purpose')}\n"
        f"执行流程:{workflow}\n"
        f"建议输出结构:{output_sections}\n"
        f"质量检查:{checks}\n"
        f"可按需调用:{tools}；教学方法:{method_text}\n"
        f"能力边界:{limitations}"
        f"{method_instruction}"
        f"{_TEACHING_RESPONSE_CONTRACT}"
    )
