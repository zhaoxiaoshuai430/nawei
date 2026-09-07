"""Knowledge-weaving Skill for natural life-knowledge explanations."""

from __future__ import annotations

import os
from typing import Any


_ACTION_REQUEST_MARKERS = (
    "怎么办", "该怎么办", "应该怎么办", "怎么处理", "如何处理", "帮我做", "替我写", "给我出题", "算一下", "等于多少",
)
_EXPLANATION_MARKERS = (
    "为什么", "什么原理", "科学原理", "背后的原理", "原因", "怎么回事", "是怎么形成", "如何形成", "是否", "会不会",
    "还是", "有什么区别", "区别", "有什么应用", "生活应用", "属于什么学科", "知识脉络",
)
_CONVERSATION_ABOUT_AGENT_MARKERS = (
    "你为什么", "小维为什么", "为什么不回答", "为什么不回复", "为什么不理我",
)
_LIFE_PHENOMENON_MARKERS = (
    "冬天", "夏天", "温度", "金属", "木头", "玻璃", "食盐", "食物", "受潮", "植物", "绿植", "睡眠", "熬夜", "构图",
    "商品", "涨价", "黑暗", "光线", "雷雨", "天空", "声音", "气味", "颜色", "合同", "承诺",
)
_SUPPORTED_SUBJECTS = {"物理", "化学", "生物", "地理", "道德与法治", "医学", "法学", "艺术", "经济", "心理", "社会学"}


def is_knowledge_weaving_query(query: str, *, subject: str = "unknown") -> bool:
    """Detect explanatory life-knowledge questions without hijacking advice or calculations."""

    if os.getenv("KNOWLEDGE_WEAVING_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return False
    normalized = query.strip()
    if not normalized or any(marker in normalized for marker in _ACTION_REQUEST_MARKERS):
        return False
    if any(marker in normalized for marker in ("请用", "请讲清", "请各举", "请把", "这道题", "解题", "初中生")):
        return False
    if "是什么" in normalized and normalized.count("？") + normalized.count("?") > 1:
        return False
    if not any(marker in normalized for marker in _EXPLANATION_MARKERS):
        return False
    if subject in {"数学", "语文", "英语", "历史"}:
        return False
    if subject in _SUPPORTED_SUBJECTS or any(marker in normalized for marker in _LIFE_PHENOMENON_MARKERS):
        return True
    return not any(marker in normalized for marker in _CONVERSATION_ABOUT_AGENT_MARKERS)


def build_knowledge_weaving_skill(query: str, *, intent: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Build a Skill that explains life knowledge and suggests related next steps."""

    intent = intent or {}
    if str(intent.get("category") or "explain") != "knowledge_exploration":
        return None
    return {
        "id": "knowledge_weaving",
        "name": "知识纵横引导",
        "purpose": "先概括生活知识问题的核心结论，再自然展开解释，并用两个相关方向开启下一轮探索。",
        "activation_reason": "检测到生活知识或现象原理解释需求",
        "workflow": [
            "先用一句话准确概括当前问题的核心结论",
            "根据问题和检索资料详细解释最关键的概念和因果关系",
            "优先依据本轮知识图谱关系和资料层级，选出两个用户尚未询问且值得继续展开的方向",
            "最后把两个探索方向自然放进同一个问题中",
        ],
        "output_sections": ["自然核心首句", "内容展开", "结尾关联问句"],
        "quality_checks": [
            "正文首先解决当前问题，不能为了铺设脉络而喧宾夺主",
            "不要使用固定的纵向知识脉络、横向场景应用标题",
            "两个探索方向必须贴合当前知识点，不能使用还想了解更多吗之类的空泛追问",
            "探索方向按内容动态选择，不能固定为深入原理和跨场景应用",
            "结尾只邀请用户继续了解新内容，不检查理解，不要求用户解释、判断或做题",
            "图谱或资料关系不足时使用建议方向等审慎表达，不编造知识层级",
            "医学、法律、心理和安全问题必须保留必要边界",
        ],
        "recommended_tools": [],
        "unavailable_capabilities": ["本轮知识检索未返回可靠关系时，不得把模型推测写成知识图谱事实"],
    }


def format_knowledge_weaving_prompt(active_skill: dict[str, Any]) -> str:
    """Render knowledge-weaving rules that supplement the shared teaching contract."""

    return (
        "\n知识纵横引导补充规则：正文结构由当前内容自然决定，不输出‘纵向知识脉络’或‘横向场景应用’等内部标签，"
        "不主动铺开完整学科树或罗列大量应用。优先使用本轮知识图谱关系和参考资料筛选有价值的关联引导点，"
        "图谱没有提供的关系只能审慎建议，不能伪装成已验证事实；最终输出遵守教学回答统一规则。"
    )
