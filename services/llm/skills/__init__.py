"""AgentSkill package owned by the LLM service."""

from typing import Any

__all__ = ["AgentSkillContext", "AgentSkillManager"]


def __getattr__(name: str) -> Any:
    if name == "AgentSkillContext":
        from services.llm.skills.base import AgentSkillContext

        return AgentSkillContext
    if name == "AgentSkillManager":
        from services.llm.skills.manager import AgentSkillManager

        return AgentSkillManager
    raise AttributeError(name)
