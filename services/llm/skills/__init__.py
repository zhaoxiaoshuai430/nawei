"""AgentSkill package owned by the LLM service."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.llm.skills.base import AgentSkillContext
    from services.llm.skills.manager import AgentSkillManager

__all__ = ["AgentSkillContext", "AgentSkillManager"]


def __getattr__(name: str) -> Any:
    if name == "AgentSkillContext":
        from services.llm.skills.base import AgentSkillContext

        return AgentSkillContext
    if name == "AgentSkillManager":
        from services.llm.skills.manager import AgentSkillManager

        return AgentSkillManager
    raise AttributeError(name)
