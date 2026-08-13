"""Shared types for LLM AgentSkill orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from services.llm.memory import LearningConcept, MemoryContext


@dataclass(frozen=True)
class AgentSkillContext:
    """Personalization context assembled before an LLM turn."""

    memory: MemoryContext
    concepts: list[LearningConcept] = field(default_factory=list)
    tutor_strategy: dict[str, Any] = field(default_factory=dict)
    skill_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def user_id(self) -> str:
        return self.memory.user_id

    @property
    def memory_ids(self) -> list[str]:
        return [item.id for item in self.memory.memories]

    @property
    def concept_ids(self) -> list[str]:
        return [item.id for item in self.concepts]


class AgentSkill(Protocol):
    """Minimal skill protocol used by AgentSkillManager."""

    name: str

    async def prepare(self, *, user_id: str, session_id: str, query: str) -> dict[str, Any]:
        """Return context contribution before the LLM call."""

    async def after_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
        context: AgentSkillContext | None,
    ) -> dict[str, Any]:
        """Persist post-turn updates."""
