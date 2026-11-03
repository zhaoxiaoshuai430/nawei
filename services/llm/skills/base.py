"""Shared types for LLM AgentSkill orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from services.llm.memory import LearningConcept, MemoryContext
from services.knowledge.schemas import KnowledgeContext
from services.llm.summary.context import SummaryContext


@dataclass(frozen=True)
class AgentSkillContext:
    """Personalization context assembled before an LLM turn."""

    memory: MemoryContext
    concepts: list[LearningConcept] = field(default_factory=list)
    knowledge: KnowledgeContext = field(default_factory=KnowledgeContext)
    tutor_strategy: dict[str, Any] = field(default_factory=dict)
    skill_metadata: dict[str, Any] = field(default_factory=dict)
    summary_context: SummaryContext | None = None

    @property
    def user_id(self) -> str:
        return self.memory.user_id

    @property
    def memory_ids(self) -> list[str]:
        return [item.id for item in self.memory.memories]

    @property
    def concept_ids(self) -> list[str]:
        return [item.id for item in self.concepts]

    @property
    def knowledge_chunk_ids(self) -> list[str]:
        return [item.chunk_id for item in self.knowledge.chunks]


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
