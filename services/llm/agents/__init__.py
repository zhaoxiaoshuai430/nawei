"""Agent orchestration package owned by the LLM service."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.llm.agents.runtime_tool_agent import LangChainRuntimeToolAgent, RuntimeToolAgentResult
    from services.llm.agents.tutor_graph import TutorGraph

__all__ = ["LangChainRuntimeToolAgent", "RuntimeToolAgentResult", "TutorGraph"]


def __getattr__(name: str) -> Any:
    if name == "TutorGraph":
        from services.llm.agents.tutor_graph import TutorGraph

        return TutorGraph
    if name in {"LangChainRuntimeToolAgent", "RuntimeToolAgentResult"}:
        from services.llm.agents.runtime_tool_agent import LangChainRuntimeToolAgent, RuntimeToolAgentResult

        return {"LangChainRuntimeToolAgent": LangChainRuntimeToolAgent, "RuntimeToolAgentResult": RuntimeToolAgentResult}[name]
    raise AttributeError(name)
