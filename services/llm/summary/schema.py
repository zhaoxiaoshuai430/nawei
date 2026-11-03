"""Model output contains semantic candidates only; metadata belongs to the server."""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field

Text = Annotated[str, Field(min_length=1, max_length=400)]
Evidence = Annotated[list[UUID], Field(min_length=1, max_length=12)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Topic(StrictModel):
    topic: Text
    evidence_message_ids: Evidence


class Claim(StrictModel):
    kind: Literal["open_task", "goal", "communication_preference", "preference", "learning_observation",
                  "teaching_strategy_observation", "profile_change_proposal", "temporary_state", "personal_fact"]
    subject: Annotated[str, Field(min_length=1, max_length=80)]
    content: Text
    origin: Literal["user_explicit", "observed", "inferred"]
    status: Literal["supported", "tentative", "contradicted", "expired"]
    evidence_message_ids: Evidence
    counterevidence_message_ids: list[UUID] = Field(default_factory=list, max_length=12)
    progress: Literal["open", "partial", "resolved", "cancelled", "changed"] | None = None
    trend: Literal["improving", "stagnant", "possible_forgetting", "recurring_error", "independent"] | None = None
    scope: Literal["subject", "general"] = "subject"
    preference_type: Literal["person", "thing", "topic", "idea", "communication"] | None = None
    polarity: Literal["like", "dislike", "neutral"] | None = None
    target: Annotated[str, Field(min_length=1, max_length=120)] | None = None


class SummaryCandidate(StrictModel):
    topics: list[Topic] = Field(default_factory=list, max_length=8)
    personal_facts: list[Claim] = Field(default_factory=list, max_length=8)
    claims: list[Claim] = Field(default_factory=list, max_length=12)
    task_outcomes: list[Claim] = Field(default_factory=list, max_length=24)
    preference_observations: list[Claim] = Field(default_factory=list, max_length=8)
    learning_observations: list[Claim] = Field(default_factory=list, max_length=8)
    teaching_strategy_observations: list[Claim] = Field(default_factory=list, max_length=4)
    profile_change_proposals: list[Claim] = Field(default_factory=list, max_length=4)
    temporary_states: list[Claim] = Field(default_factory=list, max_length=4)
    limitations: list[Text] = Field(default_factory=list, max_length=8)


class SummaryPreferences(StrictModel):
    enabled: bool | None = None
    apply_to_prompt: bool | None = None
