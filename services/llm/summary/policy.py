"""Validated rollout switches and conservative resource budgets."""
from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SummarySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    apply_to_prompt: bool = False
    profile_promotion_enabled: bool = True
    trigger_turns: int = Field(default=50, ge=1, le=1000)
    window_turns: int = Field(default=100, ge=1, le=1000)
    window_max_days: int = Field(default=30, ge=1, le=365)
    global_concurrency: int = Field(default=1, ge=1, le=1)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_attempts: int = Field(default=3, ge=1, le=10)
    input_max_tokens: int = Field(default=12000, ge=1000, le=128000)
    output_max_tokens: int = Field(default=2000, ge=256, le=8000)
    prompt_max_tokens: int = Field(default=600, ge=100, le=2000)
    read_timeout_seconds: float = Field(default=0.1, gt=0, le=1)

    @model_validator(mode="after")
    def rollout(self) -> SummarySettings:
        if self.apply_to_prompt and not self.enabled:
            raise ValueError("SUMMARY_APPLY_TO_PROMPT requires SUMMARY_ENABLED")
        return self

    @classmethod
    def from_env(cls) -> SummarySettings:
        return cls.model_validate({key: os.environ[f"SUMMARY_{key.upper()}"]
                                   for key in cls.model_fields if f"SUMMARY_{key.upper()}" in os.environ})

    @property
    def window_version(self) -> str:
        return f"recent-v2:{self.window_turns}:{self.window_max_days}:{self.input_max_tokens}"


def token_bound(text: str) -> int:
    # UTF-8 byte count is a conservative upper bound for byte-based tokenizers.
    # Deliberately avoids a guessed chars/token ratio for mixed Chinese text.
    return len(text.encode("utf-8"))
