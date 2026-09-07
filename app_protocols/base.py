"""Shared Pydantic base classes for protocol DTOs."""

from __future__ import annotations

from typing import Any

import pydantic
from pydantic import BaseModel, Field


Metadata = dict[str, Any]


class ProtocolModel(BaseModel):
    """Base model with conservative validation for protocol DTOs."""

    if hasattr(pydantic, "ConfigDict"):
        model_config = pydantic.ConfigDict(extra="forbid", use_enum_values=True)
    else:  # pragma: no cover - Pydantic 1.x
        class Config:
            extra = "forbid"
            use_enum_values = True


def metadata_field() -> Metadata:
    """Return an empty metadata object for Pydantic default_factory."""

    return {}


def dump_json(model: BaseModel) -> str:
    """Return compact JSON across Pydantic 1.x and 2.x."""

    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()  # type: ignore[attr-defined]
    return model.json(ensure_ascii=False)


RequestId = Field(..., min_length=1, description="Unique request identifier.")
TraceId = Field(..., min_length=1, description="Distributed trace identifier.")
SessionId = Field(..., min_length=1, description="Conversation session ID.")
