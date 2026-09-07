"""Build deterministic graph candidates from TutorGraph intent output."""

from __future__ import annotations

from typing import Any

from services.knowledge.schemas import GraphCandidate, KnowledgeExpansionOptions


_CONCEPT_PATHS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "三角函数": (
        ("三角函数", ("Concept",)),
        ("三角学", ("Domain", "KnowledgeNode")),
        ("初等数学", ("Domain", "KnowledgeNode")),
        ("数学", ("KnowledgeNode",)),
    ),
}


def build_semantic_route(*, query: str, intent: dict[str, Any]) -> dict[str, Any]:
    subject = str(intent.get("subject") or "unknown").strip()
    concept = str(intent.get("concept") or "").strip()
    aliases = _dedupe([str(item).strip() for item in intent.get("concept_aliases") or []])
    candidates: list[GraphCandidate] = []

    for priority, (name, entity_types) in enumerate(_CONCEPT_PATHS.get(concept, ()), start=1):
        candidates.append(
            GraphCandidate(
                name=name,
                entity_types=list(entity_types),
                aliases=aliases if priority == 1 else [],
                priority=priority,
            )
        )

    if concept and not candidates:
        candidates.append(
            GraphCandidate(
                name=concept,
                entity_types=["Concept"],
                aliases=aliases,
                priority=1,
            )
        )
    if concept and subject != "unknown" and all(item.name != subject for item in candidates):
        candidates.append(
            GraphCandidate(
                name=subject,
                entity_types=["KnowledgeNode", "Subject", "Discipline"],
                priority=len(candidates) + 1,
            )
        )

    dimensions = ["taxonomy", "prerequisites", "history", "applications"] if candidates else []
    expansion = KnowledgeExpansionOptions(
        taxonomy=bool(candidates),
        prerequisites=bool(concept),
        history=bool(concept),
        applications=bool(concept),
        taxonomy_limit=12,
        prerequisite_limit=6,
        history_limit=6,
        application_limit=8,
    )
    return {
        "query": query,
        "subject": subject,
        "concept": concept or None,
        "aliases": aliases,
        "graph_candidates": [item.model_dump() for item in candidates],
        "requested_dimensions": dimensions,
        "expansion": expansion.model_dump(),
    }


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result[:20]
