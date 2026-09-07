"""Plan bounded graph expansion and Qdrant filters from resolved entities."""

from __future__ import annotations

from services.knowledge.schemas import KnowledgeExpansionOptions, ResolvedKnowledgeEntity


class KnowledgeExpansionPlanner:
    _RELATION_GROUPS = {
        "taxonomy": [
            "BELONGS_TO",
            "CONTAINS",
            "HAS_SUBCONCEPT",
            "HAS_TEACHING_DOMAIN",
            "SPECIALIZATION_OF",
            "MAPPED_TO",
        ],
        "prerequisites": ["PREREQUISITE_OF", "REQUIRES"],
        "history": [
            "EVOLVED_FROM",
            "DEVELOPED_DURING",
            "PRECEDES",
            "CONTRIBUTED_TO",
            "ENABLED_BY",
        ],
        "applications": ["APPLIED_IN", "EXPLAINS", "USES", "AFFECTS", "PRACTICES"],
    }

    def relation_groups(
        self,
        expansion: KnowledgeExpansionOptions,
    ) -> dict[str, tuple[list[str], int]]:
        groups: dict[str, tuple[list[str], int]] = {}
        if expansion.taxonomy and expansion.taxonomy_limit:
            groups["taxonomy"] = (self._RELATION_GROUPS["taxonomy"], expansion.taxonomy_limit)
        if expansion.prerequisites and expansion.prerequisite_limit:
            groups["prerequisites"] = (
                self._RELATION_GROUPS["prerequisites"],
                expansion.prerequisite_limit,
            )
        if expansion.history and expansion.history_limit:
            groups["history"] = (self._RELATION_GROUPS["history"], expansion.history_limit)
        if expansion.applications and expansion.application_limit:
            groups["applications"] = (
                self._RELATION_GROUPS["applications"],
                expansion.application_limit,
            )
        return groups

    @staticmethod
    def qdrant_filters(resolved: list[ResolvedKnowledgeEntity]) -> dict[str, str]:
        if not resolved:
            return {}
        entity = resolved[0]
        if entity.entity_type == "Concept":
            return {"concept": entity.name}
        if entity.entity_type == "Domain":
            return {"domain": entity.name}
        if entity.entity_type in {"KnowledgeNode", "Subject", "Discipline"}:
            return {"subject": entity.name}
        return {}
