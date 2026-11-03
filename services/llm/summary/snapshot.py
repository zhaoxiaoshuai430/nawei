"""Pack complete conversations and revalidated previous evidence within one budget."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from services.llm.summary.policy import token_bound
from services.llm.summary.privacy import contains_secret

GROUPS = ("personal_facts", "task_outcomes", "claims", "preference_observations", "learning_observations", "teaching_strategy_observations",
          "profile_change_proposals", "temporary_states", "topics")


def pack(rows: list[Any], previous: dict[str, Any] | None, references: list[Any],
         old_snapshot: dict[str, Any], budget: int) -> dict[str, Any]:
    from services.llm.summary.repository import digest

    def turn(row: Any) -> dict[str, Any]:
        item = {key: str(row[key]) if isinstance(row[key], UUID) else row[key] for key in
                ("turn_id", "session_id", "completed_seq", "user_message_id", "assistant_message_id",
                 "user_text", "assistant_text")}
        item["completed_at"] = row["completed_at"].isoformat()
        item["source_hash"] = digest([item["user_text"], item["assistant_text"]])
        return item

    def safe(item: dict[str, Any]) -> bool:
        return not contains_secret(item["user_text"] + "\n" + item["assistant_text"])

    def size(value: Any) -> int:
        return token_bound(json.dumps(value, ensure_ascii=False)) + 2

    old_hashes = {t["turn_id"]: t["source_hash"] for t in
                  old_snapshot.get("turns", []) + old_snapshot.get("reference_turns", [])}
    verified = [t for row in references if safe(t := turn(row)) and old_hashes.get(t["turn_id"]) == t["source_hash"]]
    sources = {t[k]: t for t in verified for k in ("user_message_id", "assistant_message_id")}
    prior: dict[str, Any] = {"summary_revision": previous["summary_revision"]} if previous else {}
    refs: dict[str, Any] = {}
    prior_budget = max(0, budget // 3 - size(prior))
    now = datetime.now(timezone.utc)
    for group in GROUPS:
        for item in (previous or {}).get(group, []):
            if item.get("expires_at") and datetime.fromisoformat(item["expires_at"]) <= now:
                continue
            mids = set(item["evidence_message_ids"] + item.get("counterevidence_message_ids", []))
            if not mids <= sources.keys():
                continue
            additions = {sources[mid]["turn_id"]: sources[mid] for mid in mids if sources[mid]["turn_id"] not in refs}
            cost = size(item) + sum(size(t) for t in additions.values()) + size(group)
            if cost > prior_budget:
                continue
            prior.setdefault(group, []).append(item)
            refs.update(additions)
            prior_budget -= cost
    prior = prior if refs else {}
    remaining = budget - size(prior) - size(list(refs.values())) - 128
    selected = []
    sensitive_count = 0
    for row in rows:
        item = turn(row)
        if not safe(item):
            sensitive_count += 1
            continue
        cost = size(item)
        if cost <= remaining:
            selected.append(item)
            remaining -= cost
    selected.reverse()
    # Reference coverage is separate, and never masquerades as recent conversation.
    recent_ids = {t["turn_id"] for t in selected}
    reference_turns = sorted((t for tid, t in refs.items() if tid not in recent_ids), key=lambda t: t["completed_seq"])
    omitted = len(rows) - len(selected)
    return {"turns": selected, "reference_turns": reference_turns, "previous_summary": prior or None,
            "previous_summary_revision": prior.get("summary_revision"), "eligible_turn_count": len(rows),
            "omitted_turn_count": omitted, "sensitive_turn_count": sensitive_count, "truncated": bool(omitted),
            "truncation_reason": "privacy_or_input_budget" if omitted else None}
