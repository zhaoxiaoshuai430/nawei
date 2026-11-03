"""Resolve repeated stateful claims by evidence time, never model array order."""
from datetime import datetime
from typing import Any


def claim_key(item: dict[str, Any]) -> str:
    if item["kind"] in {"preference", "communication_preference"}:
        identity = f"{item.get('preference_type') or 'communication'}|{(item.get('target') or item['subject']).strip().lower()}"
    else:
        identity = f"{item['kind']}|{item['subject'].strip().lower()}"
    return f"{identity}|{item.get('scope', 'subject')}"


def latest_claims(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stateful = {"open_task", "goal", "personal_fact", "temporary_state", "preference", "communication_preference"}
    latest: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["kind"] not in stateful:
            continue
        key = claim_key(item)
        def order(value: dict[str, Any]) -> tuple:
            return (datetime.fromisoformat(value["observed_at"]), value.get("observed_seq", 0),
                    value.get("progress") in {"resolved", "cancelled"},
                    value.get("status") in {"contradicted", "expired"})
        if key not in latest or order(item) > order(latest[key]):
            latest[key] = item
    return [item for item in items if item["kind"] not in stateful or latest[claim_key(item)] is item]
