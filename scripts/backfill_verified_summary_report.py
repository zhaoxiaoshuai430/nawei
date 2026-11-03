"""Register a verified legacy test report without duplicating original messages.

Explicit operation only; never automatically guesses pairs from arbitrary history.
Run as a module with --report PATH and DATABASE_URL in the environment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import asyncpg

from services.llm.summary.repository import digest


def verify_report(report, messages):
    turns = report["results"]
    count = report["requested_rounds"]
    if not turns or report["completed_rounds"] != count or len(turns) != count:
        raise ValueError("Report is incomplete")
    if len(messages) != 2 * count:
        raise ValueError("Session message count differs from report")
    for index, turn in enumerate(turns):
        metadata = turn["metadata"]
        if (turn["round"] != index + 1 or not metadata.get("persistence", {}).get("saved_turn")
                or metadata.get("fallback") or not turn["prompt"] or not turn["answer"]):
            raise ValueError("Report contains an unverified turn")
        for message, role, content in (
            (messages[2 * index], "user", turn["prompt"]),
            (messages[2 * index + 1], "assistant", turn["answer"]),
        ):
            if (message["user_id"] != report["user_id"] or message["session_id"] != report["session_id"]
                    or message["role"] != role or message["content"] != content):
                raise ValueError("Original database messages do not exactly match report")
    return turns


async def backfill(conn, report):
    uid, session = report["user_id"], report["session_id"]
    source = "verified_report:" + digest(report)
    async with conn.transaction():
        await conn.execute("INSERT INTO navi_user_memory_state(user_id) VALUES($1) ON CONFLICT DO NOTHING", uid)
        state = await conn.fetchrow("SELECT * FROM navi_user_memory_state WHERE user_id=$1 FOR UPDATE", uid)
        messages = await conn.fetch("""SELECT * FROM navi_messages WHERE user_id=$1 AND session_id=$2
            ORDER BY created_at,id FOR SHARE""", uid, session)
        turns = verify_report(report, messages)
        existing = await conn.fetch("SELECT * FROM navi_conversation_turns WHERE user_id=$1 ORDER BY completed_seq", uid)
        if existing:
            if len(existing) == len(turns) and all(
                row["source"] == source and row["memory_epoch"] == state["memory_epoch"]
                and row["completed_seq"] == index + 1
                and row["user_message_id"] == messages[2 * index]["id"]
                and row["assistant_message_id"] == messages[2 * index + 1]["id"]
                for index, row in enumerate(existing)
            ):
                return {"registered_turns": len(existing), "already_registered": True}
            raise ValueError("Existing ledger conflicts; refusing to merge or rewrite it")
        if any(state[key] != 0 for key in ("memory_epoch", "completed_seq", "excluded_seq", "summary_seq", "summary_revision")):
            raise ValueError("User has prior ledger/deletion state; refusing historical resurrection")
        if not state["enabled"]:
            raise ValueError("User summaries are disabled")
        for index, turn in enumerate(turns):
            u, a = messages[2 * index:2 * index + 2]
            await conn.execute("""INSERT INTO navi_conversation_turns
                (turn_id,user_id,session_id,request_id,request_fingerprint,memory_epoch,status,completed_seq,
                 user_message_id,assistant_message_id,completed_at,source)
                VALUES($1,$2,$3,$4,$5,$6,'completed',$7,$8,$9,$10,$11)""",
                uuid4(), uid, session, f"{source}:{index + 1}",
                digest([turn["prompt"], turn["answer"]]), state["memory_epoch"], index + 1,
                u["id"], a["id"], a["created_at"], source)
        await conn.execute("""UPDATE navi_user_memory_state SET completed_seq=$2,desired_target_seq=$2
            WHERE user_id=$1""", uid, len(turns))
        return {"registered_turns": len(turns), "already_registered": False}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        result = await backfill(conn, report)
    finally:
        await conn.close()
    result.update(user_id=report["user_id"], session_id=report["session_id"], report_sha256=digest(report))
    args.report.with_name("summary-backfill.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
