"""Explicit additive summary migration: python -m deploy.scripts.migrate_conversation_summary.

Uses DATABASE_URL from the invoking environment; does not load or print secrets.
The existing LLM memory schema must already have been initialized.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg


async def migrate() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")
    connection = await asyncpg.connect(dsn, timeout=10)
    try:
        if not await connection.fetchval("SELECT to_regclass('navi_messages')"):
            raise RuntimeError("Initialize the existing LLM memory schema first")
        await connection.execute("SELECT pg_advisory_lock(hashtext('nawei_summary_migration_v1'))")
        directory = Path(__file__).resolve().parents[1] / "migrations"
        for name in ("001_conversation_summary.sql", "002_summary_understanding.sql", "003_teaching_preferences.sql"):
            await connection.execute((directory / name).read_text(encoding="utf-8"))
        print("Conversation summary migrations v1-v3 applied.")
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(migrate())
