"""Real PostgreSQL concurrency gates. Set SUMMARY_TEST_DATABASE_URL to an isolated test server.

Each test owns a random schema; never loads the application's .env DSN.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from services.llm.summary.policy import SummarySettings
from services.llm.summary.repository import EpochChanged, RequestConflict, SummaryRepository, decode, guard_write, write_epoch
from services.llm.summary.schema import SummaryCandidate


@pytest_asyncio.fixture
async def repository():
    dsn = os.getenv("SUMMARY_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("SUMMARY_TEST_DATABASE_URL required for real PostgreSQL gates")
    import asyncpg
    schema = "summary_test_" + uuid4().hex
    admin = await asyncpg.connect(dsn)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    pool = None
    try:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=6, server_settings={"search_path": schema})
        async with pool.acquire() as conn:
            # Minimal pre-existing message table; migration itself runs unchanged.
            await conn.execute("""CREATE TABLE navi_messages(id uuid PRIMARY KEY,user_id text NOT NULL,
                session_id text NOT NULL,role text NOT NULL,content text NOT NULL,input_type text,
                created_at timestamptz NOT NULL DEFAULT now())""")
            await conn.execute(Path("deploy/migrations/001_conversation_summary.sql").read_text(encoding="utf-8"))
            await conn.execute(Path("deploy/migrations/002_summary_understanding.sql").read_text(encoding="utf-8"))
        yield SummaryRepository(SimpleNamespace(_pool=pool), SummarySettings(enabled=True, trigger_turns=2))
    finally:
        if pool:
            await pool.close()
        # Random schema created above, never a user supplied identifier.
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()


async def commit(repo, request_id="r1", user_id="owner", epoch=0, fingerprint="same"):
    await repo.state(user_id)
    return await repo.commit_turn(user_id=user_id, session_id="shared-session", request_id=request_id,
        fingerprint=fingerprint, epoch=epoch, user_text="数学请分步骤讲解", assistant_text="好的")


async def test_atomic_turn_idempotency_and_request_conflict(repository):
    a, b = await asyncio.gather(commit(repository), commit(repository))
    assert sorted([a["duplicate"], b["duplicate"]]) == [False, True]
    assert str(a["turn_id"]) == str(b["turn_id"])
    async with repository.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM navi_messages") == 2
    with pytest.raises(RequestConflict):
        await commit(repository, fingerprint="different")
    assert (await repository.state("owner"))["completed_seq"] == 1


async def test_verified_legacy_backfill_is_atomic_and_idempotent(repository):
    from scripts.backfill_verified_summary_report import backfill
    from tests.test_summary_report_backfill import fixture

    report, messages = fixture()
    async with repository.pool.acquire() as conn:
        for message in messages:
            await conn.execute("""INSERT INTO navi_messages(id,user_id,session_id,role,content)
                VALUES($1,$2,$3,$4,$5)""", uuid4(), message["user_id"], message["session_id"],
                message["role"], message["content"])
        report["results"][0]["answer"] = "mismatch"
        with pytest.raises(ValueError):
            await backfill(conn, report)
        assert await conn.fetchval("SELECT count(*) FROM navi_user_memory_state") == 0
        report["results"][0]["answer"] = messages[1]["content"]
        assert (await backfill(conn, report))["already_registered"] is False
        assert (await backfill(conn, report))["already_registered"] is True
        assert await conn.fetchval("SELECT count(*) FROM navi_messages") == 2
        assert await conn.fetchval("SELECT count(*) FROM navi_conversation_turns") == 1
    await repository.reset("owner")
    async with repository.pool.acquire() as conn:
        with pytest.raises(ValueError):
            await backfill(conn, report)


async def test_two_workers_share_one_global_slot_and_recover(repository):
    await commit(repository)
    await commit(repository, user_id="other")
    await repository.enqueue("owner")
    await repository.enqueue("other")
    jobs = await asyncio.gather(repository.claim("worker-a"), repository.claim("worker-b"))
    assert sum(job is not None for job in jobs) == 1
    old = next(job for job in jobs if job)
    async with repository.pool.acquire() as conn:
        await conn.execute("UPDATE navi_summary_slots SET lease_expires_at=now()-interval '1 second'")
        await conn.execute("UPDATE navi_summary_jobs SET lease_expires_at=now()-interval '1 second' WHERE job_id=$1", old["job_id"])
    new = await repository.claim("worker-c")
    assert new and new["job_id"] == old["job_id"] and new["lease_token"] != old["lease_token"]
    assert decode(new["input_snapshot"]) == decode(old["input_snapshot"])
    assert not await repository.publish(old, SummaryCandidate(), "test")
    assert await repository.publish(new, SummaryCandidate(), "test")


async def test_delete_fences_running_publish_and_legacy_writer(repository):
    await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    await repository.reset("owner")
    assert not await repository.publish(job, SummaryCandidate(), "test")
    assert (await repository.read("owner"))["summary"] is None
    assert await repository.enqueue("owner") is None
    with pytest.raises(EpochChanged):
        await commit(repository, request_id="late")
    token = write_epoch.set(("owner", 0))
    try:
        async with repository.pool.acquire() as conn:
            async with conn.transaction():
                with pytest.raises(EpochChanged):
                    await guard_write(conn, "owner")
    finally:
        write_epoch.reset(token)


async def test_crash_after_commit_discovered_and_manual_coalesces(repository):
    await commit(repository)
    await repository.discover()
    assert await repository.claim("worker") is None
    await commit(repository, request_id="r2")
    await repository.discover()
    a, b = await asyncio.gather(repository.enqueue("owner"), repository.enqueue("owner"))
    assert a["job_id"] == b["job_id"]
    assert await repository.job("other", a["job_id"]) is None
    assert (await repository.claim("worker"))["target_completed_seq"] == 2


async def test_publish_evidence_and_source_deletion(repository):
    turn = await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    candidate = SummaryCandidate.model_validate({"claims": [{"kind": "communication_preference", "subject": "数学",
        "content": "请分步骤", "origin": "user_explicit", "status": "supported",
        "evidence_message_ids": [str(turn["user_message_id"])]}]})
    assert await repository.publish(job, candidate, "test")
    body = (await repository.read("owner"))["summary"]
    assert body["coverage"]["actual_input_turn_count"] == 1
    async with repository.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM navi_summary_evidence") == 1
    await repository.release(job)
    await commit(repository, request_id="r2")
    await repository.enqueue("owner")
    second = await repository.claim("worker")
    async with repository.pool.acquire() as conn:
        await conn.execute("DELETE FROM navi_messages WHERE id=$1", turn["user_message_id"] if not isinstance(turn["user_message_id"], str)
                           else UUID(turn["user_message_id"]))
    assert (await repository.read("owner"))["summary"] is None
    assert not await repository.publish(second, SummaryCandidate(), "test")


async def install_memory_schema(repository):
    from services.llm.memory import _SCHEMA_SQL

    # Include the real legacy tables: a summary-only schema skips memory promotion.
    async with repository.pool.acquire() as conn:
        # Resolve only pgcrypto's function through its installed schema; keep
        # table lookup isolated so missing legacy tables never fall back to public.
        crypto_schema = await conn.fetchval("""SELECT n.nspname FROM pg_extension e
            JOIN pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='pgcrypto'""")
        if crypto_schema:
            quoted = '"' + crypto_schema.replace('"', '""') + '"'
            await conn.execute(f"""CREATE FUNCTION digest(text, text) RETURNS bytea
                LANGUAGE SQL IMMUTABLE STRICT AS 'SELECT {quoted}.digest($1, $2)'""")
        await conn.execute(_SCHEMA_SQL)
        await conn.execute(Path("deploy/migrations/003_teaching_preferences.sql").read_text(encoding="utf-8"))


async def test_topics_and_optional_preference_fields_publish_to_full_memory_schema(repository):
    await install_memory_schema(repository)
    first = await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    claim = {"kind": "communication_preference", "subject": "数学", "content": "请分步骤讲解",
             "origin": "user_explicit", "status": "supported",
             "evidence_message_ids": [first["user_message_id"]]}
    candidate = SummaryCandidate.model_validate({
        "topics": [{"topic": "数学", "evidence_message_ids": [first["user_message_id"]]}],
        "claims": [claim],
    })
    assert await repository.publish(job, candidate, "test")
    assert (await repository.read("owner"))["summary"]["topics"][0]["topic"] == "数学"
    async with repository.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id,content FROM navi_memories WHERE user_id='owner' AND NOT is_deleted")
        assert row["content"] == claim["content"]
    await repository.release(job)

    # Explicit defaults in the next model response must update the same memory.
    second = await commit(repository, request_id="r2")
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    claim.update(preference_type="communication", target="数学", polarity="like",
                 evidence_message_ids=[second["user_message_id"]], content="数学每次只讲一个步骤")
    assert await repository.publish(job, SummaryCandidate.model_validate({"claims": [claim]}), "test")
    async with repository.pool.acquire() as conn:
        rows = await conn.fetch("SELECT id,content FROM navi_memories WHERE user_id='owner' AND NOT is_deleted")
        assert len(rows) == 1
        assert rows[0]["id"] == row["id"] and rows[0]["content"] == claim["content"]
    await repository.release(job)
    await repository.reset("owner")
    async with repository.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM navi_memories WHERE user_id='owner' AND NOT is_deleted") == 0
    assert (await repository.read("owner"))["summary"] is None


async def test_summary_plan_projection_retires_omissions_without_touching_other_sources(repository):
    from services.llm.summary.repository import normalize
    from tests.test_summary_understanding import claim, multi_turn

    await install_memory_schema(repository)
    snapshot = multi_turn()
    first, _ = normalize(SummaryCandidate(claims=[claim(snapshot, "open_task", subject="海报", progress="open")]), snapshot)
    first["summary_revision"] = 1
    async with repository.pool.acquire() as conn:
        await repository._promote_durable_memories(conn, {"user_id": "owner"}, uuid4(), first, snapshot)
        await conn.execute("""INSERT INTO navi_memories(user_id,memory_type,content,source_agent)
            VALUES('owner','plan','用户单独保存的旅行计划','memory'),
                  ('other','plan','另一用户的计划','conversation_summary')""")
        second, _ = normalize(SummaryCandidate(claims=[claim(snapshot, "open_task", subject="封面", progress="open")]), snapshot)
        second["summary_revision"] = 2
        await repository._promote_durable_memories(conn, {"user_id": "owner"}, uuid4(), second, snapshot)
        rows = await conn.fetch("SELECT user_id,summary,source_agent,is_deleted,deleted_reason FROM navi_memories")
        old = next(row for row in rows if row["summary"] == "海报")
        assert old["is_deleted"] and old["deleted_reason"] == "summary_plan_not_current"
        assert not next(row for row in rows if row["summary"] == "封面")["is_deleted"]
        assert all(not row["is_deleted"] for row in rows if row["source_agent"] == "memory" or row["user_id"] == "other")


async def test_retry_uses_same_snapshot_and_reenable_retains_shadow(repository):
    await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    await repository.fail(job, "timeout")
    await repository.release(job)
    await commit(repository, request_id="r2")
    async with repository.pool.acquire() as conn:
        await conn.execute("UPDATE navi_summary_jobs SET next_attempt_at=now()")
    retry = await repository.claim("worker2")
    assert retry["input_snapshot_id"] == job["input_snapshot_id"]
    assert decode(retry["input_snapshot"]) == decode(job["input_snapshot"])
    assert await repository.publish(retry, SummaryCandidate(), "test")
    await repository.reset("owner", {"apply_to_prompt": True})
    assert (await repository.read("owner"))["summary"] is not None


async def test_deleted_source_is_rejected_before_retry_model_input(repository):
    turn = await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    await repository.fail(job, "timeout")
    await repository.release(job)
    async with repository.pool.acquire() as conn:
        await conn.execute("UPDATE navi_summary_jobs SET next_attempt_at=now()")
        await conn.execute("DELETE FROM navi_messages WHERE id=$1", UUID(turn["user_message_id"]))
    assert await repository.claim("worker2") is None
    status = await repository.job("owner", job["job_id"])
    assert status["status"] == "obsolete" and status["error_code"] == "source_changed"


async def test_real_published_summary_automatic_online_and_delete(repository):
    from services.llm.summary.context import read_context
    repository.settings = repository.settings.model_copy(update={"apply_to_prompt": True, "read_timeout_seconds": 1})
    turn = await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    candidate = SummaryCandidate.model_validate({"claims": [{"kind": "communication_preference", "subject": "数学",
        "content": "数学请分步骤", "origin": "user_explicit", "status": "supported",
        "evidence_message_ids": [turn["user_message_id"]]}]})
    assert await repository.publish(job, candidate, "test")
    assert (await repository.state("owner"))["apply_to_prompt"] is True
    selected = await read_context(repository, "owner", "数学", "full", expected_epoch=0)
    assert selected and selected.entries == ("数学请分步骤",)
    assert await read_context(repository, "other", "数学", "full") is None
    await repository.reset("owner")
    assert await read_context(repository, "owner", "数学", "full") is None


async def test_previous_summary_reference_outside_window_and_deletion_fence(repository):
    repository.settings = SummarySettings(enabled=True, window_turns=1, input_max_tokens=30000)
    first = await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    candidate = SummaryCandidate.model_validate({"claims": [{"kind": "open_task", "subject": "数学",
        "content": "数学分数还没完成", "origin": "user_explicit", "status": "supported", "progress": "partial",
        "evidence_message_ids": [first["user_message_id"]]}]})
    assert await repository.publish(job, candidate, "test")
    await repository.release(job)
    await commit(repository, request_id="r2")
    await repository.enqueue("owner")
    second = await repository.claim("worker")
    snapshot = decode(second["input_snapshot"])
    assert snapshot["previous_summary"]["summary_revision"] == 1
    assert [t["turn_id"] for t in snapshot["reference_turns"]] == [first["turn_id"]]
    assert await repository.publish(second, candidate, "test")
    body = (await repository.read("owner"))["summary"]
    assert body["schema_version"] == 2 and body["coverage"]["previous_summary_revision"] == 1
    assert "previous_summary" not in body["coverage"]
    async with repository.pool.acquire() as conn:
        await conn.execute("DELETE FROM navi_messages WHERE id=$1", UUID(first["user_message_id"]))
    assert (await repository.read("owner"))["summary"] is None


async def test_changed_reference_invalidates_retry_and_publish(repository):
    repository.settings = SummarySettings(enabled=True, window_turns=1, input_max_tokens=30000)
    first = await commit(repository)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    candidate = SummaryCandidate.model_validate({"claims": [{"kind": "open_task", "subject": "数学",
        "content": "数学分数还没完成", "origin": "user_explicit", "status": "supported",
        "evidence_message_ids": [first["user_message_id"]]}]})
    assert await repository.publish(job, candidate, "test")
    await repository.release(job)
    await commit(repository, request_id="r2")
    await repository.enqueue("owner")
    second = await repository.claim("worker")
    async with repository.pool.acquire() as conn:
        await conn.execute("UPDATE navi_messages SET content='原文更正' WHERE id=$1", UUID(first["user_message_id"]))
    assert not await repository.publish(second, candidate, "test")
    assert (await repository.job("owner", second["job_id"]))["status"] == "obsolete"


async def test_migration_preserves_opt_out_and_retires_old_jobs(repository):
    await commit(repository)
    await repository.reset("owner", {"apply_to_prompt": False})
    async with repository.pool.acquire() as conn:
        await conn.execute("""INSERT INTO navi_summary_jobs(job_id,user_id,memory_epoch,target_completed_seq,
            window_policy_version,prompt_version) VALUES($1,'owner',1,1,'old','summary-v1')""", uuid4())
        migration = Path("deploy/migrations/002_summary_understanding.sql").read_text(encoding="utf-8")
        await conn.execute(migration)
        await conn.execute(migration)
        assert await conn.fetchval("SELECT status FROM navi_summary_jobs WHERE prompt_version='summary-v1'") == "obsolete"
    assert not (await repository.state("owner"))["apply_to_prompt"]
    assert (await repository.state("new-owner"))["apply_to_prompt"]
    await repository.check_schema()
    # A still-running legacy process must not accidentally create v2 work.
    async with repository.pool.acquire() as conn:
        default = await conn.fetchval("""SELECT column_default FROM information_schema.columns
            WHERE table_schema=current_schema() AND table_name='navi_summary_jobs' AND column_name='prompt_version'""")
        assert "summary-v1" in default
    await commit(repository, user_id="new-owner")
    new_job = await repository.enqueue("new-owner")
    assert new_job["prompt_version"] == "summary-v2" and new_job["schema_version"] == 2


async def test_trigger_twenty_and_hundred_turn_window(repository):
    repository.settings = SummarySettings(enabled=True, trigger_turns=20, input_max_tokens=64000)
    for index in range(19):
        await commit(repository, request_id=f"r{index}")
    await repository.discover()
    assert await repository.claim("worker") is None
    await commit(repository, request_id="r19")
    await repository.discover()
    first = await repository.claim("worker")
    assert first["target_completed_seq"] == 20
    assert len(decode(first["input_snapshot"])["turns"]) == 20
    assert await repository.publish(first, SummaryCandidate(), "test")
    await repository.release(first)
    for index in range(20, 105):
        await commit(repository, request_id=f"r{index}")
    await repository.discover()
    latest = await repository.claim("worker")
    turns = decode(latest["input_snapshot"])["turns"]
    assert len(turns) == 100
    assert turns[0]["completed_seq"] == 6 and turns[-1]["completed_seq"] == 105


async def test_window_age_and_oversized_whole_turn_omission(repository):
    await commit(repository, request_id="old")
    big = await commit(repository, request_id="large")
    small = await commit(repository, request_id="recent")
    async with repository.pool.acquire() as conn:
        await conn.execute("UPDATE navi_conversation_turns SET completed_at=now()-interval '31 days' WHERE request_id='old'")
        await conn.execute("UPDATE navi_messages SET content=$2 WHERE id=$1", UUID(big["user_message_id"]), "x"*20000)
    await repository.enqueue("owner")
    job = await repository.claim("worker")
    snapshot = decode(job["input_snapshot"])
    assert snapshot["eligible_turn_count"] == 2
    assert snapshot["omitted_turn_count"] == 1 and snapshot["truncated"]
    assert [item["turn_id"] for item in snapshot["turns"]] == [small["turn_id"]]


async def test_teaching_preferences_publish_timestamp_and_refresh(repository):
    """The third migration stores native timestamptz, not ISO text parameters."""
    async with repository.pool.acquire() as conn:
        await conn.execute(Path("deploy/migrations/003_teaching_preferences.sql").read_text(encoding="utf-8"))
    for index in range(2):
        turn = await commit(repository, request_id=f"teaching-{index}")
        await repository.enqueue("owner")
        job = await repository.claim("worker")
        candidate = SummaryCandidate.model_validate({"teaching_strategy_observations": [{
            "kind": "teaching_strategy_observation", "subject": "数学分步骤",
            "content": "数学分步骤讲解有助于理解", "origin": "user_explicit", "status": "supported",
            "evidence_message_ids": [turn["user_message_id"]],
        }]})
        assert await repository.publish(job, candidate, "test")
        await repository.release(job)
        async with repository.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM navi_teaching_preferences WHERE user_id='owner'")
            expected_time = await conn.fetchval("SELECT completed_at FROM navi_conversation_turns WHERE turn_id=$1",
                                                UUID(turn["turn_id"]))
        assert len(rows) == 1 and rows[0]["observed_at"] == expected_time
        assert rows[0]["source_summary_id"] is not None
        result = await repository.read("owner")
        assert len(result["teaching_preferences"]) == 1
        assert result["teaching_preferences"][0]["observed_at"] == expected_time.isoformat()
    await repository.reset("owner")
    assert not (await repository.read("owner"))["teaching_preferences"]
