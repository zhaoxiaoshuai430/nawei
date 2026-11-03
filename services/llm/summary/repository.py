"""Short PostgreSQL transactions; no model calls or volatile job queues here."""
from __future__ import annotations

import hashlib
import json
import random
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from services.llm.summary.policy import SummarySettings
from services.llm.summary.claims import claim_key, latest_claims
from services.llm.summary.schema import SummaryCandidate
from services.llm.summary.privacy import prohibited_conclusion

write_epoch: ContextVar[tuple[str, int] | None] = ContextVar("summary_write_epoch", default=None)


async def guard_write(conn: Any, user_id: str) -> None:
    expected = write_epoch.get()
    if expected is None:
        return
    if conn is None:
        raise EpochChanged("Cannot verify background write epoch")
    epoch = await conn.fetchval("SELECT memory_epoch FROM navi_user_memory_state WHERE user_id=$1 FOR UPDATE", user_id)
    if expected != (user_id, epoch):
        raise EpochChanged("Discarding obsolete background memory write")


class EpochChanged(RuntimeError):
    pass


class RequestConflict(ValueError):
    pass


def decode(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def memory_content_hash(value: str) -> str:
    """Match the content hash used by MemoryStore for durable memories."""
    return hashlib.sha1(value.strip().lower().encode("utf-8")).hexdigest()


def _teaching_dict(row: Any) -> dict[str, Any]:
    item = dict(row)
    for key in ("observed_at", "expires_at"):
        if item.get(key) is not None and hasattr(item[key], "isoformat"):
            item[key] = item[key].isoformat()
    return item


class SummaryRepository:
    def __init__(self, memory: Any, settings: SummarySettings) -> None:
        self.memory = memory
        self.settings = settings

    @property
    def pool(self) -> Any:
        if self.memory._pool is None:
            raise RuntimeError("Summary PostgreSQL is unavailable")
        return self.memory._pool

    async def check_schema(self) -> None:
        async with self.pool.acquire() as conn:
            if not await conn.fetchval("SELECT to_regclass('navi_summary_slots')"):
                raise RuntimeError("Apply deploy/migrations/001_conversation_summary.sql before enabling summaries")
            default = await conn.fetchval("""SELECT column_default FROM information_schema.columns
                WHERE table_schema=current_schema() AND table_name='navi_user_memory_state' AND column_name='apply_to_prompt'""")
            if default != "true":
                raise RuntimeError("Apply deploy/migrations/002_summary_understanding.sql before enabling summary v2")

    async def state(self, user_id: str) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            await conn.execute("INSERT INTO navi_user_memory_state(user_id) VALUES($1) ON CONFLICT DO NOTHING", user_id)
            return dict(await conn.fetchrow("SELECT * FROM navi_user_memory_state WHERE user_id=$1", user_id))

    async def assert_epoch(self, user_id: str, epoch: int) -> None:
        async with self.pool.acquire() as conn:
            actual = await conn.fetchval("SELECT memory_epoch FROM navi_user_memory_state WHERE user_id=$1", user_id)
        if actual != epoch:
            raise EpochChanged("User memory changed during the request")

    async def previous_turn(self, user_id: str, request_id: str, fingerprint: str) -> str | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""SELECT t.request_fingerprint,a.content FROM navi_conversation_turns t
                JOIN navi_messages a ON a.id=t.assistant_message_id AND a.user_id=t.user_id
                JOIN navi_user_memory_state s ON s.user_id=t.user_id AND s.memory_epoch=t.memory_epoch
                WHERE t.user_id=$1 AND t.request_id=$2""", user_id, request_id)
        if row and row["request_fingerprint"] != fingerprint:
            raise RequestConflict("request_id was already used for different input")
        return str(row["content"]) if row else None

    async def commit_turn(self, *, user_id: str, session_id: str, request_id: str,
                          fingerprint: str, epoch: int, user_text: str, assistant_text: str) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await conn.fetchrow("SELECT * FROM navi_user_memory_state WHERE user_id=$1 FOR UPDATE", user_id)
                if not state or state["memory_epoch"] != epoch:
                    raise EpochChanged("User memory changed before history commit")
                previous = await conn.fetchrow(
                    "SELECT * FROM navi_conversation_turns WHERE user_id=$1 AND request_id=$2", user_id, request_id)
                if previous:
                    if previous["memory_epoch"] != epoch:
                        raise RequestConflict("request_id belongs to an obsolete memory epoch")
                    if previous["request_fingerprint"] != fingerprint:
                        raise RequestConflict("request_id was already used for different input")
                    return {**dict(previous), "duplicate": True}
                user_mid, assistant_mid, turn_id = uuid4(), uuid4(), uuid4()
                await conn.executemany(
                    "INSERT INTO navi_messages(id,user_id,session_id,role,content,input_type,created_at) VALUES($1,$2,$3,$4,$5,'text',clock_timestamp())",
                    [(user_mid, user_id, session_id, "user", user_text),
                     (assistant_mid, user_id, session_id, "assistant", assistant_text)])
                seq = state["completed_seq"] + 1
                await conn.execute(
                    """INSERT INTO navi_conversation_turns(turn_id,user_id,session_id,request_id,request_fingerprint,
                    memory_epoch,status,completed_seq,user_message_id,assistant_message_id)
                    VALUES($1,$2,$3,$4,$5,$6,'completed',$7,$8,$9)""",
                    turn_id, user_id, session_id, request_id, fingerprint, epoch, seq, user_mid, assistant_mid)
                await conn.execute("""UPDATE navi_user_memory_state SET completed_seq=$2,desired_target_seq=$2
                                    WHERE user_id=$1""", user_id, seq)
                return {"turn_id": str(turn_id), "completed_seq": seq, "duplicate": False,
                        "user_message_id": str(user_mid), "assistant_message_id": str(assistant_mid)}

    async def enqueue(self, user_id: str, *, manual: bool = True) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await conn.fetchrow("SELECT * FROM navi_user_memory_state WHERE user_id=$1 FOR UPDATE", user_id)
                if not state or not state["enabled"] or state["desired_target_seq"] <= state["excluded_seq"]:
                    return None
                if not manual and state["desired_target_seq"] - max(state["summary_seq"], state["excluded_seq"]) < self.settings.trigger_turns:
                    return None
                active = await conn.fetchrow("""SELECT * FROM navi_summary_jobs WHERE user_id=$1 AND memory_epoch=$2
                    AND prompt_version='summary-v2' AND window_policy_version=$3
                    AND status IN ('pending','running','retry_wait') ORDER BY created_at DESC LIMIT 1""",
                    user_id, state["memory_epoch"], self.settings.window_version)
                if active:
                    return dict(active)
                await conn.execute("""INSERT INTO navi_summary_jobs(job_id,user_id,memory_epoch,target_completed_seq,window_policy_version,prompt_version,schema_version)
                    VALUES($1,$2,$3,$4,$5,'summary-v2',2) ON CONFLICT DO NOTHING""",
                    uuid4(), user_id, state["memory_epoch"], state["desired_target_seq"], self.settings.window_version)
                row = await conn.fetchrow("""SELECT * FROM navi_summary_jobs WHERE user_id=$1 AND memory_epoch=$2
                    AND target_completed_seq=$3 AND window_policy_version=$4 AND prompt_version='summary-v2'""",
                    user_id, state["memory_epoch"], state["desired_target_seq"], self.settings.window_version)
                return dict(row)

    async def discover(self) -> None:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""SELECT s.user_id FROM navi_user_memory_state s WHERE enabled
                AND desired_target_seq - GREATEST(summary_seq,excluded_seq) >= $1
                AND NOT EXISTS (SELECT 1 FROM navi_summary_jobs j WHERE j.user_id=s.user_id AND j.memory_epoch=s.memory_epoch
                    AND j.prompt_version='summary-v2' AND j.window_policy_version=$2
                    AND (j.status IN ('pending','running','retry_wait') OR
                        (j.target_completed_seq=s.desired_target_seq AND j.window_policy_version=$2 AND j.prompt_version='summary-v2')))
                ORDER BY user_id LIMIT 100""", self.settings.trigger_turns, self.settings.window_version)
        for row in rows:
            await self.enqueue(row["user_id"], manual=False)

    async def _snapshot(self, conn: Any, job: Any) -> dict[str, Any]:
        rows = await conn.fetch("""SELECT t.*,u.content AS user_text,a.content AS assistant_text
            FROM navi_conversation_turns t JOIN navi_messages u ON u.id=t.user_message_id AND u.user_id=t.user_id
            JOIN navi_messages a ON a.id=t.assistant_message_id AND a.user_id=t.user_id
            JOIN navi_user_memory_state s ON s.user_id=t.user_id
            WHERE t.user_id=$1 AND t.memory_epoch=$2 AND t.status='completed'
            AND t.completed_seq <= $3 AND t.completed_seq > s.excluded_seq
            AND t.completed_at >= now()-$4::interval ORDER BY t.completed_seq DESC LIMIT $5""",
            job["user_id"], job["memory_epoch"], job["target_completed_seq"],
            timedelta(days=self.settings.window_max_days), self.settings.window_turns)
        from services.llm.agents.summary_agent import prompt_overhead
        from services.llm.summary.snapshot import pack
        previous = await conn.fetchrow("""SELECT d.summary_id,d.document,j.input_snapshot
            FROM navi_conversation_summaries d JOIN navi_user_memory_state s ON s.user_id=d.user_id
            JOIN navi_summary_jobs j ON j.job_id=d.job_id
            WHERE d.user_id=$1 AND d.memory_epoch=$2 AND d.summary_revision=s.summary_revision""",
            job["user_id"], job["memory_epoch"])
        references = []
        if previous:
            references = await conn.fetch("""SELECT t.*,u.content AS user_text,a.content AS assistant_text
                FROM navi_conversation_turns t JOIN navi_messages u ON u.id=t.user_message_id AND u.user_id=t.user_id
                JOIN navi_messages a ON a.id=t.assistant_message_id AND a.user_id=t.user_id
                JOIN navi_user_memory_state s ON s.user_id=t.user_id
                WHERE t.user_id=$1 AND t.memory_epoch=$2 AND t.completed_seq>s.excluded_seq AND t.completed_seq<=$3
                AND EXISTS (SELECT 1 FROM navi_summary_evidence e WHERE e.summary_id=$4 AND e.turn_id=t.turn_id)
                ORDER BY t.completed_seq DESC LIMIT 500""", job["user_id"], job["memory_epoch"],
                job["target_completed_seq"], previous["summary_id"])
        return pack(rows, decode(previous["document"]) if previous else None, references,
                    (decode(previous["input_snapshot"]) or {}) if previous else {},
                    self.settings.input_max_tokens - prompt_overhead())

    async def claim(self, owner: str) -> dict[str, Any] | None:
        lease = timedelta(seconds=self.settings.timeout_seconds + 15)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                slot = await conn.fetchrow("""SELECT * FROM navi_summary_slots WHERE slot_id=1
                    AND (lease_expires_at IS NULL OR lease_expires_at <= now()) FOR UPDATE SKIP LOCKED""")
                if not slot:
                    return None
                await conn.execute("""UPDATE navi_summary_jobs SET status='failed',error_code='lease_expired',finished_at=now()
                    WHERE status='running' AND lease_expires_at <= now() AND attempt_count >= $1""", self.settings.max_attempts)
                job = await conn.fetchrow("""SELECT j.* FROM navi_summary_jobs j
                    JOIN navi_user_memory_state s ON s.user_id=j.user_id AND s.memory_epoch=j.memory_epoch AND s.enabled
                    WHERE j.attempt_count < $1 AND j.window_policy_version=$2 AND j.prompt_version='summary-v2'
                    AND ((j.status IN ('pending','retry_wait') AND j.next_attempt_at<=now())
                        OR (j.status='running' AND j.lease_expires_at<=now()))
                    ORDER BY j.created_at LIMIT 1 FOR UPDATE OF j SKIP LOCKED""",
                    self.settings.max_attempts, self.settings.window_version)
                if not job:
                    return None
                snapshot = decode(job["input_snapshot"]) if job["input_snapshot"] else await self._snapshot(conn, job)
                if job["input_snapshot"] and not await self._sources_valid(conn, job, snapshot):
                    await conn.execute("""UPDATE navi_summary_jobs SET status='obsolete',input_snapshot=NULL,
                        error_code='source_changed',finished_at=now() WHERE job_id=$1""", job["job_id"])
                    return None
                revision = await conn.fetchval("SELECT summary_revision FROM navi_user_memory_state WHERE user_id=$1", job["user_id"])
                token = uuid4()
                row = await conn.fetchrow("""UPDATE navi_summary_jobs SET status='running',attempt_count=attempt_count+1,
                    lease_owner=$2,lease_token=$3,lease_expires_at=now()+$4::interval,
                    input_snapshot=$5::jsonb,input_snapshot_id=COALESCE(input_snapshot_id,$6),
                    expected_summary_revision=COALESCE(expected_summary_revision,$7) WHERE job_id=$1 RETURNING *""",
                    job["job_id"], owner, token, lease, json.dumps(snapshot, ensure_ascii=False), uuid4(), revision)
                await conn.execute("UPDATE navi_summary_slots SET lease_token=$1,lease_expires_at=now()+$2::interval WHERE slot_id=1", token, lease)
                return dict(row)

    async def _sources_valid(self, conn: Any, job: Any, snapshot: dict[str, Any]) -> bool:
        for turn in snapshot["turns"] + snapshot.get("reference_turns", []):
            row = await conn.fetchrow("""SELECT u.content AS u,a.content AS a FROM navi_conversation_turns t
                JOIN navi_messages u ON u.id=t.user_message_id AND u.user_id=t.user_id
                JOIN navi_messages a ON a.id=t.assistant_message_id AND a.user_id=t.user_id
                WHERE t.turn_id=$1 AND t.user_id=$2 AND t.memory_epoch=$3""",
                UUID(turn["turn_id"]), job["user_id"], job["memory_epoch"])
            if not row or digest([row["u"], row["a"]]) != turn["source_hash"]:
                return False
        return True

    async def renew(self, job: dict[str, Any]) -> bool:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                token = await conn.fetchval("""UPDATE navi_summary_slots SET lease_expires_at=now()+$2::interval
                    WHERE slot_id=1 AND lease_token=$1 AND lease_expires_at>now() RETURNING lease_token""",
                    job["lease_token"], timedelta(seconds=self.settings.timeout_seconds+15))
                if not token:
                    return False
                return bool(await conn.fetchval("""UPDATE navi_summary_jobs SET lease_expires_at=now()+$3::interval
                    WHERE job_id=$1 AND lease_token=$2 AND status='running' AND lease_expires_at>now() RETURNING job_id""",
                    job["job_id"], token, timedelta(seconds=self.settings.timeout_seconds+15)))

    async def release(self, job: dict[str, Any]) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE navi_summary_slots SET lease_token=NULL,lease_expires_at=NULL WHERE lease_token=$1", job["lease_token"])

    async def fail(self, job: dict[str, Any], code: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""UPDATE navi_summary_jobs SET status=$3,error_code=$4,next_attempt_at=now()+$5::interval,
                finished_at=CASE WHEN $3='failed' THEN now() ELSE NULL END
                WHERE job_id=$1 AND lease_token=$2 AND status='running' AND lease_expires_at>now()""",
                job["job_id"], job["lease_token"], "failed" if job["attempt_count"] >= self.settings.max_attempts else "retry_wait",
                code[:80], timedelta(seconds=min(300, 2 ** job["attempt_count"] + random.uniform(0, 3))))

    async def publish(self, job: dict[str, Any], candidate: SummaryCandidate, model_id: str) -> bool:
        snapshot = decode(job["input_snapshot"])
        document, evidence = normalize(candidate, snapshot, promote=self.settings.profile_promotion_enabled)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                state = await conn.fetchrow("SELECT * FROM navi_user_memory_state WHERE user_id=$1 FOR UPDATE", job["user_id"])
                owned = await conn.fetchval("""SELECT job_id FROM navi_summary_jobs WHERE job_id=$1 AND lease_token=$2
                    AND status='running' AND lease_expires_at>now() FOR UPDATE""", job["job_id"], job["lease_token"])
                if not owned:
                    return False
                valid = (state["enabled"] and state["memory_epoch"] == job["memory_epoch"]
                         and state["summary_revision"] == job["expected_summary_revision"]
                         and state["summary_seq"] < job["target_completed_seq"])
                valid = valid and await self._sources_valid(conn, job, snapshot)
                if not valid:
                    await conn.execute("UPDATE navi_summary_jobs SET status='obsolete',finished_at=now(),input_snapshot=NULL WHERE job_id=$1", owned)
                    return False
                revision, summary_id = state["summary_revision"]+1, uuid4()
                document.update(schema_version=2, summary_revision=revision, memory_epoch=job["memory_epoch"],
                                prompt_version=job["prompt_version"], window_policy_version=job["window_policy_version"],
                                input_snapshot_id=str(job["input_snapshot_id"]))
                document["coverage"] = {key: value for key, value in snapshot.items()
                                        if key not in {"turns", "reference_turns", "previous_summary"}}
                document["coverage"].update(actual_input_turn_count=len(snapshot["turns"]),
                    actual_input_turn_ids=[t["turn_id"] for t in snapshot["turns"]],
                    reference_turn_ids=[t["turn_id"] for t in snapshot.get("reference_turns", [])],
                    from_completed_seq=min((t["completed_seq"] for t in snapshot["turns"]), default=None),
                    to_completed_seq=max((t["completed_seq"] for t in snapshot["turns"]), default=None),
                    target_completed_seq=job["target_completed_seq"])
                await conn.execute("""INSERT INTO navi_conversation_summaries(summary_id,user_id,memory_epoch,summary_revision,job_id,document)
                    VALUES($1,$2,$3,$4,$5,$6::jsonb)""", summary_id, job["user_id"], job["memory_epoch"], revision, owned,
                    json.dumps(document, ensure_ascii=False))
                await conn.executemany("""INSERT INTO navi_summary_evidence(summary_id,user_id,claim_id,message_id,turn_id)
                    VALUES($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING""",
                    [(summary_id, job["user_id"], claim, UUID(mid), UUID(tid)) for claim, mid, tid in evidence])
                await self._promote_durable_memories(conn, job, summary_id, document, snapshot)
                await self._promote_teaching_preferences(conn, job, summary_id, document, snapshot)
                await conn.execute("UPDATE navi_user_memory_state SET summary_revision=$2,summary_seq=$3 WHERE user_id=$1",
                                   job["user_id"], revision, job["target_completed_seq"])
                await conn.execute("UPDATE navi_summary_jobs SET status='succeeded',model_id=$2,error_code=NULL,finished_at=now() WHERE job_id=$1", owned, model_id)
                return True

    async def _promote_durable_memories(self, conn: Any, job: dict[str, Any], summary_id: UUID,
                                        document: dict[str, Any], snapshot: dict[str, Any]) -> None:
        """Persist confirmed summary claims as independently retrievable memories.

        The summary document remains the versioned audit record. These rows are the
        durable memory projection used by the rest of MemoryStore. A deterministic
        claim key makes later summaries update the same memory instead of creating
        one row per summary revision.
        """
        # Summary migrations are additive and some isolated summary-only databases
        # intentionally do not load the legacy MemoryStore schema.
        if not await conn.fetchval("SELECT to_regclass('navi_memories')"):
            return
        source_by_message = {
            str(turn["user_message_id"]): (turn, True)
            for turn in snapshot.get("turns", []) + snapshot.get("reference_turns", [])
        }
        source_by_message.update({
            str(turn["assistant_message_id"]): (turn, False)
            for turn in snapshot.get("turns", []) + snapshot.get("reference_turns", [])
        })
        groups = {
            "personal_facts": {"personal_fact": "profile_fact"},
            "claims": {"open_task": "plan", "goal": "plan", "communication_preference": "preference"},
            "task_outcomes": {"open_task": "plan", "goal": "plan"},
            "preference_observations": {"preference": "preference"},
            "profile_change_proposals": {"profile_change_proposal": "behavior_observation"},
        }
        latest = latest_claims([item for group in groups for item in document.get(group, [])])
        for group, kinds in groups.items():
            for item in document.get(group, []):
                if not any(current is item for current in latest):
                    continue
                key = hashlib.sha256(claim_key(item).encode("utf-8")).hexdigest()
                evidence_ids = [str(value) for value in item.get("evidence_message_ids", [])]
                user_sources = [source_by_message[mid][0] for mid in evidence_ids
                                if mid in source_by_message and source_by_message[mid][1]]
                if item.get("status") in {"contradicted", "expired"} or item.get("progress") in {"resolved", "cancelled"}:
                    await conn.execute(
                        """UPDATE navi_memories SET is_deleted=TRUE, deleted_at=now(),
                                deleted_reason='summary_claim_retracted', updated_at=now()
                           WHERE user_id=$1 AND source_agent='conversation_summary'
                             AND evidence->>'summary_claim_key'=$2 AND NOT is_deleted""", job["user_id"], key)
                    continue
                # normalize() has already checked this, but keep the promotion
                # conservative if a future schema adds another summary section.
                if item.get("status") == "supported" and item.get("origin") != "inferred" \
                        and not item.get("counterevidence_message_ids") and user_sources \
                        and (item["kind"] not in {"profile_change_proposal", "communication_preference", "preference"}
                             or item.get("corroborated") or item.get("origin") == "user_explicit"):
                    memory_type = kinds.get(item["kind"])
                    if not memory_type:
                        continue
                    content = str(item["content"]).strip()
                    evidence_payload = {
                        "summary_id": str(summary_id), "summary_revision": document["summary_revision"],
                        "summary_claim_key": key, "claim_id": item["claim_id"],
                        "message_ids": evidence_ids,
                        "turn_ids": [str(source["turn_id"]) for source in user_sources],
                        "origin": item["origin"], "status": item["status"],
                        "preference_type": item.get("preference_type"),
                        "polarity": item.get("polarity"), "target": item.get("target"),
                    }
                    existing = await conn.fetchval(
                        """SELECT id FROM navi_memories
                           WHERE user_id=$1 AND source_agent='conversation_summary'
                             AND NOT is_deleted AND evidence->>'summary_claim_key'=$2
                           ORDER BY updated_at DESC LIMIT 1""", job["user_id"], key)
                    if existing:
                        await conn.execute(
                            """UPDATE navi_memories SET memory_type=$2, content=$3, summary=$4,
                                importance=GREATEST(importance, $5), confidence=$6,
                                evidence=$7::jsonb, source_session_id=$8, content_hash=$9,
                                is_deleted=FALSE, deleted_at=NULL, deleted_reason=NULL, updated_at=now()
                               WHERE id=$1""", existing, memory_type, content, item["subject"],
                            8 if item["kind"] in {"goal", "open_task"} else 7,
                            0.9 if item["origin"] == "user_explicit" else 0.8,
                            json.dumps(evidence_payload, ensure_ascii=False), user_sources[0]["session_id"],
                            memory_content_hash(content))
                    else:
                        await conn.execute(
                            """INSERT INTO navi_memories(user_id,memory_type,content,summary,importance,
                                sensitivity,confidence,tags,evidence,decay_category,source_agent,
                                source_session_id,content_hash)
                               VALUES($1,$2,$3,$4,$5,'normal',$6,$7::text[],$8::jsonb,'low',
                                      'conversation_summary',$9,$10)""", job["user_id"], memory_type, content,
                            item["subject"], 8 if item["kind"] in {"goal", "open_task"} else 7,
                            0.9 if item["origin"] == "user_explicit" else 0.8,
                            ["summary", item["kind"]], json.dumps(evidence_payload, ensure_ascii=False),
                            user_sources[0]["session_id"], memory_content_hash(content))

        # Plans are a projection of the current summary, not an append-only task
        # ledger. An omitted old plan is no longer verified as actionable. Retire
        # only summary-derived plans; omission does NOT assert task completion.
        await conn.execute("""UPDATE navi_memories SET is_deleted=TRUE, deleted_at=now(),
                deleted_reason='summary_plan_not_current', updated_at=now()
            WHERE user_id=$1 AND source_agent='conversation_summary' AND memory_type='plan'
              AND NOT is_deleted AND (evidence->>'summary_id') IS DISTINCT FROM $2""",
            job["user_id"], str(summary_id))

    async def _promote_teaching_preferences(self, conn: Any, job: dict[str, Any], summary_id: UUID,
                                             document: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if not await conn.fetchval("SELECT to_regclass('navi_teaching_preferences')"):
            return
        turns = snapshot.get("turns", []) + snapshot.get("reference_turns", [])
        source_by_message = {str(turn["user_message_id"]): turn for turn in turns}
        for item in document.get("teaching_strategy_observations", []):
            key = hashlib.sha256(
                f"{item['subject'].strip().lower()}|{item['content'].strip().lower()}".encode("utf-8")
            ).hexdigest()
            if item.get("status") in {"contradicted", "expired"} or item.get("counterevidence_message_ids"):
                await conn.execute("""UPDATE navi_teaching_preferences SET is_deleted=TRUE,
                        deleted_at=now(), deleted_reason='summary_claim_retracted', updated_at=now()
                    WHERE user_id=$1 AND preference_key=$2 AND NOT is_deleted""", job["user_id"], key)
                continue
            if item.get("status") != "supported" or item.get("origin") == "inferred":
                continue
            sources = [source_by_message[mid] for mid in map(str, item.get("evidence_message_ids", []))
                       if mid in source_by_message]
            if not sources:
                continue
            evidence_payload = {"summary_id": str(summary_id), "summary_revision": document["summary_revision"],
                                "claim_id": item["claim_id"],
                                "message_ids": [str(value) for value in item.get("evidence_message_ids", [])],
                                "turn_ids": [str(source["turn_id"]) for source in sources]}
            await conn.execute(
                """INSERT INTO navi_teaching_preferences
                    (user_id,preference_key,subject,method,content,status,confidence,evidence,source_summary_id,observed_at)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10::timestamptz)
                   ON CONFLICT(user_id,preference_key) DO UPDATE SET
                    subject=EXCLUDED.subject, method=EXCLUDED.method, content=EXCLUDED.content,
                    status=EXCLUDED.status, confidence=EXCLUDED.confidence, evidence=EXCLUDED.evidence,
                    source_summary_id=EXCLUDED.source_summary_id, observed_at=EXCLUDED.observed_at,
                    is_deleted=FALSE, deleted_at=NULL, deleted_reason=NULL, updated_at=now()""",
                job["user_id"], key, item["subject"], item["subject"], item["content"], item["status"],
                0.9 if item.get("origin") == "user_explicit" else 0.8,
                json.dumps(evidence_payload, ensure_ascii=False), summary_id,
                datetime.fromisoformat(item["observed_at"]))

    async def read(self, user_id: str) -> dict[str, Any]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""SELECT s.*,d.document,d.created_at FROM navi_user_memory_state s
                LEFT JOIN navi_conversation_summaries d ON d.user_id=s.user_id AND d.memory_epoch=s.memory_epoch
                AND d.summary_revision=s.summary_revision
                AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(
                    (d.document->'coverage'->'actual_input_turn_ids') || COALESCE(d.document->'coverage'->'reference_turn_ids','[]'::jsonb)) ids(id)
                    WHERE NOT EXISTS (SELECT 1 FROM navi_conversation_turns t WHERE t.turn_id::text=ids.id AND t.user_id=s.user_id))
                WHERE s.user_id=$1""", user_id)
            teaching_rows = []
            if await conn.fetchval("SELECT to_regclass('navi_teaching_preferences')"):
                teaching_rows = await conn.fetch("""SELECT preference_key,subject,method,content,status,
                        confidence,evidence,observed_at,expires_at
                    FROM navi_teaching_preferences
                    WHERE user_id=$1 AND NOT is_deleted AND status='supported'
                      AND (expires_at IS NULL OR expires_at > now())
                    ORDER BY updated_at DESC LIMIT 20""", user_id)
        if not row:
            return {"status": "empty", "summary": None, "teaching_preferences": [_teaching_dict(item) for item in teaching_rows],
                    "enabled": True, "apply_to_prompt": True}
        document = decode(row["document"]) if row["enabled"] and row["document"] else None
        return {"status": "disabled" if not row["enabled"] else "empty" if not document else
                "stale" if row["summary_seq"] < row["completed_seq"] or (row["created_at"] and
                    datetime.now(timezone.utc)-row["created_at"] > timedelta(days=self.settings.window_max_days)) else "ready",
                "summary": document, "generated_at": row["created_at"], "memory_epoch": row["memory_epoch"],
                "teaching_preferences": [_teaching_dict(item) for item in teaching_rows],
                "enabled": row["enabled"], "apply_to_prompt": row["apply_to_prompt"]}

    async def job(self, user_id: str, job_id: UUID) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""SELECT job_id,user_id,status,attempt_count,target_completed_seq,error_code,
                created_at,finished_at FROM navi_summary_jobs WHERE user_id=$1 AND job_id=$2""", user_id, job_id)
        return dict(row) if row else None

    async def reset(self, user_id: str, preferences: dict[str, bool] | None = None) -> dict[str, Any]:
        await self.state(user_id)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Enabling injection must retain the reviewed shadow summary.
                state = await conn.fetchrow("SELECT enabled FROM navi_user_memory_state WHERE user_id=$1 FOR UPDATE", user_id)
                if (not preferences or preferences.get("enabled") is False or preferences.get("apply_to_prompt") is False
                        or (preferences.get("enabled") is True and not state["enabled"])):
                    await invalidate(conn, user_id)
                if preferences:
                    await conn.execute("""UPDATE navi_user_memory_state SET enabled=COALESCE($2,enabled),
                        apply_to_prompt=COALESCE($3,apply_to_prompt) WHERE user_id=$1""",
                        user_id, preferences.get("enabled"), preferences.get("apply_to_prompt"))
        return await self.read(user_id)


async def invalidate(conn: Any, user_id: str) -> None:
    """Caller owns a transaction; state lock fences turns, old extractors and publishers."""
    await conn.execute("INSERT INTO navi_user_memory_state(user_id) VALUES($1) ON CONFLICT DO NOTHING", user_id)
    await conn.execute("""UPDATE navi_user_memory_state SET memory_epoch=memory_epoch+1,
        excluded_seq=completed_seq,summary_seq=completed_seq,summary_revision=summary_revision+1 WHERE user_id=$1""", user_id)
    await conn.execute("DELETE FROM navi_conversation_summaries WHERE user_id=$1", user_id)
    if await conn.fetchval("SELECT to_regclass('navi_teaching_preferences')"):
        await conn.execute("""UPDATE navi_teaching_preferences SET is_deleted=TRUE, deleted_at=now(),
                deleted_reason='summary_memory_reset', updated_at=now()
            WHERE user_id=$1 AND NOT is_deleted""", user_id)
    if await conn.fetchval("SELECT to_regclass('navi_memories')"):
        await conn.execute("""UPDATE navi_memories SET is_deleted=TRUE, deleted_at=now(),
                deleted_reason='summary_memory_reset', updated_at=now()
            WHERE user_id=$1 AND source_agent='conversation_summary' AND NOT is_deleted""", user_id)
    await conn.execute("""UPDATE navi_summary_jobs SET status='cancelled',input_snapshot=NULL,error_code='memory_changed',finished_at=now()
        WHERE user_id=$1""", user_id)


def normalize(candidate: SummaryCandidate, snapshot: dict[str, Any], *, promote: bool = True) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    sources = {t[key]: (t, role) for t in snapshot["turns"] + snapshot.get("reference_turns", [])
               for key, role in (("user_message_id", "user"), ("assistant_message_id", "assistant"))}
    body = candidate.model_dump(mode="json")
    evidence: list[tuple[str, str, str]] = []
    if prohibited_conclusion(json.dumps(body, ensure_ascii=False)):
        raise ValueError("Summary contains private data or prohibited labels")
    for group, items in body.items():
        if group == "limitations":
            continue
        for item in items:
            mids = set(item["evidence_message_ids"])
            counters = set(item.get("counterevidence_message_ids", []))
            if not (mids | counters) <= sources.keys():
                raise ValueError("Evidence is outside the user snapshot")
            if group != "topics" and not any(sources[mid][1] == "user" for mid in mids):
                raise ValueError("Assistant statements cannot establish user facts")
            if item.get("origin") == "user_explicit" and any(sources[mid][1] != "user" for mid in mids):
                raise ValueError("Explicit claims require user sources")
            if item.get("origin") == "inferred" and item["status"] == "supported":
                raise ValueError("Inferences must remain tentative")
            if group == "claims" and item["kind"] not in {"goal", "open_task", "communication_preference"}:
                raise ValueError("Claim in wrong section")
            if group == "task_outcomes" and (item["kind"] not in {"goal", "open_task"}
                                             or item.get("progress") not in {"resolved", "cancelled"}):
                raise ValueError("Task outcomes require a completed or cancelled task")
            expected = {"personal_facts": "personal_fact", "learning_observations": "learning_observation", "teaching_strategy_observations": "teaching_strategy_observation",
                        "profile_change_proposals": "profile_change_proposal", "temporary_states": "temporary_state",
                        "preference_observations": "preference"}.get(group)
            if expected and item["kind"] != expected:
                raise ValueError("Observation in wrong section")
            if group == "personal_facts" and item["origin"] != "user_explicit":
                raise ValueError("Personal facts require explicit user statements")
            if item.get("progress") and item.get("kind") not in {"open_task", "goal"}:
                raise ValueError("Progress only applies to tasks and goals")
            if item.get("trend") and group != "learning_observations":
                raise ValueError("Trends only apply to learning observations")
            user_turns = {sources[mid][0]["turn_id"] for mid in mids if sources[mid][1] == "user"}
            user_sessions = {sources[mid][0]["session_id"] for mid in mids if sources[mid][1] == "user"}
            user_times = {sources[mid][0]["completed_at"] for mid in mids if sources[mid][1] == "user"}
            if item.get("trend") and (len(user_turns) < 2 or len(user_times) < 2):
                raise ValueError("Trends require longitudinal user evidence")
            if group == "teaching_strategy_observations" and item["status"] == "supported":
                feedback_seq = max(sources[mid][0]["completed_seq"] for mid in mids if sources[mid][1] == "user")
                if any(sources[mid][1] == "assistant" and sources[mid][0]["completed_seq"] >= feedback_seq for mid in mids):
                    raise ValueError("Teaching effectiveness requires feedback after the cited explanation")
            if item.get("scope") == "general" and item.get("kind") not in {
                    "communication_preference", "preference", "temporary_state", "profile_change_proposal", "personal_fact"}:
                raise ValueError("Only preferences, states and behavioral profiles may be general")
            if item.get("kind") == "preference" and not all(item.get(key) for key in ("preference_type", "polarity", "target")):
                raise ValueError("Structured preferences require preference_type, polarity and target")
            if item.get("preference_type") or item.get("polarity") or item.get("target"):
                if item["kind"] not in {"communication_preference", "preference", "profile_change_proposal"}:
                    raise ValueError("Preference fields only apply to preference claims")
                item["preference_type"] = item.get("preference_type") or "communication"
                item["polarity"] = item.get("polarity") or "like"
                item["target"] = item.get("target") or item["subject"]
            claim_id = str(uuid4())
            item["claim_id"] = claim_id
            item["evidence_message_ids"] = sorted(mids)
            # An assistant reply must not refresh the age of a user fact or temporary state.
            item["observed_at"] = max(user_times or {sources[mid][0]["completed_at"] for mid in mids})
            item["observed_seq"] = max(sources[mid][0]["completed_seq"] for mid in mids
                                       if sources[mid][1] == "user" or not user_times)
            item["expires_at"] = ((datetime.fromisoformat(item["observed_at"]) + timedelta(hours=2)).isoformat()
                                  if group == "temporary_states" else None)
            item["evidence_turn_count"] = len({sources[mid][0]["turn_id"] for mid in mids})
            item["evidence_session_count"] = len({sources[mid][0]["session_id"] for mid in mids})
            item["corroborated"] = (len(user_turns) >= 3 and len(user_sessions) >= 2 and
                                    item.get("status") == "supported" and item.get("origin") != "inferred" and not counters)
            if item.get("kind") in {"open_task", "goal"} and not item.get("progress"):
                item["progress"] = "open"
            evidence.extend((claim_id, mid, sources[mid][0]["turn_id"]) for mid in mids | counters)
    body["stable_profile"] = [dict(item) for item in body["profile_change_proposals"] if promote and item["corroborated"]]
    return body, evidence
