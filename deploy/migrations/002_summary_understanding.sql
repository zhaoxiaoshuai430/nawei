-- Preserve all existing user preferences, including explicit opt-outs.
-- New users automatically inherit the enabled deployment's summary feature.
BEGIN;
-- Fail without waiting if an old scheduler owns either table. Retrying after
-- drain is safer than deadlocking against the worker's state/job lock order.
LOCK TABLE navi_user_memory_state, navi_summary_jobs IN ACCESS EXCLUSIVE MODE NOWAIT;
ALTER TABLE navi_user_memory_state ALTER COLUMN apply_to_prompt SET DEFAULT true;
-- v2 explicitly stamps its inserts. Retain legacy defaults so an old worker
-- awaiting restart cannot accidentally enqueue a v2 job it cannot read.
ALTER TABLE navi_summary_jobs ALTER COLUMN prompt_version SET DEFAULT 'summary-v1';
ALTER TABLE navi_summary_jobs ALTER COLUMN schema_version SET DEFAULT 1;
-- A different prompt must never consume a frozen v1 snapshot. Fences running
-- v1 publishers as well; a v2 worker will discover the outstanding demand.
UPDATE navi_summary_jobs SET status='obsolete',input_snapshot=NULL,
    error_code='summary_version_changed',finished_at=now()
WHERE prompt_version='summary-v1' AND status IN ('pending','running','retry_wait');
COMMIT;
