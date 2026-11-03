-- Explicit additive migration. Run after the existing MemoryStore schema exists.
-- No historical messages are guessed into completed turns.
BEGIN;
CREATE TABLE IF NOT EXISTS navi_user_memory_state (
    user_id text PRIMARY KEY,
    memory_epoch bigint NOT NULL DEFAULT 0,
    profile_revision bigint NOT NULL DEFAULT 0,
    completed_seq bigint NOT NULL DEFAULT 0,
    desired_target_seq bigint NOT NULL DEFAULT 0,
    summary_seq bigint NOT NULL DEFAULT 0,
    excluded_seq bigint NOT NULL DEFAULT 0,
    summary_revision bigint NOT NULL DEFAULT 0,
    enabled boolean NOT NULL DEFAULT true,
    apply_to_prompt boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS navi_conversation_turns (
    turn_id uuid PRIMARY KEY,
    user_id text NOT NULL REFERENCES navi_user_memory_state(user_id),
    session_id text NOT NULL,
    request_id text NOT NULL,
    request_fingerprint text NOT NULL,
    memory_epoch bigint NOT NULL,
    status text NOT NULL CHECK (status IN ('completed', 'interrupted', 'failed')),
    completed_seq bigint NOT NULL,
    user_message_id uuid NOT NULL REFERENCES navi_messages(id) ON DELETE CASCADE,
    assistant_message_id uuid NOT NULL REFERENCES navi_messages(id) ON DELETE CASCADE,
    completed_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL DEFAULT 'conversation',
    UNIQUE(user_id, request_id), UNIQUE(user_id, completed_seq)
);
CREATE INDEX IF NOT EXISTS navi_turns_window ON navi_conversation_turns(user_id, memory_epoch, completed_seq DESC);
CREATE TABLE IF NOT EXISTS navi_summary_jobs (
    job_id uuid PRIMARY KEY,
    user_id text NOT NULL REFERENCES navi_user_memory_state(user_id),
    memory_epoch bigint NOT NULL,
    target_completed_seq bigint NOT NULL,
    window_policy_version text NOT NULL,
    schema_version integer NOT NULL DEFAULT 1,
    prompt_version text NOT NULL DEFAULT 'summary-v1',
    status text NOT NULL DEFAULT 'pending' CHECK (status IN
        ('pending', 'running', 'succeeded', 'retry_wait', 'failed', 'obsolete', 'cancelled')),
    attempt_count integer NOT NULL DEFAULT 0,
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    lease_owner text,
    lease_token uuid,
    lease_expires_at timestamptz,
    input_snapshot_id uuid,
    input_snapshot jsonb,
    expected_summary_revision bigint,
    model_id text,
    error_code text,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    UNIQUE(user_id, memory_epoch, target_completed_seq, window_policy_version, prompt_version)
);
CREATE INDEX IF NOT EXISTS navi_summary_jobs_pending ON navi_summary_jobs(next_attempt_at, created_at)
    WHERE status IN ('pending', 'running', 'retry_wait');
CREATE TABLE IF NOT EXISTS navi_summary_slots (
    slot_id integer PRIMARY KEY CHECK (slot_id = 1),
    lease_token uuid,
    lease_expires_at timestamptz
);
INSERT INTO navi_summary_slots(slot_id) VALUES(1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS navi_conversation_summaries (
    summary_id uuid PRIMARY KEY,
    user_id text NOT NULL REFERENCES navi_user_memory_state(user_id),
    memory_epoch bigint NOT NULL,
    summary_revision bigint NOT NULL,
    job_id uuid NOT NULL REFERENCES navi_summary_jobs(job_id),
    document jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, memory_epoch, summary_revision)
);
CREATE TABLE IF NOT EXISTS navi_summary_evidence (
    summary_id uuid NOT NULL REFERENCES navi_conversation_summaries(summary_id) ON DELETE CASCADE,
    user_id text NOT NULL,
    claim_id text NOT NULL,
    message_id uuid NOT NULL REFERENCES navi_messages(id) ON DELETE CASCADE,
    turn_id uuid NOT NULL REFERENCES navi_conversation_turns(turn_id) ON DELETE CASCADE,
    PRIMARY KEY(summary_id, claim_id, message_id)
);
CREATE INDEX IF NOT EXISTS navi_summary_evidence_message ON navi_summary_evidence(user_id, message_id);
COMMIT;
