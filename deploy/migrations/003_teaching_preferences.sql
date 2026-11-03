-- Durable teaching-method preferences promoted from reviewed conversation summaries.
BEGIN;
CREATE TABLE IF NOT EXISTS navi_teaching_preferences (
    preference_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NOT NULL,
    preference_key text NOT NULL,
    subject text NOT NULL,
    method text NOT NULL,
    content text NOT NULL,
    status text NOT NULL DEFAULT 'supported' CHECK (status IN ('supported', 'tentative', 'contradicted', 'expired')),
    confidence float NOT NULL DEFAULT 0.75,
    evidence jsonb NOT NULL DEFAULT '{}',
    source_summary_id uuid,
    observed_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz,
    is_deleted boolean NOT NULL DEFAULT false,
    deleted_at timestamptz,
    deleted_reason text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(user_id, preference_key)
);
CREATE INDEX IF NOT EXISTS idx_navi_teaching_preferences_active
    ON navi_teaching_preferences(user_id, updated_at DESC) WHERE NOT is_deleted;
COMMIT;
