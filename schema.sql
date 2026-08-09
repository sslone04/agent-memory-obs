-- Core agent memory
CREATE TABLE memory_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id STRING NOT NULL,
    session_id STRING NOT NULL,
    kind STRING NOT NULL,              -- 'conversation' | 'snapshot' | 'fact'
    content STRING NOT NULL,
    embedding VECTOR(1024),            -- Bedrock Titan v2 default dim
    metadata JSONB DEFAULT '{}',

    -- observability-critical columns
    written_at TIMESTAMPTZ NOT NULL DEFAULT now(),   -- when the row landed
    effective_as_of TIMESTAMPTZ NOT NULL,            -- when the FACT was true (staleness)
    expires_at TIMESTAMPTZ,                          -- eviction pressure
    last_accessed_at TIMESTAMPTZ,
    access_count INT NOT NULL DEFAULT 0,
    INDEX (agent_id, session_id, written_at DESC),
    INDEX (expires_at)
);

-- Health check results written by the Lambda checks
CREATE TABLE memory_health_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    check_name STRING NOT NULL,        -- 'staleness' | 'eviction_pressure' | 'vector_drift' | 'empty_resolve' | 'near_miss'
    severity STRING NOT NULL,          -- 'ok' | 'warn' | 'critical'
    agent_id STRING,
    detail JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX (check_name, observed_at DESC)
) WITH (ttl_expire_after = '7 days');   -- row-level TTL: the layer prunes itself

-- Every retrieval attempt, so "resolved to empty" is detectable
CREATE TABLE memory_retrievals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id STRING NOT NULL,
    session_id STRING NOT NULL,
    query_text STRING,
    results_returned INT NOT NULL,
    top_similarity FLOAT,
    latency_ms INT,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_candidates INT,                -- candidates before the similarity floor
    applied_floor FLOAT,               -- the floor THIS retrieval used, so a
                                       -- config change never silently rewrites
                                       -- what past rows meant
    INDEX (agent_id, retrieved_at DESC)
) WITH (ttl_expire_after = '30 days');  -- row-level TTL: the layer prunes itself

-- One row per agent turn, linked to the retrieval that fed it. This is what
-- makes a bad answer traceable to the memory failure that caused it.
CREATE TABLE agent_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id STRING NOT NULL,
    session_id STRING NOT NULL,
    query STRING NOT NULL,
    response STRING NOT NULL,
    -- ON DELETE SET NULL, not CASCADE: memory_retrievals has a 30-day TTL, and
    -- an expiring retrieval must not delete the record of what the agent said.
    retrieval_id UUID REFERENCES memory_retrievals(id) ON DELETE SET NULL,
    memories_used INT NOT NULL DEFAULT 0,
    model_id STRING,
    latency_ms INT,
    -- clock_timestamp(), NOT now(): now() is the TRANSACTION timestamp in
    -- CockroachDB, so several turns written in one transaction would share a
    -- byte-identical created_at and become unorderable in the UI.
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    INDEX (agent_id, created_at DESC),
    INDEX (retrieval_id)
) WITH (ttl_expire_after = '30 days');

-- Per-agent retrieval policy. Absent row = DEFAULT_MIN_SIMILARITY.
CREATE TABLE agent_config (
    agent_id STRING PRIMARY KEY,
    min_similarity FLOAT NOT NULL DEFAULT 0.35,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE VECTOR INDEX ON memory_records (embedding);