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
    check_name STRING NOT NULL,        -- 'staleness' | 'eviction_pressure' | 'vector_drift' | 'empty_resolve'
    severity STRING NOT NULL,          -- 'ok' | 'warn' | 'critical'
    agent_id STRING,
    detail JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    INDEX (check_name, observed_at DESC)
);

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
    INDEX (agent_id, retrieved_at DESC)
);

CREATE VECTOR INDEX ON memory_records (embedding);