"""CockroachDB-backed agent memory layer.

embed()        -- Bedrock Titan v2 text embeddings
write_memory() -- persist a memory row together with its embedding
recall()       -- vector similarity search, instrumented so that every attempt
                  (including ones that resolve to nothing) lands in
                  memory_retrievals, and every hit bumps its access counters.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

import boto3
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024  # must match memory_records.embedding VECTOR(1024)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Cosine floor a candidate must clear to be handed back to the caller.  This is
# the fallback only -- the effective floor comes from agent_config, per agent.
DEFAULT_MIN_SIMILARITY = 0.35

_bedrock = None


def _bedrock_client():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (expected in .env)")
    return url


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Open a connection with dict rows. Commits on clean exit."""
    with psycopg.connect(_database_url(), row_factory=dict_row) as conn:
        yield conn


def _to_vector(values: Sequence[float]) -> str:
    """Render a VECTOR literal -- psycopg has no native adapter for the type."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _similarity(l2_distance: float) -> float:
    """Titan v2 vectors are unit-normalized, so cosine = 1 - d^2 / 2."""
    return 1.0 - (l2_distance * l2_distance) / 2.0


def get_floor(agent_id: str, conn: psycopg.Connection | None = None) -> float:
    """The similarity floor in force for this agent.

    Falls back to DEFAULT_MIN_SIMILARITY when the agent has no config row, so
    an unconfigured agent behaves exactly as it did before agent_config existed.
    """

    def _run(c: psycopg.Connection) -> float:
        with c.cursor() as cur:
            cur.execute(
                "SELECT min_similarity FROM agent_config WHERE agent_id = %s", (agent_id,)
            )
            row = cur.fetchone()
        return float(row["min_similarity"]) if row else DEFAULT_MIN_SIMILARITY

    if conn is not None:
        return _run(conn)
    with connect() as owned:
        return _run(owned)


def set_floor(
    agent_id: str, min_similarity: float, conn: psycopg.Connection | None = None
) -> None:
    """Set (or change) one agent's retrieval floor."""

    def _run(c: psycopg.Connection) -> None:
        with c.cursor() as cur:
            cur.execute(
                """
                UPSERT INTO agent_config (agent_id, min_similarity, updated_at)
                VALUES (%s, %s, now())
                """,
                (agent_id, float(min_similarity)),
            )

    if conn is not None:
        _run(conn)
        return
    with connect() as owned:
        _run(owned)


def embed(text: str) -> list[float]:
    """Embed `text` with Titan v2, normalized to EMBED_DIM dimensions."""
    response = _bedrock_client().invoke_model(
        modelId=EMBED_MODEL_ID,
        accept="application/json",
        contentType="application/json",
        body=json.dumps(
            {"inputText": text, "dimensions": EMBED_DIM, "normalize": True}
        ),
    )
    vector = json.loads(response["body"].read())["embedding"]
    if len(vector) != EMBED_DIM:
        raise ValueError(f"expected a {EMBED_DIM}-dim embedding, got {len(vector)}")
    return vector


def write_memory(
    agent_id: str,
    session_id: str,
    kind: str,
    content: str,
    *,
    effective_as_of: datetime | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    embedding: Sequence[float] | None = None,
    conn: psycopg.Connection | None = None,
) -> uuid.UUID:
    """Insert one memory row and return its id.

    `effective_as_of` defaults to now, but pass the real as-of time whenever the
    fact was true earlier than the write -- that gap is what the staleness check
    reads.  Pass `embedding` to reuse a vector you already computed.
    """
    vector = embedding if embedding is not None else embed(content)
    params = (
        agent_id,
        session_id,
        kind,
        content,
        _to_vector(vector),
        json.dumps(metadata or {}),
        effective_as_of or datetime.now(timezone.utc),
        expires_at,
    )
    sql = """
        INSERT INTO memory_records
            (agent_id, session_id, kind, content, embedding, metadata,
             effective_as_of, expires_at)
        VALUES (%s, %s, %s, %s, %s::VECTOR, %s::JSONB, %s, %s)
        RETURNING id
    """

    def _run(c: psycopg.Connection) -> uuid.UUID:
        with c.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()["id"]

    if conn is not None:
        return _run(conn)
    with connect() as owned:
        return _run(owned)


def recall(
    agent_id: str,
    query_text: str,
    *,
    session_id: str,
    limit: int = 5,
    min_similarity: float | None = None,
    include_expired: bool = False,
    stats: dict[str, Any] | None = None,
    conn: psycopg.Connection | None = None,
) -> list[dict[str, Any]]:
    """Nearest-neighbour search over one agent's memories, floored.

    The vector search returns the `limit` nearest rows whatever their scores, so
    anything below the floor is dropped before the caller sees it.  Both counts
    are logged: `raw_candidates` pre-filter, `results_returned` post.  A widening
    gap between them is the degradation signal -- the search is still finding
    rows, they are just getting worse.

    The floor comes from agent_config unless `min_similarity` is passed
    explicitly.  Whichever value applied is written to `applied_floor` on the
    retrieval row: changing an agent's floor must not silently rewrite what
    every past row meant.

    Always writes a memory_retrievals row, including when the filtered result is
    empty.  `top_similarity` is the best *pre-filter* score, so a retrieval that
    floors out still records how close it came.  Only rows that survive the
    floor have their access counters bumped.

    Pass a dict as `stats` to receive the telemetry for this call (retrieval id,
    top_similarity, raw_candidates, applied_floor).  A floored-out retrieval
    returns [], so without this there is no way for a caller to learn how close
    it came -- and the retrieval row cannot be found again by timestamp, since
    CockroachDB gives every row in a transaction the same now().
    """

    def _run(c: psycopg.Connection) -> list[dict[str, Any]]:
        started = time.perf_counter()
        floor = min_similarity if min_similarity is not None else get_floor(agent_id, c)
        vector = _to_vector(embed(query_text))
        expiry_clause = "" if include_expired else "AND (expires_at IS NULL OR expires_at > now())"
        search_sql = f"""
            SELECT id, agent_id, session_id, kind, content, metadata,
                   written_at, effective_as_of, expires_at, access_count,
                   embedding <-> %s::VECTOR AS distance
            FROM memory_records
            WHERE agent_id = %s
              AND embedding IS NOT NULL
              {expiry_clause}
            ORDER BY embedding <-> %s::VECTOR
            LIMIT %s
        """
        with c.cursor() as cur:
            cur.execute(search_sql, (vector, agent_id, vector, limit))
            candidates = cur.fetchall()

            for row in candidates:
                row["similarity"] = _similarity(float(row["distance"]))

            # Best score before the floor -- kept even when nothing survives, so
            # a floored-out retrieval still says how near it got.
            top_similarity = candidates[0]["similarity"] if candidates else None
            rows = [r for r in candidates if r["similarity"] >= floor]

            if rows:
                cur.execute(
                    """
                    UPDATE memory_records
                    SET access_count = access_count + 1,
                        last_accessed_at = now()
                    WHERE id = ANY(%s)
                    """,
                    ([row["id"] for row in rows],),
                )

            latency_ms = int((time.perf_counter() - started) * 1000)
            cur.execute(
                """
                INSERT INTO memory_retrievals
                    (agent_id, session_id, query_text, raw_candidates,
                     results_returned, top_similarity, latency_ms, applied_floor)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    agent_id,
                    session_id,
                    query_text,
                    len(candidates),
                    len(rows),
                    top_similarity,
                    latency_ms,
                    floor,
                ),
            )
            retrieval_id = cur.fetchone()["id"]

        if stats is not None:
            stats.update(
                {
                    "retrieval_id": retrieval_id,
                    "top_similarity": top_similarity,
                    "raw_candidates": len(candidates),
                    "results_returned": len(rows),
                    "applied_floor": floor,
                    "latency_ms": latency_ms,
                }
            )
        return rows

    if conn is not None:
        return _run(conn)
    with connect() as owned:
        return _run(owned)


def _smoke_test() -> None:
    """Write a few memories, recall against them, and assert the telemetry."""
    agent_id = f"smoke-{uuid.uuid4().hex[:8]}"
    session_id = "smoke-session"
    print(f"agent_id = {agent_id}")

    facts = [
        ("fact", "The production database runs CockroachDB v26.2 in us-east-1."),
        ("fact", "Embeddings come from Bedrock Titan v2 at 1024 dimensions."),
        ("conversation", "The user prefers terse status updates over long reports."),
    ]

    with connect() as conn:
        for kind, content in facts:
            memory_id = write_memory(agent_id, session_id, kind, content, conn=conn)
            print(f"  wrote {memory_id}  {content[:48]}...")

        print("\nrecall: 'which database are we running?'")
        hits = recall(
            agent_id, "which database are we running?", session_id=session_id, conn=conn
        )
        assert hits, "expected at least one hit"
        for hit in hits:
            print(f"  {hit['similarity']:.4f}  {hit['content'][:56]}")
        top = hits[0]
        assert "CockroachDB" in top["content"], f"unexpected top hit: {top['content']}"
        assert top["access_count"] == 0, "pre-update snapshot should read 0"
        assert all(h["similarity"] >= DEFAULT_MIN_SIMILARITY for h in hits), "floor leaked"

        print("\nrecall with an off-topic query (candidates found, none clear the floor)")
        floored = recall(
            agent_id,
            "sourdough starter hydration ratios",
            session_id=session_id,
            conn=conn,
        )
        assert floored == [], f"expected the floor to reject everything, got {len(floored)}"
        print("  0 results survived")

        print("\nrecall against an agent with no memories (no candidates at all)")
        empty = recall(
            f"{agent_id}-unknown", "anything at all", session_id=session_id, conn=conn
        )
        assert empty == [], f"expected no hits, got {len(empty)}"
        print("  0 candidates")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT access_count, last_accessed_at FROM memory_records WHERE id = %s",
                (top["id"],),
            )
            row = cur.fetchone()
            assert row["access_count"] == 1, f"access_count = {row['access_count']}"
            assert row["last_accessed_at"] is not None, "last_accessed_at not set"
            print(f"\naccess_count bumped to {row['access_count']}")

            cur.execute(
                """
                SELECT query_text, raw_candidates, results_returned,
                       top_similarity, latency_ms
                FROM memory_retrievals
                WHERE agent_id LIKE %s
                ORDER BY retrieved_at
                """,
                (f"{agent_id}%",),
            )
            logged = cur.fetchall()
            assert len(logged) == 3, f"expected 3 retrieval rows, got {len(logged)}"
            assert all(e["raw_candidates"] is not None for e in logged), "raw_candidates unset"
            print("memory_retrievals:")
            for entry in logged:
                similarity = (
                    f"{entry['top_similarity']:.4f}"
                    if entry["top_similarity"] is not None
                    else "none"
                )
                print(
                    f"  raw={entry['raw_candidates']} -> returned={entry['results_returned']}"
                    f"  top={similarity}  {entry['latency_ms']}ms  {entry['query_text']!r}"
                )

    print("\nsmoke test passed")


if __name__ == "__main__":
    _smoke_test()
