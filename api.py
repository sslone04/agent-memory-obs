"""FastAPI backend for the Memory Health dashboard.

Read-only views over the same tables checks.py writes to.  Thresholds are
imported from checks.py rather than restated, so the dashboard can never
disagree with the checker about what "warn" means.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import psycopg
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from checks import (
    EMPTY_RESOLVE_CRITICAL_PCT,
    EMPTY_RESOLVE_WARN_PCT,
    EVICTION_CRITICAL_PCT,
    EVICTION_WARN_PCT,
    EVICTION_HORIZON_MINUTES,
    NEAR_MISS_BAND,
    NEAR_MISS_CRITICAL_PCT,
    NEAR_MISS_WARN_PCT,
    STALENESS_THRESHOLD_HOURS,
    VECTOR_NORM_TOLERANCE,
)
from memory import DEFAULT_MIN_SIMILARITY, EMBED_DIM, connect

# The scheduled Lambda fires every 5 minutes.  Three consecutive misses and we
# stop calling the data current -- the dashboard says "unknown", never "healthy".
HEALTH_MAX_AGE_SECONDS = 15 * 60

EXPECTED_CHECKS = (
    "staleness",
    "eviction_pressure",
    "empty_resolve",
    "near_miss",
    "vector_drift",
)
SEVERITY_RANK = {"ok": 0, "warn": 1, "critical": 2}

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Agent Memory Health", docs_url="/api/docs")


@contextmanager
def read_only_connect() -> Iterator[psycopg.Connection]:
    """A connection the database itself refuses to let write.

    This dashboard is deployed publicly, so "we only wrote SELECTs" is not a
    strong enough guarantee -- a future endpoint, or a bug, should not be able
    to mutate the corpus it is supposed to be observing.  Setting the session
    read-only pushes the guarantee down to CockroachDB, which rejects any
    INSERT/UPDATE/DELETE/DDL regardless of what this process asks for.
    """
    with connect() as conn:
        conn.autocommit = True
        conn.execute("SET default_transaction_read_only = on")
        yield conn


def _rows(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with read_only_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def _one(sql: str, params: tuple = ()) -> dict[str, Any]:
    result = _rows(sql, params)
    return result[0] if result else {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@app.exception_handler(psycopg.Error)
async def _db_error_handler(request, exc: psycopg.Error) -> JSONResponse:
    """A dead database is an 'unknown' health state, not a 500 with no meaning.

    The message is the exception class only -- psycopg errors can quote the
    connection string, which must never reach the browser.
    """
    return JSONResponse(
        status_code=503,
        content={"error": "database_unavailable", "detail": type(exc).__name__},
    )


@app.get("/api/health/current")
def health_current() -> dict[str, Any]:
    """Latest result per check, plus one honest overall verdict.

    The overall status is deliberately NOT "green because the query worked".
    It degrades when any check warns, and it reads 'unknown' -- never
    'healthy' -- when the health data itself has gone stale or a check is
    missing entirely.  Stale health data is a failure state.
    """
    rows = _rows(
        """
        SELECT DISTINCT ON (check_name)
               check_name, severity, agent_id, detail, observed_at
        FROM memory_health_events
        ORDER BY check_name, observed_at DESC
        """
    )
    by_name = {r["check_name"]: r for r in rows}
    now = _now()

    checks: list[dict[str, Any]] = []
    for name in EXPECTED_CHECKS:
        row = by_name.get(name)
        if row is None:
            checks.append(
                {
                    "check_name": name,
                    "status": "unknown",
                    "severity": None,
                    "reason": "no health event has ever been written",
                    "observed_at": None,
                    "age_seconds": None,
                    "agent_id": None,
                    "detail": {},
                }
            )
            continue
        age = (now - row["observed_at"]).total_seconds()
        stale = age > HEALTH_MAX_AGE_SECONDS
        checks.append(
            {
                "check_name": name,
                "status": "unknown" if stale else row["severity"],
                "severity": row["severity"],
                "reason": (
                    f"last reported {int(age // 60)}m ago, older than the "
                    f"{HEALTH_MAX_AGE_SECONDS // 60}m freshness window"
                    if stale
                    else None
                ),
                "observed_at": row["observed_at"],
                "age_seconds": round(age, 1),
                "agent_id": row["agent_id"],
                "detail": row["detail"],
            }
        )

    ages = [c["age_seconds"] for c in checks if c["age_seconds"] is not None]
    data_age = min(ages) if ages else None

    unknown = [c for c in checks if c["status"] == "unknown"]
    if unknown:
        # Missing or stale data never resolves to healthy.
        status = "unknown"
        if len(unknown) == len(checks):
            reason = (
                "no health data within the freshness window -- the checker may not be running"
                if data_age is not None
                else "no health events have ever been written"
            )
        else:
            reason = f"{len(unknown)} of {len(checks)} checks have no recent data"
    else:
        worst = max(checks, key=lambda c: SEVERITY_RANK.get(c["status"], 0))
        status = {"ok": "healthy", "warn": "warn", "critical": "critical"}[worst["status"]]
        # Name each check's own severity -- listing several checks under one
        # worst-case word overstates the ones that are only warning.
        degraded = [
            f"{c['check_name']} {c['status']}" for c in checks if c["status"] != "ok"
        ]
        reason = (
            ", ".join(degraded) if degraded else f"all {len(checks)} checks reporting ok"
        )

    return {
        "generated_at": now,
        "overall": {
            "status": status,
            "reason": reason,
            "data_age_seconds": round(data_age, 1) if data_age is not None else None,
            "freshness_window_seconds": HEALTH_MAX_AGE_SECONDS,
        },
        "checks": checks,
        "thresholds": {
            "staleness_hours": STALENESS_THRESHOLD_HOURS,
            "eviction_horizon_minutes": EVICTION_HORIZON_MINUTES,
            "eviction_warn_pct": EVICTION_WARN_PCT,
            "eviction_critical_pct": EVICTION_CRITICAL_PCT,
            "empty_resolve_warn_pct": EMPTY_RESOLVE_WARN_PCT,
            "empty_resolve_critical_pct": EMPTY_RESOLVE_CRITICAL_PCT,
            "min_similarity": DEFAULT_MIN_SIMILARITY,
            "near_miss_band": NEAR_MISS_BAND,
            "near_miss_warn_pct": NEAR_MISS_WARN_PCT,
            "near_miss_critical_pct": NEAR_MISS_CRITICAL_PCT,
            "embed_dim": EMBED_DIM,
            "norm_tolerance": VECTOR_NORM_TOLERANCE,
        },
    }


@app.get("/api/retrievals/similarity")
def retrievals_similarity(
    hours: int = Query(24, ge=1, le=168),
    bins: int = Query(24, ge=6, le=60),
) -> dict[str, Any]:
    """Distribution of top_similarity, split by whether the floor kept it.

    This is the calibration view: a bimodal distribution with the floor sitting
    in the wrong valley is a config problem you can see, not infer.
    """
    rows = _rows(
        """
        SELECT top_similarity,
               coalesce(applied_floor, %s) AS floor,
               results_returned
        FROM memory_retrievals
        WHERE extract(epoch FROM (now() - retrieved_at)) <= %s * 3600
          AND top_similarity IS NOT NULL
        """,
        (DEFAULT_MIN_SIMILARITY, hours),
    )

    width = 1.0 / bins
    histogram = [
        {
            "lo": round(i * width, 4),
            "hi": round((i + 1) * width, 4),
            "kept": 0,
            "rejected": 0,
        }
        for i in range(bins)
    ]
    floors: dict[float, int] = {}
    for row in rows:
        score = float(row["top_similarity"])
        floor = round(float(row["floor"]), 4)
        floors[floor] = floors.get(floor, 0) + 1
        index = min(bins - 1, max(0, int(score / width)))
        histogram[index]["kept" if row["results_returned"] > 0 else "rejected"] += 1

    # Latest near_miss verdict supplies the suggested floor, if it produced one.
    latest = _one(
        """
        SELECT severity, detail, observed_at
        FROM memory_health_events
        WHERE check_name = 'near_miss'
        ORDER BY observed_at DESC
        LIMIT 1
        """
    )
    detail = latest.get("detail") or {}

    return {
        "generated_at": _now(),
        "hours": hours,
        "bins": bins,
        "samples": len(rows),
        "histogram": histogram,
        "floors_in_force": sorted(floors),
        "default_floor": DEFAULT_MIN_SIMILARITY,
        "band": NEAR_MISS_BAND,
        "near_miss": {
            "severity": latest.get("severity"),
            "observed_at": latest.get("observed_at"),
            "near_miss": detail.get("near_miss"),
            "floored_out_total": detail.get("floored_out_total"),
            "near_miss_pct_of_floored": detail.get("near_miss_pct_of_floored"),
            "suggested_floor": detail.get("suggested_floor"),
            "suggestion_basis": detail.get("suggestion_basis"),
            "examples": detail.get("examples", []),
        },
    }


@app.get("/api/health/history")
def health_history(
    hours: int = Query(24, ge=1, le=168),
    buckets: int = Query(48, ge=12, le=240),
) -> dict[str, Any]:
    """Severity timeline per check, bucketed for the status strips.

    Buckets with no health event are 'unknown', not 'ok' -- a gap in coverage
    renders as a gap, which is the whole point of the strip.
    """
    rows = _rows(
        """
        SELECT check_name, severity, observed_at
        FROM memory_health_events
        WHERE extract(epoch FROM (now() - observed_at)) <= %s * 3600
        ORDER BY observed_at
        """,
        (hours,),
    )

    bucket_count = buckets
    now = _now()
    window = timedelta(hours=hours)
    start = now - window
    bucket_seconds = window.total_seconds() / bucket_count

    series: dict[str, list[dict[str, Any]]] = {
        name: [
            {
                "t": start + timedelta(seconds=bucket_seconds * i),
                "severity": "unknown",
                "events": 0,
            }
            for i in range(bucket_count)
        ]
        for name in EXPECTED_CHECKS
    }

    for row in rows:
        name = row["check_name"]
        if name not in series:
            continue
        index = int((row["observed_at"] - start).total_seconds() // bucket_seconds)
        index = max(0, min(bucket_count - 1, index))
        bucket = series[name][index]
        bucket["events"] += 1
        # worst severity in the bucket wins -- never average away a critical
        if bucket["severity"] == "unknown" or SEVERITY_RANK.get(
            row["severity"], 0
        ) > SEVERITY_RANK.get(bucket["severity"], 0):
            bucket["severity"] = row["severity"]

    return {
        "hours": hours,
        "bucket_count": bucket_count,
        "bucket_seconds": round(bucket_seconds),
        "start": start,
        "end": now,
        "series": series,
    }


@app.get("/api/memory/stats")
def memory_stats() -> dict[str, Any]:
    """Corpus shape: totals, composition by kind, expiry pressure, bad vectors."""
    totals = _one(
        """
        SELECT
            count(*) AS total_rows,
            count(*) FILTER (WHERE expires_at IS NULL OR expires_at > now()) AS live_rows,
            count(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at <= now()) AS expired_rows,
            count(*) FILTER (
                WHERE expires_at IS NOT NULL AND expires_at > now()
                  AND extract(epoch FROM (expires_at - now())) <= %s * 60
            ) AS expiring_soon,
            count(*) FILTER (WHERE expires_at IS NULL) AS never_expire,
            count(*) FILTER (WHERE embedding IS NULL) AS null_embeddings,
            count(*) FILTER (
                WHERE embedding IS NOT NULL AND vector_dims(embedding) <> %s
            ) AS wrong_dim,
            count(*) FILTER (
                WHERE embedding IS NOT NULL
                  AND abs(vector_norm(embedding) - 1.0) > %s
            ) AS bad_norm,
            coalesce(sum(access_count), 0) AS total_accesses,
            count(*) FILTER (WHERE access_count = 0) AS never_accessed
        FROM memory_records
        """,
        (EVICTION_HORIZON_MINUTES, EMBED_DIM, VECTOR_NORM_TOLERANCE),
    )
    by_kind = _rows(
        """
        SELECT kind, count(*) AS rows,
               count(*) FILTER (WHERE embedding IS NULL) AS null_embeddings
        FROM memory_records
        GROUP BY kind
        ORDER BY count(*) DESC
        """
    )
    agents = _one("SELECT count(DISTINCT agent_id) AS agents FROM memory_records")

    live = totals.get("live_rows") or 0
    expiring = totals.get("expiring_soon") or 0
    return {
        "generated_at": _now(),
        **{k: (v or 0) for k, v in totals.items()},
        "distinct_agents": agents.get("agents") or 0,
        "expiring_pct": round(100.0 * expiring / live, 2) if live else 0.0,
        "by_kind": by_kind,
        "horizon_minutes": EVICTION_HORIZON_MINUTES,
    }


@app.get("/api/retrievals/recent")
def retrievals_recent(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    """Most recent retrievals, raw vs filtered counts side by side."""
    rows = _rows(
        """
        SELECT id, agent_id, session_id, query_text, raw_candidates,
               results_returned, top_similarity, latency_ms, retrieved_at
        FROM memory_retrievals
        ORDER BY retrieved_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    for row in rows:
        raw = row["raw_candidates"]
        row["floored_out"] = bool(
            row["results_returned"] == 0 and (raw or 0) > 0
        )
        row["dropped"] = (raw - row["results_returned"]) if raw is not None else None
    return {
        "generated_at": _now(),
        "limit": limit,
        "min_similarity": DEFAULT_MIN_SIMILARITY,
        "retrievals": rows,
    }


@app.get("/api/retrievals/degradation")
def retrievals_degradation(hours: int = Query(24, ge=1, le=168)) -> dict[str, Any]:
    """Floored-out rate and average top similarity over time.

    Two measures on deliberately separate series -- the frontend plots them as
    two charts, never one dual-axis plot.
    """
    bucket_seconds = max(300, int(hours * 3600 / 48))
    rows = _rows(
        """
        SELECT
            to_timestamp(floor(extract(epoch FROM retrieved_at) / %s::FLOAT) * %s::FLOAT) AS bucket,
            count(*) AS retrievals,
            count(*) FILTER (
                WHERE results_returned = 0 AND coalesce(raw_candidates, 0) > 0
            ) AS floored_out,
            count(*) FILTER (WHERE results_returned = 0) AS empty_results,
            avg(top_similarity) FILTER (WHERE top_similarity IS NOT NULL) AS avg_top_similarity,
            avg(latency_ms) FILTER (WHERE latency_ms IS NOT NULL) AS avg_latency_ms
        FROM memory_retrievals
        WHERE extract(epoch FROM (now() - retrieved_at)) <= %s * 3600
        GROUP BY bucket
        ORDER BY bucket
        """,
        (bucket_seconds, bucket_seconds, hours),
    )

    points = []
    for row in rows:
        retrievals = row["retrievals"] or 0
        floored = row["floored_out"] or 0
        points.append(
            {
                "t": row["bucket"],
                "retrievals": retrievals,
                "floored_out": floored,
                "floored_rate": round(100.0 * floored / retrievals, 2) if retrievals else 0.0,
                "empty_results": row["empty_results"] or 0,
                "avg_top_similarity": (
                    round(float(row["avg_top_similarity"]), 4)
                    if row["avg_top_similarity"] is not None
                    else None
                ),
                "avg_latency_ms": (
                    round(float(row["avg_latency_ms"]), 1)
                    if row["avg_latency_ms"] is not None
                    else None
                ),
            }
        )

    total = sum(p["retrievals"] for p in points)
    floored_total = sum(p["floored_out"] for p in points)
    return {
        "generated_at": _now(),
        "hours": hours,
        "bucket_seconds": bucket_seconds,
        "min_similarity": DEFAULT_MIN_SIMILARITY,
        "warn_pct": EMPTY_RESOLVE_WARN_PCT,
        "critical_pct": EMPTY_RESOLVE_CRITICAL_PCT,
        "totals": {
            "retrievals": total,
            "floored_out": floored_total,
            "floored_rate": round(100.0 * floored_total / total, 2) if total else 0.0,
        },
        "points": points,
    }


@app.get("/api/agent/impact")
def agent_impact(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(25, ge=1, le=200),
) -> dict[str, Any]:
    """Turns where a memory failure became an agent failure.

    Two classes, deliberately separated because they have different fixes:
    `blind` turns answered with zero memories (the agent had nothing), and
    `near_miss_fed` turns answered blind because the floor discarded a
    probably-relevant result (the agent had something and policy hid it).
    """
    totals = _one(
        """
        SELECT
            count(*) AS turns,
            count(*) FILTER (WHERE memories_used = 0) AS blind_turns,
            count(*) FILTER (WHERE memories_used > 0) AS grounded_turns,
            count(*) FILTER (WHERE retrieval_id IS NULL) AS unlinked_turns,
            avg(memories_used) AS avg_memories_used,
            avg(latency_ms) AS avg_latency_ms
        FROM agent_turns
        WHERE extract(epoch FROM (now() - created_at)) <= %s * 3600
        """,
        (hours,),
    )

    rows = _rows(
        """
        SELECT t.id, t.agent_id, t.session_id, t.query, t.response,
               t.memories_used, t.model_id, t.latency_ms, t.created_at,
               t.retrieval_id,
               r.raw_candidates, r.results_returned, r.top_similarity,
               coalesce(r.applied_floor, %s) AS applied_floor
        FROM agent_turns t
        LEFT JOIN memory_retrievals r ON r.id = t.retrieval_id
        WHERE extract(epoch FROM (now() - t.created_at)) <= %s * 3600
          AND t.memories_used = 0
        ORDER BY t.created_at DESC
        LIMIT %s
        """,
        (DEFAULT_MIN_SIMILARITY, hours, limit),
    )

    near_miss_fed = 0
    for row in rows:
        floor = float(row["applied_floor"]) if row["applied_floor"] is not None else None
        top = row["top_similarity"]
        row["near_miss_fed"] = bool(
            top is not None
            and floor is not None
            and (row["raw_candidates"] or 0) > 0
            and floor - NEAR_MISS_BAND <= float(top) < floor
        )
        row["short_by"] = (
            round(floor - float(top), 4) if row["near_miss_fed"] and top is not None else None
        )
        near_miss_fed += 1 if row["near_miss_fed"] else 0

    turns = totals.get("turns") or 0
    blind = totals.get("blind_turns") or 0
    avg_mem = totals.get("avg_memories_used")
    avg_lat = totals.get("avg_latency_ms")
    return {
        "generated_at": _now(),
        "hours": hours,
        "totals": {
            "turns": turns,
            "blind_turns": blind,
            "grounded_turns": totals.get("grounded_turns") or 0,
            "blind_pct": round(100.0 * blind / turns, 2) if turns else 0.0,
            "near_miss_fed": near_miss_fed,
            "unlinked_turns": totals.get("unlinked_turns") or 0,
            "avg_memories_used": round(float(avg_mem), 2) if avg_mem is not None else None,
            "avg_latency_ms": round(float(avg_lat), 1) if avg_lat is not None else None,
        },
        "band": NEAR_MISS_BAND,
        "blind": rows,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness only -- deliberately does not touch the database.

    This answers "is the web process up", which is what a platform health
    check needs. Whether the *memory* is healthy is a different question with
    a different answer, and /api/health/current is the honest one. Wiring a
    platform restart to that endpoint would restart the service every time the
    checker fell behind, which fixes nothing.
    """
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )
