"""Observability checks over the agent memory tables.

Four independent checks, each writing one row to memory_health_events:

    staleness         -- rows that landed recently but carry an old as-of time
    eviction_pressure -- share of live rows about to expire
    empty_resolve     -- retrievals that found candidates but cleared none
    near_miss         -- retrievals the floor probably discarded by mistake
    vector_drift      -- embeddings that are missing, mis-sized, or un-normalised

empty_resolve and near_miss look at the same rows and mean opposite things.
empty_resolve is a DATA problem: nothing relevant was there. near_miss is a
CONFIG problem: something relevant was there and the floor rejected it. Keeping
them separate is the point -- they have different fixes.

Every check writes a row on every run, including when it finds nothing.  A
missing health record means the checker did not run; it must never be readable
as "the system was healthy".

Importable as a module (a Lambda handler can call run_all directly) with a CLI
entrypoint under __main__.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

import psycopg

from memory import DEFAULT_MIN_SIMILARITY, EMBED_DIM, connect

# --- staleness -------------------------------------------------------------
# A row is stale when the fact it carries was true this long ago...
STALENESS_THRESHOLD_HOURS = 24.0
# ...but only counts if the row itself landed within this window, which is what
# makes it the "looks fresh, carries an old effective age" case.
STALENESS_RECENT_WRITE_MINUTES = 60
STALENESS_WARN_COUNT = 1
STALENESS_CRITICAL_COUNT = 10

# --- eviction pressure -----------------------------------------------------
EVICTION_HORIZON_MINUTES = 60
EVICTION_WARN_PCT = 10.0
EVICTION_CRITICAL_PCT = 25.0

# --- empty resolve ---------------------------------------------------------
EMPTY_RESOLVE_WINDOW_MINUTES = 60
EMPTY_RESOLVE_WARN_PCT = 10.0
EMPTY_RESOLVE_CRITICAL_PCT = 30.0
# Below this many retrievals in the window, percentages are noise -- stay ok.
EMPTY_RESOLVE_MIN_SAMPLE = 5

# --- near miss / false negatives -------------------------------------------
NEAR_MISS_WINDOW_MINUTES = 60
# How far below the applied floor a score can sit and still be "probably relevant".
NEAR_MISS_BAND = 0.15
# Share OF FLOORED-OUT RETRIEVALS that are near misses, not share of all traffic.
NEAR_MISS_WARN_PCT = 25.0
NEAR_MISS_CRITICAL_PCT = 50.0
# Below this many floored-out retrievals, a percentage means nothing.
NEAR_MISS_MIN_SAMPLE = 3

# --- vector drift ----------------------------------------------------------
VECTOR_EXPECTED_DIM = EMBED_DIM
# Titan v2 returns unit vectors; anything outside 1.0 +/- this wrote a bad one.
VECTOR_NORM_TOLERANCE = 0.05
VECTOR_DRIFT_WARN_PCT = 1.0
VECTOR_DRIFT_CRITICAL_PCT = 5.0

SEVERITY_ORDER = {"ok": 0, "warn": 1, "critical": 2}


def _severity(value: float, warn: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warn:
        return "warn"
    return "ok"


def _pct(numerator: int, denominator: int) -> float:
    """Percentage that is 0.0 rather than an exception on an empty table."""
    if not denominator:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _record(
    conn: psycopg.Connection,
    check_name: str,
    severity: str,
    agent_id: str | None,
    detail: dict[str, Any],
) -> dict[str, Any]:
    """Persist one health event and return it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO memory_health_events (check_name, severity, agent_id, detail)
            VALUES (%s, %s, %s, %s::JSONB)
            RETURNING id, observed_at
            """,
            (check_name, severity, agent_id, json.dumps(detail, default=str)),
        )
        row = cur.fetchone()
    return {
        "id": row["id"],
        "observed_at": row["observed_at"],
        "check_name": check_name,
        "severity": severity,
        "agent_id": agent_id,
        "detail": detail,
    }


def _with_conn(
    fn: Callable[[psycopg.Connection], dict[str, Any]],
    conn: psycopg.Connection | None,
) -> dict[str, Any]:
    if conn is not None:
        return fn(conn)
    with connect() as owned:
        return fn(owned)


def check_staleness(
    conn: psycopg.Connection | None = None,
    *,
    threshold_hours: float = STALENESS_THRESHOLD_HOURS,
    recent_write_minutes: int = STALENESS_RECENT_WRITE_MINUTES,
) -> dict[str, Any]:
    """Recently-written rows whose effective_as_of is already old.

    These are the dangerous ones: written_at says fresh, so nothing downstream
    treats them with suspicion, while the fact they carry has aged out.
    """

    def _run(c: psycopg.Connection) -> dict[str, Any]:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(*) AS recent_writes,
                    count(*) FILTER (
                        WHERE extract(epoch FROM (now() - effective_as_of)) > %s * 3600
                    ) AS stale_count
                FROM memory_records
                WHERE extract(epoch FROM (now() - written_at)) <= %s * 60
                """,
                (threshold_hours, recent_write_minutes),
            )
            totals = cur.fetchone()

            cur.execute(
                """
                SELECT id, agent_id, session_id, kind, effective_as_of, written_at,
                       extract(epoch FROM (now() - effective_as_of)) / 3600.0 AS stale_hours
                FROM memory_records
                WHERE extract(epoch FROM (now() - written_at)) <= %s * 60
                  AND extract(epoch FROM (now() - effective_as_of)) > %s * 3600
                ORDER BY effective_as_of ASC
                LIMIT 1
                """,
                (recent_write_minutes, threshold_hours),
            )
            worst = cur.fetchone()

        stale_count = totals["stale_count"] or 0
        recent_writes = totals["recent_writes"] or 0
        severity = _severity(stale_count, STALENESS_WARN_COUNT, STALENESS_CRITICAL_COUNT)
        detail = {
            "threshold_hours": threshold_hours,
            "recent_write_window_minutes": recent_write_minutes,
            "recent_writes": recent_writes,
            "stale_count": stale_count,
            "stale_pct_of_recent": _pct(stale_count, recent_writes),
            "worst_offender": (
                {
                    "id": str(worst["id"]),
                    "agent_id": worst["agent_id"],
                    "session_id": worst["session_id"],
                    "kind": worst["kind"],
                    "effective_as_of": worst["effective_as_of"],
                    "written_at": worst["written_at"],
                    "stale_hours": round(float(worst["stale_hours"]), 2),
                }
                if worst
                else None
            ),
        }
        return _record(
            c, "staleness", severity, worst["agent_id"] if worst else None, detail
        )

    return _with_conn(_run, conn)


def check_eviction_pressure(
    conn: psycopg.Connection | None = None,
    *,
    horizon_minutes: int = EVICTION_HORIZON_MINUTES,
) -> dict[str, Any]:
    """How much of the live corpus is about to expire."""

    def _run(c: psycopg.Connection) -> dict[str, Any]:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(*) AS live_rows,
                    count(*) FILTER (
                        WHERE expires_at IS NOT NULL
                          AND extract(epoch FROM (expires_at - now())) <= %s * 60
                    ) AS expiring_soon,
                    count(*) FILTER (WHERE expires_at IS NULL) AS never_expire
                FROM memory_records
                WHERE expires_at IS NULL OR expires_at > now()
                """,
                (horizon_minutes,),
            )
            totals = cur.fetchone()

            cur.execute(
                """
                SELECT agent_id, count(*) AS expiring
                FROM memory_records
                WHERE expires_at IS NOT NULL
                  AND expires_at > now()
                  AND extract(epoch FROM (expires_at - now())) <= %s * 60
                GROUP BY agent_id
                ORDER BY expiring DESC
                LIMIT 1
                """,
                (horizon_minutes,),
            )
            worst = cur.fetchone()

        live_rows = totals["live_rows"] or 0
        expiring = totals["expiring_soon"] or 0
        share = _pct(expiring, live_rows)
        severity = _severity(share, EVICTION_WARN_PCT, EVICTION_CRITICAL_PCT)
        detail = {
            "horizon_minutes": horizon_minutes,
            "live_rows": live_rows,
            "expiring_within_horizon": expiring,
            "expiring_pct": share,
            "never_expire": totals["never_expire"] or 0,
            "warn_pct": EVICTION_WARN_PCT,
            "critical_pct": EVICTION_CRITICAL_PCT,
            "worst_offender": (
                {"agent_id": worst["agent_id"], "expiring": worst["expiring"]}
                if worst
                else None
            ),
        }
        return _record(
            c,
            "eviction_pressure",
            severity,
            worst["agent_id"] if worst else None,
            detail,
        )

    return _with_conn(_run, conn)


def check_empty_resolve(
    conn: psycopg.Connection | None = None,
    *,
    window_minutes: int = EMPTY_RESOLVE_WINDOW_MINUTES,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> dict[str, Any]:
    """Retrievals that found candidates but handed back nothing.

    raw_candidates > 0 with results_returned = 0 means the vector search is
    still working and the floor is eating everything -- the memory is present
    but no longer relevant.  That degrades silently: the caller just sees an
    agent that has forgotten things, with no error anywhere.
    """

    def _run(c: psycopg.Connection) -> dict[str, Any]:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(*) AS retrievals,
                    count(*) FILTER (
                        WHERE results_returned = 0 AND coalesce(raw_candidates, 0) > 0
                    ) AS floored_out,
                    count(*) FILTER (
                        WHERE results_returned = 0 AND coalesce(raw_candidates, 0) = 0
                    ) AS no_candidates,
                    count(*) FILTER (
                        WHERE top_similarity IS NOT NULL AND top_similarity < %s
                    ) AS below_floor,
                    count(*) FILTER (WHERE raw_candidates IS NULL) AS unattributed,
                    avg(top_similarity) FILTER (WHERE top_similarity IS NOT NULL)
                        AS avg_top_similarity
                FROM memory_retrievals
                WHERE extract(epoch FROM (now() - retrieved_at)) <= %s * 60
                """,
                (min_similarity, window_minutes),
            )
            totals = cur.fetchone()

            cur.execute(
                """
                SELECT agent_id, count(*) AS floored_out, max(top_similarity) AS best_seen
                FROM memory_retrievals
                WHERE extract(epoch FROM (now() - retrieved_at)) <= %s * 60
                  AND results_returned = 0
                  AND coalesce(raw_candidates, 0) > 0
                GROUP BY agent_id
                ORDER BY floored_out DESC
                LIMIT 1
                """,
                (window_minutes,),
            )
            worst = cur.fetchone()

        retrievals = totals["retrievals"] or 0
        floored_out = totals["floored_out"] or 0
        share = _pct(floored_out, retrievals)
        if retrievals < EMPTY_RESOLVE_MIN_SAMPLE:
            # Too few retrievals for a percentage to mean anything.
            severity = "ok"
        else:
            severity = _severity(
                share, EMPTY_RESOLVE_WARN_PCT, EMPTY_RESOLVE_CRITICAL_PCT
            )
        avg_top = totals["avg_top_similarity"]
        detail = {
            "window_minutes": window_minutes,
            "min_similarity": min_similarity,
            "retrievals": retrievals,
            "floored_out": floored_out,
            "floored_out_pct": share,
            "no_candidates": totals["no_candidates"] or 0,
            "below_floor": totals["below_floor"] or 0,
            "avg_top_similarity": round(float(avg_top), 4) if avg_top is not None else None,
            "rows_missing_raw_candidates": totals["unattributed"] or 0,
            "min_sample": EMPTY_RESOLVE_MIN_SAMPLE,
            "sample_too_small": retrievals < EMPTY_RESOLVE_MIN_SAMPLE,
            "worst_offender": (
                {
                    "agent_id": worst["agent_id"],
                    "floored_out": worst["floored_out"],
                    "best_similarity_seen": (
                        round(float(worst["best_seen"]), 4)
                        if worst["best_seen"] is not None
                        else None
                    ),
                }
                if worst
                else None
            ),
        }
        return _record(
            c, "empty_resolve", severity, worst["agent_id"] if worst else None, detail
        )

    return _with_conn(_run, conn)


def check_near_miss(
    conn: psycopg.Connection | None = None,
    *,
    window_minutes: int = NEAR_MISS_WINDOW_MINUTES,
    band: float = NEAR_MISS_BAND,
) -> dict[str, Any]:
    """Retrievals the floor probably threw away by mistake.

    This is deliberately NOT empty_resolve.  empty_resolve says "nothing in the
    corpus was relevant" -- a data problem, fixed by writing better memories.
    near_miss says "something almost certainly WAS relevant and the retrieval
    policy discarded it" -- a config problem, fixed by moving the floor.  A
    retrieval scoring 0.34 against a 0.35 floor is a false negative; one
    scoring 0.04 is a genuine miss.  Averaging them together hides both.
    """

    def _run(c: psycopg.Connection) -> dict[str, Any]:
        floored_sql = """
            WITH floored AS (
                SELECT agent_id, query_text, top_similarity,
                       coalesce(applied_floor, %s) AS floor
                FROM memory_retrievals
                WHERE extract(epoch FROM (now() - retrieved_at)) <= %s * 60
                  AND results_returned = 0
                  AND coalesce(raw_candidates, 0) > 0
                  AND top_similarity IS NOT NULL
            )
        """
        with c.cursor() as cur:
            cur.execute(
                floored_sql
                + """
                SELECT
                    count(*) AS floored_total,
                    count(*) FILTER (
                        WHERE top_similarity >= floor - %s AND top_similarity < floor
                    ) AS near_miss,
                    count(*) FILTER (WHERE top_similarity < floor - %s) AS clear_miss,
                    max(top_similarity) FILTER (
                        WHERE top_similarity >= floor - %s AND top_similarity < floor
                    ) AS max_near_miss,
                    min(top_similarity) FILTER (
                        WHERE top_similarity >= floor - %s AND top_similarity < floor
                    ) AS min_near_miss,
                    max(top_similarity) FILTER (WHERE top_similarity < floor - %s) AS max_clear_miss
                FROM floored
                """,
                (DEFAULT_MIN_SIMILARITY, window_minutes, band, band, band, band, band),
            )
            totals = cur.fetchone()

            cur.execute(
                floored_sql
                + """
                SELECT top_similarity, floor, agent_id, query_text
                FROM floored
                WHERE top_similarity >= floor - %s AND top_similarity < floor
                ORDER BY top_similarity DESC
                LIMIT 100
                """,
                (DEFAULT_MIN_SIMILARITY, window_minutes, band),
            )
            near_rows = cur.fetchall()

            cur.execute(
                floored_sql
                + """
                SELECT agent_id, count(*) AS near_miss,
                       max(top_similarity) AS best_rejected
                FROM floored
                WHERE top_similarity >= floor - %s AND top_similarity < floor
                GROUP BY agent_id ORDER BY count(*) DESC LIMIT 1
                """,
                (DEFAULT_MIN_SIMILARITY, window_minutes, band),
            )
            worst = cur.fetchone()

            cur.execute(
                """
                SELECT DISTINCT coalesce(applied_floor, %s) AS floor
                FROM memory_retrievals
                WHERE extract(epoch FROM (now() - retrieved_at)) <= %s * 60
                ORDER BY 1
                """,
                (DEFAULT_MIN_SIMILARITY, window_minutes),
            )
            floors_seen = [round(float(r["floor"]), 4) for r in cur.fetchall()]

        floored_total = totals["floored_total"] or 0
        near_miss = totals["near_miss"] or 0
        share = _pct(near_miss, floored_total)

        if floored_total < NEAR_MISS_MIN_SAMPLE:
            severity = "ok"
        else:
            severity = _severity(share, NEAR_MISS_WARN_PCT, NEAR_MISS_CRITICAL_PCT)

        scores = [round(float(r["top_similarity"]), 4) for r in near_rows]
        max_near = totals["max_near_miss"]
        max_clear = totals["max_clear_miss"]

        # A suggested floor only means something when both populations are
        # present: the highest score we rejected as relevant, and the highest
        # score we are confident was noise. The midpoint separates them.
        suggested = None
        suggestion_note = None
        if near_miss == 0:
            suggestion_note = "no near misses in the window -- nothing to suggest"
        elif max_clear is None:
            suggestion_note = (
                "no clearly-off-topic retrievals in the window to separate from, "
                "so a midpoint would be guesswork"
            )
        elif len(floors_seen) > 1:
            suggested = round((float(max_near) + float(max_clear)) / 2, 4)
            suggestion_note = (
                f"multiple floors in force in this window ({floors_seen}); the "
                "suggestion is an aggregate and should be applied per agent"
            )
        else:
            suggested = round((float(max_near) + float(max_clear)) / 2, 4)
            suggestion_note = (
                f"midpoint between the best rejected-but-relevant score "
                f"({float(max_near):.4f}) and the best clearly-off-topic score "
                f"({float(max_clear):.4f})"
            )

        detail = {
            "window_minutes": window_minutes,
            "band": band,
            "floors_in_force": floors_seen,
            "floored_out_total": floored_total,
            "near_miss": near_miss,
            "near_miss_pct_of_floored": share,
            "clear_miss": totals["clear_miss"] or 0,
            "warn_pct": NEAR_MISS_WARN_PCT,
            "critical_pct": NEAR_MISS_CRITICAL_PCT,
            "min_sample": NEAR_MISS_MIN_SAMPLE,
            "sample_too_small": floored_total < NEAR_MISS_MIN_SAMPLE,
            "score_distribution": {
                "near_miss_scores": scores,
                "max_near_miss": round(float(max_near), 4) if max_near is not None else None,
                "min_near_miss": (
                    round(float(totals["min_near_miss"]), 4)
                    if totals["min_near_miss"] is not None
                    else None
                ),
                "max_clear_miss": round(float(max_clear), 4) if max_clear is not None else None,
            },
            # A suggestion, never applied automatically. Moving a retrieval floor
            # changes what every future query returns; that is a human decision.
            "suggested_floor": suggested,
            "suggested_floor_is_advisory": True,
            "suggestion_basis": suggestion_note,
            "examples": [
                {
                    "agent_id": r["agent_id"],
                    "query_text": r["query_text"],
                    "top_similarity": round(float(r["top_similarity"]), 4),
                    "floor": round(float(r["floor"]), 4),
                }
                for r in near_rows[:5]
            ],
            "worst_offender": (
                {
                    "agent_id": worst["agent_id"],
                    "near_miss": worst["near_miss"],
                    "best_rejected": round(float(worst["best_rejected"]), 4),
                }
                if worst
                else None
            ),
        }
        return _record(
            c, "near_miss", severity, worst["agent_id"] if worst else None, detail
        )

    return _with_conn(_run, conn)


def check_vector_drift(
    conn: psycopg.Connection | None = None,
    *,
    expected_dim: int = VECTOR_EXPECTED_DIM,
    norm_tolerance: float = VECTOR_NORM_TOLERANCE,
) -> dict[str, Any]:
    """Embeddings that are missing, the wrong size, or not unit-normalised."""

    def _run(c: psycopg.Connection) -> dict[str, Any]:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(*) AS total_rows,
                    count(*) FILTER (WHERE embedding IS NULL) AS null_embeddings,
                    count(*) FILTER (
                        WHERE embedding IS NOT NULL AND vector_dims(embedding) <> %s
                    ) AS wrong_dim,
                    count(*) FILTER (
                        WHERE embedding IS NOT NULL
                          AND abs(vector_norm(embedding) - 1.0) > %s
                    ) AS bad_norm,
                    min(vector_norm(embedding)) AS min_norm,
                    max(vector_norm(embedding)) AS max_norm,
                    avg(vector_norm(embedding)) AS avg_norm
                FROM memory_records
                """,
                (expected_dim, norm_tolerance),
            )
            totals = cur.fetchone()

            cur.execute(
                """
                SELECT agent_id, count(*) AS bad_rows
                FROM memory_records
                WHERE embedding IS NULL
                   OR vector_dims(embedding) <> %s
                   OR abs(vector_norm(embedding) - 1.0) > %s
                GROUP BY agent_id
                ORDER BY bad_rows DESC
                LIMIT 1
                """,
                (expected_dim, norm_tolerance),
            )
            worst = cur.fetchone()

        total = totals["total_rows"] or 0
        bad = (
            (totals["null_embeddings"] or 0)
            + (totals["wrong_dim"] or 0)
            + (totals["bad_norm"] or 0)
        )
        share = _pct(bad, total)
        severity = _severity(share, VECTOR_DRIFT_WARN_PCT, VECTOR_DRIFT_CRITICAL_PCT)

        def _round(value: Any) -> float | None:
            return round(float(value), 6) if value is not None else None

        detail = {
            "expected_dim": expected_dim,
            "norm_tolerance": norm_tolerance,
            "total_rows": total,
            "null_embeddings": totals["null_embeddings"] or 0,
            "wrong_dim": totals["wrong_dim"] or 0,
            "bad_norm": totals["bad_norm"] or 0,
            "bad_rows": bad,
            "bad_pct": share,
            "min_norm": _round(totals["min_norm"]),
            "max_norm": _round(totals["max_norm"]),
            "avg_norm": _round(totals["avg_norm"]),
            "worst_offender": (
                {"agent_id": worst["agent_id"], "bad_rows": worst["bad_rows"]}
                if worst
                else None
            ),
        }
        return _record(
            c, "vector_drift", severity, worst["agent_id"] if worst else None, detail
        )

    return _with_conn(_run, conn)


CHECKS: tuple[Callable[..., dict[str, Any]], ...] = (
    check_staleness,
    check_eviction_pressure,
    check_empty_resolve,
    check_near_miss,
    check_vector_drift,
)


def run_all(conn: psycopg.Connection | None = None) -> list[dict[str, Any]]:
    """Run every check on one connection and return their results in order."""

    def _run(c: psycopg.Connection) -> Any:
        return [check(c) for check in CHECKS]

    return _with_conn(_run, conn)  # type: ignore[return-value]


def worst_severity(results: list[dict[str, Any]]) -> str:
    """The most severe outcome across a run -- handy as a Lambda exit signal."""
    if not results:
        return "ok"
    return max((r["severity"] for r in results), key=lambda s: SEVERITY_ORDER[s])


def lambda_handler(event: Any = None, context: Any = None) -> dict[str, Any]:
    """AWS Lambda entrypoint: run every check, return a JSON-safe summary.

    Never logs DATABASE_URL or any part of the connection string.
    """
    results = run_all()
    summary = {
        "worst_severity": worst_severity(results),
        "checks_run": len(results),
        "checks": [
            {
                "check_name": r["check_name"],
                "severity": r["severity"],
                "agent_id": r["agent_id"],
                "observed_at": r["observed_at"],
                "detail": r["detail"],
            }
            for r in results
        ],
    }
    for result in results:
        print(f"{result['check_name']}: {result['severity']}")
    print(f"worst_severity: {summary['worst_severity']}")
    # default=str flattens UUIDs/datetimes so the return value stays JSON-safe
    return json.loads(json.dumps(summary, default=str))


def _print_human(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"\n[{result['severity'].upper():>8}]  {result['check_name']}")
        if result["agent_id"]:
            print(f"           agent: {result['agent_id']}")
        for key, value in result["detail"].items():
            if isinstance(value, dict):
                print(f"           {key}:")
                for sub_key, sub_value in value.items():
                    print(f"             {sub_key}: {sub_value}")
            else:
                print(f"           {key}: {value}")
    print(f"\noverall: {worst_severity(results)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json", action="store_true", help="emit results as JSON instead of text"
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "warn", "critical"),
        default="never",
        help="exit non-zero when the worst severity reaches this level",
    )
    args = parser.parse_args(argv)

    results = run_all()

    if args.json:
        print(json.dumps(results, default=str, indent=2))
    else:
        _print_human(results)

    if args.fail_on != "never":
        threshold = SEVERITY_ORDER[args.fail_on]
        if SEVERITY_ORDER[worst_severity(results)] >= threshold:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
