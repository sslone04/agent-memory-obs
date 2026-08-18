"""Keep the demo failures fresh for as long as judging is open.

Every check in `checks.py` looks back 60 minutes.  A failure injected once is
invisible an hour later, so a judge opening the dashboard at an arbitrary time
would land on a green board sitting under a banner that promises injected
failures.  This Lambda closes that gap: every 30 minutes it re-injects the two
retrieval failures, so the live check window is populated whenever anyone looks.

It calls `demo_harness.py` rather than restating the injections.  There is one
definition of what a near miss is, and the scheduled demo cannot drift from the
one a reader runs locally.

Safety is inherited from the harness and re-asserted here:

* `require_demo()` -- nothing is written without a ``demo-`` agent_id prefix.
* `protected_counts()` / `assert_untouched()` wrap the whole run, so a single
  changed non-demo row aborts it.
* Every DELETE the pruner issues is predicated on ``agent_id LIKE 'demo-%'``.
* Nothing here logs the connection string, and psycopg errors -- which quote
  the conninfo -- are redacted before they reach CloudWatch.

    python demo_injector.py          # same run, locally
"""

from __future__ import annotations

import os
import re
from typing import Any

import psycopg

from demo_harness import (
    AGENT_MAIN,
    DEMO_LIKE,
    assert_untouched,
    demo_counts,
    do_seed,
    inject_degradation,
    inject_near_miss,
    protected_counts,
    require_demo,
    step,
)
from memory import connect

# How much demo history to keep.  24h is not arbitrary: it is the widest window
# any panel on the dashboard reads, so a demo row older than this is invisible
# everywhere and only costs storage.
#
# Pruning by age rather than capping total rows is the choice here.  A cap has
# to decide which rows to drop and would quietly delete the oldest points out
# from under the 24h charts mid-window; an age bound makes the steady state a
# function of the schedule (~23 retrievals x 48 runs/day) instead of a function
# of how long judging happens to run.
PRUNE_HOURS = float(os.getenv("DEMO_PRUNE_HOURS", "24"))

# Healthy recalls per run.  Not part of the two injections, but without them the
# only demo retrievals inside the window after a prune are failures, and the
# calibration histogram loses the "returned to agent" cluster that makes the
# discarded ones legible as a separate mode.
HEALTHY_RECALLS = int(os.getenv("DEMO_HEALTHY_RECALLS", "4"))

_URL_RE = re.compile(r"postgres(?:ql)?://[^\s'\"]+", re.I)


def _redact(text: str) -> str:
    """psycopg quotes the connection string in its errors.  It must not leave."""
    return _URL_RE.sub("postgres://<redacted>", text)


def prune(conn: psycopg.Connection) -> dict[str, int]:
    """Drop demo rows older than the retention bound.

    memory_records is deliberately never pruned -- it is the corpus the
    injections retrieve *against*, and deleting it would turn every near miss
    into an empty-corpus miss, which is the other failure class entirely.

    Retrievals are kept while any agent_turn still points at them.  The FK is
    ON DELETE SET NULL, so pruning a referenced retrieval would silently break
    the turn -> retrieval trace that the Agent impact panel is built on.
    """
    removed: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_turns "
            "WHERE agent_id LIKE %s "
            "  AND extract(epoch FROM (now() - created_at)) > %s * 3600",
            (DEMO_LIKE, PRUNE_HOURS),
        )
        removed["agent_turns"] = cur.rowcount

        cur.execute(
            "DELETE FROM memory_retrievals r "
            "WHERE r.agent_id LIKE %s "
            "  AND extract(epoch FROM (now() - r.retrieved_at)) > %s * 3600 "
            "  AND NOT EXISTS (SELECT 1 FROM agent_turns t WHERE t.retrieval_id = r.id)",
            (DEMO_LIKE, PRUNE_HOURS),
        )
        removed["memory_retrievals"] = cur.rowcount

        cur.execute(
            "DELETE FROM memory_health_events "
            "WHERE agent_id LIKE %s "
            "  AND extract(epoch FROM (now() - observed_at)) > %s * 3600",
            (DEMO_LIKE, PRUNE_HOURS),
        )
        removed["memory_health_events"] = cur.rowcount
    return removed


def run() -> dict[str, Any]:
    """Prune, top up the healthy baseline, then re-inject both failures."""
    require_demo(AGENT_MAIN)

    with connect() as conn:
        before = protected_counts(conn)

        removed = prune(conn)
        step(
            f"pruned demo rows older than {PRUNE_HOURS:g}h: "
            + " · ".join(f"{k.split('_')[-1]} {v}" for k, v in removed.items())
        )

        # Rebuilds the corpus only if it is missing; otherwise just runs the
        # healthy recalls that keep the returned-to-agent cluster populated.
        do_seed(conn, recalls=HEALTHY_RECALLS)

        # Order matters for the story, not for correctness: near_miss re-arms
        # the miscalibrated floor, and degradation is read against that floor.
        inject_near_miss(conn)
        inject_degradation(conn)

        after = protected_counts(conn)
        assert_untouched(before, after)

        demo = demo_counts(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n, "
                "       count(*) FILTER (WHERE results_returned = 0 "
                "                          AND coalesce(raw_candidates, 0) > 0) AS floored "
                "FROM memory_retrievals "
                "WHERE agent_id LIKE %s "
                "  AND extract(epoch FROM (now() - retrieved_at)) <= 3600",
                (DEMO_LIKE,),
            )
            window = cur.fetchone()

        step(
            f"last 60m: {window['n']} demo retrievals, {window['floored']} floored out"
        )
        step("safety check passed: no non-demo rows were modified")

        return {
            "ok": True,
            "pruned": removed,
            "demo_rows": demo,
            "window_60m": {
                "retrievals": window["n"],
                "floored_out": window["floored"],
            },
            "protected_rows_unchanged": True,
        }


def lambda_handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    try:
        return run()
    except psycopg.Error as exc:
        # Raise, so the invocation is recorded as a failure and the schedule's
        # retry applies -- but without the conninfo psycopg puts in the message.
        raise RuntimeError(
            f"{type(exc).__name__}: {_redact(str(exc))}"
        ) from None


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, default=str))
