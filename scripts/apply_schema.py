#!/usr/bin/env python
"""Apply schema.sql to the cluster in DATABASE_URL.

    ./.venv/bin/python scripts/apply_schema.py          # create tables
    ./.venv/bin/python scripts/apply_schema.py --check  # report only, no writes

Safe to re-run: every statement is IF NOT EXISTS or already-exists tolerant,
and the script reports what was already there instead of failing.
"""

from __future__ import annotations

import os
import pathlib
import sys

import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPECTED = [
    "memory_records",
    "memory_health_events",
    "memory_retrievals",
    "agent_turns",
    "agent_config",
]


def main() -> int:
    load_dotenv(ROOT / ".env")
    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
        return 1

    check_only = "--check" in sys.argv

    with psycopg.connect(url, autocommit=True) as conn:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ).fetchall()
        }
        missing = [t for t in EXPECTED if t not in existing]

        if check_only:
            for table in EXPECTED:
                print(f"  {'present' if table in existing else 'MISSING':>8}  {table}")
            return 0 if not missing else 1

        if not missing:
            print("All tables already present; nothing to do.")
            print("(To rebuild from scratch, drop the database and re-run.)")
            return 0

        print(f"Missing tables: {', '.join(missing)}")
        conn.execute((ROOT / "schema.sql").read_text())
        print("Applied schema.sql")

        after = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ).fetchall()
        }
        for table in EXPECTED:
            print(f"  {'ok' if table in after else 'FAILED':>6}  {table}")
        return 0 if all(t in after for t in EXPECTED) else 1


if __name__ == "__main__":
    sys.exit(main())
