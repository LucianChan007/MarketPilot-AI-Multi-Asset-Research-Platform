"""One-time repair for the QQQM project's FRED/BLS SQLite table collision.

Run from project root:
    .\\.venv\\Scripts\\python.exe repair_macro_tables.py

The script makes a timestamped SQLite backup first, restores the original FRED
macro_observations table from macro_observations_legacy, moves BLS observations
into bls_macro_observations, and isolates the BLS sync state table.
It never deletes the mixed table; that table is archived under a new name.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/market_data.db").resolve()
BLS_SERIES_IDS = ("CUSR0000SA0", "LNS14000000", "CES0000000001")
BLS_OBSERVATION_COLUMNS = {
    "series_id", "series_name", "category", "year", "period", "period_name",
    "value", "unit", "footnotes", "source_name", "source_url", "fetched_at",
}
FRED_OBSERVATION_COLUMNS = {"series_id", "observation_date", "value", "retrieved_at"}
BLS_STATE_COLUMNS = {
    "series_id", "series_name", "category", "source_url", "last_attempt_at",
    "last_success_at", "last_observation_count", "last_error",
}


def tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    safe = table.replace('"', '""')
    return {str(row[1]).lower() for row in conn.execute(f'PRAGMA table_info("{safe}")')}


def row_count(conn: sqlite3.Connection, table: str) -> int:
    safe = table.replace('"', '""')
    return int(conn.execute(f'SELECT COUNT(*) FROM "{safe}"').fetchone()[0])


def unique_name(existing: set[str], base: str) -> str:
    if base not in existing:
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def main() -> int:
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}\nRun this from the project root.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DB_PATH.with_name(f"{DB_PATH.stem}.before_macro_split_{stamp}{DB_PATH.suffix}")

    # Make a consistent SQLite-level backup before attempting any schema changes.
    source = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        backup = sqlite3.connect(str(backup_path))
        try:
            source.backup(backup)
        finally:
            backup.close()
    finally:
        source.close()
    print(f"Backup created: {backup_path}")

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        names = tables(conn)
        if "macro_observations" not in names:
            # Already repaired? Validate target schema and return idempotently.
            if "bls_macro_observations" in names and FRED_OBSERVATION_COLUMNS.issubset(columns(conn, "macro_observations")):
                print("macro_observations seems already restored; no table rename needed.")
                print("FRED rows:", row_count(conn, "macro_observations"))
                if "bls_macro_observations" in names:
                    print("BLS rows:", row_count(conn, "bls_macro_observations"))
                return 0
            raise SystemExit("Expected macro_observations table was not found; no changes made.")

        current_cols = columns(conn, "macro_observations")
        legacy_exists = "macro_observations_legacy" in names
        legacy_cols = columns(conn, "macro_observations_legacy") if legacy_exists else set()
        state_exists = "macro_sync_state" in names
        state_cols = columns(conn, "macro_sync_state") if state_exists else set()

        if not BLS_OBSERVATION_COLUMNS.issubset(current_cols):
            raise SystemExit(
                "Current macro_observations does not match the known BLS-style schema. "
                "No changes made. Columns: " + repr(sorted(current_cols))
            )
        if not legacy_exists or not FRED_OBSERVATION_COLUMNS.issubset(legacy_cols):
            raise SystemExit(
                "macro_observations_legacy does not contain the expected original FRED schema. "
                "No changes made. Columns: " + repr(sorted(legacy_cols))
            )
        if not state_exists or not BLS_STATE_COLUMNS.issubset(state_cols):
            raise SystemExit(
                "macro_sync_state does not match the known BLS sync-state schema. "
                "No changes made. Columns: " + repr(sorted(state_cols))
            )

        existing = set(names)
        archive_table = unique_name(existing, "macro_observations_bls_mixed_archive")
        existing.add(archive_table)

        conn.execute("BEGIN IMMEDIATE")

        # Build an isolated BLS table. If the table is already present, keep its data.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS bls_macro_observations (
                series_id TEXT NOT NULL,
                series_name TEXT NOT NULL,
                category TEXT NOT NULL,
                year INTEGER NOT NULL,
                period TEXT NOT NULL,
                period_name TEXT NOT NULL DEFAULT '',
                value TEXT NOT NULL,
                unit TEXT NOT NULL DEFAULT '',
                footnotes TEXT NOT NULL DEFAULT '',
                source_name TEXT NOT NULL DEFAULT 'U.S. Bureau of Labor Statistics',
                source_url TEXT NOT NULL DEFAULT '',
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (series_id, year, period)
            )"""
        )
        # Only copy BLS's known series IDs out of the mixed table. Other rows stay archived.
        conn.executemany(
            """INSERT OR IGNORE INTO bls_macro_observations
               (series_id, series_name, category, year, period, period_name, value,
                unit, footnotes, source_name, source_url, fetched_at)
               SELECT series_id, series_name, category, year, period, period_name, value,
                      unit, footnotes, source_name, source_url, fetched_at
               FROM macro_observations WHERE series_id = ?""",
            [(sid,) for sid in BLS_SERIES_IDS],
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bls_macro_observations_series_period "
            "ON bls_macro_observations(series_id, year DESC, period DESC)"
        )

        # Move the mixed 579-row table aside intact, then restore the original FRED table name.
        # The known index belonged to the mixed/BLS table and is no longer needed under its old name.
        conn.execute("DROP INDEX IF EXISTS idx_macro_observations_series_period")
        conn.execute(f'ALTER TABLE macro_observations RENAME TO "{archive_table}"')
        conn.execute('ALTER TABLE macro_observations_legacy RENAME TO "macro_observations"')

        # The current macro_sync_state belongs to BLS, so move it into the BLS namespace.
        if "bls_macro_sync_state" not in tables(conn):
            conn.execute("ALTER TABLE macro_sync_state RENAME TO bls_macro_sync_state")
        else:
            # Merge sync metadata conservatively if a BLS-specific table already exists.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS bls_macro_sync_state (
                    series_id TEXT PRIMARY KEY,
                    series_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL DEFAULT '',
                    last_success_at TEXT NOT NULL DEFAULT '',
                    last_observation_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT ''
                )"""
            )
            conn.execute(
                """INSERT OR REPLACE INTO bls_macro_sync_state
                   SELECT series_id, series_name, category, source_url, last_attempt_at,
                          last_success_at, last_observation_count, last_error
                   FROM macro_sync_state"""
            )
            state_archive = unique_name(tables(conn), "macro_sync_state_bls_archive")
            conn.execute(f'ALTER TABLE macro_sync_state RENAME TO "{state_archive}"')

        # Restore a familiar index for the FRED table, without assuming prior index names.
        fred_cols = columns(conn, "macro_observations")
        if FRED_OBSERVATION_COLUMNS.issubset(fred_cols):
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fred_macro_observations_series_date "
                "ON macro_observations(series_id, observation_date DESC)"
            )

        check = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise sqlite3.DatabaseError(f"SQLite integrity_check failed: {check}")

        conn.commit()
        final_tables = tables(conn)
        print("Migration completed.")
        print("FRED macro_observations rows:", row_count(conn, "macro_observations"))
        print("BLS bls_macro_observations rows:", row_count(conn, "bls_macro_observations"))
        print("Archived mixed table:", archive_table, f"({row_count(conn, archive_table)} rows)")
        print("BLS sync table present:", "bls_macro_sync_state" in final_tables)
        print("SQLite integrity_check:", check)
        print("Original backup retained at:", backup_path)
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
