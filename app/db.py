from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.config import DB_PATH, ensure_directories


def get_connection() -> sqlite3.Connection:
    ensure_directories()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_daily (
                ticker TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                dividends REAL DEFAULT 0,
                stock_splits REAL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'unknown',
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (ticker, trade_date)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_daily_date ON market_daily(trade_date)"
        )


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def upsert_history(ticker: str, history: pd.DataFrame, source: str = "unknown") -> int:
    if history.empty:
        return 0

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[tuple] = []
    for timestamp, row in history.iterrows():
        trade_date = pd.Timestamp(timestamp).date().isoformat()
        adj_close = row.get("Adj Close", row.get("Close"))
        volume_value = row.get("Volume")
        volume = None if volume_value is None or pd.isna(volume_value) else int(volume_value)
        dividends = _as_float(row.get("Dividends", 0)) or 0.0
        splits = _as_float(row.get("Stock Splits", 0)) or 0.0
        records.append(
            (
                ticker.upper(), trade_date,
                _as_float(row.get("Open")), _as_float(row.get("High")),
                _as_float(row.get("Low")), _as_float(row.get("Close")),
                _as_float(adj_close), volume, dividends, splits,
                source, fetched_at,
            )
        )

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO market_daily (
                ticker, trade_date, open, high, low, close, adj_close,
                volume, dividends, stock_splits, source, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, trade_date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                adj_close=excluded.adj_close,
                volume=excluded.volume,
                dividends=excluded.dividends,
                stock_splits=excluded.stock_splits,
                source=excluded.source,
                fetched_at=excluded.fetched_at
            """,
            records,
        )
    return len(records)


def load_history(ticker: str) -> pd.DataFrame:
    initialize_database()
    with get_connection() as conn:
        df = pd.read_sql_query(
            """SELECT trade_date, open, high, low, close, adj_close, volume,
                      dividends, stock_splits, source, fetched_at
               FROM market_daily WHERE ticker = ? ORDER BY trade_date""",
            conn,
            params=(ticker.upper(),),
        )
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
    return df


def database_status() -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ticker, COUNT(*) AS rows_count,
                   MIN(trade_date) AS first_date,
                   MAX(trade_date) AS last_date,
                   MAX(fetched_at) AS last_fetched_at
            FROM market_daily
            GROUP BY ticker
            ORDER BY ticker
            """
        ).fetchall()
    return [dict(row) for row in rows]
