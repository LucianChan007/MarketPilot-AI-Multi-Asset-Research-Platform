from __future__ import annotations

"""Local-only initialization for clean project installs.

The distributable intentionally excludes SQLite databases. If the legacy market
DB has no QQQM rows, import the bundled Tiingo history CSV into it. This function
never makes a network request and never overwrites an existing QQQM date.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "market_data.db"
DEFAULT_HISTORY_PATH = ROOT / "data" / "QQQM_tiingo_history.csv"
SOURCE_NAME = "bundled_tiingo_history_csv"


def _initialize_market_table(conn: sqlite3.Connection) -> None:
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market_daily_date ON market_daily(trade_date)")


def ensure_bundled_qqqm_history(
    db_path: str | Path = DEFAULT_DB_PATH,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
) -> dict[str, Any]:
    """Ensure local legacy pages have historical QQQM data on a fresh install.

    Existing QQQM data is never replaced. This is a local CSV bootstrap only; it
    does not claim to be a current quote or successful provider synchronization.
    """
    db_path = Path(db_path)
    history_path = Path(history_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path, timeout=15) as conn:
        _initialize_market_table(conn)
        existing = conn.execute(
            "SELECT COUNT(*), MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'"
        ).fetchone()
        if int(existing[0] or 0) > 0:
            return {
                "applied": False,
                "reason": "existing_qqqm_data",
                "rows_added": 0,
                "latest_date": str(existing[1])[:10] if existing[1] else None,
                "source": None,
                "db_path": str(db_path),
            }
        if not history_path.is_file():
            return {
                "applied": False,
                "reason": "bundled_csv_missing",
                "rows_added": 0,
                "latest_date": None,
                "source": None,
                "db_path": str(db_path),
                "error": f"找不到本地历史数据文件：{history_path}",
            }

        raw = pd.read_csv(history_path)
        columns = {str(column).strip().lower(): column for column in raw.columns}
        date_column = columns.get("date")
        required = ("open", "high", "low", "close", "volume")
        if date_column is None or any(key not in columns for key in required):
            return {
                "applied": False,
                "reason": "bundled_csv_invalid_schema",
                "rows_added": 0,
                "latest_date": None,
                "source": None,
                "db_path": str(db_path),
                "error": f"历史 CSV 缺少必要字段。实际字段：{list(raw.columns)}",
            }

        dates = pd.to_datetime(raw[date_column], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
        adj_column = columns.get("adjclose") or columns.get("adj_close") or columns.get("adj close")
        div_column = columns.get("divcash") or columns.get("dividends")
        split_column = columns.get("splitfactor") or columns.get("stock_splits")
        records: list[tuple[Any, ...]] = []
        imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        for index, row in raw.iterrows():
            trade_date = dates.iloc[index]
            if not trade_date or pd.isna(trade_date):
                continue
            try:
                open_price = float(row[columns["open"]])
                high_price = float(row[columns["high"]])
                low_price = float(row[columns["low"]])
                close_price = float(row[columns["close"]])
                volume_value = pd.to_numeric(row[columns["volume"]], errors="coerce")
                adj_value = pd.to_numeric(row[adj_column], errors="coerce") if adj_column else close_price
                div_value = pd.to_numeric(row[div_column], errors="coerce") if div_column else 0.0
                split_value = pd.to_numeric(row[split_column], errors="coerce") if split_column else 0.0
            except (TypeError, ValueError, KeyError):
                continue
            if min(open_price, high_price, low_price, close_price) <= 0:
                continue
            if high_price < max(open_price, low_price, close_price) or low_price > min(open_price, high_price, close_price):
                continue
            records.append((
                "QQQM", str(trade_date), open_price, high_price, low_price, close_price,
                float(adj_value) if pd.notna(adj_value) else close_price,
                int(volume_value) if pd.notna(volume_value) and float(volume_value) >= 0 else None,
                float(div_value) if pd.notna(div_value) else 0.0,
                float(split_value) if pd.notna(split_value) else 0.0,
                SOURCE_NAME, imported_at,
            ))

        if records:
            conn.executemany(
                """INSERT OR IGNORE INTO market_daily
                (ticker,trade_date,open,high,low,close,adj_close,volume,dividends,stock_splits,source,fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                records,
            )
        latest = conn.execute(
            "SELECT MAX(trade_date), COUNT(*) FROM market_daily WHERE UPPER(ticker)='QQQM'"
        ).fetchone()
        conn.commit()

    return {
        "applied": bool(records),
        "reason": "seeded_from_bundled_csv" if records else "csv_has_no_valid_rows",
        "rows_added": len(records),
        "latest_date": str(latest[0])[:10] if latest and latest[0] else None,
        "source": SOURCE_NAME if records else None,
        "db_path": str(db_path),
        "history_path": str(history_path),
        "note": "仅导入随项目附带的历史 CSV，不联网、不代表实时行情。",
    }


def mirror_qqqm_daily_from_multi_asset(
    legacy_db_path: str | Path = DEFAULT_DB_PATH,
    multi_asset_db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Mirror QQQM daily bars into the legacy database for existing pages.

    This is local-only. It never fetches prices and only copies already stored rows.
    Current provider rows can replace bundled/static snapshots, but old provider rows
    are not overwritten by a lower-confidence bundled row.
    """
    from app.multi_asset_engine import DB_PATH as MULTI_DB_PATH

    legacy_db_path = Path(legacy_db_path)
    multi_asset_db_path = Path(multi_asset_db_path or MULTI_DB_PATH)
    ensure_bundled_qqqm_history(legacy_db_path)
    if not multi_asset_db_path.is_file():
        return {"applied": False, "rows_copied": 0, "reason": "multi_asset_db_missing"}

    with sqlite3.connect(multi_asset_db_path, timeout=15) as source_conn:
        table = source_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='asset_daily'"
        ).fetchone()
        if not table:
            return {"applied": False, "rows_copied": 0, "reason": "asset_daily_table_missing"}
        source_rows = source_conn.execute(
            "SELECT trade_date,open,high,low,close,adj_close,volume,feed,fetched_at "
            "FROM asset_daily WHERE UPPER(symbol)='QQQM' ORDER BY trade_date"
        ).fetchall()

    if not source_rows:
        return {"applied": False, "rows_copied": 0, "reason": "no_multi_asset_qqqm_rows"}

    with sqlite3.connect(legacy_db_path, timeout=15) as target_conn:
        _initialize_market_table(target_conn)
        copied = 0
        for row in source_rows:
            trade_date, open_price, high_price, low_price, close_price, adj_close, volume, feed, fetched_at = row
            source = f"multi_asset_{str(feed or 'unknown').lower()}"
            cursor = target_conn.execute(
                """INSERT INTO market_daily
                (ticker,trade_date,open,high,low,close,adj_close,volume,dividends,stock_splits,source,fetched_at)
                VALUES ('QQQM',?,?,?,?,?,?,?,0,0,?,?)
                ON CONFLICT(ticker,trade_date) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                    adj_close=excluded.adj_close, volume=excluded.volume, source=excluded.source, fetched_at=excluded.fetched_at
                WHERE market_daily.source LIKE 'bundled_%'
                   OR market_daily.source LIKE 'public_snapshot_%'
                   OR market_daily.source LIKE 'multi_asset_%'
                """,
                (trade_date, open_price, high_price, low_price, close_price, adj_close, volume, source, fetched_at),
            )
            copied += max(0, cursor.rowcount)
        latest = target_conn.execute(
            "SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'"
        ).fetchone()[0]
        target_conn.commit()
    return {"applied": bool(copied), "rows_copied": copied, "latest_date": latest, "source": "multi_asset_database"}
