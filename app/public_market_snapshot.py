from __future__ import annotations

"""Transparent, one-time public quote snapshot used only to unblock an offline demo.

The snapshot is not described as a live API sync. It is inserted only when the local
QQQM database does not yet contain these sessions; provider sync can replace those
rows later. Source: https://stockanalysis.com/etf/qqqm/history/ (checked 2026-10-24).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.db import DB_PATH, initialize_database, upsert_history

SNAPSHOT_SOURCE = "public_snapshot_stockanalysis_20261023"
SNAPSHOT_AS_OF = "2026-10-23"
SNAPSHOT_SOURCE_URL = "https://stockanalysis.com/etf/qqqm/history/"

# Date, Open, High, Low, Close, Volume. Public history page was checked on 2026-10-24.
_SNAPSHOT_ROWS = [
    ("2026-10-20", 294.54, 297.57, 292.52, 295.59, 3_634_194),
    ("2026-10-21", 297.45, 299.74, 296.50, 298.64, 2_459_361),
    ("2026-10-22", 293.86, 294.56, 289.01, 290.13, 4_266_191),
    ("2026-10-23", 289.10, 292.02, 288.74, 290.06, 2_509_040),
]


def bootstrap_public_snapshot(db_path: str | Path = DB_PATH) -> dict[str, Any]:
    """Insert only missing snapshot dates when the local QQQM series stops before 10/23."""
    from app.config import DB_PATH as CONFIG_DB_PATH
    import sqlite3

    path = Path(db_path or CONFIG_DB_PATH).resolve()
    if path != Path(CONFIG_DB_PATH).resolve():
        # The current app/database helper is bound to the configured project database.
        # This guard avoids accidentally writing the default helper into a different DB.
        raise ValueError("bootstrap_public_snapshot only supports the configured market_data.db")

    initialize_database()
    with sqlite3.connect(path) as conn:
        latest_row = conn.execute(
            "SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'"
        ).fetchone()
        latest = str(latest_row[0])[:10] if latest_row and latest_row[0] else None
        existing = {
            str(row[0])[:10]
            for row in conn.execute(
                "SELECT trade_date FROM market_daily WHERE UPPER(ticker)='QQQM' AND trade_date BETWEEN '2026-10-20' AND '2026-10-23'"
            ).fetchall()
        }

    if latest and latest >= SNAPSHOT_AS_OF:
        return {
            "applied": False,
            "latest_before": latest,
            "latest_after": latest,
            "added_dates": [],
            "source": SNAPSHOT_SOURCE,
            "source_url": SNAPSHOT_SOURCE_URL,
        }

    rows = [row for row in _SNAPSHOT_ROWS if row[0] not in existing]
    if rows:
        index = pd.to_datetime([row[0] for row in rows])
        frame = pd.DataFrame(
            {
                "Open": [row[1] for row in rows],
                "High": [row[2] for row in rows],
                "Low": [row[3] for row in rows],
                "Close": [row[4] for row in rows],
                "Adj Close": [row[4] for row in rows],
                "Volume": [row[5] for row in rows],
                "Dividends": [0.0] * len(rows),
                "Stock Splits": [0.0] * len(rows),
            },
            index=index,
        )
        upsert_history("QQQM", frame, source=SNAPSHOT_SOURCE)

    with sqlite3.connect(path) as conn:
        latest_row = conn.execute(
            "SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'"
        ).fetchone()
        latest_after = str(latest_row[0])[:10] if latest_row and latest_row[0] else None

    return {
        "applied": bool(rows),
        "latest_before": latest,
        "latest_after": latest_after,
        "added_dates": [row[0] for row in rows],
        "source": SNAPSHOT_SOURCE,
        "source_url": SNAPSHOT_SOURCE_URL,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "仅作为 API 暂不可用时的公开历史行情快照；不是实时行情，也不替代后续供应商同步。",
    }
