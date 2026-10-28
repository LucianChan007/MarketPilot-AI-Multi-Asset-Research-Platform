from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.services.local_data_bootstrap import (
    SOURCE_NAME,
    ensure_bundled_qqqm_history,
    mirror_qqqm_daily_from_multi_asset,
)


class LocalDataBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.legacy_db = self.root / "data" / "market_data.db"
        self.multi_db = self.root / "data" / "multi_asset_research.db"
        self.csv = self.root / "data" / "QQQM_tiingo_history.csv"
        self.csv.parent.mkdir(parents=True)
        pd.DataFrame([
            {"date": "2026-10-07T00:00:00.000Z", "open": 100, "high": 102, "low": 99, "close": 101, "adjClose": 101, "volume": 1000, "divCash": 0, "splitFactor": 1},
            {"date": "2026-10-08T00:00:00.000Z", "open": 101, "high": 103, "low": 100, "close": 102, "adjClose": 102, "volume": 1100, "divCash": 0, "splitFactor": 1},
        ]).to_csv(self.csv, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_fresh_install_seeds_bundled_csv_without_network(self):
        result = ensure_bundled_qqqm_history(self.legacy_db, self.csv)
        self.assertTrue(result["applied"])
        self.assertEqual(result["rows_added"], 2)
        self.assertEqual(result["latest_date"], "2026-10-08")
        with sqlite3.connect(self.legacy_db) as conn:
            rows = conn.execute("SELECT ticker,trade_date,source FROM market_daily ORDER BY trade_date").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row[0] == "QQQM" and row[2] == SOURCE_NAME for row in rows))

    def test_bootstrap_is_idempotent(self):
        first = ensure_bundled_qqqm_history(self.legacy_db, self.csv)
        second = ensure_bundled_qqqm_history(self.legacy_db, self.csv)
        self.assertTrue(first["applied"])
        self.assertFalse(second["applied"])
        self.assertEqual(second["latest_date"], "2026-10-08")
        with sqlite3.connect(self.legacy_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM market_daily WHERE ticker='QQQM'").fetchone()[0]
        self.assertEqual(count, 2)

    def test_existing_market_rows_are_not_overwritten_by_seed(self):
        ensure_bundled_qqqm_history(self.legacy_db, self.csv)
        with sqlite3.connect(self.legacy_db) as conn:
            conn.execute("UPDATE market_daily SET close=999,source='official_test_feed' WHERE ticker='QQQM' AND trade_date='2026-10-08'")
        second = ensure_bundled_qqqm_history(self.legacy_db, self.csv)
        self.assertFalse(second["applied"])
        with sqlite3.connect(self.legacy_db) as conn:
            row = conn.execute("SELECT close,source FROM market_daily WHERE ticker='QQQM' AND trade_date='2026-10-08'").fetchone()
        self.assertEqual(row, (999.0, "official_test_feed"))

    def test_mirror_copies_multi_asset_qqqm_rows_and_refreshes_bundled_rows(self):
        ensure_bundled_qqqm_history(self.legacy_db, self.csv)
        with sqlite3.connect(self.multi_db) as conn:
            conn.execute("""CREATE TABLE asset_daily(
                symbol TEXT, trade_date TEXT, feed TEXT, open REAL, high REAL, low REAL,
                close REAL, adj_close REAL, volume REAL, fetched_at TEXT,
                PRIMARY KEY(symbol,trade_date,feed))""")
            conn.execute("INSERT INTO asset_daily VALUES(?,?,?,?,?,?,?,?,?,?)", (
                "QQQM", "2026-10-08", "yahoo", 101, 104, 100, 103, 103, 1500, "2026-10-28T00:00:00Z"
            ))
            conn.execute("INSERT INTO asset_daily VALUES(?,?,?,?,?,?,?,?,?,?)", (
                "QQQM", "2026-10-09", "yahoo", 103, 105, 102, 104, 104, 1600, "2026-10-28T00:00:00Z"
            ))
        result = mirror_qqqm_daily_from_multi_asset(self.legacy_db, self.multi_db)
        self.assertTrue(result["applied"])
        self.assertEqual(result["latest_date"], "2026-10-09")
        with sqlite3.connect(self.legacy_db) as conn:
            corrected = conn.execute("SELECT close,source FROM market_daily WHERE ticker='QQQM' AND trade_date='2026-10-08'").fetchone()
            newest = conn.execute("SELECT close,source FROM market_daily WHERE ticker='QQQM' AND trade_date='2026-10-09'").fetchone()
        self.assertEqual(corrected, (103.0, "multi_asset_yahoo"))
        self.assertEqual(newest, (104.0, "multi_asset_yahoo"))

    def test_mirror_does_not_replace_existing_provider_row(self):
        ensure_bundled_qqqm_history(self.legacy_db, self.csv)
        with sqlite3.connect(self.legacy_db) as conn:
            conn.execute("UPDATE market_daily SET close=777,source='eodhd' WHERE ticker='QQQM' AND trade_date='2026-10-08'")
        with sqlite3.connect(self.multi_db) as conn:
            conn.execute("""CREATE TABLE asset_daily(
                symbol TEXT, trade_date TEXT, feed TEXT, open REAL, high REAL, low REAL,
                close REAL, adj_close REAL, volume REAL, fetched_at TEXT,
                PRIMARY KEY(symbol,trade_date,feed))""")
            conn.execute("INSERT INTO asset_daily VALUES(?,?,?,?,?,?,?,?,?,?)", (
                "QQQM", "2026-10-08", "stooq", 101, 104, 100, 103, 103, 1500, "2026-10-28T00:00:00Z"
            ))
        mirror_qqqm_daily_from_multi_asset(self.legacy_db, self.multi_db)
        with sqlite3.connect(self.legacy_db) as conn:
            row = conn.execute("SELECT close,source FROM market_daily WHERE ticker='QQQM' AND trade_date='2026-10-08'").fetchone()
        self.assertEqual(row, (777.0, "eodhd"))


if __name__ == "__main__":
    unittest.main()
