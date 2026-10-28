from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.intraday_data import ensure_intraday_table, parse_bar_timestamp, save_intraday_bars


class IntradayDataTests(unittest.TestCase):
    def test_parse_utc_timestamp_to_new_york_date_and_time(self):
        utc, ny, trade_date = parse_bar_timestamp("2026-10-13T13:30:00Z")
        self.assertEqual(utc, "2026-10-13T13:30:00Z")
        self.assertTrue(ny.startswith("2026-10-13T09:30:00-04:00"))
        self.assertEqual(trade_date, "2026-10-13")

    def test_creates_separate_feed_labeled_table_and_upserts_idempotently(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            bar = {"t": "2026-10-13T13:30:00Z", "o": 304.5, "h": 305.0, "l": 304.0, "c": 304.8, "v": 1234, "n": 10, "vw": 304.7}
            first = save_intraday_bars([bar], symbol="QQQM", feed="iex", timeframe="5Min", db_path=db_path)
            second = save_intraday_bars([bar], symbol="QQQM", feed="iex", timeframe="5Min", db_path=db_path)
            self.assertEqual(first["total_rows"], 1)
            self.assertEqual(second["total_rows"], 1)
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute("SELECT symbol,feed,timeframe,trade_date_ny,session_time_ny,close FROM intraday_bars").fetchone()
                self.assertEqual(row, ("QQQM", "iex", "5Min", "2026-10-13", "09:30", 304.8))
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM intraday_bars").fetchone()[0], 1)
            finally:
                conn.close()

    def test_rejects_bad_ohlc(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            bad = {"t": "2026-10-13T13:30:00Z", "o": 305, "h": 304, "l": 303, "c": 304.5, "v": 1}
            with self.assertRaises(ValueError):
                save_intraday_bars([bad], symbol="QQQM", feed="iex", timeframe="5Min", db_path=db_path)


if __name__ == "__main__":
    unittest.main()
