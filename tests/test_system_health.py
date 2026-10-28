from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.services.system_health import credential_status, inspect_project, status_label, expected_last_completed_session

NY = ZoneInfo("America/New_York")


class SystemHealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "reports" / "intraday").mkdir(parents=True)
        (self.root / "reports" / "video_scripts" / "20261020").mkdir(parents=True)
        (self.root / "reports" / "model_lab").mkdir(parents=True)
        self.db = self.root / "data" / "market_data.db"
        with closing(sqlite3.connect(self.db)) as conn:
            conn.executescript("""
                CREATE TABLE market_daily(ticker TEXT, trade_date TEXT, close REAL, source TEXT, fetched_at TEXT);
                INSERT INTO market_daily VALUES('QQQM','2026-10-19',295.78,'yahoo','2026-10-20T08:17:50+00:00');
                CREATE TABLE intraday_bars(symbol TEXT, feed TEXT, timeframe TEXT, trade_date_ny TEXT,
                    session_time_ny TEXT, fetched_at_utc TEXT, open REAL, high REAL, low REAL, close REAL);
            """)
            for i in range(26):
                mins = 9 * 60 + 30 + 15 * i
                hh, mm = divmod(mins, 60)
                conn.execute("INSERT INTO intraday_bars VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    "QQQM", "yahoo", "15Min", "2026-10-16", f"{hh:02d}:{mm:02d}",
                    "2026-10-17T01:41:50Z", 295, 296, 294, 295.5,
                ))
            conn.executescript("""
                CREATE TABLE news_items(published_at TEXT, fetched_at TEXT);
                INSERT INTO news_items VALUES('2026-10-13T18:00:00+00:00','2026-10-20T04:05:28+00:00');
                CREATE TABLE macro_observations(observation_date TEXT, retrieved_at TEXT);
                INSERT INTO macro_observations VALUES('2026-09-01','2026-10-14T03:25:27+00:00');
                CREATE TABLE forecast_feedback(ticker TEXT, status TEXT);
                INSERT INTO forecast_feedback VALUES('QQQM','evaluated');
                INSERT INTO forecast_feedback VALUES('QQQM','pending');
            """)
        (self.root / "reports" / "video_scripts" / "20261020" / "每日行情视频脚本与两张配图.zip").write_bytes(b"zip")
        (self.root / "reports" / "intraday" / "QQQM_video_forecast_latest.json").write_text(
            json.dumps({"as_of_date": "2026-10-19", "forecast_date_weekday_estimate": None, "available": False}), encoding="utf-8"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_reads_dates_not_just_last_fetch(self):
        with patch.dict(os.environ, {}, clear=True):
            health = inspect_project(self.root, now_ny=datetime(2026, 10, 21, 8, 0, tzinfo=NY))
        self.assertEqual(health["daily"]["latest_date"], "2026-10-19")
        self.assertEqual(health["intraday"]["latest_complete_date"], "2026-10-16")
        self.assertEqual(health["news"]["latest_published_date"], "2026-10-13")
        self.assertEqual(health["news"]["latest_fetched_at"], "2026-10-20T04:05:28+00:00")
        self.assertEqual(health["forecast"]["video_forecast"]["available"], False)

    def test_status_label_distinguishes_stale_and_missing(self):
        self.assertEqual(status_label(None, "2026-10-20")[0], "缺数据")
        self.assertEqual(status_label("2026-10-19", "2026-10-20")[0], "有滞后")
        self.assertEqual(status_label("2026-10-20", "2026-10-20")[0], "已到最近交易日")

    def test_credentials_only_report_presence_and_never_return_secret_values(self):
        (self.root / ".env").write_text(
            "ALPACA_API_KEY_ID=example-public-key\nALPACA_API_SECRET_KEY=super-secret-value\n", encoding="utf-8"
        )
        status = credential_status(self.root)
        self.assertTrue(status["alpaca_configured"])
        self.assertNotIn("super-secret-value", json.dumps(status))
        self.assertNotIn("example-public-key", json.dumps(status))

    def test_missing_credentials_are_reported_false(self):
        # Isolate the test from credentials exported by the developer's shell or .env loader.
        with patch.dict(os.environ, {}, clear=True):
            status = credential_status(self.root)
        self.assertFalse(status["alpaca_configured"])


if __name__ == "__main__":
    unittest.main()
