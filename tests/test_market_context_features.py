from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.market_context_features import build_market_context_snapshot


class MarketContextFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_market.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE market_daily (
                ticker TEXT, trade_date TEXT, close REAL, adj_close REAL,
                volume INTEGER, source TEXT, fetched_at TEXT
            );
            CREATE TABLE news_items (
                news_id TEXT, source_name TEXT, category TEXT, title TEXT,
                summary TEXT, url TEXT, published_at TEXT, fetched_at TEXT
            );
            CREATE TABLE macro_observations (
                series_id TEXT, observation_date TEXT, value REAL, retrieved_at TEXT
            );
            """
        )
        for day, price in [
            ("2026-10-09", 300.0),
            ("2026-10-12", 303.0),
            ("2026-10-13", 306.0),
            ("2026-10-14", 305.0),
        ]:
            conn.execute(
                "INSERT INTO market_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("QQQM", day, price, price, 100000, "test", "2026-10-14T00:00:00Z"),
            )
        # 2026-10-13 14:00 New York local time (18:00 UTC) is available
        # for that session; 17:00 New York is assigned to the next session.
        conn.execute(
            "INSERT INTO news_items VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("n1", "Federal Reserve", "monetary_policy", "FOMC statement: inflation remains elevated",
             "Hawkish policy text", "https://example.test/1", "2026-10-13T18:00:00+00:00", "2026-10-14T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO news_items VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("n2", "Federal Reserve", "monetary_policy", "Rate cut decision",
             "Dovish decision", "https://example.test/2", "2026-10-13T21:00:00+00:00", "2026-10-14T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO macro_observations VALUES (?, ?, ?, ?)",
            ("DGS10", "2026-10-13", 4.2, "2026-10-14T00:00:00Z"),
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_builds_context_and_uses_adjusted_price(self) -> None:
        data = build_market_context_snapshot(self.db_path, "QQQM")
        self.assertEqual(data["as_of_date"], "2026-10-14")
        self.assertEqual(data["latest_price"], 305.0)
        self.assertEqual(data["market"]["sessions_available"], 4)
        self.assertFalse(data["macro"]["historical_backtest_eligible"])

    def test_after_close_news_is_assigned_to_next_session(self) -> None:
        data = build_market_context_snapshot(self.db_path, "QQQM", "2026-10-13")
        self.assertEqual(data["news"]["news_count_30_calendar_days"], 1)
        self.assertEqual(data["news"]["keyword_counts"]["inflation_keyword_count"], 1)
        self.assertEqual(data["news"]["keyword_counts"]["dovish_keyword_count"], 0)

    def test_snapshot_warns_against_training_with_sparse_news(self) -> None:
        data = build_market_context_snapshot(self.db_path, "QQQM")
        self.assertFalse(data["modeling_policy"]["news_features_allowed_in_training_by_default"])


if __name__ == "__main__":
    unittest.main()
