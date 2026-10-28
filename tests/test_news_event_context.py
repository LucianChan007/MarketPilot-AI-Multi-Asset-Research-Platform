from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.news_event_context import build_point_in_time_news_context


class NewsEventContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE news_items (
                news_id TEXT PRIMARY KEY, title TEXT, url TEXT
            );
            CREATE TABLE news_event_features (
                news_id TEXT, feature_version TEXT, source_name TEXT,
                published_at TEXT, classified_at_utc TEXT, classifier_model TEXT,
                event_type TEXT, nasdaq_bias TEXT, impact_score REAL,
                relevance_score REAL, confidence REAL, time_horizon TEXT,
                facts_summary TEXT, PRIMARY KEY (news_id, feature_version)
            );
            """
        )
        news_rows = [
            ("eligible", "Published and classified in time", "https://example.com/1"),
            ("late_classification", "Classification completed too late", "https://example.com/2"),
            ("future_publish", "Not published yet", "https://example.com/3"),
            ("old", "Outside lookback window", "https://example.com/4"),
        ]
        conn.executemany("INSERT INTO news_items VALUES (?, ?, ?)", news_rows)
        feature_rows = [
            ("eligible", "news-event-v1", "Fed", "2026-10-13T16:00:00+00:00", "2026-10-13T16:05:00+00:00", "deepseek-flash", "monetary_policy", "bearish", -1.0, 0.9, 0.8, "1_5_sessions", "Policy statement",),
            ("late_classification", "news-event-v1", "Fed", "2026-10-13T16:00:00+00:00", "2026-10-14T12:00:00+00:00", "deepseek-flash", "monetary_policy", "bullish", 1.0, 0.9, 0.8, "1_5_sessions", "Too late",),
            ("future_publish", "news-event-v1", "BLS", "2026-10-14T13:00:00+00:00", "2026-10-14T13:01:00+00:00", "deepseek-flash", "employment", "neutral", 0.0, 0.9, 0.8, "intraday", "Future publish",),
            ("old", "news-event-v1", "Fed", "2026-08-01T16:00:00+00:00", "2026-08-01T16:05:00+00:00", "deepseek-flash", "monetary_policy", "neutral", 0.0, 0.9, 0.8, "1_5_sessions", "Too old",),
        ]
        conn.executemany(
            "INSERT INTO news_event_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            feature_rows,
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_only_published_and_classified_before_cutoff_are_included(self) -> None:
        result = build_point_in_time_news_context(
            self.db_path,
            forecast_cutoff="2026-10-14T10:00:00+08:00",
            window_days=30,
        )
        self.assertEqual(result["included_article_count"], 1)
        self.assertEqual(result["nasdaq_bias_counts"], {"bearish": 1})
        self.assertEqual(result["mean_impact_score_high_confidence_relevant"], -1.0)
        self.assertEqual(result["excluded_counts"]["classified_after_cutoff"], 1)
        self.assertEqual(result["excluded_counts"]["published_after_cutoff"], 1)
        self.assertEqual(result["excluded_counts"]["outside_window"], 1)
        self.assertFalse(result["eligible_for_historical_training"])

    def test_missing_cutoff_fails_closed(self) -> None:
        result = build_point_in_time_news_context(self.db_path, forecast_cutoff=None)
        self.assertFalse(result["available"])
        self.assertEqual(result["included_article_count"], 0)
        self.assertIsNotNone(result.get("reason"))

    def test_missing_feature_table_is_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "empty.db"
            sqlite3.connect(db).close()
            result = build_point_in_time_news_context(
                db, forecast_cutoff="2026-10-14T10:00:00+08:00"
            )
            self.assertFalse(result["available"])
            self.assertEqual(result["included_article_count"], 0)


if __name__ == "__main__":
    unittest.main()
