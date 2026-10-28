from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.context_agent import build_context_agent_payload, build_local_analysis


class ContextAgentStructuredNewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "reports" / "context_features").mkdir(parents=True)
        (self.root / "reports" / "forecast").mkdir(parents=True)
        (self.root / "data").mkdir()
        context = {
            "ticker": "QQQM",
            "as_of_date": "2026-10-13",
            "latest_price": 306.16,
            "market": {"return_1_session_pct": -0.33, "return_5_sessions_pct": -2.1,
                       "return_20_sessions_pct": 5.65, "annualized_volatility_20_session_pct": 15.0},
            "news": {"news_count_30_calendar_days": 5, "recent_items": []},
            "macro": {"available": True},
            "modeling_policy": {"macro_features_allowed_in_historical_backtest": False,
                                "news_features_allowed_in_training_by_default": False},
        }
        forecast = {
            "ticker": "QQQM", "as_of_date": "2026-10-13", "latest_adjusted_close": 306.16,
            "generated_at_local": "2026-10-14T09:39:32+08:00",
            "point_forecasts": {"historical_mean_return_pct": 0.07, "ridge_full_return_pct": 0.06,
                                "ridge_returns_only_return_pct": 0.14},
            "historical_return_bands": {"last_60_sessions": {"p10_price_scenario": 302.8,
                                      "p50_price_scenario": 306.2, "p90_price_scenario": 310.4}},
        }
        (self.root / "reports" / "context_features" / "QQQM_context_snapshot_latest.json").write_text(
            json.dumps(context), encoding="utf-8"
        )
        (self.root / "reports" / "forecast" / "QQQM_return_forecast_latest_20261014_093932.json").write_text(
            json.dumps(forecast), encoding="utf-8"
        )
        conn = sqlite3.connect(self.root / "data" / "market_data.db")
        conn.execute("""CREATE TABLE forecast_feedback (
            id INTEGER PRIMARY KEY, ticker TEXT, as_of_date TEXT, status TEXT, actual_target_date TEXT,
            actual_return_pct REAL, error_zero_return_bps REAL, error_historical_mean_bps REAL,
            error_ridge_full_bps REAL, error_ridge_returns_only_bps REAL,
            direction_correct_zero_return INTEGER, direction_correct_historical_mean INTEGER,
            direction_correct_ridge_full INTEGER, direction_correct_ridge_returns_only INTEGER,
            interval_20_covered INTEGER, interval_60_covered INTEGER, interval_252_covered INTEGER
        )""")
        conn.execute("""INSERT INTO forecast_feedback (
            id,ticker,as_of_date,status,actual_target_date,actual_return_pct,
            error_zero_return_bps,error_historical_mean_bps,error_ridge_full_bps,error_ridge_returns_only_bps,
            direction_correct_zero_return,direction_correct_historical_mean,direction_correct_ridge_full,
            direction_correct_ridge_returns_only,interval_20_covered,interval_60_covered,interval_252_covered
        ) VALUES (1,'QQQM','2026-10-12','evaluated','2026-10-13',-0.3353,33.53,40.80,50.10,47.65,1,0,0,0,1,1,1)""")

        conn.execute("""CREATE TABLE news_items (
            news_id TEXT PRIMARY KEY, title TEXT, url TEXT
        )""")
        conn.execute("""CREATE TABLE news_event_features (
            news_id TEXT NOT NULL, feature_version TEXT NOT NULL, source_id TEXT, source_name TEXT,
            published_at TEXT, classified_at_utc TEXT NOT NULL, classifier_model TEXT NOT NULL,
            event_type TEXT NOT NULL, nasdaq_bias TEXT NOT NULL, impact_score REAL,
            relevance_score REAL, confidence REAL, time_horizon TEXT NOT NULL,
            facts_summary TEXT NOT NULL, evidence_json TEXT NOT NULL, raw_classification_json TEXT NOT NULL,
            PRIMARY KEY(news_id, feature_version)
        )""")
        now = datetime.now(timezone.utc)
        published = (now - timedelta(hours=5)).isoformat(timespec="seconds")
        classified = (now - timedelta(hours=2)).isoformat(timespec="seconds")
        future_published = (now + timedelta(days=1)).isoformat(timespec="seconds")
        conn.execute("INSERT INTO news_items VALUES (?,?,?)", ("known-1", "Fed statement published", "https://example.test/known"))
        conn.execute("INSERT INTO news_items VALUES (?,?,?)", ("future-1", "Future event", "https://example.test/future"))
        conn.execute("""INSERT INTO news_event_features VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("known-1", "news-event-v1", "fed", "Federal Reserve", published, classified,
             "unit-test", "monetary_policy", "neutral", 0.0, 0.8, 0.8, "1_5_sessions",
             "The statement indicates no immediate rate change.", "[]", "{}"))
        conn.execute("""INSERT INTO news_event_features VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("future-1", "news-event-v1", "fed", "Federal Reserve", future_published, classified,
             "unit-test", "monetary_policy", "bullish", 1.0, 0.9, 0.9, "1_5_sessions",
             "This must be excluded because it has not yet been published.", "[]", "{}"))
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_payload_includes_only_news_published_and_classified_by_analysis_cutoff(self) -> None:
        payload = build_context_agent_payload(self.root)
        news = payload["structured_news_context_current"]
        self.assertEqual(news["included_article_count"], 1)
        self.assertEqual(news["items"][0]["news_id"], "known-1")
        self.assertGreaterEqual(news["excluded_counts"]["published_after_cutoff"], 1)

    def test_local_report_mentions_structured_news_and_temporal_caveat(self) -> None:
        payload = build_context_agent_payload(self.root)
        report = build_local_analysis(payload)
        self.assertIn("已结构化新闻事件", report)
        self.assertIn("Fed statement published", report)
        self.assertIn("不得声称它们影响了那次旧预测", report)


if __name__ == "__main__":
    unittest.main()
