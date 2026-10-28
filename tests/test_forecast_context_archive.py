from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.forecast_context_archive import archive_forecast_context, get_context_archive_summary


TEST_CONTEXT = {
    "ticker": "QQQM",
    "as_of_date": "2026-10-13",
    "latest_price": 306.16,
    "market": {
        "latest_trade_date": "2026-10-13",
        "return_1_session_pct": -0.3353,
        "return_5_sessions_pct": -2.11,
        "return_20_sessions_pct": 5.65,
        "annualized_volatility_20_session_pct": 15.01,
        "price_vs_sma20_pct": 0.31,
        "sessions_available": 253,
    },
    "news": {
        "news_count_30_calendar_days": 5,
        "keyword_counts": {
            "hawkish_keyword_count": 0,
            "dovish_keyword_count": 0,
            "inflation_keyword_count": 0,
            "labor_keyword_count": 1,
            "growth_risk_keyword_count": 0,
        },
        "recent_items": [],
    },
    "macro": {"available": True, "historical_backtest_eligible": False, "items": []},
    "modeling_policy": {
        "macro_features_allowed_in_historical_backtest": False,
        "news_features_allowed_in_training_by_default": False,
    },
}


class ForecastContextArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_market_data.db"
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """CREATE TABLE forecast_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    UNIQUE(ticker, as_of_date)
                )"""
            )
            conn.execute(
                "INSERT INTO forecast_feedback (ticker, as_of_date, current_price, status) VALUES (?, ?, ?, ?)",
                ("QQQM", "2026-10-13", 306.16, "pending"),
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_archives_context_linked_to_forecast_and_marks_training_boundaries(self):
        result = archive_forecast_context(
            {"ticker": "QQQM", "as_of_date": "2026-10-13", "latest_adjusted_close": 306.16},
            db_path=self.db_path,
            capture_mode="live_prediction",
            context_snapshot=TEST_CONTEXT,
        )
        self.assertTrue(result["inserted"])
        self.assertEqual(result["forecast_feedback_id"], 1)

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT feature_vector_json, context_json, training_eligibility_json FROM forecast_context_snapshots WHERE id=?",
                (result["id"],),
            ).fetchone()
        finally:
            conn.close()

        features = json.loads(row[0])
        context = json.loads(row[1])
        eligibility = json.loads(row[2])
        self.assertEqual(features["market_return_1_session_pct"], -0.3353)
        self.assertEqual(features["news_count_30_calendar_days"], 5)
        self.assertEqual(context["macro"]["historical_backtest_eligible"], False)
        self.assertTrue(eligibility["market"]["eligible_for_candidate_training"])
        self.assertFalse(eligibility["news"]["eligible_for_candidate_training"])
        self.assertFalse(eligibility["macro"]["eligible_for_candidate_training"])

    def test_duplicate_date_keeps_original_context_immutable(self):
        payload = {"ticker": "QQQM", "as_of_date": "2026-10-13", "latest_adjusted_close": 306.16}
        first = archive_forecast_context(payload, db_path=self.db_path, context_snapshot=TEST_CONTEXT)
        changed = {**TEST_CONTEXT, "latest_price": 306.16, "market": {**TEST_CONTEXT["market"], "return_1_session_pct": 99.0}}
        second = archive_forecast_context(payload, db_path=self.db_path, context_snapshot=changed)
        self.assertTrue(first["inserted"])
        self.assertFalse(second["inserted"])
        self.assertEqual(first["context_sha256"], second["context_sha256"])
        self.assertEqual(get_context_archive_summary(self.db_path)["snapshots"], 1)

    def test_mismatched_context_date_is_rejected(self):
        wrong = {**TEST_CONTEXT, "as_of_date": "2026-10-12"}
        with self.assertRaises(ValueError):
            archive_forecast_context(
                {"ticker": "QQQM", "as_of_date": "2026-10-13", "latest_adjusted_close": 306.16},
                db_path=self.db_path,
                context_snapshot=wrong,
            )

    def test_missing_forecast_record_is_rejected(self):
        with self.assertRaises(RuntimeError):
            archive_forecast_context(
                {"ticker": "QQQ", "as_of_date": "2026-10-13", "latest_adjusted_close": 306.16},
                db_path=self.db_path,
                context_snapshot={**TEST_CONTEXT, "ticker": "QQQ"},
            )


if __name__ == "__main__":
    unittest.main()
