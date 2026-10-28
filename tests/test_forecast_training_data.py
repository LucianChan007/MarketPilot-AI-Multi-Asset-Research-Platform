from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.forecast_training_data import FEATURE_COLUMNS, build_training_dataset, export_training_dataset


class ForecastTrainingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.db"
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE forecast_feedback (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                as_of_date TEXT,
                status TEXT,
                current_price REAL,
                actual_target_date TEXT,
                actual_adjusted_close REAL,
                actual_return_pct REAL
            );
            CREATE TABLE forecast_context_snapshots (
                id INTEGER PRIMARY KEY,
                forecast_feedback_id INTEGER,
                ticker TEXT,
                as_of_date TEXT,
                capture_mode TEXT,
                context_schema_version TEXT,
                feature_vector_json TEXT,
                training_eligibility_json TEXT
            );
            """
        )
        good_vector = {
            "market_return_1_session_pct": -0.3,
            "market_return_5_sessions_pct": 0.7,
            "market_return_20_sessions_pct": 2.1,
            "market_annualized_volatility_20_session_pct": 15.0,
            "market_price_vs_sma20_pct": 0.2,
            "news_hawkish_keyword_count": 1,
        }
        live_eligibility = {
            "market": {"eligible_for_candidate_training": True, "point_in_time_captured": True},
            "news": {"eligible_for_candidate_training": False},
            "macro": {"eligible_for_candidate_training": False},
        }
        reconstructed_eligibility = {
            "market": {"eligible_for_candidate_training": True, "point_in_time_captured": False},
            "news": {"eligible_for_candidate_training": False},
            "macro": {"eligible_for_candidate_training": False},
        }
        rows = [
            (1, "QQQM", "2026-10-13", "evaluated", 306.16, "2026-10-14", 307.0, 0.274),
            (2, "QQQM", "2026-10-14", "pending", 307.0, None, None, None),
            (3, "QQQM", "2026-10-12", "evaluated", 307.19, "2026-10-13", 306.16, -0.335),
        ]
        conn.executemany(
            "INSERT INTO forecast_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        snapshots = [
            (1, 1, "QQQM", "2026-10-13", "live_prediction", "v1", json.dumps(good_vector), json.dumps(live_eligibility)),
            (2, 2, "QQQM", "2026-10-14", "live_prediction", "v1", json.dumps(good_vector), json.dumps(live_eligibility)),
            (3, 3, "QQQM", "2026-10-12", "reconstructed", "v1", json.dumps(good_vector), json.dumps(reconstructed_eligibility)),
        ]
        conn.executemany(
            "INSERT INTO forecast_context_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)", snapshots
        )
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_only_evaluated_live_point_in_time_market_features_are_eligible(self) -> None:
        rows, report = build_training_dataset(self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["eligible_training_rows"], 1)
        self.assertEqual(report["exclusion_counts"]["reconstructed_snapshot"], 1)
        self.assertEqual(report["exclusion_counts"]["not_evaluated_yet"], 1)
        self.assertEqual(set(FEATURE_COLUMNS).issubset(rows[0]), True)
        self.assertNotIn("news_hawkish_keyword_count", rows[0])
        self.assertEqual(rows[0]["target_date"], "2026-10-14")

    def test_export_writes_header_and_readiness_even_with_small_sample(self) -> None:
        out = Path(self.temp.name) / "output"
        report = export_training_dataset(self.db_path, out)
        self.assertTrue(Path(report["dataset_csv"]).exists())
        self.assertTrue(Path(report["readiness_json"]).exists())
        self.assertEqual(report["readiness"], "collecting_only")
        self.assertFalse(report["promotion_enabled"])


if __name__ == "__main__":
    unittest.main()
