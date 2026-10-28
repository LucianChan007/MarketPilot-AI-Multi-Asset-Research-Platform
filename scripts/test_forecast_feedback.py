from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.forecast_feedback import (
    evaluate_pending_forecasts,
    get_feedback_summary,
    list_forecasts,
    record_forecast_payload,
)


class ForecastFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_market_data.db"
        self.payload = {
            "as_of_date": "2026-10-12",
            "latest_adjusted_close": 307.19,
            "point_forecasts": {
                "zero_return_baseline_pct": 0.0,
                "historical_mean_return_pct": 0.1,
                "ridge_full_return_pct": 0.2,
                "ridge_returns_only_return_pct": 0.15,
            },
            "historical_return_bands": {
                "last_20_sessions": {"p10_return_pct": -1.0, "p50_return_pct": 0.2, "p90_return_pct": 1.0},
                "last_60_sessions": {"p10_return_pct": -1.2, "p50_return_pct": 0.1, "p90_return_pct": 1.4},
                "last_252_sessions": {"p10_return_pct": -1.5, "p50_return_pct": 0.1, "p90_return_pct": 1.6},
            },
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_is_idempotent_per_ticker_and_as_of_date(self) -> None:
        first = record_forecast_payload(self.payload, db_path=self.db_path, source_file="first.json")
        second_payload = dict(self.payload)
        second_payload["latest_adjusted_close"] = 999
        second = record_forecast_payload(second_payload, db_path=self.db_path, source_file="second.json")
        self.assertTrue(first["inserted"])
        self.assertFalse(second["inserted"])
        rows = list_forecasts(db_path=self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["current_price"], 307.19)
        self.assertEqual(rows[0]["source_file"], "first.json")

    def test_evaluates_first_later_trading_date_and_summarizes(self) -> None:
        record_forecast_payload(self.payload, db_path=self.db_path)
        history = [
            {"date": "2026-10-12", "adjusted_close": 307.19},
            # The evaluator naturally skips non-trading days because only actual rows are supplied.
            {"date": "2026-10-14", "adjusted_close": 310.2619},
        ]
        evaluated = evaluate_pending_forecasts(history, db_path=self.db_path)
        self.assertEqual(len(evaluated), 1)
        self.assertEqual(evaluated[0]["target_date"], "2026-10-14")
        self.assertAlmostEqual(evaluated[0]["actual_return_pct"], 1.0, places=5)
        summary = get_feedback_summary(db_path=self.db_path)
        self.assertEqual(summary["evaluated_forecasts"], 1)
        self.assertEqual(summary["models"]["ridge_full"]["n"], 1)
        rows = list_forecasts(db_path=self.db_path)
        self.assertEqual(rows[0]["status"], "evaluated")

    def test_does_not_evaluate_without_later_actual_price(self) -> None:
        record_forecast_payload(self.payload, db_path=self.db_path)
        evaluated = evaluate_pending_forecasts(
            [{"date": "2026-10-12", "adjusted_close": 307.19}],
            db_path=self.db_path,
        )
        self.assertEqual(evaluated, [])
        self.assertEqual(list_forecasts(db_path=self.db_path)[0]["status"], "pending")

    def test_invalid_price_is_rejected(self) -> None:
        broken = dict(self.payload)
        broken["latest_adjusted_close"] = 0
        with self.assertRaises(ValueError):
            record_forecast_payload(broken, db_path=self.db_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
