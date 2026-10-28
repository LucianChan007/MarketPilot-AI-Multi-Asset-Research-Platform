from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import run_public_intraday_learning_cycle as cycle


class PublicIntradayLearningCycleTests(unittest.TestCase):
    def test_public_cycle_trains_candidate_and_records_no_auto_promotion(self):
        expected = "2026-10-21"
        eligible_sessions = [
            {"trade_date": f"2026-09-{(i % 28) + 1:02d}", "path_eligible": True}
            for i in range(25)
        ]
        eligible_sessions[-1]["trade_date"] = expected

        def fake_run_prototype(**kwargs):
            target = kwargs["forecast_target_date"]
            return {
                "forecast_json": "candidate.json",
                "archive_json": "archive.json",
                "backtest_summary": {
                    "ridge_ml_path_test_sessions": 5,
                    "ridge_ml_path_mae_bps": 89.2,
                    "ridge_ml_improvement_pct_vs_historical_profile": 5.1,
                },
                "forecast": {
                    "as_of_date": expected,
                    "forecast_date_weekday_estimate": target,
                    "main_path": {
                        "model_family": "ridge_regression_per_time_slot",
                        "price_by_time": {
                            f"{9 + (30 + 15 * i) // 60:02d}:{(30 + 15 * i) % 60:02d}": 300.0 + i * 0.01
                            for i in range(20)
                        },
                    },
                    "capture_mode": "live_prediction",
                    "pre_session_snapshot": True,
                },
            }

        with (
            patch.object(cycle, "refresh_market_data", return_value={
                "daily_latest_date": expected, "intraday_latest_date": expected,
                "daily_provider": "公开免密行情", "intraday_provider": "雅虎财经公开接口",
                "errors": [], "attempt_notes": [],
            }),
            patch.object(cycle, "_latest_daily_date", return_value=expected),
            patch("app.news_data.sync_news", return_value={"failed": False}),
            patch.object(cycle, "expected_last_completed_session", return_value=expected),
            patch.object(cycle, "choose_intraday_feed", return_value=("yahoo", eligible_sessions)),
            patch.object(cycle, "build_dataset", side_effect=[{"settled": 1}, {"settled": 2}]),
            patch.object(cycle, "run_intraday_prototype", side_effect=fake_run_prototype),
        ):
            status = cycle._run()

        self.assertTrue(status["overall_success"])
        self.assertTrue(status["model_training_performed"])
        self.assertFalse(status["model_promotion_performed"])
        self.assertEqual(status["feed"], "yahoo")
        self.assertEqual(status["forecast_as_of_date"], expected)
        self.assertEqual(status["latest_complete_intraday_date"], expected)
        self.assertEqual(status["completed_training_sessions"], 25)
        self.assertIsNotNone(status["walk_forward_backtest"])


if __name__ == "__main__":
    unittest.main()
