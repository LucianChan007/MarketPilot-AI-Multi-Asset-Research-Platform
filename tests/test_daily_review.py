from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from app.services.daily_review import (
    build_daily_review_context,
    load_recent_human_feedback,
    save_human_feedback,
)


class DailyReviewTests(unittest.TestCase):
    def make_forecast(self, target_date="2026-10-15"):
        return {
            "as_of_date": "2026-10-14",
            "forecast_date_weekday_estimate": target_date,
            "generated_at_utc": "2026-10-14T20:01:00Z",
            "curve": [{"time_ny": f"{hour:02d}:{minute:02d}"} for hour, minute in [(9, 30), (9, 45), (10, 0), (15, 45)]],
            "main_path": {
                "model_name": "Ridge V2",
                "model_family": "ridge_regression_per_time_slot",
                "predicted_open_price": 305.0,
                "predicted_close_price": 307.0,
            },
        }

    def test_matching_summary_uses_actuals(self):
        forecast = self.make_forecast("2026-10-15")
        summary = {
            "target_date": "2026-10-15",
            "actual_points_found": 4,
            "actual_open": 304.5,
            "actual_close_near_last_bar": 307.6,
            "main_path_curve_mae_bps_vs_actual": 72.0,
            "p50_curve_mae_bps_vs_actual": 85.0,
        }
        context = build_daily_review_context(forecast, summary)
        self.assertTrue(context["summary_matches_target"])
        self.assertEqual(context["actual_points_found"], 4)
        self.assertAlmostEqual(context["open_error_usd_actual_minus_predicted"], -0.5)
        self.assertAlmostEqual(context["close_error_usd_actual_minus_predicted"], 0.6)
        self.assertAlmostEqual(context["main_path_mae_pct"], 0.72)

    def test_stale_summary_is_not_joined_to_new_forecast(self):
        forecast = self.make_forecast("2026-10-16")
        summary = {
            "target_date": "2026-10-15",
            "actual_points_found": 26,
            "actual_open": 304.5,
            "actual_close_near_last_bar": 307.6,
            "main_path_curve_mae_bps_vs_actual": 72.0,
        }
        context = build_daily_review_context(forecast, summary)
        self.assertFalse(context["summary_matches_target"])
        self.assertEqual(context["actual_points_found"], 0)
        self.assertIsNone(context["actual_open_price"])
        self.assertIsNotNone(context["previous_completed_summary"])
        self.assertEqual(context["previous_completed_summary"]["target_date"], "2026-10-15")

    def test_missing_summary_gives_waiting_status(self):
        context = build_daily_review_context(self.make_forecast(), None)
        self.assertFalse(context["summary_matches_target"])
        self.assertEqual(context["actual_points_found"], 0)
        self.assertIn("不会把旧交易日", context["explanation"])

    def test_no_actual_points_waits_even_when_dates_match(self):
        context = build_daily_review_context(self.make_forecast(), {
            "target_date": "2026-10-15", "actual_points_found": 0
        })
        self.assertEqual(context["status"], "尚未匹配到实际盘中价格")

    def test_save_feedback_appends_and_can_be_read_back(self):
        forecast = self.make_forecast()
        context = build_daily_review_context(forecast, {
            "target_date": "2026-10-15", "actual_points_found": 4,
            "actual_open": 304.5, "actual_close_near_last_bar": 307.6,
        })
        with tempfile.TemporaryDirectory() as temp:
            first_path = save_human_feedback(Path(temp), context, {
                "overall_match": "有些地方接近",
                "biggest_miss": "午后走势",
                "news_helpfulness": "帮助不明显",
                "notes": "午后反弹比预测明显。",
            })
            save_human_feedback(Path(temp), context, {
                "overall_match": "走势差异较大",
                "biggest_miss": "收盘方向或价格",
                "news_helpfulness": "暂时无法判断",
                "notes": "复盘第二条。",
            })
            with first_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["user_biggest_miss"], "午后走势")
            recent = load_recent_human_feedback(Path(temp), limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["user_notes"], "复盘第二条。")


if __name__ == "__main__":
    unittest.main()
