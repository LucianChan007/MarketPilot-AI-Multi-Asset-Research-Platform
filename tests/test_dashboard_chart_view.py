from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gui.components.chart import build_forecast_figure


SLOTS = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
    "11:00", "11:15", "11:30", "11:45", "12:00", "12:15",
    "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
    "14:00", "14:15", "14:30", "14:45", "15:00", "15:15",
    "15:30", "15:45",
]


class DashboardChartViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        report_dir = self.root / "reports" / "intraday"
        report_dir.mkdir(parents=True)
        curve = []
        path_by_time = {}
        path = []
        for i, slot in enumerate(SLOTS):
            base = 306.0 + 0.10 * __import__("math").sin(i / 3.2) + (i * 0.002)
            curve.append({"time_ny": slot, "low_price": base - 1.0, "median_price": base,
                          "high_price": base + 1.0, "historical_slot_samples": 50})
            predicted = 306.1 + 0.13 * __import__("math").sin(i / 4.0) - (i * 0.001)
            path_by_time[slot] = round(predicted, 4)
            path.append({"time_ny": slot, "predicted_price": round(predicted, 4)})
        forecast = {
            "ticker": "QQQM", "as_of_date": "2026-10-13", "forecast_date_weekday_estimate": "2026-10-14",
            "previous_session_close": 306.22,
            "estimated_open": {"p10_price": 303.0, "p50_price": 306.0, "p90_price": 309.0},
            "estimated_close_from_path": {"p10_price": 302.0, "p50_price": 306.0, "p90_price": 311.0},
            "curve": curve,
            "main_path": {
                "model_name": "main-path-ridge-market-v1", "model_family": "ridge_regression_per_time_slot",
                "independent_of_quantile_curves": True, "predicted_open_price": 306.1,
                "predicted_close_price": path[-1]["predicted_price"], "path": path,
                "price_by_time": path_by_time, "news_adjustment_applied": False,
            },
        }
        (report_dir / "QQQM_intraday_forecast_latest.json").write_text(json.dumps(forecast), encoding="utf-8")
        (report_dir / "QQQM_intraday_forecast_vs_actual_summary_latest.json").write_text(
            json.dumps({"target_date": "2026-10-14", "actual_points_found": 0}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_default_chart_shows_main_path_without_scenario_lines(self):
        fig, note = build_forecast_figure(self.root, show_scenarios=False)
        self.assertIsNotNone(fig)
        labels = [line.get_label() for line in fig.axes[0].lines]
        self.assertTrue(any("主预测路径" in label for label in labels))
        self.assertFalse(any("P50" in label or "P10" in label or "P90" in label for label in labels))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_scenario_lines_are_optional_and_explained(self):
        fig, note = build_forecast_figure(self.root, show_scenarios=True)
        labels = [line.get_label() for line in fig.axes[0].lines]
        self.assertTrue(any("P10" in label for label in labels))
        self.assertTrue(any("P50" in label for label in labels))
        self.assertTrue(any("P90" in label for label in labels))
        self.assertIn("不是整条曲线的概率", note)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
