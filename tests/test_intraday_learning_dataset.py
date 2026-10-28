from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.build_intraday_learning_dataset import build_dataset

NY = ZoneInfo("America/New_York")
SLOTS = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
    "11:00", "11:15", "11:30", "11:45", "12:00", "12:15",
    "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
    "14:00", "14:15", "14:30", "14:45", "15:00", "15:15",
    "15:30", "15:45",
]


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE intraday_bars (
        symbol TEXT, feed TEXT, timeframe TEXT, trade_date_ny TEXT,
        session_time_ny TEXT, open REAL, close REAL
    )""")
    target_day = date(2026, 10, 14)
    for index, slot in enumerate(SLOTS):
        open_price = 100.0 + index * 0.01
        close_price = 100.0 + index * 0.025
        conn.execute(
            "INSERT INTO intraday_bars VALUES (?,?,?,?,?,?,?)",
            ("QQQM", "iex", "15Min", target_day.isoformat(), slot, open_price, close_price),
        )
    conn.commit()
    conn.close()


def _make_forecast(archive: Path, live: bool = True) -> Path:
    curve = []
    for index, slot in enumerate(SLOTS):
        median = 100.0 + index * 0.02
        curve.append({"time_ny": slot, "low_price": median - 0.4, "median_price": median, "high_price": median + 0.4})
    prices = {row["time_ny"]: 100.0 + i * 0.02 for i, row in enumerate(curve)}
    forecast = {
        "ticker": "QQQM",
        "feed": "iex",
        "timeframe": "15Min",
        "as_of_date": "2026-10-13",
        "forecast_date_weekday_estimate": "2026-10-14",
        "generated_at_utc": "2026-10-13T22:00:00Z",
        "previous_session_close": 100.0,
        "estimated_open": {"p10_price": 99.5, "p50_price": 100.0, "p90_price": 100.5},
        "estimated_close_from_path": {"p10_price": 100.0, "p50_price": 100.5, "p90_price": 101.0},
        "curve": curve,
        "main_path": {
            "selected_neighbor_date": "2026-09-25", "predicted_open_price": 100.0,
            "predicted_close_price": 100.5, "cluster_member_count": 3,
            "cluster_weight_share_not_calibrated_probability": 0.5,
            "price_by_time": prices,
        },
        "market_only_main_path": {
            "selected_neighbor_date": "2026-09-23", "predicted_open_price": 100.1,
            "predicted_close_price": 100.4, "cluster_member_count": 2,
            "cluster_weight_share_not_calibrated_probability": 0.35,
            "price_by_time": {k: v + 0.1 for k, v in prices.items()},
        },
        "path_selection_diagnostics": {"news_adjustment_applied": False, "news_signal": {"signal": 0.0}},
        "forecast_snapshot_version": "intraday-learning-snapshot-v1" if live else None,
        "capture_mode": "live_prediction" if live else "reconstructed",
        "market_state_features": {"last_close_return_pct": 0.1, "return_5_sessions_pct": 1.0},
        "news_context": {
            "available": True,
            "included_article_count": 2,
            "mean_impact_score": 0.2,
            "mean_relevance_score": 0.7,
            "mean_confidence": 0.6,
            "event_type_counts": {"monetary_policy": 1, "employment": 1},
            "nasdaq_bias_counts": {"neutral": 2},
            "items": [],
        },
    }
    name = "QQQM_intraday_forecast_2026-10-14.json" if live else "QQQM_intraday_forecast_2026-10-14_legacy.json"
    path = archive / name
    path.write_text(json.dumps(forecast), encoding="utf-8")
    return path


class IntradayDatasetTests(unittest.TestCase):
    def test_only_point_in_time_live_snapshot_becomes_training_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / "test.db"
            archive = root / "archive"
            reports = root / "reports" / "model_lab"
            archive.mkdir()
            _make_db(db)
            _make_forecast(archive, live=True)
            _make_forecast(archive, live=False)

            readiness = build_dataset(db, archive, reports)
            self.assertEqual(readiness["eligible_live_prediction_training_rows"], 1)
            self.assertEqual(readiness["evaluation_rows_including_legacy_forecasts"], 2)
            self.assertTrue((reports / "QQQM_intraday_path_training_dataset_latest.csv").exists())
            with (reports / "QQQM_intraday_path_training_dataset_latest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["png_is_model_input"], "False")
            self.assertIn("target_actual_return_1545_from_open_pct", rows[0])

            feedback_path = reports / "QQQM_intraday_human_feedback.csv"
            with feedback_path.open("r", encoding="utf-8-sig", newline="") as handle:
                feedback_rows = list(csv.DictReader(handle))
                fields = feedback_rows[0].keys()
                feedback_rows[0]["human_note"] = "预测拐点出现太早，下一版重点检查"
            with feedback_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fields))
                writer.writeheader()
                writer.writerows(feedback_rows)

            build_dataset(db, archive, reports)
            with feedback_path.open("r", encoding="utf-8-sig", newline="") as handle:
                preserved = list(csv.DictReader(handle))
            self.assertEqual(preserved[0]["human_note"], "预测拐点出现太早，下一版重点检查")


if __name__ == "__main__":
    unittest.main()
