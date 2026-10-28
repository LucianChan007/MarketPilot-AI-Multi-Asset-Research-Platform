import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.plot_intraday_forecast import create_chart, _interpolate_short_visual_gaps


class PlotIntradayForecastTests(unittest.TestCase):
    def test_short_gaps_are_interpolated_for_display_only(self):
        raw = [100.0, None, 104.0, None, None, 110.0, None, None, None, None, 120.0]
        visual, interpolated = _interpolate_short_visual_gaps(raw, max_gap_points=2)
        self.assertEqual(visual[1], 102.0)
        self.assertEqual(visual[3], 106.0)
        self.assertEqual(visual[4], 108.0)
        # Four consecutive missing slots are too long to bridge, and edge gaps lack anchors.
        self.assertIsNone(visual[6])
        self.assertIsNone(visual[9])
        self.assertEqual(interpolated, [1, 3, 4])
        self.assertEqual(raw, [100.0, None, 104.0, None, None, 110.0, None, None, None, None, 120.0])

    def test_generates_chart_and_representative_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports").mkdir()
            (root / "data").mkdir()
            db = root / "data" / "market_data.db"
            con = sqlite3.connect(db)
            con.execute(
                """CREATE TABLE intraday_bars (
                    symbol TEXT, timestamp_utc TEXT, timeframe TEXT, feed TEXT,
                    open REAL, close REAL
                )"""
            )
            # Two completed New York sessions with 26 15-minute bars each.
            for day, base in (("2026-10-12", 300.0), ("2026-10-13", 301.0)):
                start = datetime.fromisoformat(day + "T13:30:00+00:00")
                for idx in range(26):
                    stamp = start + timedelta(minutes=15 * idx)
                    price = base * (1 + idx * 0.0001)
                    con.execute(
                        "INSERT INTO intraday_bars VALUES (?,?,?,?,?,?)",
                        ("QQQM", stamp.isoformat(), "15Min", "iex", price, price * (1 + 0.00005)),
                    )
            con.commit()
            con.close()

            curve = []
            for idx in range(26):
                minute = 30 if idx == 0 else (idx * 15 + 30) % 60
                hour = 9 + (idx * 15 + 0) // 60
                # Use a direct slot table, matching the normal session.
                all_times = [
                    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
                    "11:00", "11:15", "11:30", "11:45", "12:00", "12:15",
                    "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
                    "14:00", "14:15", "14:30", "14:45", "15:00", "15:15",
                    "15:30", "15:45",
                ]
                slot = all_times[idx]
                median = 302.0 * (1 + idx * 0.0001)
                curve.append({
                    "time_ny": slot,
                    "p10_return_from_open_pct": -0.2,
                    "p50_return_from_open_pct": idx * 0.01,
                    "p90_return_from_open_pct": 0.2,
                    "low_price": median - 1.0,
                    "median_price": median,
                    "high_price": median + 1.0,
                    "historical_slot_samples": 2,
                })
            forecast = {
                "ticker": "QQQM", "as_of_date": "2026-10-13",
                "forecast_date_weekday_estimate": "2026-10-14", "previous_session_close": 301.5,
                "estimated_open": {"p10_price": 300.0, "p50_price": 302.0, "p90_price": 304.0},
                "estimated_close_from_path": {"p10_price": 300.0, "p50_price": 302.2, "p90_price": 305.0},
                "curve": curve,
                # Deliberately omit one 15-minute slot to verify plotting-only interpolation.
                "main_path": {
                    "selected_neighbor_date": "2026-10-13",
                    "selection_method": "test_medoid",
                    "candidate_count": 2,
                    "cluster_member_count": 1,
                    "cluster_weight_share_not_calibrated_probability": 0.5,
                    "predicted_open_price": 302.0,
                    "predicted_close_price": 302.2,
                    "price_by_time": {
                        point["time_ny"]: point["median_price"] + 0.2
                        for idx, point in enumerate(curve) if idx != 7
                    },
                },
            }
            forecast_path = root / "reports" / "forecast.json"
            forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
            summary = create_chart(forecast_path, db, root / "reports")
            self.assertTrue(Path(summary["chart_latest"]).exists())
            self.assertTrue(Path(summary["comparison_csv"]).exists())
            self.assertIsNotNone(summary["representative_historical_session"])
            self.assertEqual(summary["target_date"], "2026-10-14")
            self.assertEqual(summary["main_path_visual_interpolated_count"], 1)
            self.assertFalse(summary["main_path_visual_interpolation_is_training_data"])
            rows = list(__import__("csv").DictReader(Path(summary["comparison_csv"]).open(encoding="utf-8-sig")))
            gap_row = next(row for row in rows if row["time_ny"] == curve[7]["time_ny"])
            self.assertEqual(gap_row["main_path_price"], "")
            self.assertNotEqual(gap_row["main_path_visual_price"], "")
            self.assertEqual(gap_row["main_path_visual_interpolated"], "yes")


if __name__ == "__main__":
    unittest.main()
