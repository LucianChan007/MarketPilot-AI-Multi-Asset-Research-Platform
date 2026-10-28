from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, time
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from scripts import generate_video_script as video

NY = ZoneInfo("America/New_York")


class VideoForecastArchiveV2Tests(unittest.TestCase):
    def test_review_chart_points_use_the_same_ridge_main_path_as_forecast_chart(self):
        payload = {
            "curve": [{"time_ny": "09:30", "median_price": 100.0, "low_price": 99.0, "high_price": 101.0}],
            "main_path": {"model_family": "ridge_intraday_v1", "price_by_time": {"09:30": 102.25}},
        }
        points = video._forecast_points_for_plot(payload)
        self.assertAlmostEqual(points["09:30"]["main_path"], 102.25)
        self.assertAlmostEqual(points["09:30"]["median"], 102.25)

    def test_previous_video_folder_forecast_json_is_reused_for_matching_target_day(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "intraday_archive"
            out = root / "video_scripts"
            (out / "forecast_archive").mkdir(parents=True)
            target = "2026-10-21"
            created = datetime.combine(date.fromisoformat(target), time(8, 0), tzinfo=NY).isoformat()
            curve = [{"time_ny": slot, "median_price": 100 + i * 0.01, "low_price": 99, "high_price": 101} for i, slot in enumerate(video.SLOTS)]
            payload = {
                "ticker": "QQQM", "as_of_date": "2026-10-20", "forecast_date_weekday_estimate": target,
                "generated_at_utc": created, "forecast_snapshot_version": 1, "pre_session_snapshot": True,
                "curve": curve,
            }
            path = out / "forecast_archive" / f"QQQM_forecast_asof_2026-10-20_target_{target}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(video, "ARCHIVE", archive), patch.object(video, "OUTROOT", out):
                result = video._archive_forecast(target)
            self.assertIsNotNone(result)
            self.assertEqual(result["_archive_source_type"], "video_forecast_archive")
            self.assertEqual(result["forecast_date_weekday_estimate"], target)
            self.assertEqual(Path(result["_archive_source_path"]), path.resolve())


if __name__ == "__main__":
    unittest.main()
