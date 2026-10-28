from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.generate_video_script import (
    _chinese_news_context,
    _next_target_day,
    _valid_intraday_sessions,
    build_profile_forecast,
    plot_next,
    plot_review,
    read_json,
    _archive_forecast,
    translate_news,
    _make_script,
)
from scripts.video_market_data import _payload_to_frame, fetch_eodhd_intraday

NY = ZoneInfo("America/New_York")


def sample_intraday_frame(day_count: int = 12) -> pd.DataFrame:
    rows = []
    start = date(2026, 9, 1)
    days = []
    current = start
    while len(days) < day_count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    for day_idx, trading_day in enumerate(days):
        opening = 300.0 + day_idx * 0.55
        for slot_idx in range(26):
            time_value = (datetime.combine(trading_day, time(9, 30)) + timedelta(minutes=15 * slot_idx)).replace(tzinfo=NY)
            price = opening * (1 + (0.05 * slot_idx / 100) + (0.08 * ((slot_idx % 6) - 2) / 100))
            rows.append({
                "timestamp_utc": time_value.astimezone(ZoneInfo("UTC")),
                "timestamp_ny": time_value,
                "trade_date_ny": trading_day.isoformat(),
                "session_time_ny": time_value.strftime("%H:%M"),
                "open": price - 0.03, "high": price + 0.12, "low": price - 0.12,
                "close": price, "volume": 1000,
                "feed": "yahoo", "timeframe": "15Min",
            })
    return pd.DataFrame(rows)


class VideoScriptStudioTests(unittest.TestCase):

    def test_next_target_day_before_us_open_includes_todays_session(self):
        # 08:00 ET on Oct 21, after Oct 20 data is complete: target Oct 21, not Oct 22.
        now = datetime(2026, 10, 21, 8, 0, tzinfo=NY)
        self.assertEqual(_next_target_day("2026-10-20", now), "2026-10-21")

    def test_next_target_day_after_open_skips_current_partial_session(self):
        now = datetime(2026, 10, 21, 10, 0, tzinfo=NY)
        self.assertEqual(_next_target_day("2026-10-20", now), "2026-10-22")

    def test_next_target_day_after_close_uses_following_session(self):
        now = datetime(2026, 10, 21, 17, 0, tzinfo=NY)
        self.assertEqual(_next_target_day("2026-10-21", now), "2026-10-22")

    def test_next_target_day_when_next_session_has_opened_targets_next_full_session(self):
        now = datetime(2026, 10, 22, 10, 0, tzinfo=NY)
        self.assertEqual(_next_target_day("2026-10-21", now), "2026-10-23")
    def test_full_sessions_exclude_incomplete_current_day(self):
        frame = sample_intraday_frame(7)
        day = frame["trade_date_ny"].max()
        current = datetime.fromisoformat(day + "T12:00:00-04:00")
        sessions = _valid_intraday_sessions(frame, now_ny=current)
        self.assertEqual(len(sessions), 6)
        self.assertEqual(sessions[-1]["date"], sorted(set(frame["trade_date_ny"]))[-2])

    def test_profile_forecast_uses_prior_sessions_and_has_curve(self):
        frame = sample_intraday_frame(12)
        now = datetime(2026, 10, 1, 18, 0, tzinfo=NY)
        sessions = _valid_intraday_sessions(frame, now_ny=now)
        target = (date.fromisoformat(sessions[-1]["date"]) + timedelta(days=1)).isoformat()
        forecast = build_profile_forecast(sessions, target, lookback=10)
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast["feed"], "yahoo")
        self.assertGreaterEqual(len(forecast["curve"]), 18)
        self.assertEqual(forecast["forecast_date_weekday_estimate"], target)
        self.assertFalse(forecast["news_used_as_direct_price_signal"])

    def test_chinese_news_translation_does_not_copy_english_headline(self):
        item = translate_news({
            "title": "Federal Reserve issues FOMC statement",
            "summary": "The Federal Open Market Committee issued a statement.",
            "source": "Federal Reserve / Monetary Policy",
            "category": "monetary_policy",
            "published_at": "2026-10-13T18:00:00+00:00",
        })
        joined = " ".join(item.values())
        self.assertIn("美联储", joined)
        self.assertNotIn("Federal Reserve", joined)
        self.assertNotIn("Monetary Policy", joined)

    def test_chart_images_are_generated_from_intraday_paths(self):
        frame = sample_intraday_frame(12)
        sessions = _valid_intraday_sessions(frame, now_ny=datetime(2026, 10, 1, 18, 0, tzinfo=NY))
        review_session = sessions[-1]
        review_reference = build_profile_forecast(sessions, review_session["date"], lookback=10)
        next_forecast = build_profile_forecast(sessions, "2026-10-01", lookback=10)
        # The next target in this fixture predates all data, so make a valid target after the final session.
        next_target = (date.fromisoformat(sessions[-1]["date"]) + timedelta(days=1)).isoformat()
        next_forecast = build_profile_forecast(sessions, next_target, lookback=10)
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.png"
            second = Path(tmp) / "second.png"
            review = plot_review(review_session, review_reference, first, "模拟公开行情源")
            available = plot_next(next_forecast, second, "新闻仅用于中文背景说明")
            self.assertTrue(first.exists() and first.stat().st_size > 10000)
            self.assertTrue(second.exists() and second.stat().st_size > 10000)
            self.assertEqual(review["review_method"], "historical_backcast")
            self.assertTrue(available)

    def test_published_2026_10_20_chart_is_used_as_image_restored_review_source(self):
        # Synthetic bars are test fixtures only. The production workflow still requires
        # actual refreshed public 15-minute bars before it draws the real-price line.
        day = date(2026, 10, 20)
        rows = []
        for idx in range(26):
            stamp = (datetime.combine(day, time(9, 30)) + timedelta(minutes=15 * idx)).replace(tzinfo=NY)
            price = 294.5 + 0.02 * idx + 0.15 * ((idx % 5) - 2)
            rows.append({
                "timestamp_utc": stamp.astimezone(ZoneInfo("UTC")),
                "timestamp_ny": stamp, "trade_date_ny": day.isoformat(),
                "session_time_ny": stamp.strftime("%H:%M"),
                "open": price - 0.03, "high": price + 0.12,
                "low": price - 0.12, "close": price, "volume": 1000,
                "feed": "synthetic_test_fixture", "timeframe": "15Min",
            })
        session = _valid_intraday_sessions(pd.DataFrame(rows), now_ny=datetime(2026, 10, 21, 18, 0, tzinfo=NY))[-1]
        archived = _archive_forecast("2026-10-20")
        self.assertIsNotNone(archived)
        self.assertEqual(archived["source_kind"], "published_chart_digitized")
        self.assertEqual(archived["estimated_open"]["p50_price"], 295.16)
        self.assertEqual(archived["estimated_close_from_path"]["p50_price"], 294.55)
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "review.png"
            review = plot_review(session, archived, img, "测试用合成数据（仅测试）")
            self.assertTrue(img.exists() and img.stat().st_size > 10000)
            self.assertEqual(review["review_method"], "published_chart_digitized")
            self.assertIsNotNone(review["mae_usd"])

    def test_public_chart_payload_parser_keeps_ny_session_time(self):
        import calendar
        from datetime import timezone

        stamps = []
        for hh, mm in ((13, 30), (13, 45)):
            stamps.append(int(datetime(2026, 10, 20, hh, mm, tzinfo=timezone.utc).timestamp()))
        payload = {
            "timestamp": stamps,
            "indicators": {"quote": [{
                "open": [300.0, 300.2], "high": [300.3, 300.4],
                "low": [299.9, 300.1], "close": [300.2, 300.3], "volume": [100, 120],
            }]},
        }
        frame = _payload_to_frame(payload, daily=False)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["trade_date_ny"], "2026-10-20")
        self.assertEqual(frame.iloc[0]["session_time_ny"], "09:30")
        self.assertAlmostEqual(float(frame.iloc[-1]["close"]), 300.3)

    def test_eodhd_five_minute_bars_aggregate_to_real_fifteen_minute_path(self):
        class FakeResponse:
            status_code = 200
            def json(self):
                return [
                    {"datetime": "2026-10-20 09:30:00", "open": 300.0, "high": 300.4, "low": 299.9, "close": 300.2, "volume": 10},
                    {"datetime": "2026-10-20 09:35:00", "open": 300.2, "high": 300.5, "low": 300.1, "close": 300.3, "volume": 12},
                    {"datetime": "2026-10-20 09:40:00", "open": 300.3, "high": 300.7, "low": 300.2, "close": 300.6, "volume": 13},
                    {"datetime": "2026-10-20 09:45:00", "open": 300.6, "high": 300.8, "low": 300.4, "close": 300.7, "volume": 14},
                    {"datetime": "2026-10-20 09:50:00", "open": 300.7, "high": 300.9, "low": 300.5, "close": 300.8, "volume": 15},
                    {"datetime": "2026-10-20 09:55:00", "open": 300.8, "high": 301.0, "low": 300.7, "close": 300.9, "volume": 16},
                ]
        with patch.dict("os.environ", {"EODHD_API_KEY": "test-token"}), patch(
            "scripts.video_market_data.requests.get", return_value=FakeResponse()
        ) as request:
            bars = fetch_eodhd_intraday(period_days=3)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars.iloc[0]["session_time_ny"], "09:30")
        self.assertAlmostEqual(float(bars.iloc[0]["open"]), 300.0)
        self.assertAlmostEqual(float(bars.iloc[0]["high"]), 300.7)
        self.assertAlmostEqual(float(bars.iloc[0]["close"]), 300.6)
        self.assertEqual(int(bars.iloc[0]["volume"]), 35)
        self.assertEqual(bars.iloc[0]["feed"], "eodhd")
        # The token is sent as a request parameter, but no source code prints it.
        self.assertTrue(request.called)

    def test_voiceover_is_short_and_focuses_on_two_charts(self):
        review = {
            "review_date": "2026-10-20", "review_method": "actual_only",
            "actual_close": 295.78, "reference_close": None, "mae_usd": None,
            "day_change_pct": -0.2,
        }
        md, txt, meta = _make_script(
            "2026-10-21", review, None, False, [],
            "目前没有可确认发布时间的新闻记录，因此本期不编造新闻依据。",
            "2026-10-19", "测试数据源", [],
        )
        self.assertLess(len("".join(txt.split())), 320)
        self.assertIn("第一张图", txt)
        self.assertIn("第二张图", txt)
        self.assertNotIn("模型用最近完整交易日", txt)
        self.assertIn("视频标题", md)
        self.assertFalse(meta["next_forecast_available"])

    def test_voiceover_uses_ridge_main_path_price_instead_of_historical_median(self):
        frame = sample_intraday_frame(12)
        sessions = _valid_intraday_sessions(frame, now_ny=datetime(2026, 10, 1, 18, 0, tzinfo=NY))
        target = "2026-10-02"
        curve = []
        prices = {}
        for idx in range(26):
            slot = (datetime.combine(date(2026, 10, 2), time(9, 30)) + timedelta(minutes=15 * idx)).strftime("%H:%M")
            prices[slot] = 301.0 + idx * 0.03
            curve.append({"time_ny": slot, "low_price": 295.0 + idx * 0.01,
                          "median_price": 300.0 + idx * 0.01, "high_price": 305.0 + idx * 0.01})
        forecast = {
            "as_of_date": "2026-10-01", "forecast_date_weekday_estimate": target,
            "feed": "yahoo", "curve_method": "conditional_neighbors", "video_source_kind": "trained_intraday_candidate",
            "curve": curve,
            "estimated_open": {"p50_price": 294.88},
            "estimated_close_from_path": {"p50_price": 294.43},
            "main_path": {"model_family": "ridge_regression_per_time_slot", "price_by_time": prices,
                          "predicted_open_price": 301.25, "predicted_close_price": 301.75},
        }
        review = {"review_date": "2026-10-01", "review_method": "actual_only", "actual_close": 300.0,
                  "reference_close": None, "mae_usd": None, "day_change_pct": 0.2}
        _, txt, meta = _make_script("2026-10-02", review, forecast, True, [], "暂无可核验新闻", "2026-10-01", "测试源", [])
        self.assertIn("301.25美元", txt)
        self.assertIn("301.75美元", txt)
        self.assertNotIn("294.43美元", txt)
        self.assertTrue(meta["model_candidate_used"])
        self.assertEqual(meta["forecast_source_kind"], "trained_intraday_candidate")

    def test_stale_news_is_marked_as_background_only(self):
        translated, note, stale = _chinese_news_context([{
            "title": "Federal Reserve issues FOMC statement", "summary": "",
            "source": "Federal Reserve / Monetary Policy", "category": "monetary_policy",
            "published_at": "2026-10-01T18:00:00+00:00", "url": "",
        }], "2026-10-20")
        self.assertTrue(stale)
        self.assertIn("只作为宏观背景", note)
        self.assertTrue(translated)


if __name__ == "__main__":
    unittest.main()
