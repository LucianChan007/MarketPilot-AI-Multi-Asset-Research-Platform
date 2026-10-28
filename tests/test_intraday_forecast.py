from __future__ import annotations

import json
import sqlite3
from contextlib import closing
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app.intraday_forecast import (
    EXPECTED_SLOTS,
    load_intraday_sessions,
    predict_intraday_path,
    run_intraday_prototype,
    run_walk_forward_backtest,
    _conditional_profile,
    choose_intraday_feed,
)

NY = ZoneInfo("America/New_York")


def _create_test_db(path: Path, n_days: int = 35) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("""CREATE TABLE intraday_bars (
            symbol TEXT NOT NULL, feed TEXT NOT NULL, timeframe TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL, trade_date_ny TEXT NOT NULL,
            timestamp_ny TEXT NOT NULL, session_time_ny TEXT NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume INTEGER NOT NULL DEFAULT 0,
            trade_count INTEGER, vwap REAL, fetched_at_utc TEXT NOT NULL,
            PRIMARY KEY(symbol,feed,timeframe,timestamp_utc))""")
        day = date(2026, 7, 1)
        sessions = []
        while len(sessions) < n_days:
            if day.weekday() < 5:
                sessions.append(day)
            day += timedelta(days=1)
        for i, session in enumerate(sessions):
            start_price = 300.0 + i * 0.05
            for slot_i, slot in enumerate(EXPECTED_SLOTS):
                # Deterministic intraday path with a small wave and time trend.
                op = start_price * (1 + slot_i * 0.0002 + ((i % 4) - 1.5) * 0.0001)
                close = op * (1 + ((slot_i % 5) - 2) * 0.00015)
                high = max(op, close) + 0.02
                low = min(op, close) - 0.02
                dt_ny = datetime.combine(session, time.fromisoformat(slot), tzinfo=NY)
                dt_utc = dt_ny.astimezone(ZoneInfo("UTC"))
                timestamp_utc = dt_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
                timestamp_ny = dt_ny.isoformat(timespec="seconds")
                conn.execute("""INSERT INTO intraday_bars
                    (symbol,feed,timeframe,timestamp_utc,trade_date_ny,timestamp_ny,session_time_ny,
                     open,high,low,close,volume,trade_count,vwap,fetched_at_utc)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    ("QQQM", "iex", "15Min", timestamp_utc, session.isoformat(), timestamp_ny, slot,
                     op, high, low, close, 1000, 10, close, timestamp_utc))
        conn.commit()
    finally:
        conn.close()


class IntradayForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "intraday.db"
        _create_test_db(self.db_path)
        # Force dates to be closed so today's historical row isn't excluded.
        self.now_ny = datetime(2026, 10, 20, 18, 0, tzinfo=NY)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_public_feed_selection_prefers_one_fresh_complete_source_without_blending(self) -> None:
        # Add a complete Yahoo copy of the history, then make IEX stale by one session.
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("SELECT symbol,feed,timeframe,timestamp_utc,trade_date_ny,timestamp_ny,session_time_ny,open,high,low,close,volume,trade_count,vwap,fetched_at_utc FROM intraday_bars").fetchall()
            yahoo_rows = [tuple([row[0], "yahoo", *row[2:]]) for row in rows]
            conn.executemany("INSERT INTO intraday_bars (symbol,feed,timeframe,timestamp_utc,trade_date_ny,timestamp_ny,session_time_ny,open,high,low,close,volume,trade_count,vwap,fetched_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", yahoo_rows)
            latest_day = conn.execute("SELECT MAX(trade_date_ny) FROM intraday_bars WHERE feed='iex'").fetchone()[0]
            conn.execute("DELETE FROM intraday_bars WHERE feed='iex' AND trade_date_ny=?", (latest_day,))
            conn.commit()
        feed, sessions = choose_intraday_feed(
            self.db_path, expected_latest_date=latest_day, min_training_sessions=25
        )
        self.assertEqual(feed, "yahoo")
        self.assertEqual(max(row["trade_date"] for row in sessions if row["path_eligible"]), latest_day)
        self.assertTrue(all(row["feed"] == "yahoo" for row in sessions))

    def test_run_intraday_prototype_can_override_target_to_next_full_session(self) -> None:
        report = Path(self.temp.name) / "reports"
        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        latest_day = max(row["trade_date"] for row in sessions if row["path_eligible"])
        from app.services.market_calendar import next_nasdaq_session
        from scripts.generate_video_script import _next_target_day
        target_day = _next_target_day(latest_day, datetime.now(NY))
        result = run_intraday_prototype(
            db_path=self.db_path, report_dir=report, feed="iex", timeframe="15Min",
            min_training_sessions=20, forecast_target_date=target_day,
        )
        self.assertEqual(result["forecast"]["forecast_date_weekday_estimate"], target_day)
        self.assertEqual(result["forecast"]["as_of_date"], latest_day)
        self.assertTrue(result["forecast"]["forecast_target_override_reason"].startswith("根据当前"))

    def test_loads_sessions_and_builds_forecast_curve(self) -> None:
        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        self.assertGreaterEqual(sum(s["path_eligible"] for s in sessions), 30)
        forecast = predict_intraday_path(sessions, min_training_sessions=20)
        self.assertGreater(len(forecast["curve"]), 20)
        self.assertGreater(forecast["estimated_open"]["p50_price"], 0)
        self.assertEqual(forecast["feed"], "iex")

    def test_conditional_neighbor_forecast_has_curve_and_method_label(self) -> None:
        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        forecast = predict_intraday_path(
            sessions, min_training_sessions=20, method="conditional_neighbors"
        )
        self.assertEqual(forecast["curve_method"], "conditional_neighbors")
        self.assertGreaterEqual(forecast["similar_session_count"], 8)
        self.assertGreater(len(forecast["curve"]), 20)
        self.assertGreater(forecast["estimated_open"]["p50_price"], 0)
        self.assertGreaterEqual(len(forecast.get("candidate_paths", [])), 8)
        self.assertIsInstance(forecast.get("market_only_main_path"), dict)
        self.assertIsInstance(forecast.get("main_path"), dict)
        self.assertGreater(len(forecast["main_path"].get("path", [])), 20)
        self.assertEqual(forecast["main_path"].get("model_family"), "ridge_regression_per_time_slot")
        self.assertTrue(forecast["main_path"].get("independent_of_quantile_curves"))

    def test_high_confidence_news_adjusts_ridge_path_as_experimental_overlay(self) -> None:
        from app.intraday_forecast import _attach_path_selection

        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        forecast = predict_intraday_path(
            sessions, min_training_sessions=20, method="conditional_neighbors"
        )
        news_context = {
            "available": True,
            "included_article_count": 2,
            "items": [
                {"nasdaq_bias": "bullish", "confidence": 0.9, "relevance_score": 0.9, "impact_score": 1.0},
                {"nasdaq_bias": "bullish", "confidence": 0.8, "relevance_score": 0.8, "impact_score": 1.0},
            ],
        }
        _attach_path_selection(forecast, news_context)
        diagnostics = forecast["path_selection_diagnostics"]
        self.assertTrue(diagnostics["news_adjustment_applied"])
        self.assertEqual(diagnostics["news_signal"]["qualified_directional_articles"], 2)
        self.assertEqual(diagnostics["selection_version"], "ridge-main-path-news-shadow-v2")
        self.assertTrue(forecast["main_path"].get("news_adjustment_applied"))
        self.assertIn("news-overlay", forecast["main_path"].get("model_name", ""))

    def test_conditional_p50_is_actual_median_not_weighted_mean(self) -> None:
        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        profile, ranked, _ = _conditional_profile(sessions, min_training_sessions=20)
        slot = next(slot for slot in EXPECTED_SLOTS if slot in profile)
        values = sorted(float(item["target"]["path_returns_pct"][slot]) for item in ranked
                        if slot in item["target"]["path_returns_pct"])
        n = len(values)
        expected = (values[(n - 1) // 2] + values[n // 2]) / 2.0
        self.assertAlmostEqual(profile[slot]["p50_return_pct"], expected, places=8)

    def test_walk_forward_metrics_have_test_sessions(self) -> None:
        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        summary, daily = run_walk_forward_backtest(sessions, min_training_sessions=20)
        self.assertGreater(summary["walk_forward_test_sessions"], 0)
        self.assertGreater(summary["evaluated_intraday_points"], 0)
        self.assertIn("ridge_ml_path_mae_bps", summary)
        self.assertGreater(len(daily), 0)

    def test_prototype_archives_candidate_paths_and_point_in_time_context(self) -> None:
        report_dir = Path(self.temp.name) / "reports" / "intraday"
        result = run_intraday_prototype(db_path=self.db_path, report_dir=report_dir)
        forecast = json.loads(Path(result["forecast_json"]).read_text(encoding="utf-8"))
        self.assertTrue(Path(result["archive_json"]).exists())
        self.assertEqual(forecast["forecast_snapshot_version"], "intraday-learning-snapshot-v2-main-path-ridge")
        self.assertRegex(forecast["forecast_date_weekday_estimate"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertGreater(len(forecast.get("candidate_paths", [])), 0)
        self.assertIsInstance(forecast.get("news_context"), dict)
        self.assertIsInstance(forecast.get("market_state_features"), dict)
        # The test database is old relative to wall-clock time, so it must not fake a live sample.
        self.assertEqual(forecast["capture_mode"], "late_or_stale_forecast")

    def test_repeated_generation_reuses_the_immutable_published_snapshot(self) -> None:
        from scripts.generate_video_script import _next_target_day
        report_dir = Path(self.temp.name) / "reports" / "intraday_repeat"
        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        latest_day = max(row["trade_date"] for row in sessions if row["path_eligible"])
        target = _next_target_day(latest_day, datetime.now(NY))
        first = run_intraday_prototype(
            db_path=self.db_path, report_dir=report_dir, feed="iex", timeframe="15Min",
            min_training_sessions=20, forecast_target_date=target,
        )
        first_forecast = json.loads(Path(first["forecast_json"]).read_text(encoding="utf-8"))
        second = run_intraday_prototype(
            db_path=self.db_path, report_dir=report_dir, feed="iex", timeframe="15Min",
            min_training_sessions=20, forecast_target_date=target,
        )
        second_forecast = json.loads(Path(second["forecast_json"]).read_text(encoding="utf-8"))
        self.assertEqual(first["archive_json"], second["archive_json"])
        self.assertEqual(first_forecast["generated_at_utc"], second_forecast["generated_at_utc"])
        self.assertEqual(first_forecast["main_path"]["price_by_time"], second_forecast["main_path"]["price_by_time"])

    def test_missing_middle_bar_is_not_fabricated(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM intraday_bars WHERE trade_date_ny=? AND session_time_ny=?", ("2026-07-01", "12:00"))
            conn.commit()
        finally:
            conn.close()
        sessions = load_intraday_sessions(self.db_path, now_ny=self.now_ny)
        first = next(s for s in sessions if s["trade_date"] == "2026-07-01")
        self.assertNotIn("12:00", first["bars"])
        self.assertNotIn("12:00", first["path_returns_pct"])


if __name__ == "__main__":
    unittest.main()
