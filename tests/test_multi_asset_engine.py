from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app import multi_asset_engine as engine

NY = ZoneInfo("America/New_York")


def make_market_data(start: date = date(2026, 8, 25), sessions: int = 40, symbol: str = "AAPL") -> tuple[pd.DataFrame, pd.DataFrame]:
    days = []
    d = start
    while len(days) < sessions:
        if d.weekday() < 5 and d not in {date(2026, 9, 7), date(2026, 10, 12)}:
            days.append(d)
        d += timedelta(days=1)
    daily_rows = []
    intraday_rows = []
    previous_close = {"QQQM": 500.0, "AAPL": 180.0, "MSFT": 420.0, "NVDA": 130.0, "SPY": 600.0}.get(symbol.upper(), 75.0)
    for idx, day in enumerate(days):
        open_price = previous_close * (1 + 0.001 * np.sin(idx / 3))
        bar_closes = []
        prices = []
        for slot_index in range(26):
            t = slot_index / 25
            price = open_price * (1 + .0015 * t + .002 * np.sin(idx / 4 + t * np.pi * 1.5))
            bar_closes.append(float(price))
            start_dt = datetime.combine(day, time(9, 30), tzinfo=NY) + timedelta(minutes=15 * slot_index)
            intraday_rows.append({
                "timestamp_ny": pd.Timestamp(start_dt),
                "timestamp_utc": pd.Timestamp(start_dt).tz_convert("UTC"),
                "trade_date_ny": day.isoformat(), "session_time_ny": start_dt.strftime("%H:%M"),
                "open": float(open_price if slot_index == 0 else bar_closes[-2]),
                "high": float(max(open_price if slot_index == 0 else bar_closes[-2], price) * 1.0005),
                "low": float(min(open_price if slot_index == 0 else bar_closes[-2], price) * .9995),
                "close": float(price), "volume": float(100000 + idx * 100 + slot_index),
            })
            prices.extend([price])
        close = bar_closes[-1]
        daily_rows.append({"trade_date_ny": day.isoformat(), "open": open_price, "high": max(prices) * 1.001,
                           "low": min(prices) * .999, "close": close, "adj_close": close, "volume": 2600000 + idx * 1000})
        previous_close = close
    return pd.DataFrame(daily_rows), pd.DataFrame(intraday_rows)


class MultiAssetEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "multi_asset.db"
        self.config = self.root / "instruments.json"
        self.report_root = self.root / "reports"
        self.archive_root = self.report_root / "forecast_archive"
        self.news_root = self.report_root / "news"
        self.config.write_text(json.dumps({"schema_version": 1, "instruments": [
            {"symbol": "QQQM", "name": "QQQM", "type": "etf", "enabled": True},
            {"symbol": "AAPL", "name": "Apple", "type": "stock", "enabled": True},
        ]}), encoding="utf-8")
        engine.init_db(self.db)

    def tearDown(self):
        self.temp.cleanup()

    def _seed(self, symbol: str = "AAPL", sessions: int = 40):
        daily, intraday = make_market_data(sessions=sessions, symbol=symbol)
        engine.save_daily(symbol, daily, self.db)
        engine.save_intraday(symbol, intraday, self.db)
        return daily, intraday

    def test_instrument_config_add_symbol_without_duplicate(self):
        engine.add_instrument("AMD", "Advanced Micro Devices", "stock", self.config)
        engine.add_instrument("AMD", "AMD", "stock", self.config)
        rows = engine.load_instruments(self.config, enabled_only=False)
        self.assertEqual(sum(x["symbol"] == "AMD" for x in rows), 1)

    def test_symbols_are_isolated_in_database(self):
        daily, intraday = self._seed("AAPL")
        q_daily, q_intraday = self._seed("QQQM")
        self.assertEqual(engine.latest_dates("AAPL", self.db)["daily_rows"], len(daily))
        self.assertEqual(engine.latest_dates("QQQM", self.db)["daily_rows"], len(q_daily))
        aapl_prices = engine._daily_rows("AAPL", self.db)["close"].tolist()
        qqqm_prices = engine._daily_rows("QQQM", self.db)["close"].tolist()
        self.assertNotEqual(aapl_prices, qqqm_prices)
        self.assertEqual(len(engine.complete_sessions("AAPL", self.db)), len(daily))

    def test_daily_chart_date_uses_utc_exchange_date_not_shifted_new_york_date(self):
        stamp = int(pd.Timestamp("2026-10-20T00:00:00Z").timestamp())
        payload = {"timestamp": [stamp], "indicators": {"quote": [{"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [500]}], "adjclose": [{"adjclose": [100.5]}]}}
        frame = engine._payload_frame(payload, "AAPL", "1d")
        self.assertEqual(frame.iloc[0]["trade_date_ny"], "2026-10-20")

    def test_ridge_forecast_has_27_points_and_is_archived_immutably(self):
        self._seed("AAPL", 40)
        with patch.object(engine, "ARCHIVE_ROOT", self.archive_root), patch.object(engine, "REPORT_ROOT", self.report_root), patch.object(engine, "_next_session_date", return_value="2099-01-02"):
            first = engine.generate_forecast("AAPL", self.db)
            second = engine.generate_forecast("AAPL", self.db)
        self.assertTrue(first["success"])
        self.assertEqual(len(first["forecast"]["curve"]), 27)
        self.assertEqual(first["forecast"]["model_version"], "ridge-multi-asset-v2.1")
        self.assertEqual(first["forecast_file"], second["forecast_file"])
        self.assertTrue(second["reused_immutable_snapshot"])
        self.assertTrue(Path(first["forecast_file"]).is_file())

    def test_due_forecast_settlement_uses_prospective_snapshot(self):
        daily, intraday = self._seed("AAPL", 40)
        target = daily.iloc[-2]["trade_date_ny"]
        target_session = [s for s in engine.complete_sessions("AAPL", self.db) if s["date"] == target][0]
        prev_close = float(daily[daily["trade_date_ny"] < target].iloc[-1]["close"])
        actual_path = engine._bar_path(target_session)
        times = engine.TIME_SLOTS
        curve = [{"time_ny": t, "predicted_price": round(p * 1.0001, 4), "median_price": round(p * 1.0001, 4), "low_price": round(p * .99, 4), "high_price": round(p * 1.01, 4)} for t, p in zip(times, actual_path)]
        asof = daily[daily["trade_date_ny"] < target].iloc[-1]["trade_date_ny"]
        path = self.archive_root / f"AAPL_asof_{asof}_target_{target}.json"
        # The creation time is before the target NY market open, proving this is prospective.
        created = (datetime.combine(date.fromisoformat(target), time(8, 0), tzinfo=NY)).astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 2, "symbol": "AAPL", "target_date": target, "as_of_date": asof,
                                    "feed": "yahoo", "previous_close": prev_close, "created_at_utc": created,
                                    "immutable_snapshot": True, "curve": curve}), encoding="utf-8")
        with patch.object(engine, "ARCHIVE_ROOT", self.archive_root), patch.object(engine, "REPORT_ROOT", self.report_root):
            results = engine.settle_due_forecasts("AAPL", self.db)
        row = next((x for x in results if x.get("target_date") == target and x.get("status") == "evaluated"), None)
        self.assertIsNotNone(row)
        self.assertGreater(row["bar_count"], 20)
        self.assertGreaterEqual(row["mae_usd"], 0)
        self.assertEqual(len(engine.get_evaluations("AAPL", db_path=self.db)), 1)
        self.assertTrue((self.report_root / "AAPL" / f"review_{target}.png").is_file())

    def test_settlement_refuses_to_compare_different_market_data_feeds(self):
        daily, intraday = self._seed("AAPL", 40)
        target = daily.iloc[-2]["trade_date_ny"]
        target_session = [s for s in engine.complete_sessions("AAPL", self.db) if s["date"] == target][0]
        prev_close = float(daily[daily["trade_date_ny"] < target].iloc[-1]["close"])
        actual_path = engine._bar_path(target_session)
        curve = [{"time_ny": slot, "predicted_price": float(price), "median_price": float(price), "low_price": float(price), "high_price": float(price)} for slot, price in zip(engine.TIME_SLOTS, actual_path)]
        asof = daily[daily["trade_date_ny"] < target].iloc[-1]["trade_date_ny"]
        path = self.archive_root / f"AAPL_asof_{asof}_target_{target}.json"
        created = datetime.combine(date.fromisoformat(target), time(8, 0), tzinfo=NY).astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 2, "symbol": "AAPL", "target_date": target, "as_of_date": asof,
                                    "feed": "eodhd", "previous_close": prev_close, "created_at_utc": created,
                                    "immutable_snapshot": True, "curve": curve}), encoding="utf-8")
        with patch.object(engine, "ARCHIVE_ROOT", self.archive_root), patch.object(engine, "REPORT_ROOT", self.report_root):
            results = engine.settle_due_forecasts("AAPL", self.db, feed="yahoo")
        row = next((x for x in results if x.get("target_date") == target), None)
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "skipped_feed_mismatch")
        self.assertEqual(engine.get_evaluations("AAPL", db_path=self.db), [])

    def test_ai_key_presence_is_not_required_for_market_model(self):
        daily, intraday = self._seed("AAPL")
        dates = engine.latest_dates("AAPL", self.db)
        self.assertEqual(dates["daily_latest_date"], dates["intraday_latest_complete_date"])
        self.assertGreaterEqual(dates["complete_intraday_sessions"], 20)


    def test_daily_fetch_falls_back_to_yfinance_when_direct_chart_api_fails(self):
        frame = pd.DataFrame({
            "trade_date_ny": ["2026-10-26"], "open": [100.0], "high": [102.0],
            "low": [99.0], "close": [101.0], "adj_close": [101.0], "volume": [1000],
        })
        with patch.object(engine, "_fetch_chart_payload", side_effect=RuntimeError("direct API blocked")), \
             patch.object(engine, "_yfinance_daily_frame", return_value=frame):
            result = engine.fetch_daily("AAPL")
        self.assertEqual(result.attrs["fetch_method"], "yfinance_fallback")
        self.assertEqual(result.attrs["fetch_warnings"][0].split(":", 1)[0], "chart_api")
        self.assertEqual(result.iloc[0]["trade_date_ny"], "2026-10-26")

    def test_intraday_fetch_falls_back_to_yfinance_when_direct_chart_api_fails(self):
        start = datetime(2026, 10, 26, 9, 30, tzinfo=NY)
        rows = []
        for i in range(26):
            stamp = start + timedelta(minutes=15 * i)
            price = 100.0 + i * .1
            rows.append({
                "timestamp_utc": pd.Timestamp(stamp).tz_convert("UTC"),
                "timestamp_ny": pd.Timestamp(stamp), "trade_date_ny": stamp.date().isoformat(),
                "session_time_ny": stamp.strftime("%H:%M"), "open": price, "high": price + .1,
                "low": price - .1, "close": price, "volume": 500,
            })
        frame = pd.DataFrame(rows)
        with patch.object(engine, "_fetch_chart_payload", side_effect=RuntimeError("direct API blocked")), \
             patch.object(engine, "_yfinance_intraday_frame", return_value=frame):
            result = engine.fetch_intraday("AAPL")
        self.assertEqual(result.attrs["fetch_method"], "yfinance_fallback")
        self.assertEqual(len(result), 26)
        self.assertEqual(result.iloc[0]["session_time_ny"], "09:30")

    def test_daily_only_fallback_is_visible_but_not_mixed_into_forecast_feed(self):
        daily, _ = self._seed("AAPL", 40)
        last_date = date.fromisoformat(str(daily.iloc[-1]["trade_date_ny"]))
        fallback_date = last_date + timedelta(days=2)
        stooq = pd.DataFrame([{
            "trade_date_ny": fallback_date.isoformat(), "open": 200.0, "high": 202.0,
            "low": 199.0, "close": 201.0, "adj_close": 201.0, "volume": 10000,
        }])
        engine.save_daily("AAPL", stooq, self.db, feed="stooq")
        result = engine.latest_dates("AAPL", self.db)
        self.assertEqual(result["feed"], "yahoo")
        self.assertTrue(result["dates_aligned"])
        self.assertEqual(result["daily_latest_date"], str(daily.iloc[-1]["trade_date_ny"]))
        self.assertEqual(result["latest_daily_any_feed_date"], fallback_date.isoformat())
        self.assertEqual(result["latest_daily_any_feed_source"], "stooq")

    def test_sync_only_workflow_does_not_run_news_or_generate_forecasts(self):
        config = self.root / "instruments.json"
        config.write_text(json.dumps({"schema_version": 1, "instruments": [
            {"symbol": "AAPL", "name": "Apple", "type": "stock", "enabled": True},
        ]}), encoding="utf-8")
        latest = {
            "symbol": "AAPL", "daily_latest_date": "2026-10-27",
            "intraday_latest_complete_date": "2026-10-27", "daily_rows": 250,
            "complete_intraday_sessions": 50, "feed": "yahoo", "dates_aligned": True,
            "latest_daily_any_feed_date": "2026-10-27", "latest_daily_any_feed_source": "yahoo",
            "latest_daily_any_feed_rows": 250, "daily_fetched_at": "2026-10-28T03:00:00Z",
            "intraday_fetched_at": "2026-10-28T03:00:00Z", "provider_candidates": [],
        }
        attempt = {"feed": "yahoo", "daily_rows_updated": 250, "intraday_rows_updated": 1200,
                   "daily_latest_date": "2026-10-27", "intraday_latest_complete_date": "2026-10-27",
                   "dates_aligned": True, "errors": [], "warnings": []}
        with patch.object(engine, "CONFIG_PATH", config), \
             patch.object(engine, "REPORT_ROOT", self.report_root), \
             patch.object(engine, "_expected_completed_date", return_value="2026-10-27"), \
             patch.object(engine, "_provider_order", return_value=["yahoo"]), \
             patch.object(engine, "_fetch_provider_pair", return_value=attempt), \
             patch.object(engine, "_refresh_stooq_daily_if_needed", return_value=None), \
             patch.object(engine, "latest_dates", return_value=latest), \
             patch.object(engine, "sync_news", side_effect=AssertionError("sync-only must not sync news")), \
             patch.object(engine, "generate_forecast", side_effect=AssertionError("sync-only must not forecast")):
            result = engine.sync_market_data(["AAPL"], self.db)
        self.assertTrue(result["overall_success"])
        self.assertTrue(result["forecast_ready_all"])
        self.assertEqual(result["workflow"], "market_data_sync_only")
        self.assertTrue((self.report_root / "latest_status.json").is_file())


if __name__ == "__main__":
    unittest.main()
