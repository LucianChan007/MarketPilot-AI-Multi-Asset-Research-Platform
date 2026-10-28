from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.context_agent import build_context_agent_payload, build_local_analysis, run_context_agent


class ContextAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "reports" / "context_features").mkdir(parents=True)
        (self.root / "reports" / "forecast").mkdir(parents=True)
        (self.root / "data").mkdir()
        context = {
            "ticker": "QQQM",
            "as_of_date": "2026-10-13",
            "latest_price": 306.16,
            "market": {"return_1_session_pct": -0.33, "return_5_sessions_pct": -2.1,
                       "return_20_sessions_pct": 5.65, "annualized_volatility_20_session_pct": 15.0},
            "news": {"news_count_30_calendar_days": 5, "recent_items": []},
            "macro": {"available": True},
            "modeling_policy": {"macro_features_allowed_in_historical_backtest": False,
                                "news_features_allowed_in_training_by_default": False},
        }
        forecast = {
            "ticker": "QQQM", "as_of_date": "2026-10-13", "latest_adjusted_close": 306.16,
            "point_forecasts": {"historical_mean_return_pct": 0.07, "ridge_full_return_pct": 0.06,
                                 "ridge_returns_only_return_pct": 0.14},
            "historical_return_bands": {"last_60_sessions": {"p10_price_scenario": 302.8,
                                      "p50_price_scenario": 306.2, "p90_price_scenario": 310.4}},
        }
        (self.root / "reports" / "context_features" / "QQQM_context_snapshot_latest.json").write_text(
            json.dumps(context), encoding="utf-8"
        )
        (self.root / "reports" / "forecast" / "QQQM_return_forecast_latest_20261014_093932.json").write_text(
            json.dumps(forecast), encoding="utf-8"
        )
        conn = sqlite3.connect(self.root / "data" / "market_data.db")
        conn.execute("""CREATE TABLE forecast_feedback (
            id INTEGER PRIMARY KEY, ticker TEXT, as_of_date TEXT, status TEXT, actual_target_date TEXT,
            actual_return_pct REAL, error_zero_return_bps REAL, error_historical_mean_bps REAL,
            error_ridge_full_bps REAL, error_ridge_returns_only_bps REAL,
            direction_correct_zero_return INTEGER, direction_correct_historical_mean INTEGER,
            direction_correct_ridge_full INTEGER, direction_correct_ridge_returns_only INTEGER,
            interval_20_covered INTEGER, interval_60_covered INTEGER, interval_252_covered INTEGER
        )""")
        conn.execute("""INSERT INTO forecast_feedback (
            id,ticker,as_of_date,status,actual_target_date,actual_return_pct,
            error_zero_return_bps,error_historical_mean_bps,error_ridge_full_bps,error_ridge_returns_only_bps,
            direction_correct_zero_return,direction_correct_historical_mean,direction_correct_ridge_full,
            direction_correct_ridge_returns_only,interval_20_covered,interval_60_covered,interval_252_covered
        ) VALUES (1,'QQQM','2026-10-12','evaluated','2026-10-13',-0.3353,33.53,40.80,50.10,47.65,1,0,0,0,1,1,1)""")
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_payload_reads_context_forecast_and_feedback(self) -> None:
        payload = build_context_agent_payload(self.root)
        self.assertEqual(payload["context"]["as_of_date"], "2026-10-13")
        self.assertEqual(payload["feedback"]["evaluated_forecasts"], 1)
        self.assertFalse(payload["constraints"]["historical_macro_data_eligible_for_training"])

    def test_local_report_warns_about_small_sample(self) -> None:
        payload = build_context_agent_payload(self.root)
        report = build_local_analysis(payload)
        self.assertIn("样本不足", report)
        self.assertIn("不是已校准的未来概率区间", report)

    def test_offline_run_writes_latest_report(self) -> None:
        result = run_context_agent(self.root, offline=True)
        self.assertEqual(result["mode"], "offline")
        self.assertTrue(Path(result["latest_report_path"]).exists())
        self.assertTrue(Path(result["payload_path"]).exists())


if __name__ == "__main__":
    unittest.main()
