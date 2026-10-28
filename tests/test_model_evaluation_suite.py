from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_model_evaluation_suite import prepare_extended_history, summarize_live_shadow


class ModelEvaluationSuiteTests(unittest.TestCase):
    def test_appends_only_newer_local_database_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "history.csv"
            database = root / "market.db"
            output = root / "out" / "extended.csv"
            dates = pd.bdate_range("2024-01-01", periods=360)
            history = pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "close": [100 + i * 0.01 for i in range(len(dates))],
                "adjClose": [100 + i * 0.01 for i in range(len(dates))],
                "open": [100 + i * 0.01 for i in range(len(dates))],
                "high": [101 + i * 0.01 for i in range(len(dates))],
                "low": [99 + i * 0.01 for i in range(len(dates))],
                "volume": [1000] * len(dates),
            })
            history.to_csv(source, index=False)
            last = dates[-1]
            new_day = (last + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
            conn = sqlite3.connect(database)
            conn.execute("""CREATE TABLE market_daily (
                ticker TEXT, trade_date TEXT, open REAL, high REAL, low REAL,
                close REAL, adj_close REAL, volume INTEGER
            )""")
            conn.execute(
                "INSERT INTO market_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("QQQM", new_day, 105, 107, 104, 106, 106, 1200),
            )
            conn.commit()
            conn.close()

            report = prepare_extended_history(source, database, output)
            extended = pd.read_csv(output)
            self.assertEqual(report["base_latest_date"], last.strftime("%Y-%m-%d"))
            self.assertEqual(report["appended_database_rows"], 1)
            self.assertEqual(report["combined_latest_date"], new_day)
            self.assertEqual(len(extended), len(history) + 1)
            self.assertEqual(len(pd.read_csv(source)), len(history), "original history CSV must stay unchanged")

    def test_shadow_gate_requires_minimum_prospective_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "feedback.db"
            conn = sqlite3.connect(database)
            conn.execute("""CREATE TABLE forecast_feedback (
                ticker TEXT, status TEXT, actual_target_date TEXT,
                error_zero_return_bps REAL, direction_correct_zero_return INTEGER,
                error_historical_mean_bps REAL, direction_correct_historical_mean INTEGER,
                error_ridge_full_bps REAL, direction_correct_ridge_full INTEGER,
                error_ridge_returns_only_bps REAL, direction_correct_ridge_returns_only INTEGER
            )""")
            row = ("QQQM", "evaluated", "2026-01-01", 10, 1, 10, 1, 8, 1, 7, 1)
            conn.executemany(
                "INSERT INTO forecast_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row] * 29,
            )
            conn.commit()
            conn.close()

            result = summarize_live_shadow(database, minimum_samples=30)
            self.assertEqual(result["comparable_samples"], 29)
            self.assertFalse(result["gate_passed"])

    def test_shadow_gate_can_pass_candidate_screen_but_never_auto_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "feedback.db"
            conn = sqlite3.connect(database)
            conn.execute("""CREATE TABLE forecast_feedback (
                ticker TEXT, status TEXT, actual_target_date TEXT,
                error_zero_return_bps REAL, direction_correct_zero_return INTEGER,
                error_historical_mean_bps REAL, direction_correct_historical_mean INTEGER,
                error_ridge_full_bps REAL, direction_correct_ridge_full INTEGER,
                error_ridge_returns_only_bps REAL, direction_correct_ridge_returns_only INTEGER
            )""")
            rows = []
            for i in range(30):
                rows.append(("QQQM", "evaluated", f"2026-01-{(i % 28) + 1:02d}", 10, 1, 10, 1, 8, 1, 7, 1))
            conn.executemany(
                "INSERT INTO forecast_feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            conn.commit()
            conn.close()

            result = summarize_live_shadow(database, minimum_samples=30)
            self.assertTrue(result["gate_passed"])
            self.assertFalse(result["automatic_production_replacement"])
            self.assertEqual(result["comparable_samples"], 30)


if __name__ == "__main__":
    unittest.main()
