from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_champion_challenger import run_tournament


class ChampionChallengerTests(unittest.TestCase):
    def _frame(self, n: int = 240) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-01", periods=n)
        actual = [((i % 7) - 3) * 0.1 for i in range(n)]
        baseline = [v + 0.08 for v in actual]
        # Challenger has slightly lower noise on all samples.
        challenger = [v + (0.015 if i % 2 else -0.015) for i, v in enumerate(actual)]
        frame = pd.DataFrame({
            "target_date": dates.strftime("%Y-%m-%d"),
            "actual_return_pct": actual,
            "zero_return_baseline_predicted_return_pct": [0.0] * n,
            "expanding_mean_baseline_predicted_return_pct": baseline,
            "rolling_20d_mean_baseline_predicted_return_pct": [v + 0.06 for v in actual],
            "ridge_full_predicted_return_pct": challenger,
            "ridge_returns_only_predicted_return_pct": [v + 0.04 for v in actual],
        })
        return frame

    def test_splits_chronologically_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "predictions.csv"
            self._frame().to_csv(source, index=False)
            report = run_tournament(source, root / "out")
            self.assertLess(
                report["development"]["end_target_date"],
                report["holdout"]["start_target_date"],
            )
            self.assertEqual(report["holdout"]["incumbent"], "expanding_mean_baseline")
            self.assertTrue((root / "out" / "QQQM_champion_challenger_latest.json").exists())
            self.assertTrue((root / "out" / "QQQM_champion_challenger_monthly_latest.csv").exists())
            self.assertFalse(report["promotion_gate"]["automatic_production_replacement"])

    def test_rejects_too_few_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "predictions.csv"
            self._frame(100).to_csv(source, index=False)
            with self.assertRaises(ValueError):
                run_tournament(source, root / "out")


if __name__ == "__main__":
    unittest.main()
