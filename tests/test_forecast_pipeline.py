import unittest

import numpy as np
import pandas as pd

from app.forecast_pipeline import merge_training_and_recent_history


class ForecastPipelineMergeTests(unittest.TestCase):
    def _series(self, start: str, periods: int) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, periods=periods)
        close = 100.0 * np.cumprod(1.0 + np.linspace(-0.002, 0.003, periods))
        return pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "close": close,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "volume": np.linspace(1_000_000, 1_100_000, periods),
            }
        )

    def test_appends_newer_sqlite_bars_after_median_scale_alignment(self):
        true_series = self._series("2025-01-02", 320)
        csv_history = true_series.iloc[:300].copy().reset_index(drop=True)
        sqlite_history = true_series.iloc[250:].copy().reset_index(drop=True)
        price_columns = ["close", "open", "high", "low"]
        for column in price_columns:
            sqlite_history[column] = sqlite_history[column] / 1.0032

        merged, metadata = merge_training_and_recent_history(csv_history, sqlite_history)

        self.assertEqual(len(merged), 320)
        self.assertEqual(metadata["overlap_rows"], 50)
        self.assertEqual(metadata["appended_sqlite_rows"], 20)
        self.assertAlmostEqual(metadata["sqlite_to_csv_price_scale_ratio"], 1.0032, places=4)
        self.assertEqual(merged["date"].nunique(), 320)
        self.assertAlmostEqual(
            float(merged.iloc[-1]["close"]),
            float(true_series.iloc[-1]["close"]),
            places=6,
        )

    def test_refuses_to_append_unaligned_sources_without_enough_overlap(self):
        csv_history = self._series("2025-01-02", 300)
        recent_history = self._series("2026-03-02", 30)
        with self.assertRaisesRegex(ValueError, "重叠交易日"):
            merge_training_and_recent_history(csv_history, recent_history, minimum_overlap=20)

    def test_no_new_sqlite_dates_does_not_duplicate_rows(self):
        history = self._series("2025-01-02", 300)
        merged, metadata = merge_training_and_recent_history(
            history,
            history.iloc[-40:].copy(),
        )
        self.assertEqual(len(merged), len(history))
        self.assertEqual(merged["date"].nunique(), len(history))
        self.assertEqual(metadata["appended_sqlite_rows"], 0)


if __name__ == "__main__":
    unittest.main()
