from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.next_day_model import (
    NumpyLogisticRegression,
    build_feature_frame,
    forecast_next_session,
    run_walk_forward_backtest,
)


def synthetic_history(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(20261013)
    dates = pd.bdate_range("2022-01-03", periods=rows)
    daily = rng.normal(0.0003, 0.012, size=rows)
    close = 100 * np.cumprod(1 + daily)
    open_price = close * (1 + rng.normal(0, 0.003, size=rows))
    high = np.maximum(close, open_price) * (1 + rng.uniform(0.001, 0.012, size=rows))
    low = np.minimum(close, open_price) * (1 - rng.uniform(0.001, 0.012, size=rows))
    volume = rng.integers(1_000_000, 10_000_000, size=rows)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": close,
        "high": high,
        "low": low,
        "open": open_price,
        "volume": volume,
    })


class NextDayModelTests(unittest.TestCase):
    def test_feature_frame_creates_next_day_target(self):
        history = synthetic_history(400)
        frame = build_feature_frame(history)
        self.assertEqual(frame.iloc[0]["target_date"], history.iloc[1]["date"])
        expected_return = history.iloc[1]["close"] / history.iloc[0]["close"] - 1
        self.assertAlmostEqual(frame.iloc[0]["target_return"], expected_return, places=10)
        self.assertTrue(pd.isna(frame.iloc[-1]["target_return"]))

    def test_logistic_regression_returns_bounded_scores(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(500, 4))
        y = (x[:, 0] - 0.3 * x[:, 1] > 0).astype(int)
        model = NumpyLogisticRegression(iterations=500).fit(x, y)
        probabilities = model.predict_proba_up(x[:20])
        self.assertEqual(len(probabilities), 20)
        self.assertTrue(np.all(probabilities > 0))
        self.assertTrue(np.all(probabilities < 1))


    def test_forecast_can_use_newer_feature_history_without_training_on_unknown_label(self):
        train = synthetic_history(800)
        newer = synthetic_history(820)
        # Shift dates of the final 20 bars to make the separate feature snapshot later.
        newer["date"] = pd.bdate_range("2022-01-03", periods=820).strftime("%Y-%m-%d")
        forecast = forecast_next_session(train, newer, "test newer feature frame")
        self.assertEqual(forecast["as_of_date"], newer.iloc[-1]["date"])
        self.assertEqual(forecast["feature_source"], "test newer feature frame")
        self.assertIn("next_session_directional_score_up_raw_uncalibrated", forecast)

    def test_walk_forward_returns_out_of_sample_rows(self):
        history = synthetic_history(900)
        predictions, summary = run_walk_forward_backtest(
            history, train_fraction=0.7, minimum_train_rows=300, refit_every=20
        )
        self.assertGreater(len(predictions), 100)
        self.assertIn("model_metrics", summary)
        self.assertIn("historical_majority_baseline_metrics", summary)
        self.assertLess(
            predictions.iloc[0]["feature_date"],
            predictions.iloc[0]["target_date"],
        )
        self.assertTrue(predictions["model_probability_up_raw"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
