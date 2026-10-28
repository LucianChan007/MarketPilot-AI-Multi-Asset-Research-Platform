from __future__ import annotations

"""QQQM next-session return forecast research script.

Compares simple point-forecast baselines with a small NumPy ridge regression,
then evaluates empirical return bands based only on returns available by each
feature date. Walk-forward test results are exploratory and not trade signals.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.next_day_model import (  # noqa: E402
    FEATURE_COLUMNS,
    build_feature_frame,
    load_history,
    load_recent_history_from_sqlite,
)

RETURN_FEATURE_GROUPS: dict[str, list[str]] = {
    "ridge_full": list(FEATURE_COLUMNS),
    "ridge_returns_only": ["ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d"],
}


class NumpyRidgeRegression:
    """Small standardized ridge regression; no sklearn dependency."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = float(alpha)
        self.mean_x: np.ndarray | None = None
        self.scale_x: np.ndarray | None = None
        self.mean_y: float | None = None
        self.weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRidgeRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(y) < 30:
            raise ValueError("Ridge 训练数据无效，至少需要 30 条样本。")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Ridge 训练数据包含缺失或非有限值。")
        self.mean_x = x.mean(axis=0)
        self.scale_x = x.std(axis=0)
        self.scale_x[self.scale_x < 1e-10] = 1.0
        x_std = (x - self.mean_x) / self.scale_x
        self.mean_y = float(y.mean())
        y_centered = y - self.mean_y
        identity = np.eye(x_std.shape[1], dtype=float)
        self.weights = np.linalg.solve(x_std.T @ x_std + self.alpha * identity, x_std.T @ y_centered)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean_x is None or self.scale_x is None or self.mean_y is None or self.weights is None:
            raise RuntimeError("Ridge 模型尚未训练。")
        x = np.asarray(x, dtype=float)
        return self.mean_y + ((x - self.mean_x) / self.scale_x) @ self.weights


def _point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return {"n": 0}
    correlation: float | None = None
    if len(actual) > 1 and np.std(predicted) > 1e-12 and np.std(actual) > 1e-12:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    # Convert decimal returns to basis points: 1 bp = 0.01 percentage point.
    sign_actual = actual > 0
    sign_predicted = predicted > 0
    return {
        "n": int(len(actual)),
        "mae_bps": float(np.mean(np.abs(predicted - actual)) * 10000),
        "rmse_bps": float(np.sqrt(np.mean((predicted - actual) ** 2)) * 10000),
        "mean_actual_return_pct": float(np.mean(actual) * 100),
        "mean_predicted_return_pct": float(np.mean(predicted) * 100),
        "mean_absolute_prediction_bps": float(np.mean(np.abs(predicted)) * 10000),
        "sign_accuracy": float(np.mean(sign_actual == sign_predicted)),
        "prediction_actual_correlation": correlation,
    }


def _interval_metrics(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, label: str) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    actual, lower, upper = actual[valid], lower[valid], upper[valid]
    if len(actual) == 0:
        return {"interval": label, "n": 0}
    return {
        "interval": label,
        "n": int(len(actual)),
        "nominal_coverage": 0.80,
        "observed_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width_pct_points": float(np.mean(upper - lower) * 100),
        "median_lower_pct": float(np.median(lower) * 100),
        "median_upper_pct": float(np.median(upper) * 100),
    }


def _history_context(history: pd.DataFrame) -> pd.DataFrame:
    """Rolling return summaries available after close on each feature date."""
    out = history[["date", "close"]].copy().sort_values("date").reset_index(drop=True)
    daily = pd.to_numeric(out["close"], errors="coerce").pct_change()
    out["known_daily_return"] = daily
    out["rolling_mean_20"] = daily.rolling(20, min_periods=15).mean()
    for window in (20, 60, 252):
        out[f"q10_{window}"] = daily.rolling(window, min_periods=max(15, window // 2)).quantile(0.10)
        out[f"q50_{window}"] = daily.rolling(window, min_periods=max(15, window // 2)).quantile(0.50)
        out[f"q90_{window}"] = daily.rolling(window, min_periods=max(15, window // 2)).quantile(0.90)
    return out


def run_backtest(
    csv_path: Path,
    train_fraction: float = 0.70,
    minimum_train_rows: int = 300,
    refit_every: int = 20,
    ridge_alpha: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    history = load_history(csv_path)
    engineered = build_feature_frame(history)
    context = _history_context(history)
    engineered = engineered.merge(context.drop(columns=["close"]), on="date", how="left")
    needed = list(FEATURE_COLUMNS) + ["target_return", "target_date"]
    labeled = engineered.dropna(subset=needed).copy().reset_index(drop=True)
    if len(labeled) < minimum_train_rows + 50:
        raise ValueError(f"有效样本只有 {len(labeled)} 条；至少需要 {minimum_train_rows + 50} 条。")
    first_test = max(int(len(labeled) * train_fraction), minimum_train_rows)
    if first_test >= len(labeled) - 10:
        raise ValueError("样本外测试数据不足，请增加历史数据或调整参数。")

    feature_all = labeled[FEATURE_COLUMNS].to_numpy(dtype=float)
    target_all = labeled["target_return"].to_numpy(dtype=float)
    dates = labeled["date"].astype(str).to_numpy()
    target_dates = labeled["target_date"].astype(str).to_numpy()
    records: list[dict[str, Any]] = []

    for block_start in range(first_test, len(labeled), refit_every):
        block_end = min(block_start + refit_every, len(labeled))
        y_train = target_all[:block_start]
        x_test_full = feature_all[block_start:block_end]
        train_mean = float(np.mean(y_train))
        models: dict[str, np.ndarray] = {
            "zero_return_baseline": np.zeros(block_end - block_start, dtype=float),
            "expanding_mean_baseline": np.full(block_end - block_start, train_mean, dtype=float),
            "rolling_20d_mean_baseline": labeled["rolling_mean_20"].iloc[block_start:block_end].to_numpy(dtype=float),
        }
        for model_name, columns in RETURN_FEATURE_GROUPS.items():
            indices = [FEATURE_COLUMNS.index(column) for column in columns]
            model = NumpyRidgeRegression(alpha=ridge_alpha).fit(feature_all[:block_start, indices], y_train)
            models[model_name] = model.predict(x_test_full[:, indices])

        for offset, row_index in enumerate(range(block_start, block_end)):
            row: dict[str, Any] = {
                "feature_date": dates[row_index],
                "target_date": target_dates[row_index],
                "actual_return_pct": float(target_all[row_index] * 100),
                "actual_direction": "上涨" if target_all[row_index] > 0 else "下跌或持平",
            }
            for model_name, values in models.items():
                row[f"{model_name}_predicted_return_pct"] = float(values[offset] * 100) if np.isfinite(values[offset]) else np.nan
            for window in (20, 60, 252):
                row[f"historical_{window}d_q10_pct"] = float(labeled.iloc[row_index][f"q10_{window}"] * 100)
                row[f"historical_{window}d_q50_pct"] = float(labeled.iloc[row_index][f"q50_{window}"] * 100)
                row[f"historical_{window}d_q90_pct"] = float(labeled.iloc[row_index][f"q90_{window}"] * 100)
            records.append(row)

    predictions = pd.DataFrame(records)
    summary_rows: list[dict[str, Any]] = []
    actual = predictions["actual_return_pct"].to_numpy(dtype=float) / 100
    for model_name in ["zero_return_baseline", "expanding_mean_baseline", "rolling_20d_mean_baseline", *RETURN_FEATURE_GROUPS.keys()]:
        pred = predictions[f"{model_name}_predicted_return_pct"].to_numpy(dtype=float) / 100
        summary_rows.append({"experiment": model_name, **_point_metrics(actual, pred)})
    summary = pd.DataFrame(summary_rows).sort_values("mae_bps").reset_index(drop=True)

    interval_rows: list[dict[str, Any]] = []
    for window in (20, 60, 252):
        interval_rows.append(_interval_metrics(
            actual,
            predictions[f"historical_{window}d_q10_pct"].to_numpy(dtype=float) / 100,
            predictions[f"historical_{window}d_q90_pct"].to_numpy(dtype=float) / 100,
            f"rolling_{window}d_empirical_p10_p90",
        ))
    intervals = pd.DataFrame(interval_rows)

    monthly_rows: list[dict[str, Any]] = []
    predictions["target_month"] = pd.to_datetime(predictions["target_date"]).dt.to_period("M").astype(str)
    model_names = ["zero_return_baseline", "expanding_mean_baseline", "rolling_20d_mean_baseline", *RETURN_FEATURE_GROUPS.keys()]
    for (month,), frame in predictions.groupby(["target_month"], sort=True):
        y = frame["actual_return_pct"].to_numpy(dtype=float) / 100
        for model_name in model_names:
            p = frame[f"{model_name}_predicted_return_pct"].to_numpy(dtype=float) / 100
            monthly_rows.append({"month": month, "experiment": model_name, **_point_metrics(y, p)})
    monthly = pd.DataFrame(monthly_rows)
    predictions = predictions.drop(columns=["target_month"])
    meta = {
        "ticker": "QQQM",
        "csv_path": str(csv_path.resolve()),
        "history_start": str(history["date"].iloc[0]),
        "history_end": str(history["date"].iloc[-1]),
        "usable_labeled_samples": int(len(labeled)),
        "out_of_sample_samples": int(len(predictions)),
        "test_start": str(predictions["target_date"].iloc[0]),
        "test_end": str(predictions["target_date"].iloc[-1]),
        "train_fraction": train_fraction,
        "refit_every": refit_every,
        "ridge_alpha_fixed_without_test_tuning": ridge_alpha,
        "features": FEATURE_COLUMNS,
        "caveat": "Return forecasts and empirical bands are exploratory estimates, not calibrated guarantees or trade instructions. Transaction costs are not included in this return forecast test; they must be applied separately by the portfolio planner.",
    }
    return summary, predictions, intervals, meta, monthly


def make_latest_forecast(csv_path: Path, db_path: Path | None, ridge_alpha: float) -> dict[str, Any]:
    training = load_history(csv_path)
    source = "Tiingo CSV"
    latest_history = training
    if db_path is not None and db_path.exists():
        candidate = load_recent_history_from_sqlite(db_path, ticker="QQQM", limit=260)
        if str(candidate["date"].iloc[-1]) > str(training["date"].iloc[-1]):
            latest_history = candidate
            source = f"market_daily SQLite，更新至 {candidate['date'].iloc[-1]}；训练标签仍来自 Tiingo CSV"
    training_frame = build_feature_frame(training)
    training_labeled = training_frame.dropna(subset=FEATURE_COLUMNS + ["target_return"]).copy().reset_index(drop=True)
    if len(training_labeled) < 300:
        raise ValueError("用于收益率回归的有效训练样本不足 300 条。")
    latest_frame = build_feature_frame(latest_history)
    latest_row = latest_frame.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    result: dict[str, Any] = {
        "as_of_date": str(latest_row["date"]),
        "latest_adjusted_close": float(latest_row["close"]),
        "feature_source": source,
        "training_labels_known_through": str(training_labeled["target_date"].iloc[-1]),
        "estimates_are_not_guarantees": True,
        "point_forecasts": {},
        "historical_return_bands": {},
        "caveat": "历史经验区间不是经过校准的概率区间，实际下一交易日涨跌可能超出区间。模型预测未含交易成本，不能单独作为买卖信号。",
    }
    y = training_labeled["target_return"].to_numpy(dtype=float)
    x_latest = latest_row[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
    result["point_forecasts"]["zero_return_baseline_pct"] = 0.0
    result["point_forecasts"]["historical_mean_return_pct"] = float(np.mean(y) * 100)
    for model_name, columns in RETURN_FEATURE_GROUPS.items():
        indices = [FEATURE_COLUMNS.index(column) for column in columns]
        x_train = training_labeled[columns].to_numpy(dtype=float)
        model = NumpyRidgeRegression(alpha=ridge_alpha).fit(x_train, y)
        predicted = float(model.predict(latest_row[columns].to_numpy(dtype=float).reshape(1, -1))[0])
        result["point_forecasts"][f"{model_name}_return_pct"] = predicted * 100

    latest_returns = pd.to_numeric(latest_history["close"], errors="coerce").pct_change().dropna()
    for window in (20, 60, 252):
        sample = latest_returns.tail(window)
        if len(sample) >= min(window, 20):
            q10, q50, q90 = sample.quantile([0.10, 0.50, 0.90]).tolist()
            result["historical_return_bands"][f"last_{window}_sessions"] = {
                "sample_size": int(len(sample)),
                "p10_return_pct": float(q10 * 100),
                "p50_return_pct": float(q50 * 100),
                "p90_return_pct": float(q90 * 100),
                "p10_price_scenario": float(latest_row["close"] * (1 + q10)),
                "p50_price_scenario": float(latest_row["close"] * (1 + q50)),
                "p90_price_scenario": float(latest_row["close"] * (1 + q90)),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="QQQM 下一交易日收益幅度回测（探索版）")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "QQQM_tiingo_history.csv")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "market_data.db")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-train-rows", type=int, default=300)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "forecast")
    args = parser.parse_args()

    try:
        summary, predictions, intervals, metadata, monthly = run_backtest(
            args.csv, args.train_fraction, args.minimum_train_rows, args.refit_every, args.ridge_alpha
        )
        latest = make_latest_forecast(args.csv, args.db, args.ridge_alpha)
    except Exception as exc:
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    paths = {
        "summary": args.output_dir / f"QQQM_return_forecast_summary_{stamp}.csv",
        "predictions": args.output_dir / f"QQQM_return_forecast_predictions_{stamp}.csv",
        "intervals": args.output_dir / f"QQQM_return_forecast_intervals_{stamp}.csv",
        "monthly": args.output_dir / f"QQQM_return_forecast_monthly_{stamp}.csv",
        "metadata": args.output_dir / f"QQQM_return_forecast_metadata_{stamp}.json",
        "latest": args.output_dir / f"QQQM_return_forecast_latest_{stamp}.json",
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    intervals.to_csv(paths["intervals"], index=False, encoding="utf-8-sig")
    monthly.to_csv(paths["monthly"], index=False, encoding="utf-8-sig")
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["latest"].write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== QQQM 下一交易日收益幅度回测 ==========")
    print(f"样本外区间：{metadata['test_start']} 至 {metadata['test_end']}；样本数：{metadata['out_of_sample_samples']}")
    print("\n点预测指标（MAE/RMSE 越低越好；收益方向命中率只作辅助）:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n历史经验区间覆盖率（目标 P10-P90 名义覆盖率 80%；不代表已校准概率区间）:")
    print(intervals.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n最新行情估计（基于最后可用收盘数据；不是盘中预测）:")
    print(json.dumps(latest, ensure_ascii=False, indent=2))
    print("\n输出文件：")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    print("注意：该脚本不修改现有模型或数据库；未计交易手续费，尚不能生成买卖结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
