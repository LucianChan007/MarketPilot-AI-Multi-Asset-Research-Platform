from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.next_day_model import (
    FEATURE_COLUMNS,
    build_feature_frame,
    load_history,
    load_recent_history_from_sqlite,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = ROOT / "data" / "QQQM_tiingo_history.csv"
DEFAULT_DB_PATH = ROOT / "data" / "market_data.db"

RETURN_FEATURE_GROUPS: dict[str, list[str]] = {
    "ridge_full": list(FEATURE_COLUMNS),
    "ridge_returns_only": ["ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d"],
}


class NumpyRidgeRegression:
    """Small standardized Ridge regression; uses NumPy only."""

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
        self.weights = np.linalg.solve(
            x_std.T @ x_std + self.alpha * identity,
            x_std.T @ y_centered,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean_x is None or self.scale_x is None or self.mean_y is None or self.weights is None:
            raise RuntimeError("Ridge 模型尚未训练。")
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("预测特征包含缺失或非有限值。")
        return self.mean_y + ((x - self.mean_x) / self.scale_x) @ self.weights


def _normalize_history(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"date", "close", "open", "high", "low", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} 缺少必要字段：{sorted(missing)}")
    out = frame[list(required)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("close", "open", "high", "low", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close"]).copy()
    out = out[out["close"] > 0]
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{name} 没有有效行情记录。")
    return out


def merge_training_and_recent_history(
    training_history: pd.DataFrame,
    recent_history: pd.DataFrame,
    *,
    minimum_overlap: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append newer SQLite rows after aligning their adjusted-price level to CSV.

    Recent SQLite prices and long Tiingo history can differ by a small vendor
    adjustment factor. We estimate a robust multiplicative factor from overlap
    dates, use the CSV scale for training continuity, and append only dates newer
    than the CSV. All targets are still computed strictly from next-session
    returns in the merged, date-sorted history.
    """
    base = _normalize_history(training_history, "training_history")
    recent = _normalize_history(recent_history, "recent_history")
    base_max = str(base["date"].iloc[-1])
    recent_max = str(recent["date"].iloc[-1])
    overlap = base[["date", "close"]].merge(
        recent[["date", "close"]], on="date", how="inner", suffixes=("_csv", "_sqlite")
    )
    ratios = (
        overlap["close_csv"] / overlap["close_sqlite"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    ratios = ratios[ratios > 0]

    scale_ratio = 1.0
    if recent_max > base_max:
        if len(ratios) < minimum_overlap:
            raise ValueError(
                f"SQLite 行情比 Tiingo 更新，但只有 {len(ratios)} 个重叠交易日；"
                f"至少需要 {minimum_overlap} 个才能安全对齐价格口径。"
            )
        scale_ratio = float(ratios.median())
        if not np.isfinite(scale_ratio) or scale_ratio <= 0:
            raise ValueError("无法从重叠日期估算有效的跨数据源价格比例。")

    appended = recent[recent["date"] > base_max].copy()
    if not appended.empty:
        for column in ("close", "open", "high", "low"):
            appended[column] = appended[column] * scale_ratio
        merged = pd.concat([base, appended], ignore_index=True)
    else:
        merged = base.copy()

    merged = merged.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    metadata = {
        "csv_latest_date": base_max,
        "sqlite_latest_date": recent_max,
        "merged_latest_date": str(merged["date"].iloc[-1]),
        "csv_rows": int(len(base)),
        "sqlite_rows": int(len(recent)),
        "overlap_rows": int(len(overlap)),
        "appended_sqlite_rows": int(len(appended)),
        "sqlite_to_csv_price_scale_ratio": scale_ratio,
        "training_series_price_basis": "Tiingo adjusted-price level; newer SQLite bars aligned using overlap median ratio",
    }
    return merged, metadata


def build_daily_forecast(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    ticker: str = "QQQM",
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    """Fit candidate regressors on all known next-session outcomes and forecast the latest close."""
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    long_history = load_history(csv_path)
    recent_history = load_recent_history_from_sqlite(db_path, ticker=ticker, limit=260)
    merged, merge_metadata = merge_training_and_recent_history(long_history, recent_history)

    engineered = build_feature_frame(merged)
    labeled = engineered.dropna(
        subset=[*FEATURE_COLUMNS, "target_return", "target_date"]
    ).copy().reset_index(drop=True)
    live_rows = engineered.dropna(subset=FEATURE_COLUMNS).copy()
    if len(labeled) < 300:
        raise ValueError(f"可用于训练的已知收益标签只有 {len(labeled)} 条，至少需要 300 条。")
    if live_rows.empty:
        raise ValueError("没有可用于生成最新预测的有效特征行。")

    latest_feature = live_rows.iloc[-1]
    latest_date = str(latest_feature["date"])
    merged_latest_date = str(merged["date"].iloc[-1])
    if latest_date != merged_latest_date:
        raise ValueError("最新行情对应的特征无法计算，拒绝生成过期预测。")

    target = labeled["target_return"].to_numpy(dtype=float)
    x_latest_full = latest_feature[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
    forecasts: dict[str, float] = {
        "zero_return_baseline_pct": 0.0,
        "historical_mean_return_pct": float(np.mean(target) * 100.0),
    }

    for model_name, columns in RETURN_FEATURE_GROUPS.items():
        x_train = labeled[columns].to_numpy(dtype=float)
        x_latest = latest_feature[columns].to_numpy(dtype=float).reshape(1, -1)
        model = NumpyRidgeRegression(alpha=ridge_alpha).fit(x_train, target)
        forecasts[f"{model_name}_return_pct"] = float(model.predict(x_latest)[0] * 100.0)

    latest_prices = pd.to_numeric(merged["close"], errors="coerce")
    historical_returns = latest_prices.pct_change().dropna()
    # Prefer the unscaled SQLite adjusted close for the displayed current price.
    # The merged series is scaled only to avoid a vendor boundary jump in training.
    recent_last = recent_history.sort_values("date").iloc[-1]
    if str(recent_last["date"]) == latest_date:
        current_price = float(recent_last["close"])
        price_source = "SQLite market_daily adj_close basis; feature history aligned to Tiingo scale"
    else:
        current_price = float(merged.iloc[-1]["close"])
        price_source = "Tiingo adjusted CSV; SQLite has no newer session"

    bands: dict[str, Any] = {}
    for window in (20, 60, 252):
        sample = historical_returns.tail(window)
        minimum = min(window, 20)
        if len(sample) < minimum:
            continue
        q10, q50, q90 = [float(value) for value in sample.quantile([0.10, 0.50, 0.90]).tolist()]
        bands[f"last_{window}_sessions"] = {
            "sample_size": int(len(sample)),
            "p10_return_pct": q10 * 100.0,
            "p50_return_pct": q50 * 100.0,
            "p90_return_pct": q90 * 100.0,
            "p10_price_scenario": current_price * (1.0 + q10),
            "p50_price_scenario": current_price * (1.0 + q50),
            "p90_price_scenario": current_price * (1.0 + q90),
        }

    training_target_dates = labeled["target_date"].astype(str)
    return {
        "ticker": ticker.upper(),
        "as_of_date": latest_date,
        "latest_adjusted_close": current_price,
        "feature_source": price_source,
        "training_labels_known_through": str(training_target_dates.max()),
        "training_rows": int(len(labeled)),
        "training_history": merge_metadata,
        "point_forecasts": forecasts,
        "historical_return_bands": bands,
        "estimates_are_not_guarantees": True,
        "caveat": (
            "本次候选模型使用所有已知的历史下一交易日收益标签重新拟合；最新收盘日只用于构造预测特征，"
            "其下一交易日结果尚未知，不参与训练。历史经验区间未必达到名义覆盖率，不含交易成本，不构成买卖信号。"
        ),
    }
