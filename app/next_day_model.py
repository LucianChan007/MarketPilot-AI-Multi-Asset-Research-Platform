from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_2d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "close_to_sma5",
    "close_to_sma10",
    "close_to_sma20",
    "close_to_sma50",
    "volatility_5d",
    "volatility_20d",
    "rsi_14",
    "drawdown_20d",
    "intraday_range_pct",
    "overnight_gap_pct",
    "volume_change_1d",
]


@dataclass
class BacktestResult:
    summary: dict[str, Any]
    predictions: pd.DataFrame
    forecast: dict[str, Any]


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().casefold(): str(column) for column in frame.columns}


def _get_numeric(frame: pd.DataFrame, lookup: dict[str, str], candidates: tuple[str, ...]) -> pd.Series | None:
    for candidate in candidates:
        original = lookup.get(candidate.casefold())
        if original is not None:
            return pd.to_numeric(frame[original], errors="coerce")
    return None


def load_history(path: str | Path) -> pd.DataFrame:
    """Load a Tiingo-like daily CSV and normalize it into a date-indexed frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"历史行情文件不存在：{path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"历史行情文件为空：{path}")

    lookup = _column_lookup(raw)
    date_col = lookup.get("date") or lookup.get("trade_date") or lookup.get("datetime")
    if date_col is None:
        raise ValueError("行情 CSV 缺少 date/trade_date/datetime 日期字段。")

    dates = pd.to_datetime(raw[date_col], errors="coerce", utc=True)
    # Keep exchange session date rather than timezone-dependent local date.
    normalized_dates = dates.dt.strftime("%Y-%m-%d")
    close = _get_numeric(raw, lookup, ("adjclose", "adj_close", "close"))
    if close is None:
        raise ValueError("行情 CSV 缺少 adjClose/adj_close/close 收盘价字段。")

    frame = pd.DataFrame({"date": normalized_dates, "close": close})

    high = _get_numeric(raw, lookup, ("adjhigh", "adj_high", "high"))
    low = _get_numeric(raw, lookup, ("adjlow", "adj_low", "low"))
    open_price = _get_numeric(raw, lookup, ("adjopen", "adj_open", "open"))
    volume = _get_numeric(raw, lookup, ("adjvolume", "adj_volume", "volume"))

    frame["high"] = high if high is not None else np.nan
    frame["low"] = low if low is not None else np.nan
    frame["open"] = open_price if open_price is not None else np.nan
    frame["volume"] = volume if volume is not None else np.nan

    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["close"] > 0]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    if len(frame) < 260:
        raise ValueError(
            f"有效行情只有 {len(frame)} 条，至少需要 260 条才进行初步训练和时序验证。"
        )
    return frame


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Build features available after close on date t and target return for t+1."""
    required = {"date", "close", "high", "low", "open", "volume"}
    if not required.issubset(history.columns):
        raise ValueError(f"历史 DataFrame 必须包含字段：{sorted(required)}")

    frame = history.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    returns = close.pct_change()

    features = pd.DataFrame(index=frame.index)
    features["ret_1d"] = close.pct_change(1)
    features["ret_2d"] = close.pct_change(2)
    features["ret_5d"] = close.pct_change(5)
    features["ret_10d"] = close.pct_change(10)
    features["ret_20d"] = close.pct_change(20)

    for period in (5, 10, 20, 50):
        rolling_mean = close.rolling(period, min_periods=period).mean()
        features[f"close_to_sma{period}"] = close / rolling_mean - 1.0

    features["volatility_5d"] = returns.rolling(5, min_periods=5).std(ddof=1) * np.sqrt(252)
    features["volatility_20d"] = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)

    daily_change = close.diff()
    gain = daily_change.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-daily_change.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    features["rsi_14"] = rsi

    rolling_high = close.rolling(20, min_periods=20).max()
    features["drawdown_20d"] = close / rolling_high - 1.0

    open_price = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    features["intraday_range_pct"] = (high - low) / close
    features["overnight_gap_pct"] = open_price / close.shift(1) - 1.0
    features["volume_change_1d"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)

    for column in FEATURE_COLUMNS:
        if column not in features:
            features[column] = np.nan
    features = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    result = pd.concat([frame[["date", "close"]], features], axis=1)
    result["target_date"] = frame["date"].shift(-1)
    result["target_return"] = close.shift(-1) / close - 1.0
    result["target_up"] = np.where(
        result["target_return"].notna(),
        (result["target_return"] > 0).astype(float),
        np.nan,
    )
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


class NumpyLogisticRegression:
    """Small L2-regularized logistic regression implemented with NumPy."""

    def __init__(self, learning_rate: float = 0.08, iterations: int = 1200, l2: float = 0.03):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2 = l2
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.constant_probability_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(y) < 30:
            raise ValueError("训练数据形状无效，至少需要 30 条样本。")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("训练数据包含缺失或非有限值。")

        prevalence = float(np.mean(y))
        if len(np.unique(y)) < 2:
            self.constant_probability_ = float(np.clip(prevalence, 1e-4, 1 - 1e-4))
            self.mean_ = np.mean(x, axis=0)
            self.scale_ = np.std(x, axis=0)
            self.scale_[self.scale_ < 1e-8] = 1.0
            self.weights_ = np.zeros(x.shape[1] + 1, dtype=float)
            return self

        self.constant_probability_ = None
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        xs = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(xs)), xs])

        # Initialize intercept with the training prevalence.
        prevalence_safe = np.clip(prevalence, 1e-4, 1 - 1e-4)
        weights = np.zeros(design.shape[1], dtype=float)
        weights[0] = np.log(prevalence_safe / (1 - prevalence_safe))
        for _ in range(self.iterations):
            probabilities = _sigmoid(design @ weights)
            gradient = design.T @ (probabilities - y) / len(y)
            gradient[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * gradient
        self.weights_ = weights
        return self

    def predict_proba_up(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("模型尚未训练。")
        x = np.asarray(x, dtype=float)
        if self.constant_probability_ is not None:
            return np.full(x.shape[0], self.constant_probability_, dtype=float)
        xs = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(xs)), xs])
        return _sigmoid(design @ self.weights_)


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, float | int | None]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if len(actual) == 0:
        return {"n": 0, "accuracy": float("nan"), "balanced_accuracy": float("nan"), "brier_score": None, "log_loss": None}

    accuracy = float(np.mean(actual == predicted))
    recalls = []
    for label in (0, 1):
        mask = actual == label
        if mask.any():
            recalls.append(float(np.mean(predicted[mask] == label)))
    balanced = float(np.mean(recalls)) if recalls else float("nan")

    brier: float | None = None
    log_loss_value: float | None = None
    if probabilities is not None:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        brier = float(np.mean((p - actual) ** 2))
        log_loss_value = float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))

    return {
        "n": int(len(actual)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "brier_score": brier,
        "log_loss": log_loss_value,
    }


def run_walk_forward_backtest(
    history: pd.DataFrame,
    train_fraction: float = 0.70,
    minimum_train_rows: int = 300,
    refit_every: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate out-of-sample predictions with expanding-window retraining.

    At each refit boundary, only rows with target outcomes already known are used.
    Predictions within a block use the model trained at the start of that block.
    """
    if not 0.5 <= train_fraction < 0.9:
        raise ValueError("train_fraction 必须在 0.5（含）至 0.9（不含）之间。")
    if minimum_train_rows < 100 or refit_every < 1:
        raise ValueError("minimum_train_rows 至少 100，refit_every 至少 1。")

    engineered = build_feature_frame(history)
    labeled = engineered.dropna(subset=FEATURE_COLUMNS + ["target_up", "target_return", "target_date"]).copy()
    labeled = labeled.reset_index(drop=True)
    if len(labeled) < minimum_train_rows + 50:
        raise ValueError(
            f"可用训练样本仅 {len(labeled)} 条；至少需要 {minimum_train_rows + 50} 条。"
        )

    first_test = max(int(len(labeled) * train_fraction), minimum_train_rows)
    if first_test >= len(labeled) - 10:
        raise ValueError("留给样本外评估的数据太少，请增加历史数据或调整训练比例。")

    rows: list[dict[str, Any]] = []
    x_all = labeled[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_all = labeled["target_up"].to_numpy(dtype=int)
    returns_all = labeled["target_return"].to_numpy(dtype=float)

    for block_start in range(first_test, len(labeled), refit_every):
        block_end = min(block_start + refit_every, len(labeled))
        x_train = x_all[:block_start]
        y_train = y_all[:block_start]
        model = NumpyLogisticRegression().fit(x_train, y_train)
        probabilities = model.predict_proba_up(x_all[block_start:block_end])
        majority_probability = float(np.mean(y_train))
        majority_prediction = int(majority_probability >= 0.5)

        for offset, probability in enumerate(probabilities):
            index = block_start + offset
            actual = int(y_all[index])
            momentum_prediction = int(float(labeled.iloc[index]["ret_1d"]) > 0)
            rows.append({
                "feature_date": str(labeled.iloc[index]["date"]),
                "target_date": str(labeled.iloc[index]["target_date"]),
                "actual_return_pct": float(returns_all[index] * 100),
                "actual_direction": "上涨" if actual else "下跌或持平",
                "actual_up": actual,
                "model_probability_up_raw": float(probability),
                "model_prediction": int(probability >= 0.5),
                "majority_baseline_prediction": majority_prediction,
                "majority_baseline_probability_up": majority_probability,
                "previous_day_momentum_prediction": momentum_prediction,
            })

    predictions = pd.DataFrame(rows)
    actual = predictions["actual_up"].to_numpy(dtype=int)
    model_prob = predictions["model_probability_up_raw"].to_numpy(dtype=float)
    majority_prob = predictions["majority_baseline_probability_up"].to_numpy(dtype=float)
    model_metrics = _binary_metrics(actual, (model_prob >= 0.5).astype(int), model_prob)
    majority_metrics = _binary_metrics(actual, (majority_prob >= 0.5).astype(int), majority_prob)
    momentum_metrics = _binary_metrics(
        actual,
        predictions["previous_day_momentum_prediction"].to_numpy(dtype=int),
    )

    summary: dict[str, Any] = {
        "status": "completed",
        "ticker": "QQQM",
        "method": "NumPy L2 logistic regression; expanding-window walk-forward",
        "sample_count": int(len(engineered)),
        "usable_labeled_samples": int(len(labeled)),
        "first_test_index": int(first_test),
        "test_sample_count": int(len(predictions)),
        "test_start_date": str(predictions.iloc[0]["target_date"]),
        "test_end_date": str(predictions.iloc[-1]["target_date"]),
        "feature_date_start": str(labeled.iloc[0]["date"]),
        "feature_date_end": str(labeled.iloc[-1]["date"]),
        "features": FEATURE_COLUMNS,
        "refit_every_rows": int(refit_every),
        "model_metrics": model_metrics,
        "historical_majority_baseline_metrics": majority_metrics,
        "previous_day_momentum_baseline_metrics": momentum_metrics,
        "model_beats_majority_on_accuracy": bool(
            model_metrics["accuracy"] > majority_metrics["accuracy"]
        ),
        "caveat": (
            "样本外历史结果不保证未来表现；模型概率尚未校准，不能直接视为真实上涨概率。"
            "该基线未使用新闻，不能称为已验证的新闻融合模型。"
        ),
    }
    return predictions, summary


def load_recent_history_from_sqlite(
    db_path: str | Path,
    ticker: str = "QQQM",
    limit: int = 260,
) -> pd.DataFrame:
    """Load recent market_daily bars, scaling OHLC to adjusted-close basis.

    Historical training/evaluation stays on the Tiingo CSV. This separate recent
    frame is only used for today's features, so a stale CSV doesn't silently
    make the current forecast stale too. The newest DB row is not used as a
    labeled training sample until its following-session outcome is known.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"行情数据库不存在：{db_path}")
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT trade_date, open, high, low, close, adj_close, volume
            FROM market_daily
            WHERE UPPER(ticker) = UPPER(?)
            ORDER BY trade_date DESC
            LIMIT ?
            """,
            (ticker, int(limit)),
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError(f"数据库中没有 {ticker} 的行情记录。")
    frame = pd.DataFrame(
        rows,
        columns=["date", "raw_open", "raw_high", "raw_low", "raw_close", "adj_close", "volume"],
    ).sort_values("date").reset_index(drop=True)

    raw_close = pd.to_numeric(frame["raw_close"], errors="coerce")
    adjusted_close = pd.to_numeric(frame["adj_close"], errors="coerce").fillna(raw_close)
    scale = adjusted_close / raw_close.replace(0, np.nan)
    result = pd.DataFrame({
        "date": frame["date"].astype(str),
        "close": adjusted_close,
        "open": pd.to_numeric(frame["raw_open"], errors="coerce") * scale,
        "high": pd.to_numeric(frame["raw_high"], errors="coerce") * scale,
        "low": pd.to_numeric(frame["raw_low"], errors="coerce") * scale,
        "volume": pd.to_numeric(frame["volume"], errors="coerce"),
    })
    result = result.dropna(subset=["date", "close"]).reset_index(drop=True)
    if len(result) < 60:
        raise ValueError(f"数据库中 {ticker} 最近有效行情不足 60 条，不能稳定计算技术特征。")
    return result


def forecast_next_session(
    training_history: pd.DataFrame,
    feature_history: pd.DataFrame | None = None,
    feature_source: str = "Tiingo CSV",
) -> dict[str, Any]:
    """Train on historical CSV labels, optionally forecast from a newer feature frame."""
    engineered_training = build_feature_frame(training_history)
    labeled = engineered_training.dropna(subset=FEATURE_COLUMNS + ["target_up"]).copy()
    live_history = feature_history if feature_history is not None else training_history
    engineered_live = build_feature_frame(live_history)
    latest = engineered_live.dropna(subset=FEATURE_COLUMNS).sort_values("date").iloc[-1]
    if len(labeled) < 300:
        raise ValueError("可用于训练的已知标签少于 300 条，暂不生成预测。")

    model = NumpyLogisticRegression().fit(
        labeled[FEATURE_COLUMNS].to_numpy(dtype=float),
        labeled["target_up"].to_numpy(dtype=int),
    )
    probability = float(
        model.predict_proba_up(latest[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1))[0]
    )
    if probability >= 0.55:
        direction = "模型原始分数偏向上涨"
    elif probability <= 0.45:
        direction = "模型原始分数偏向下跌"
    else:
        direction = "方向不明确"

    return {
        "ticker": "QQQM",
        "as_of_date": str(latest["date"]),
        "latest_adjusted_close": float(latest["close"]),
        "feature_source": feature_source,
        "training_labels_known_through": str(labeled["target_date"].iloc[-1]),
        "next_session_directional_score_up_raw_uncalibrated": probability,
        "directional_label": direction,
        "model": "NumPy L2 logistic regression",
        "features_used": FEATURE_COLUMNS,
        "label_definition": "下一条日线的调整收盘价回报率 > 0 记为上涨，否则记为下跌或持平",
        "caveat": (
            "这是未经概率校准的探索性模型分数，并非真实概率保证或买卖建议。"
            "训练标签仅使用已知的历史 Tiingo CSV 结果；最新市场数据库只用于生成当前技术特征。"
            "需先检查样本外表现，并在更长时间与多个市场环境下持续验证。"
        ),
    }


def run_backtest_and_forecast(
    csv_path: str | Path,
    db_path: str | Path | None = None,
) -> BacktestResult:
    history = load_history(csv_path)
    predictions, summary = run_walk_forward_backtest(history)

    feature_history = history
    feature_source = "Tiingo CSV"
    if db_path is not None and Path(db_path).exists():
        try:
            candidate = load_recent_history_from_sqlite(db_path, ticker="QQQM")
            csv_latest = str(history["date"].iloc[-1])
            db_latest = str(candidate["date"].iloc[-1])
            if db_latest > csv_latest:
                feature_history = candidate
                feature_source = f"market_daily SQLite（更新至 {db_latest}）；训练标签仍来自 Tiingo CSV"
        except (sqlite3.Error, ValueError, FileNotFoundError):
            # Keep the independently usable historical CSV forecast, but identify its source.
            feature_history = history
            feature_source = "Tiingo CSV（未能读取更新的 SQLite 最新行情）"

    forecast = forecast_next_session(history, feature_history, feature_source)
    summary["historical_csv_last_date"] = str(history["date"].iloc[-1])
    summary["forecast_feature_date"] = forecast["as_of_date"]
    summary["forecast_feature_source"] = feature_source
    return BacktestResult(summary=summary, predictions=predictions, forecast=forecast)
