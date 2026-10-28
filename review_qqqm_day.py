from __future__ import annotations

"""QQQM historical daily-review utility.

This script DOES NOT create a forecast for the next trading day.
It audits one completed session, reconstructs the original close-only Ridge
forecast if the saved forecast JSON is missing, compares close-only and
candlestick-feature Ridge models with a walk-forward test, downloads optional
5-minute bars for an actual intraday chart, and updates a daily scorecard.

Run from the QQQM investment assistant project root:
    .\\.venv\\Scripts\\python.exe .\\review_qqqm_day.py --target-date 2026-10-19

Required local project modules:
    app.db.load_history
    app.indicators.add_indicators

Optional for full intraday chart:
    pip install yfinance matplotlib
"""

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "sma20_gap", "sma50_gap", "volatility20", "rsi14_scaled", "drawdown_1y",
]
CANDLE_FEATURES = [
    "candle_body_pct", "candle_range_pct", "upper_shadow_pct", "lower_shadow_pct",
    "close_location", "open_gap_pct", "volume_ratio20", "bullish_candle",
]


def configure_matplotlib_chinese(plt: Any) -> None:
    """Select an installed CJK font when available, without requiring font installation."""
    try:
        from matplotlib import font_manager
        installed = {font.name for font in font_manager.fontManager.ttflist}
        preferred = ["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        selected = next((name for name in preferred if name in installed), "DejaVu Sans")
        plt.rcParams["font.family"] = selected
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        plt.rcParams["axes.unicode_minus"] = False


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def normalize_daily_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    df = history.copy()
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_close_features(history: pd.DataFrame, add_indicators: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    df = history.copy().sort_index()
    close = pd.to_numeric(df["close"], errors="coerce")
    ind = add_indicators(df)
    daily_return = close.pct_change()
    features = pd.DataFrame(index=df.index)
    for days in (1, 2, 3, 5, 10, 20):
        features[f"ret_{days}d"] = close.pct_change(days) * 100.0
    features["sma20_gap"] = close / ind["sma_20"] - 1.0
    features["sma50_gap"] = close / ind["sma_50"] - 1.0
    features["volatility20"] = daily_return.rolling(20, min_periods=20).std() * math.sqrt(252)
    features["rsi14_scaled"] = ind["rsi_14"] / 100.0
    features["drawdown_1y"] = ind["drawdown_from_1y_high"]
    return features[BASE_FEATURES].replace([np.inf, -np.inf], np.nan)


def make_candle_features(history: pd.DataFrame) -> pd.DataFrame:
    """Candlestick features available after each daily candle has closed.

    The features on date t are used only to predict a later session, never the
    same day's close. Percent denominators use t-1 close where appropriate.
    """
    df = history.copy().sort_index()
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise RuntimeError(f"本地行情缺少 K 线所需字段：{missing}。请确认数据库有 open/high/low/close。")

    op = pd.to_numeric(df["open"], errors="coerce")
    hi = pd.to_numeric(df["high"], errors="coerce")
    lo = pd.to_numeric(df["low"], errors="coerce")
    cl = pd.to_numeric(df["close"], errors="coerce")
    prev_close = cl.shift(1).replace(0, np.nan)
    candle_range = (hi - lo).replace(0, np.nan)
    body_top = pd.concat([op, cl], axis=1).max(axis=1)
    body_bottom = pd.concat([op, cl], axis=1).min(axis=1)

    out = pd.DataFrame(index=df.index)
    out["candle_body_pct"] = (cl - op) / prev_close * 100.0
    out["candle_range_pct"] = (hi - lo) / prev_close * 100.0
    out["upper_shadow_pct"] = (hi - body_top) / prev_close * 100.0
    out["lower_shadow_pct"] = (body_bottom - lo) / prev_close * 100.0
    out["close_location"] = (cl - lo) / candle_range
    out["open_gap_pct"] = (op / prev_close - 1.0) * 100.0
    out["bullish_candle"] = (cl > op).astype(float)
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        vol_mean = vol.rolling(20, min_periods=10).mean().replace(0, np.nan)
        out["volume_ratio20"] = vol / vol_mean
    else:
        out["volume_ratio20"] = np.nan
    return out[CANDLE_FEATURES].replace([np.inf, -np.inf], np.nan)


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    x_train = (train_x - mean) / scale
    x_pred = (predict_x - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_pred = np.column_stack([np.ones(len(x_pred)), x_pred])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(x_train.T @ x_train + penalty, x_train.T @ train_y, rcond=None)[0]
    return x_pred @ coef


def supervised_frame(features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    # Feature row t predicts close-to-close return from close(t) to close(t+1).
    target = close.pct_change().shift(-1) * 100.0
    return features.join(target.rename("target_return_pct")).dropna(subset=list(features.columns) + ["target_return_pct"])


def metric_pack(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if len(actual) == 0:
        return {"n": 0, "mae_pct_points": None, "rmse_pct_points": None, "direction_accuracy": None}
    err = predicted - actual
    nonzero = np.abs(actual) > 1e-12
    direction = float(np.mean(np.sign(predicted[nonzero]) == np.sign(actual[nonzero]))) if nonzero.any() else None
    return {
        "n": int(len(actual)),
        "mae_pct_points": float(np.mean(np.abs(err))),
        "rmse_pct_points": float(np.sqrt(np.mean(err ** 2))),
        "direction_accuracy": direction,
    }


def walk_forward_compare(supervised: pd.DataFrame, feature_sets: dict[str, list[str]], min_train: int = 100, max_test: int = 60) -> tuple[dict[str, Any], pd.DataFrame]:
    available = len(supervised)
    test_count = min(max_test, max(0, available - min_train))
    if test_count < 20:
        return ({"warning": f"可用样本不足：supervised={available}，至少需要约 {min_train+20} 条有效样本。", "models": {}}, pd.DataFrame())
    start = available - test_count
    results: list[dict[str, Any]] = []
    for j in range(start, available):
        train = supervised.iloc[:j]
        test = supervised.iloc[j]
        actual = float(test["target_return_pct"])
        row: dict[str, Any] = {
            "signal_date": test.name.date().isoformat(),
            "actual_next_day_return_pct": actual,
            "zero_return_baseline_pct": 0.0,
        }
        for model_name, cols in feature_sets.items():
            pred = float(ridge_predict(
                train[cols].to_numpy(dtype=float),
                train["target_return_pct"].to_numpy(dtype=float),
                test[cols].to_numpy(dtype=float).reshape(1, -1),
            )[0])
            row[f"{model_name}_predicted_return_pct"] = pred
        results.append(row)

    detail = pd.DataFrame(results)
    actual = detail["actual_next_day_return_pct"].to_numpy(dtype=float)
    models: dict[str, Any] = {"zero_return_baseline": metric_pack(actual, np.zeros(len(actual)))}
    for name in feature_sets:
        models[name] = metric_pack(actual, detail[f"{name}_predicted_return_pct"].to_numpy(dtype=float))
        baseline_mae = models["zero_return_baseline"]["mae_pct_points"]
        mae = models[name]["mae_pct_points"]
        models[name]["beats_baseline_by_5pct"] = bool(baseline_mae is not None and mae is not None and mae < baseline_mae * 0.95)
    return ({"test_window": "latest known next-day outcomes only; chronological walk-forward", "min_train": min_train, "test_count": test_count, "models": models}, detail)


def reconstruct_original_close_forecast(history_asof: pd.DataFrame, add_indicators: Callable[[pd.DataFrame], pd.DataFrame]) -> dict[str, Any]:
    """Recreate the original close-only script's decision as-of the last history day.

    It cannot prove what was actually published; the output is explicitly marked
    as reconstructed if the saved forecast JSON is absent.
    """
    close = history_asof["close"].astype(float)
    features = make_close_features(history_asof, add_indicators)
    supervised = supervised_frame(features, close)
    latest_x = features.iloc[[-1]][BASE_FEATURES]
    if latest_x.isna().any(axis=None):
        raise RuntimeError("用于历史回放的最新特征不完整，无法重建原始收盘预测。")

    eval_summary, _ = walk_forward_compare(supervised, {"ridge_close_only": BASE_FEATURES})
    models = eval_summary.get("models", {})
    ridge_metrics = models.get("ridge_close_only", {})
    baseline_metrics = models.get("zero_return_baseline", {})
    model_wins = bool(ridge_metrics.get("beats_baseline_by_5pct", False))
    if len(supervised) >= 100:
        raw_pct = float(ridge_predict(
            supervised[BASE_FEATURES].to_numpy(dtype=float),
            supervised["target_return_pct"].to_numpy(dtype=float),
            latest_x.to_numpy(dtype=float),
        )[0])
    else:
        raw_pct = 0.0
        model_wins = False
    chosen_pct = raw_pct if model_wins else 0.0
    last_close = float(close.iloc[-1])
    return {
        "reconstruction_only": True,
        "model_used_reconstructed": "Ridge" if model_wins else "zero_return_baseline",
        "asof_date": history_asof.index[-1].date().isoformat(),
        "predicted_close_estimate_reconstructed": round(last_close * (1.0 + chosen_pct / 100.0), 4),
        "predicted_return_pct_reconstructed": round(chosen_pct, 4),
        "ridge_raw_return_estimate_pct_reconstructed": round(raw_pct, 4),
        "ridge_walk_forward_metrics": ridge_metrics,
        "zero_return_baseline_metrics": baseline_metrics,
        "note": "这是从当前数据库按历史时点重建的模型结果，不等同于原始发布快照；数据源修订、代码差异都可能造成细微差异。",
    }


def try_get_yfinance_daily(target: date) -> pd.Series | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    start = target.isoformat()
    end = (target + timedelta(days=1)).isoformat()
    try:
        bars = yf.download("QQQM", start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    except Exception:
        return None
    if bars is None or bars.empty:
        return None
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    r = bars.iloc[0]
    return pd.Series({k.lower(): float(r[k]) for k in ("Open", "High", "Low", "Close", "Volume") if k in r.index})


def try_get_yfinance_intraday(target: date) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    start = target.isoformat()
    end = (target + timedelta(days=1)).isoformat()
    try:
        try:
            bars = yf.download("QQQM", start=start, end=end, interval="5m", prepost=False, auto_adjust=False, progress=False, multi_level_index=False)
        except TypeError:
            bars = yf.download("QQQM", start=start, end=end, interval="5m", prepost=False, auto_adjust=False, progress=False)
    except Exception as exc:
        print(f"[提示] 无法从 Yahoo Finance 获取 5 分钟数据：{exc}", file=sys.stderr)
        return pd.DataFrame()
    if bars is None or bars.empty:
        return pd.DataFrame()
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    idx = pd.to_datetime(bars.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    bars.index = idx
    bars = bars[~bars.index.isna()].sort_index()
    bars = bars[bars.index.date == target]
    # Regular US session only, using New York local clock.
    bars = bars.between_time("09:30", "16:00")
    for col in ("open", "high", "low", "close", "volume"):
        # yfinance columns typically remain title-cased
        source_col = col.title()
        if source_col in bars.columns:
            bars[col] = pd.to_numeric(bars[source_col], errors="coerce")
        elif col in bars.columns:
            bars[col] = pd.to_numeric(bars[col], errors="coerce")
    needed = [c for c in ("open", "high", "low", "close", "volume") if c in bars.columns]
    return bars[needed].dropna(subset=["close"])


def read_eodhd_token() -> str | None:
    """Read an EODHD token without printing it. Environment variables take precedence."""
    for key in ("EODHD_API_KEY", "EODHD_API_TOKEN", "EODHD_TOKEN"):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip().strip('"').strip("'")
    # Look for the project's .env in the current directory and its parents.
    for base in (Path.cwd(), *Path.cwd().parents):
        env_path = base / ".env"
        if not env_path.is_file():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() in ("EODHD_API_KEY", "EODHD_API_TOKEN", "EODHD_TOKEN") and value.strip():
                    return value.strip().strip('"').strip("'")
        except OSError:
            continue
    return None


def try_get_eodhd_intraday(target: date) -> pd.DataFrame:
    """Fetch 5-minute bars using the project's EODHD token, if it has intraday access."""
    token = read_eodhd_token()
    if not token:
        return pd.DataFrame()
    ny = ZoneInfo("America/New_York")
    start_dt = datetime.combine(target, datetime.min.time(), tzinfo=ny).replace(hour=9, minute=30)
    end_dt = datetime.combine(target, datetime.min.time(), tzinfo=ny).replace(hour=16, minute=5)
    params = urllib.parse.urlencode({
        "api_token": token,
        "interval": "5m",
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
        "fmt": "json",
    })
    url = f"https://eodhd.com/api/intraday/QQQM.US?{params}"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "QQQM-Investment-Assistant/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[提示] EODHD 5 分钟数据不可用（网络、权限或额度问题）：{exc}", file=sys.stderr)
        return pd.DataFrame()
    if not isinstance(payload, list) or not payload:
        if isinstance(payload, dict) and payload.get("message"):
            print(f"[提示] EODHD 返回信息：{payload.get('message')}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    dt_col = "datetime" if "datetime" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if dt_col is None:
        print("[提示] EODHD 响应缺少 datetime/timestamp 字段。", file=sys.stderr)
        return pd.DataFrame()
    if dt_col == "timestamp" and pd.api.types.is_numeric_dtype(df[dt_col]):
        idx = pd.to_datetime(df[dt_col], unit="s", utc=True, errors="coerce").tz_convert("America/New_York").tz_localize(None)
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(df[dt_col], errors="coerce"))
        if idx.tz is None:
            idx = idx.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").tz_localize(None)
        else:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
    df.index = idx
    df = df[~df.index.isna()].sort_index()
    df = df[df.index.date == target].between_time("09:30", "16:00")
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols].dropna(subset=["close"]) if "close" in cols else pd.DataFrame()


def try_get_intraday(target: date, provider: str = "auto") -> pd.DataFrame:
    """Prefer Yahoo 5m data, then use EODHD if configured; never synthesize missing bars."""
    if provider in ("auto", "yfinance"):
        bars = try_get_yfinance_intraday(target)
        if not bars.empty:
            print(f"[盘中数据] 使用 Yahoo Finance 5 分钟行情，共 {len(bars)} 根。")
            return bars
        if provider == "yfinance":
            return pd.DataFrame()
    if provider in ("auto", "eodhd"):
        bars = try_get_eodhd_intraday(target)
        if not bars.empty:
            print(f"[盘中数据] 使用 EODHD 5 分钟行情，共 {len(bars)} 根。")
            return bars
    return pd.DataFrame()


def build_candle_descriptive_row(row: pd.Series, previous_close: float) -> dict[str, Any]:
    op, hi, lo, cl = (float(row[k]) for k in ("open", "high", "low", "close"))
    span = hi - lo
    upper = hi - max(op, cl)
    lower = min(op, cl) - lo
    body = abs(cl - op)
    return {
        "open": op, "high": hi, "low": lo, "close": cl,
        "previous_close": previous_close,
        "gap_pct_vs_previous_close": (op / previous_close - 1.0) * 100.0 if previous_close else None,
        "day_change_pct_vs_previous_close": (cl / previous_close - 1.0) * 100.0 if previous_close else None,
        "body_signed_usd_close_minus_open": cl - op,
        "body_abs_usd": body,
        "upper_shadow_usd": upper,
        "lower_shadow_usd": lower,
        "full_range_usd": span,
        "body_fraction_of_day_range": body / span if span else None,
        "upper_shadow_fraction_of_day_range": upper / span if span else None,
        "lower_shadow_fraction_of_day_range": lower / span if span else None,
        "descriptive_note": "向上跳空后盘中冲高回落，收盘低于开盘但高于前收；单根 K 线不构成可靠方向预测。",
    }


def plot_candle_history(history: pd.DataFrame, target: date, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        configure_matplotlib_chinese(plt)
    except ImportError as exc:
        raise RuntimeError("需要 matplotlib：.\\.venv\\Scripts\\python.exe -m pip install matplotlib") from exc

    data = history.dropna(subset=["open", "high", "low", "close"]).tail(35).copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    fig.patch.set_facecolor("#F5F7FB")
    ax.set_facecolor("white")
    x = np.arange(len(data))
    width = 0.62
    for i, (_, r) in enumerate(data.iterrows()):
        op, hi, lo, cl = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        up = cl >= op
        color = "#159A75" if up else "#D94F58"  # US-market convention: green up, red down
        ax.vlines(i, lo, hi, color=color, linewidth=1.2, zorder=2)
        body_low = min(op, cl)
        body_h = max(abs(cl - op), 0.015)
        ax.add_patch(Rectangle((i - width / 2, body_low), width, body_h, facecolor=color, edgecolor=color, linewidth=0.8, zorder=3))
    target_idx = [i for i, d in enumerate(data.index) if d.date() == target]
    if target_idx:
        i = target_idx[0]
        ax.axvspan(i - 0.55, i + 0.55, color="#F2B84B", alpha=0.15, zorder=1)
        rr = data.iloc[i]
        ax.annotate(f"10/19  O {rr['open']:.2f}  H {rr['high']:.2f}\nL {rr['low']:.2f}  C {rr['close']:.2f}",
                    (i, float(rr["high"])), xytext=(12, 20), textcoords="offset points", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#FFF8E9", ec="#E8C477"))
    labels = [d.strftime("%m/%d") for d in data.index]
    stride = max(1, len(labels) // 9)
    ax.set_xticks(x[::stride])
    ax.set_xticklabels(labels[::stride], rotation=0)
    ax.set_title(f"QQQM 日 K 线复盘（标记 {target.isoformat()}）", loc="left", fontsize=17, fontweight="bold", pad=16, color="#1B2D45")
    ax.set_ylabel("价格（美元）")
    ax.grid(axis="y", color="#E7ECF3", linewidth=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D8E0EA"); ax.spines["bottom"].set_color("#D8E0EA")
    ax.text(0.01, -0.13, "美股图表惯例：绿涨红跌。K 线是每个时间区间的开高低收摘要，不等于区间内完整价格路径。",
            transform=ax.transAxes, fontsize=9, color="#68778C")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_intraday(bars: pd.DataFrame, target: date, previous_close: float, prediction: dict[str, Any] | None, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        configure_matplotlib_chinese(plt)
    except ImportError as exc:
        raise RuntimeError("需要 matplotlib：.\\.venv\\Scripts\\python.exe -m pip install matplotlib") from exc
    if bars.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    fig.patch.set_facecolor("#F5F7FB"); ax.set_facecolor("white")
    x = bars.index
    if {"low", "high"}.issubset(bars.columns):
        ax.fill_between(x, bars["low"].astype(float).to_numpy(), bars["high"].astype(float).to_numpy(), color="#4E86D8", alpha=0.13, label="5 分钟 K 线区间")
    ax.plot(x, bars["close"].astype(float), color="#2867B2", linewidth=1.8, label="实际 5 分钟收盘价")
    ax.axhline(previous_close, color="#7D8898", linestyle="--", linewidth=1.1, label=f"前收 ${previous_close:.2f}")
    forecast_block = (prediction or {}).get("intraday_forecast", {}) if isinstance(prediction, dict) else {}
    pred_close = None
    source = prediction or {}
    for key in ("predicted_close_estimate", "predicted_close", "forecast_close", "predicted_close_estimate_reconstructed"):
        if isinstance(source, dict) and source.get(key) is not None:
            pred_close = float(source[key]); break
    if pred_close is None and forecast_block.get("estimated_close") is not None:
        pred_close = float(forecast_block["estimated_close"])
    # Overlay the original saved path only; never construct a new forecast during review.
    path = forecast_block.get("path", []) if isinstance(forecast_block, dict) else []
    if path:
        times, prices, lows, highs = [], [], [], []
        for point in path:
            t = point.get("time_et")
            p = point.get("price_estimate")
            if not t or p is None:
                continue
            try:
                ts = pd.Timestamp(f"{target.isoformat()} {t}")
                ts = ts.tz_localize("America/New_York").tz_localize(None)
            except Exception:
                continue
            times.append(ts); prices.append(float(p))
            lows.append(float(point.get("band_low_estimate", p)))
            highs.append(float(point.get("band_high_estimate", p)))
        if times:
            ax.fill_between(times, np.asarray(lows), np.asarray(highs), color="#E6A23C", alpha=0.12, label="原始预测参考范围")
            ax.plot(times, prices, color="#D88B20", linestyle="--", marker="o", markersize=3.5, linewidth=1.6, label="原始预测盘中路径")
    elif pred_close is not None:
        ax.axhline(pred_close, color="#D88B20", linestyle="--", linewidth=1.2, label=f"收盘预测参考 ${pred_close:.2f}")
    ax.set_title(f"QQQM {target.isoformat()} 实际盘中曲线与原始预测对照（5 分钟）", loc="left", fontsize=17, fontweight="bold", pad=14, color="#1B2D45")
    ax.set_ylabel("价格（美元）"); ax.set_xlabel("美东时间")
    ax.grid(axis="y", color="#E7ECF3", linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D8E0EA"); ax.spines["bottom"].set_color("#D8E0EA")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_actual_vs_forecast_candles(pred_ohlc: dict[str, float | None], actual_ohlc: dict[str, float], previous_close: float, target: date, out_path: Path) -> None:
    """One actual candle beside the four saved forecast references, not a new forecast."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        configure_matplotlib_chinese(plt)
    except ImportError as exc:
        raise RuntimeError(r"需要 matplotlib：.\.venv\Scripts\python.exe -m pip install matplotlib") from exc
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    fig.patch.set_facecolor("#F5F7FB"); ax.set_facecolor("white")
    items = [(0, actual_ohlc, "实际行情", "#159A75" if actual_ohlc["close"] >= actual_ohlc["open"] else "#D94F58")]
    forecast_valid = all(pred_ohlc.get(k) is not None for k in ("open", "high", "low", "close"))
    if forecast_valid:
        p = {k: float(pred_ohlc[k]) for k in ("open", "high", "low", "close")}
        items.append((1, p, "原始预测参考", "#D88B20"))
    ax.axhline(previous_close, color="#7D8898", linestyle="--", linewidth=1.1, label=f"前收 ${previous_close:.2f}")
    for xpos, row, label, color in items:
        op, hi, lo, cl = (float(row[k]) for k in ("open", "high", "low", "close"))
        ax.vlines(xpos, lo, hi, color=color, linewidth=2.0, zorder=2)
        body_bottom = min(op, cl); body_height = max(abs(cl-op), 0.015)
        ax.add_patch(Rectangle((xpos-0.18, body_bottom), 0.36, body_height, facecolor=color, edgecolor=color, alpha=0.78, zorder=3))
        ax.text(xpos, hi+0.16, f"{label}\nO {op:.2f}  H {hi:.2f}\nL {lo:.2f}  C {cl:.2f}", ha="center", va="bottom", fontsize=9, color="#24354A")
    ax.set_xlim(-0.55, 1.55 if forecast_valid else 0.55)
    ax.set_xticks([0,1] if forecast_valid else [0]); ax.set_xticklabels(["实际 K 线", "预测 K 线参考"] if forecast_valid else ["实际 K 线"])
    ax.set_ylabel("价格（美元）")
    ax.set_title(f"QQQM {target.isoformat()}：原始预测与实际日 K 线对照", loc="left", fontsize=15, fontweight="bold", color="#1B2D45", pad=18)
    ax.grid(axis="y", color="#E7ECF3", linewidth=0.8); ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.01, -0.16, "注意：预测 H/L 是原始快照中的参考值，不等同于经过校准的极值概率区间。", transform=ax.transAxes, fontsize=8.5, color="#68778C")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)


def pick_prediction_value(snapshot: dict[str, Any], keys: list[str], nested_keys: list[str] | None = None) -> float | None:
    """Read top-level snapshot values first, then the nested intraday_forecast block."""
    candidates = [(snapshot, keys)]
    nested = snapshot.get("intraday_forecast", {}) if isinstance(snapshot, dict) else {}
    if isinstance(nested, dict):
        candidates.append((nested, nested_keys or keys))
    for container, candidate_keys in candidates:
        for key in candidate_keys:
            value = container.get(key) if isinstance(container, dict) else None
            try:
                if value is not None and np.isfinite(float(value)):
                    return float(value)
            except (TypeError, ValueError):
                continue
    return None


def upsert_scorecard(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()
        if not old.empty and "target_date" in old.columns:
            old = old[old["target_date"].astype(str) != str(row["target_date"])]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="复盘已完成的 QQQM 交易日；不会生成下一交易日预测")
    parser.add_argument("--target-date", default="2026-10-19", help="要复盘的已完成交易日，例如 2026-10-19")
    parser.add_argument("--reports-dir", default="reports", help="项目 reports 目录")
    parser.add_argument("--forecast-file", default=None, help="原始预测 JSON 路径；不填时查找 reports 和当前目录")
    parser.add_argument("--intraday-provider", choices=("auto", "yfinance", "eodhd"), default="auto", help="5 分钟数据来源；auto 先尝试 Yahoo，再尝试 .env 中的 EODHD token")
    parser.add_argument("--no-intraday-download", action="store_true", help="不请求 5 分钟历史数据")
    args = parser.parse_args()

    try:
        target = date.fromisoformat(args.target_date)
    except ValueError:
        print("[失败] --target-date 格式须为 YYYY-MM-DD", file=sys.stderr); return 2

    try:
        from app.db import load_history
        from app.indicators import add_indicators
    except Exception as exc:
        print(f"[失败] 请在 QQQM 项目根目录运行，无法导入 app.db / app.indicators：{exc}", file=sys.stderr)
        return 2

    try:
        history = normalize_daily_history(load_history("QQQM"))
    except Exception as exc:
        print(f"[失败] 读取本地 QQQM 数据库失败：{exc}", file=sys.stderr); return 2
    if history.empty:
        print("[失败] 本地数据库没有 QQQM 行情，请先同步至少 1 年日线。", file=sys.stderr); return 2
    if not {"close"}.issubset(history.columns):
        print("[失败] 数据库缺少 close 列。", file=sys.stderr); return 2

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    review_dir = reports / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = target.strftime("%Y%m%d")
    expected_snapshot_name = f"QQQM_forecast_{stamp}.json"
    candidate_paths = [Path(args.forecast_file)] if args.forecast_file else [reports / expected_snapshot_name, Path.cwd() / expected_snapshot_name]
    # Also accept the uploaded source snapshot name for this one historical review.
    if not args.forecast_file and target.isoformat() == "2026-10-19":
        candidate_paths.extend([Path.cwd() / "QQQM_forecast_20261019.json", Path.cwd().parent / "QQQM_forecast_20261019.json"])
    original_json_path = next((p.resolve() for p in candidate_paths if p.is_file()), reports / expected_snapshot_name)
    original_snapshot: dict[str, Any] = {}
    snapshot_found = False
    if original_json_path.exists():
        try:
            original_snapshot = json.loads(original_json_path.read_text(encoding="utf-8"))
            snapshot_found = isinstance(original_snapshot, dict) and bool(original_snapshot)
        except Exception as exc:
            print(f"[提示] 预测 JSON 存在但无法解析，将尝试重建旧模型结果：{exc}", file=sys.stderr)

    # Find/obtain actual daily OHLC. Prefer the project's own data source to avoid mixing sources.
    actual_row: pd.Series | None = history.loc[target.isoformat()] if target.isoformat() in history.index.astype(str) else None
    actual_source = "local_database"
    required_ohlc = ("open", "high", "low", "close")
    actual_is_complete = actual_row is not None and all(k in actual_row.index and pd.notna(actual_row[k]) for k in required_ohlc)
    if not actual_is_complete:
        yf_actual = try_get_yfinance_daily(target)
        if yf_actual is not None and all(k in yf_actual.index and pd.notna(yf_actual[k]) for k in required_ohlc):
            actual_row = yf_actual
            actual_source = "Yahoo Finance yfinance fallback"
            print("[提示] 本地数据库缺少目标日完整 OHLC，复盘实际行情使用 Yahoo Finance 回退数据；下次建议统一行情源。", file=sys.stderr)
        else:
            print(f"[失败] 本地数据库没有 {target} 的完整 OHLC 数据，且无法获取 Yahoo 日线。请先同步行情后重试。", file=sys.stderr)
            return 2

    prior_history = history[history.index.date < target].copy()
    if prior_history.empty:
        print("[失败] 找不到目标交易日前的历史数据。", file=sys.stderr); return 2
    previous_close = float(prior_history["close"].iloc[-1])

    # Exact saved snapshot wins. If absent, reconstruct the old close-only rule on data known before target.
    pred_close = pick_prediction_value(original_snapshot, ["predicted_close_estimate", "predicted_close", "forecast_close"], ["estimated_close", "predicted_close_estimate", "forecast_close"])
    if snapshot_found and pred_close is not None:
        prediction_source = "saved_original_snapshot"
        reconstructed = None
    else:
        reconstructed = reconstruct_original_close_forecast(prior_history, add_indicators)
        pred_close = float(reconstructed["predicted_close_estimate_reconstructed"])
        prediction_source = "reconstructed_not_original_snapshot"
        print("[提示] 未找到可读取的原始预测 JSON；收盘预测值是按目标日前数据重建的，不可冒充当时实际发布值。", file=sys.stderr)

    actual = {k: float(actual_row[k]) for k in ("open", "high", "low", "close") if k in actual_row.index and pd.notna(actual_row[k])}
    pred_open = pick_prediction_value(original_snapshot, ["predicted_open_estimate", "predicted_open", "forecast_open"], ["estimated_open", "predicted_open_estimate", "forecast_open"])
    pred_high = pick_prediction_value(original_snapshot, ["predicted_high_estimate", "predicted_high", "forecast_high"], ["estimated_high_reference", "estimated_high", "predicted_high_estimate", "forecast_high"])
    pred_low = pick_prediction_value(original_snapshot, ["predicted_low_estimate", "predicted_low", "forecast_low"], ["estimated_low_reference", "estimated_low", "predicted_low_estimate", "forecast_low"])

    # Build close-only versus candlestick-enriched model comparison using data that existed before target date.
    base_features = make_close_features(prior_history, add_indicators)
    candle_features = make_candle_features(prior_history)
    combined = base_features.join(candle_features)
    models: dict[str, Any] = {}
    details: dict[str, pd.DataFrame] = {}
    for model_name, feat, columns in (
        ("ridge_close_only", base_features, BASE_FEATURES),
        ("ridge_plus_candles", combined, BASE_FEATURES + CANDLE_FEATURES),
    ):
        sup = supervised_frame(feat, prior_history["close"].astype(float))
        eval_result, eval_detail = walk_forward_compare(sup, {model_name: columns})
        models[model_name] = eval_result
        details[model_name] = eval_detail

    comparison_rows: list[dict[str, Any]] = []
    for model_name, result in models.items():
        model_metrics = result.get("models", {})
        comparison_rows.append({
            "target_date": target.isoformat(),
            "evaluation_window": result.get("test_window"),
            "model": model_name,
            "test_count": result.get("test_count"),
            **model_metrics.get(model_name, {}),
            "baseline_mae_pct_points": model_metrics.get("zero_return_baseline", {}).get("mae_pct_points"),
            "baseline_rmse_pct_points": model_metrics.get("zero_return_baseline", {}).get("rmse_pct_points"),
        })
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(review_dir / f"QQQM_model_comparison_{stamp}.csv", index=False, encoding="utf-8-sig")
    model_history_path = reports / "QQQM_model_comparison_history.csv"
    if model_history_path.exists():
        try:
            old_models = pd.read_csv(model_history_path)
        except Exception:
            old_models = pd.DataFrame()
        if not old_models.empty and {"target_date", "model"}.issubset(old_models.columns):
            old_models = old_models[~((old_models["target_date"].astype(str) == target.isoformat()) & old_models["model"].astype(str).isin(comparison_frame["model"].astype(str)))]
            comparison_frame = pd.concat([old_models, comparison_frame], ignore_index=True)
    comparison_frame.to_csv(model_history_path, index=False, encoding="utf-8-sig")
    # Combine per-window predictions in an easy-to-inspect detail table.
    detail_frames = [df.assign(model=model_name) for model_name, df in details.items() if not df.empty]
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(review_dir / f"QQQM_walk_forward_detail_{stamp}.csv", index=False, encoding="utf-8-sig")

    candle_summary = build_candle_descriptive_row(actual_row, previous_close)
    history_with_actual = history.copy()
    if pd.Timestamp(target) not in history_with_actual.index:
        add_row = pd.DataFrame([actual_row.to_dict()], index=pd.DatetimeIndex([pd.Timestamp(target)]))
        history_with_actual = pd.concat([history_with_actual, add_row]).sort_index()
    candle_features_all = make_candle_features(history_with_actual)
    candle_features_all.to_csv(review_dir / f"QQQM_candle_features_history_{stamp}.csv", index_label="date", encoding="utf-8-sig")
    plot_candle_history(history_with_actual[history_with_actual.index <= pd.Timestamp(target)], target, review_dir / f"QQQM_daily_candles_{stamp}.png")
    prediction_ohlc = {"open": pred_open, "high": pred_high, "low": pred_low, "close": pred_close}
    actual_ohlc_for_plot = {k: float(actual_row[k]) for k in ("open", "high", "low", "close") if k in actual_row.index and pd.notna(actual_row[k])}
    if all(k in actual_ohlc_for_plot for k in ("open", "high", "low", "close")):
        plot_actual_vs_forecast_candles(prediction_ohlc, actual_ohlc_for_plot, previous_close, target, review_dir / f"QQQM_prediction_vs_actual_candle_{stamp}.png")

    intraday = pd.DataFrame()
    if not args.no_intraday_download:
        intraday = try_get_intraday(target, provider=args.intraday_provider)
    if not intraday.empty:
        intraday.to_csv(review_dir / f"QQQM_intraday_actual_{stamp}.csv", index_label="timestamp_et", encoding="utf-8-sig")
        plot_intraday(intraday, target, previous_close, original_snapshot if snapshot_found else reconstructed, review_dir / f"QQQM_intraday_actual_{stamp}.png")
    else:
        print("[提示] 未取得 5 分钟行情；日 K 线图仍会生成。可打开 MarketBeat/TradingView 手动查看当日盘中图。", file=sys.stderr)

    errors = {}
    for field, predicted, actual_key in (
        ("open", pred_open, "open"), ("high", pred_high, "high"),
        ("low", pred_low, "low"), ("close", pred_close, "close"),
    ):
        actual_value = actual.get(actual_key)
        errors[field] = {
            "predicted": predicted,
            "actual": actual_value,
            "signed_error_actual_minus_predicted": None if predicted is None or actual_value is None else round(actual_value - predicted, 4),
            "absolute_error": None if predicted is None or actual_value is None else round(abs(actual_value - predicted), 4),
            "absolute_error_pct_of_actual": None if predicted is None or actual_value in (None, 0) else round(abs(actual_value - predicted) / abs(actual_value) * 100.0, 4),
            "status": "scored" if predicted is not None and actual_value is not None else "prediction_not_available_in_original_snapshot",
        }

    review = {
        "ticker": "QQQM",
        "target_date": target.isoformat(),
        "actual_data_source": actual_source,
        "previous_close": previous_close,
        "actual_ohlc": actual,
        "actual_candle_description": candle_summary,
        "prediction_source": prediction_source,
        "original_prediction_json_path": str(original_json_path),
        "original_snapshot_found": snapshot_found,
        "original_snapshot": original_snapshot if snapshot_found else None,
        "reconstructed_historical_close_forecast": reconstructed,
        "predicted_close_return_pct_from_previous_close": round((pred_close / previous_close - 1.0) * 100.0, 4) if pred_close is not None and previous_close else None,
        "actual_close_return_pct_from_previous_close": round((actual.get("close") / previous_close - 1.0) * 100.0, 4) if actual.get("close") is not None and previous_close else None,
        "close_direction_correct_vs_previous_close": (pred_close >= previous_close) == (actual.get("close") >= previous_close) if pred_close is not None and actual.get("close") is not None and previous_close else None,
        "close_return_error_pct_points": round((pred_close / previous_close - actual.get("close") / previous_close) * 100.0, 4) if pred_close is not None and actual.get("close") is not None and previous_close else None,
        "errors": errors,
        "walk_forward_model_comparison": models,
        "intraday_5m_bars_downloaded": int(len(intraday)),
        "outputs": {
            "review_json": str(review_dir / f"QQQM_review_{stamp}.json"),
            "candlestick_chart": str(review_dir / f"QQQM_daily_candles_{stamp}.png"),
            "intraday_chart": str(review_dir / f"QQQM_intraday_actual_{stamp}.png") if not intraday.empty else None,
            "model_comparison_csv": str(review_dir / f"QQQM_model_comparison_{stamp}.csv"),
            "model_comparison_history_csv": str(reports / "QQQM_model_comparison_history.csv"),
            "prediction_vs_actual_candle_chart": str(review_dir / f"QQQM_prediction_vs_actual_candle_{stamp}.png"),
            "scorecard_csv": str(reports / "QQQM_daily_scorecard.csv"),
        },
        "note": "本脚本只复盘已完成交易日，不生成下一交易日预测。回放预测若非从原始 JSON 读取会显式标注为 reconstructed。K 线特征模型是候选实验，是否值得并入生产预测器由样本外误差决定。",
        "generated_at_hkt": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
    }
    review_path = review_dir / f"QQQM_review_{stamp}.json"
    review_path.write_text(json.dumps(json_safe(review), ensure_ascii=False, indent=2), encoding="utf-8")

    score_row = {
        "target_date": target.isoformat(),
        "prediction_source": prediction_source,
        "actual_data_source": actual_source,
        "previous_close": previous_close,
        "predicted_open": pred_open,
        "actual_open": actual.get("open"),
        "open_abs_error": errors["open"]["absolute_error"],
        "predicted_high": pred_high,
        "actual_high": actual.get("high"),
        "high_abs_error": errors["high"]["absolute_error"],
        "predicted_low": pred_low,
        "actual_low": actual.get("low"),
        "low_abs_error": errors["low"]["absolute_error"],
        "predicted_close": pred_close,
        "actual_close": actual.get("close"),
        "close_signed_error_actual_minus_predicted": errors["close"]["signed_error_actual_minus_predicted"],
        "close_abs_error": errors["close"]["absolute_error"],
        "close_abs_error_pct_of_actual": errors["close"]["absolute_error_pct_of_actual"],
        "predicted_close_return_pct": round((pred_close / previous_close - 1.0) * 100.0, 4) if pred_close is not None and previous_close else None,
        "actual_close_return_pct": round((actual.get("close") / previous_close - 1.0) * 100.0, 4) if actual.get("close") is not None and previous_close else None,
        "close_direction_correct": (pred_close >= previous_close) == (actual.get("close") >= previous_close) if pred_close is not None and actual.get("close") is not None and previous_close else None,
        "original_snapshot_found": snapshot_found,
        "close_only_walk_forward_mae_pct_points": models.get("ridge_close_only", {}).get("models", {}).get("ridge_close_only", {}).get("mae_pct_points"),
        "candle_enriched_walk_forward_mae_pct_points": models.get("ridge_plus_candles", {}).get("models", {}).get("ridge_plus_candles", {}).get("mae_pct_points"),
        "baseline_walk_forward_mae_pct_points": models.get("ridge_close_only", {}).get("models", {}).get("zero_return_baseline", {}).get("mae_pct_points"),
        "candle_candidate_beats_close_only_mae": (
            models.get("ridge_plus_candles", {}).get("models", {}).get("ridge_plus_candles", {}).get("mae_pct_points") is not None
            and models.get("ridge_close_only", {}).get("models", {}).get("ridge_close_only", {}).get("mae_pct_points") is not None
            and models.get("ridge_plus_candles", {}).get("models", {}).get("ridge_plus_candles", {}).get("mae_pct_points")
                < models.get("ridge_close_only", {}).get("models", {}).get("ridge_close_only", {}).get("mae_pct_points")
        ),
    }
    upsert_scorecard(reports / "QQQM_daily_scorecard.csv", score_row)

    print("\n[复盘完成] 没有生成下一交易日预测。")
    print(f"目标日期：{target.isoformat()} | 实际数据来源：{actual_source}")
    print("实际 OHLC：" + " | ".join(f"{k.upper()}=${v:.2f}" for k, v in actual.items()))
    print(f"预测来源：{prediction_source}")
    print(f"收盘预测：${pred_close:.2f} | 实际收盘：${actual.get('close', float('nan')):.2f} | 绝对误差：${errors['close']['absolute_error'] if errors['close']['absolute_error'] is not None else '无法评分'}")
    if pred_close is not None and actual.get("close") is not None:
        pred_ret = (pred_close / previous_close - 1.0) * 100.0
        actual_ret = (actual["close"] / previous_close - 1.0) * 100.0
        print(f"收盘涨幅：预测 {pred_ret:+.3f}% | 实际 {actual_ret:+.3f}% | 方向 {'正确' if (pred_close >= previous_close) == (actual['close'] >= previous_close) else '错误'}")
    for field in ("open", "high", "low", "close"):
        err = errors[field]
        if err["status"] == "scored":
            label = {"open": "开盘", "high": "最高价参考", "low": "最低价参考", "close": "收盘"}[field]
            print(f"{label}：预测 ${err['predicted']:.4f} | 实际 ${err['actual']:.2f} | 绝对误差 ${err['absolute_error']:.4f}")
        elif field in ("open", "high", "low"):
            print(f"{field} 误差：原始预测快照没有对应预测值，因此不会伪造误差。")
    if reconstructed:
        print("注意：上面的收盘预测为历史重建结果，不保证与当时公众号发布的原始数值完全一致。")
    print(f"复盘 JSON：{review_path}")
    print(f"成绩表：{reports / 'QQQM_daily_scorecard.csv'}")
    print(f"日 K 线图：{review_dir / f'QQQM_daily_candles_{stamp}.png'}")
    print(f"预测与实际 K 线对照图：{review_dir / f'QQQM_prediction_vs_actual_candle_{stamp}.png'}")
    print(f"K线特征模型长期对比：{model_history_path}")
    if not intraday.empty:
        print(f"盘中曲线图：{review_dir / f'QQQM_intraday_actual_{stamp}.png'}")
        print(f"盘中 5 分钟 CSV：{review_dir / f'QQQM_intraday_actual_{stamp}.csv'}")
    print(f"模型对比：{review_dir / f'QQQM_model_comparison_{stamp}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
