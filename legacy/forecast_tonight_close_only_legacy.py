from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.config import REPORTS_DIR
from app.db import load_history
from app.indicators import add_indicators

FEATURES = [
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "sma20_gap", "sma50_gap", "volatility20", "rsi14_scaled", "drawdown_1y",
]


def make_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build lagged, point-in-time features. No future data is used in X."""
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
    return features[FEATURES].replace([np.inf, -np.inf], np.nan)


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """Small closed-form Ridge implementation, avoiding a new dependency."""
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-10] = 1.0
    x_train = (train_x - mean) / scale
    x_pred = (predict_x - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_pred = np.column_stack([np.ones(len(x_pred)), x_pred])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0  # don't penalize intercept
    try:
        coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(x_train.T @ x_train + penalty, x_train.T @ train_y, rcond=None)[0]
    return x_pred @ coef


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def build_forecast(target_date: date) -> dict[str, Any]:
    history = load_history("QQQM")
    if history.empty:
        raise RuntimeError("本地数据库没有 QQQM 行情，请先运行：python main.py sync --period 1y --tickers QQQM")
    history = history[history["close"].notna()].copy().sort_index()
    latest_date = history.index[-1].date()
    if target_date <= latest_date:
        raise RuntimeError(f"预测日期 {target_date} 必须晚于最新行情日期 {latest_date}。请检查目标日期或同步数据。")
    if target_date.weekday() >= 5:
        raise RuntimeError(f"目标日期 {target_date} 是周末。请设置为实际美股交易日。")
    if (target_date - latest_date).days > 4:
        raise RuntimeError(
            f"最新行情只到 {latest_date}，距离目标日 {target_date} 已超过 4 个自然日。为避免使用过期行情生成文章，请先同步数据并检查 main.py status。"
        )

    close = history["close"].astype(float)
    features = make_features(history)
    next_return_pct = close.pct_change().shift(-1) * 100.0
    supervised = features.join(next_return_pct.rename("target_return_pct")).dropna()
    latest_x = features.iloc[[-1]][FEATURES]
    if latest_x.isna().any(axis=None):
        raise RuntimeError("最新行情的预测特征存在缺失。请同步至少约 1 年行情，并检查数据是否完整。")

    # Strict walk-forward evaluation on the latest 60 known next-day outcomes.
    min_train = 100
    test_count = min(60, max(0, len(supervised) - min_train))
    oos_rows: list[dict[str, Any]] = []
    if test_count >= 20:
        start = len(supervised) - test_count
        for j in range(start, len(supervised)):
            train = supervised.iloc[:j]
            test = supervised.iloc[j]
            if len(train) < min_train:
                continue
            pred = float(ridge_predict(
                train[FEATURES].to_numpy(dtype=float),
                train["target_return_pct"].to_numpy(dtype=float),
                test[FEATURES].to_numpy(dtype=float).reshape(1, -1),
            )[0])
            oos_rows.append({
                "signal_date": test.name.date().isoformat(),
                "actual_next_day_return_pct": float(test["target_return_pct"]),
                "ridge_predicted_return_pct": pred,
                "zero_return_baseline_pct": 0.0,
            })
    oos = pd.DataFrame(oos_rows)
    if not oos.empty:
        actual = oos["actual_next_day_return_pct"].to_numpy()
        pred = oos["ridge_predicted_return_pct"].to_numpy()
        baseline_mae = float(np.mean(np.abs(actual)))
        ridge_mae = float(np.mean(np.abs(actual - pred)))
        baseline_rmse = float(np.sqrt(np.mean(actual ** 2)))
        ridge_rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        nonzero = np.abs(actual) > 1e-10
        direction_accuracy = float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero]))) if nonzero.any() else None
        model_wins = ridge_mae < baseline_mae * 0.95
    else:
        baseline_mae = ridge_mae = baseline_rmse = ridge_rmse = None
        direction_accuracy = None
        model_wins = False

    latest_close = float(close.iloc[-1])
    model = "Ridge 回归（近期滚动验证表现优于持平基准）" if model_wins else "持平基准（Ridge 尚未证明有足够优势，自动回退）"
    if len(supervised) >= min_train:
        pred_pct_model = float(ridge_predict(
            supervised[FEATURES].to_numpy(dtype=float),
            supervised["target_return_pct"].to_numpy(dtype=float),
            latest_x.to_numpy(dtype=float),
        )[0])
    else:
        pred_pct_model = 0.0
        model = "持平基准（样本不足，未启用机器学习预测）"
        model_wins = False

    # Gate the model prediction with walk-forward performance. The fallback is an explicit baseline, not a guess.
    chosen_pred_pct = pred_pct_model if model_wins else 0.0
    forecast_close = latest_close * (1.0 + chosen_pred_pct / 100.0)

    recent_returns_pct = close.pct_change().dropna().tail(60) * 100.0
    q10 = float(recent_returns_pct.quantile(0.10)) if len(recent_returns_pct) >= 20 else float(recent_returns_pct.min())
    q90 = float(recent_returns_pct.quantile(0.90)) if len(recent_returns_pct) >= 20 else float(recent_returns_pct.max())
    range_low = latest_close * (1 + q10 / 100.0)
    range_high = latest_close * (1 + q90 / 100.0)

    indicators = add_indicators(history)
    latest = indicators.iloc[-1]
    def val(key: str) -> float | None:
        value = latest.get(key)
        return None if pd.isna(value) else float(value)

    summary = {
        "ticker": "QQQM",
        "target_date": target_date.isoformat(),
        "latest_market_date": latest_date.isoformat(),
        "latest_close": round(latest_close, 4),
        "predicted_close_estimate": round(forecast_close, 4),
        "predicted_return_pct": round(chosen_pred_pct, 4),
        "ridge_raw_return_estimate_pct": round(pred_pct_model, 4),
        "model_used": model,
        "history_rows": int(len(history)),
        "walk_forward_test_rows": int(len(oos)),
        "ridge_mae_pct_points": None if ridge_mae is None else round(ridge_mae, 4),
        "baseline_mae_pct_points": None if baseline_mae is None else round(baseline_mae, 4),
        "ridge_rmse_pct_points": None if ridge_rmse is None else round(ridge_rmse, 4),
        "baseline_rmse_pct_points": None if baseline_rmse is None else round(baseline_rmse, 4),
        "ridge_direction_accuracy": None if direction_accuracy is None else round(direction_accuracy, 4),
        "historical_60d_return_q10_pct": round(q10, 4),
        "historical_60d_return_q90_pct": round(q90, 4),
        "historical_range_low": round(range_low, 4),
        "historical_range_high": round(range_high, 4),
        "rsi_14": val("rsi_14"),
        "sma_20": val("sma_20"),
        "sma_50": val("sma_50"),
        "sma_200": val("sma_200"),
        "return_5d_pct": None if pd.isna(latest.get("return_5d")) else round(float(latest.get("return_5d"))*100, 4),
        "return_20d_pct": None if pd.isna(latest.get("return_20d")) else round(float(latest.get("return_20d"))*100, 4),
        "annualized_volatility_20d_pct": None if pd.isna(latest.get("volatility_20d_annualized")) else round(float(latest.get("volatility_20d_annualized"))*100, 4),
        "data_source": ", ".join(sorted(set(history["source"].dropna().astype(str)))) if "source" in history.columns else "本地行情数据库",
        "generated_at_local": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
        "note": "历史 60 个交易日收益率的 10%-90% 分位区间，仅作波动参考，不是经过校准的统计置信区间。模型不构成交易指令。",
    }
    summary["walk_forward_model_beats_baseline"] = bool(model_wins)
    # Only one target session is forecast. The separate intraday simulation is attached
    # after date-matched news/context is validated in main().
    summary["walk_forward_results"] = oos_rows
    return summary


def apply_intraday_forecast(summary: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
    """Attach one-session open-to-close scenario using the daily model and date-matched news.

    The within-session points are a transparent scenario curve, not observed minute bars.
    Historical OHLC ranges are used only to bound/scale the scenario, not as future facts.
    """
    history = load_history("QQQM")
    history = history[history["close"].notna()].copy().sort_index()
    if history.empty:
        raise RuntimeError("无法生成盘中模拟：本地没有 QQQM 历史行情。")

    close = float(summary["latest_close"])
    signals = (context or {}).get("signals", {})
    nq_futures = float(signals.get("nasdaq100_futures_pct", 0.0) or 0.0)
    sp_futures = float(signals.get("sp500_futures_pct", 0.0) or 0.0)
    risk_adjustment = float(signals.get("risk_adjustment_pct_points", -0.18 if signals.get("ai_capex_funding_risk") else 0.0) or 0.0)

    # Estimate a plausible opening gap from a small fraction of the premarket futures move.
    # The pass-through coefficient is an explicit heuristic, not a fitted futures/ETF beta.
    blended_futures_pct = 0.75 * nq_futures + 0.25 * sp_futures
    raw_gap_pct = 0.70 * blended_futures_pct
    if "open" in history.columns and len(history) >= 30:
        prev_close = pd.to_numeric(history["close"], errors="coerce").shift(1)
        actual_open = pd.to_numeric(history["open"], errors="coerce")
        gaps = ((actual_open / prev_close) - 1.0) * 100.0
        gaps = gaps.replace([np.inf, -np.inf], np.nan).dropna().tail(60)
        if len(gaps) >= 20:
            gap_q10 = float(gaps.quantile(0.10))
            gap_q90 = float(gaps.quantile(0.90))
        else:
            gap_q10, gap_q90 = -1.25, 1.25
    else:
        gap_q10, gap_q90 = -1.25, 1.25
    # Do not force a positive/negative opening: the live news signal may be neutral.
    opening_gap_pct = float(np.clip(raw_gap_pct, min(gap_q10, gap_q90), max(gap_q10, gap_q90)))
    open_estimate = close * (1.0 + opening_gap_pct / 100.0)

    # Use the Ridge return only when its walk-forward MAE beat the no-change baseline.
    # Then apply a deliberately modest news adjustment; the risk haircut counters the
    # tendency to extrapolate a futures bounce one-for-one into the cash-session close.
    model_return_pct = float(summary.get("ridge_raw_return_estimate_pct", 0.0) or 0.0) if summary.get("walk_forward_model_beats_baseline") else 0.0
    news_close_adjustment_pct = 0.25 * nq_futures + 0.05 * sp_futures + risk_adjustment
    raw_close_return_pct = model_return_pct + news_close_adjustment_pct

    recent_returns = pd.to_numeric(history["close"], errors="coerce").pct_change().dropna().tail(60) * 100.0
    if len(recent_returns) >= 20:
        return_q10 = float(recent_returns.quantile(0.10))
        return_q90 = float(recent_returns.quantile(0.90))
    elif len(recent_returns):
        return_q10, return_q90 = float(recent_returns.min()), float(recent_returns.max())
    else:
        return_q10, return_q90 = -2.0, 2.0
    close_return_pct = float(np.clip(raw_close_return_pct, min(return_q10, return_q90), max(return_q10, return_q90)))
    close_estimate = close * (1.0 + close_return_pct / 100.0)
    intraday_return_pct = (close_estimate / open_estimate - 1.0) * 100.0

    # Estimate a typical daily high-low span from historical OHLC, to size the uncertainty band.
    typical_range_pct = 1.25
    if {"high", "low"}.issubset(history.columns):
        highs = pd.to_numeric(history["high"], errors="coerce")
        lows = pd.to_numeric(history["low"], errors="coerce")
        ranges = ((highs - lows) / pd.to_numeric(history["close"], errors="coerce") * 100.0)
        ranges = ranges.replace([np.inf, -np.inf], np.nan).dropna().tail(60)
        if len(ranges) >= 20:
            typical_range_pct = float(ranges.median())
    typical_range_pct = float(np.clip(typical_range_pct, 0.25, 5.0))
    band_half_pct = float(np.clip(typical_range_pct * 0.22, 0.10, 0.75))

    # News context says futures are currently firmer while AI-capex/funding concerns remain.
    # This creates a modest early-up / later-fade shape when the signal blend is positive
    # and the risk adjustment is negative. All points are simulated scenario anchors.
    times = ["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]
    progress = np.linspace(0.0, 1.0, len(times))
    if nq_futures > 0 and risk_adjustment < 0:
        wiggles = np.array([0.0, 0.42, 0.18, -0.22, -0.12, 0.12, -0.10, 0.0])
    elif nq_futures < 0:
        wiggles = np.array([0.0, -0.28, -0.10, 0.16, -0.18, -0.05, 0.10, 0.0])
    else:
        wiggles = np.array([0.0, 0.20, -0.12, 0.10, -0.08, 0.12, -0.05, 0.0])
    wiggle_scale = max(typical_range_pct * 0.30, 0.08)
    values = open_estimate + (close_estimate - open_estimate) * progress
    values = values + open_estimate * (wiggles * wiggle_scale / 100.0)
    values[0] = open_estimate
    values[-1] = close_estimate

    path = []
    for i, (clock, value) in enumerate(zip(times, values)):
        price = float(value)
        path.append({
            "time_et": clock,
            "price_estimate": round(price, 4),
            "band_low_estimate": round(price * (1.0 - band_half_pct / 100.0), 4),
            "band_high_estimate": round(price * (1.0 + band_half_pct / 100.0), 4),
            "role": "estimated_open" if i == 0 else ("estimated_close" if i == len(times)-1 else "simulated_intraday_anchor"),
        })

    prices = [point["price_estimate"] for point in path]
    high_reference = max(prices) * (1.0 + band_half_pct * 0.55 / 100.0)
    low_reference = min(prices) * (1.0 - band_half_pct * 0.55 / 100.0)
    forecast = {
        "target_date": summary["target_date"],
        "previous_close": round(close, 4),
        "nasdaq100_futures_pct": nq_futures,
        "sp500_futures_pct": sp_futures,
        "opening_gap_assumption_pct": round(opening_gap_pct, 4),
        "estimated_open": round(open_estimate, 4),
        "estimated_high_reference": round(high_reference, 4),
        "estimated_low_reference": round(low_reference, 4),
        "estimated_close": round(close_estimate, 4),
        "estimated_close_return_from_previous_pct": round(close_return_pct, 4),
        "estimated_intraday_return_open_to_close_pct": round(intraday_return_pct, 4),
        "model_return_component_pct": round(model_return_pct, 4),
        "news_close_adjustment_pct_points": round(news_close_adjustment_pct, 4),
        "risk_adjustment_pct_points": round(risk_adjustment, 4),
        "typical_historical_intraday_range_pct": round(typical_range_pct, 4),
        "uncertainty_band_half_width_pct": round(band_half_pct, 4),
        "path": path,
        "method_note": "单日情景模拟：开盘缺口采用纳指/标普期指信号的保守折扣；收盘估算结合通过滚动验证的日线模型（若模型未胜过持平基准则权重为零）与较小的新闻调整；盘中折线为启发式路径锚点，不是分钟级预测或真实成交数据。",
    }
    summary["intraday_forecast"] = forecast
    # Keep compatibility with article/report metadata, but make it describe the single target day.
    summary["predicted_close_estimate"] = forecast["estimated_close"]
    summary["predicted_return_pct"] = forecast["estimated_close_return_from_previous_pct"]
    summary["historical_range_low"] = round(close * (1.0 + return_q10 / 100.0), 4)
    summary["historical_range_high"] = round(close * (1.0 + return_q90 / 100.0), 4)
    summary["forecast_path"] = []
    summary["forecast_path_note"] = "本期只模拟一个交易日的开盘至收盘路径；所有盘中点均为情景模拟锚点，不是实际盘中行情。"
    return summary


def load_market_context(context_path: Path, target_date: date) -> dict[str, Any] | None:
    """Load date-scoped, manually verified market context; never reuse stale headlines."""
    if not context_path.exists():
        return None
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[警告] 盘前背景文件无法读取：{exc}", file=sys.stderr)
        return None
    if context.get("date") != target_date.isoformat():
        print(
            f"[提示] 盘前背景文件日期为 {context.get('date')}，目标日期为 {target_date.isoformat()}。"
            "为避免引用过期新闻，本期不会自动加入这段背景；请更新 market_context.json。",
            file=sys.stderr,
        )
        return None
    return context


def article_data(summary: dict[str, Any], context: dict[str, Any] | None) -> dict[str, Any]:
    close = float(summary["latest_close"])
    intraday = summary["intraday_forecast"]
    forecast = float(intraday["estimated_close"])
    change_pct = (forecast / close - 1.0) * 100.0
    headline = (context or {}).get("article_title") or f"QQQM 下一交易日盘中走势模拟：开盘、波动与收盘估算（{summary['target_date']}）"
    intro = (context or {}).get("intro") or "这是一份结合历史行情、模型检验与盘前信息的单日情景模拟。"
    model_beats = bool(summary.get("walk_forward_model_beats_baseline"))
    if model_beats:
        model_explanation = (
            f"模型在最近 {summary['walk_forward_test_rows']} 次按时间顺序进行的历史测试中，平均绝对误差优于‘明天价格不变’的参照，"
            "因此收盘估算保留了模型信号，再叠加幅度较小的盘前新闻调整。历史测试结果并不保证这一次有效。"
        )
    else:
        model_explanation = (
            f"最近 {summary.get('walk_forward_test_rows', 0)} 次历史测试尚未证明日线模型稳定优于‘明天价格不变’基准，"
            "所以模型方向在本次收盘估算中的权重设为 0，避免把未经证实的模型输出当成优势；收盘方向主要来自明确标注的盘前期指信号和风险折扣。"
        )
    return {
        "headline": headline,
        "intro": intro,
        "close": close,
        "forecast": forecast,
        "change_pct": change_pct,
        "forecast_label": "单日综合情景估算",
        "model_explanation": model_explanation,
        "context": context,
        "range_low": float(intraday["estimated_low_reference"]),
        "range_high": float(intraday["estimated_high_reference"]),
    }


def make_chart(summary: dict[str, Any], chart_path: Path) -> None:
    """Create a single US trading-session simulated intraday path chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        from matplotlib import font_manager
    except ImportError as exc:
        raise RuntimeError("缺少 matplotlib，先执行：.\\.venv\\Scripts\\python.exe -m pip install matplotlib") from exc

    f = summary["intraday_forecast"]
    times = [p["time_et"] for p in f["path"]]
    prices = np.array([float(p["price_estimate"]) for p in f["path"]])
    band_low = np.array([float(p["band_low_estimate"]) for p in f["path"]])
    band_high = np.array([float(p["band_high_estimate"]) for p in f["path"]])
    prev_close = float(f["previous_close"])
    open_est = float(f["estimated_open"])
    close_est = float(f["estimated_close"])
    high_est = float(f["estimated_high_reference"])
    low_est = float(f["estimated_low_reference"])

    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"), Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    ]
    selected_font = next((candidate for candidate in font_candidates if candidate.exists()), None)
    if selected_font is not None:
        font_manager.fontManager.addfont(str(selected_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(selected_font)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12.5, 6.7), dpi=180)
    fig.patch.set_facecolor("#F4F7FB")
    ax.set_facecolor("#FFFFFF")
    fig.subplots_adjust(left=0.095, right=0.965, top=0.82, bottom=0.16)
    fig.text(0.095, 0.94, "QQQM 下一交易日单日走势模拟", fontsize=23, fontweight="bold", color="#15263D")
    fig.text(0.097, 0.895,
             f"目标交易日 {f['target_date']}（美东时间）  |  最近收盘 ${prev_close:.2f}  |  盘中路径为情景模拟，非真实分钟行情",
             fontsize=10.4, color="#61718A")

    x = np.arange(len(times))
    ax.fill_between(x, band_low, band_high, color="#8CB8FA", alpha=0.22,
                    label="路径波动参考带（不是置信区间）", zorder=1)
    ax.plot(x, prices, color="#087D67", linewidth=3.2, marker="o", markersize=6,
            markerfacecolor="#FFFFFF", markeredgecolor="#087D67", markeredgewidth=2,
            label="模拟盘中路径", zorder=4)
    ax.axhline(prev_close, color="#8B97A8", linestyle=(0, (5, 4)), linewidth=1.5,
               label=f"前收 ${prev_close:.2f}", zorder=2)
    ax.scatter([0], [open_est], s=125, color="#3478F6", edgecolor="#FFFFFF", linewidth=2,
               label=f"预测开盘 ${open_est:.2f}", zorder=6)
    ax.scatter([len(times)-1], [close_est], s=135, color="#F08A24", edgecolor="#FFFFFF", linewidth=2,
               label=f"预测收盘 ${close_est:.2f}", zorder=7)

    max_i = int(np.argmax(prices))
    min_i = int(np.argmin(prices))
    ax.annotate(f"开盘估算  ${open_est:.2f}", (0, open_est), xytext=(12, -42), textcoords="offset points",
                fontsize=10.2, color="#225DB6", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#EEF5FF", ec="#BAD5FF", lw=0.8))
    ax.annotate(f"收盘估算  ${close_est:.2f}\n较前收 {((close_est/prev_close)-1)*100:+.2f}%", (len(times)-1, close_est),
                xytext=(-115, -42 if close_est >= prices[max_i] else 18), textcoords="offset points",
                fontsize=10.2, color="#B76512", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFF4E8", ec="#F4C58E", lw=0.8))
    ax.text(0.99, 0.96, f"高点参考 {high_est:.2f} 美元  ·  低点参考 {low_est:.2f} 美元",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#61718A",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F7F9FC", ec="#E4EAF2", lw=0.7))

    y_all = np.concatenate([prices, band_low, band_high, np.array([prev_close, open_est, close_est])])
    pad = max((float(np.max(y_all)) - float(np.min(y_all))) * 0.22, prev_close * 0.003)
    ax.set_ylim(float(np.min(y_all))-pad, float(np.max(y_all))+pad)
    ax.set_xlim(-0.35, len(times)-0.65)
    ax.set_xticks(x, times)
    ax.set_ylabel("QQQM 每股价格（美元）", fontsize=10, color="#61718A", labelpad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.2f}"))
    ax.grid(axis="y", color="#E8EDF4", linewidth=0.9)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D7E0EB")
    ax.tick_params(axis="x", colors="#718096", labelsize=9.5, length=0, pad=8)
    ax.tick_params(axis="y", colors="#718096", labelsize=9.5, length=0, pad=8)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9.0, labelcolor="#4D5C70")
    fig.text(0.097, 0.045,
             "说明：开盘由期指信号折扣估算；收盘综合日线模型（若历史验证有效）与新闻风险调整；中间节点是启发式模拟。实际走势可能完全不同。",
             fontsize=8.7, color="#718096", wrap=True)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_markdown(summary: dict[str, Any], context: dict[str, Any] | None, chart_name: str) -> str:
    d = article_data(summary, context)
    f = summary["intraday_forecast"]
    money_ = money
    path_lines = ["## 这条预测曲线具体在模拟什么？", "",
                  "它只针对一个交易日：从预计开盘价开始，模拟盘中几个观察时点的价格路径，最后落到预计收盘价。**绿色曲线中间的点是情景模拟锚点，不是机器实时预测出来的分钟级价格，更不是真实成交行情。**", "",
                  f"- **预计开盘：{money_(f['estimated_open'])}**（相对上一交易日收盘 {pct(f['opening_gap_assumption_pct'])}）。", 
                  f"- **盘中高点参考：{money_(f['estimated_high_reference'])}；低点参考：{money_(f['estimated_low_reference'])}。** 这是基于模拟路径和近期日内波动幅度得出的参考，不是保证区间。", 
                  f"- **预计收盘：{money_(f['estimated_close'])}**（相对前收 {pct(f['estimated_close_return_from_previous_pct'])}；相对预计开盘 {pct(f['estimated_intraday_return_open_to_close_pct'])}）。", "",
                  "模拟路径按美东时间排列：" + " → ".join(f"{p['time_et']} {money_(p['price_estimate'])}" for p in f["path"]), ""]
    lines: list[str] = [
        f"# {d['headline']}", "",
        f"> 每日市场观察｜行情截至 {summary['latest_market_date']} 美股收盘｜文章生成时间（香港时间）：{summary['generated_at_local']}", "",
        d["intro"], "",
        "## 先看今天这份单日预测", "",
        f"- **上一交易日真实收盘：{money_(d['close'])}。**这是已发生的价格，不是实时盘前报价。",
        f"- **预计开盘：{money_(f['estimated_open'])}。**基于盘前期指信号做了折扣处理，期指不是 QQQM 本身，开盘时也可能出现跳空偏差。",
        f"- **预计收盘：{money_(f['estimated_close'])}。**相对上一交易日收盘的中心估算为 {pct(d['change_pct'])}。",
        f"- **盘中波动参考：{money_(f['estimated_low_reference'])} 至 {money_(f['estimated_high_reference'])}。**它不是统计置信区间，也不意味着价格一定会落在其中。", "",
        f"![QQQM 下一交易日单日走势模拟]({chart_name})", "",
        *path_lines,
        "## 最近行情发生了什么？", "",
    ]
    if context and context.get("recent_price_story"):
        lines.extend([context["recent_price_story"], ""])
    else:
        ret5 = summary.get("return_5d_pct")
        ret20 = summary.get("return_20d_pct")
        lines.extend([f"过去 5 个交易日变化约为 **{pct(float(ret5)) if ret5 is not None else '数据不足'}**；过去 20 个交易日约为 **{pct(float(ret20)) if ret20 is not None else '数据不足'}**。这些是回顾，不是未来保证。", ""])

    if context and context.get("sections"):
        lines.extend(["## 盘前新闻：反弹信号和风险同时存在", ""])
        for section in context["sections"]:
            lines.extend([f"### {section['heading']}", "", section["text"], ""])
        lines.extend(["## 盘中重点观察", ""])
        for item in context.get("watch_items", []):
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.extend(["## 盘前新闻", "", "未导入与目标交易日匹配的新闻，本期预测只使用历史行情和模型信息。发布前请更新当天的 market_context.json。", ""])

    lines.extend([
        "## 预测方法与局限", "",
        d["model_explanation"], "",
        f"开盘估算把纳斯达克 100 期指变化（{f['nasdaq100_futures_pct']:+.2f}%）和标普 500 期指变化（{f['sp500_futures_pct']:+.2f}%）组合后，只传递其中一部分到 QQQM 的开盘缺口估计；收盘估算再叠加 {f['news_close_adjustment_pct_points']:+.2f} 个百分点的新闻调整。对 AI 投入/融资风险做了 {f['risk_adjustment_pct_points']:+.2f} 个百分点的折扣。以上是可复盘的启发式权重，并非经过长期回测校准的规律。", "",
        "盘中走势形状是依据盘前偏强、风险未消除的情景生成，目的在于形成明天可以逐项核对的预测记录。日线历史数据不能真正预测每半小时的成交价格，因此请把中间点看作假设路径，把预计开盘和收盘作为主要评估项目。", "",
        "## 资料来源", "",
    ])
    sources = [
        ("QQQM 历史行情（Yahoo Finance）", "https://finance.yahoo.com/quote/QQQM/history/"),
        ("QQQM 基金跟踪目标（Invesco 官方）", "https://www.invesco.com/us/en/financial-products/etfs/invesco-nasdaq-100-etf.html"),
    ]
    if context:
        sources.extend([(item.get("title", "市场报道"), item["url"]) for item in context.get("sources", []) if item.get("url")])
    seen: set[str] = set()
    for label, url in sources:
        if url not in seen:
            lines.append(f"- [{label}]({url})")
            seen.add(url)
    lines.extend(["", "## 风险提示", "",
                  "本文是个人编程与市场研究记录，只用于学习与复盘，不构成个性化投资建议，也不是买入或卖出指令。单日价格很难稳定预测，模拟开盘、盘中路径与收盘估算都可能失准。请勿仅凭本文交易，并考虑交易费用、汇率与自身风险承受能力。", "",
                  "*QQQM Investment Assistant｜单日预测。下一交易日结束且行情同步后，将预测开盘、盘中参考区间及收盘估算与真实 OHLC 对照。*", ""])
    return "\n".join(lines)


def markdown_inline_to_html(text: str) -> str:
    """Convert the small Markdown subset used by this article into inline-styled HTML."""
    import html as html_lib
    escaped = html_lib.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#15263D;">\1</strong>', escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    escaped = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" style="color:#2875D8;text-decoration:none;">\1</a>', escaped)
    return escaped


def markdown_to_publish_html(markdown: str, chart_path: Path, summary: dict[str, Any], context: dict[str, Any] | None) -> str:
    """Make an HTML file with inline styles and the chart embedded, for rich copy/paste to WeChat."""
    import base64
    import html as html_lib
    chart_b64 = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    out = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>QQQM 单日预测</title></head>',
        '<body style="margin:0;background:#F4F7FB;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',\'Microsoft YaHei\',Arial,sans-serif;color:#26364B;line-height:1.9;">',
        '<article style="max-width:760px;margin:0 auto;background:#FFFFFF;padding:28px 24px 36px;">',
    ]
    lines = markdown.splitlines()
    in_list = False
    image_inserted = False
    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append('</ul>')
            in_list = False
    for line in lines:
        if not line.strip():
            close_list()
            continue
        if line.startswith('# '):
            close_list()
            out.append(f'<h1 style="font-size:28px;line-height:1.4;color:#15263D;margin:8px 0 12px;font-weight:800;">{markdown_inline_to_html(line[2:])}</h1>')
        elif line.startswith('## '):
            close_list()
            out.append(f'<h2 style="font-size:20px;line-height:1.5;color:#15263D;margin:30px 0 12px;padding-bottom:8px;border-bottom:1px solid #E7EDF5;">{markdown_inline_to_html(line[3:])}</h2>')
        elif line.startswith('### '):
            close_list()
            out.append(f'<h3 style="font-size:17px;color:#1B3856;margin:20px 0 8px;">{markdown_inline_to_html(line[4:])}</h3>')
        elif line.startswith('> '):
            close_list()
            out.append(f'<p style="font-size:12px;color:#75849A;margin:4px 0 20px;">{markdown_inline_to_html(line[2:])}</p>')
        elif line.startswith('- '):
            if not in_list:
                out.append('<ul style="padding-left:22px;margin:8px 0 18px;">')
                in_list = True
            out.append(f'<li style="margin:7px 0;padding-left:3px;">{markdown_inline_to_html(line[2:])}</li>')
        elif line.startswith('![QQQM'):
            close_list()
            out.append(f'<div style="margin:16px 0 8px;"><img alt="QQQM 下一交易日预测曲线" src="data:image/png;base64,{chart_b64}" style="display:block;width:100%;height:auto;border-radius:12px;border:1px solid #E6ECF4;"/></div>')
            out.append('<p style="font-size:12px;color:#8190A4;text-align:center;margin:0 0 24px;">绿色为单日模拟盘中路径，蓝色点为预计开盘，橙色点为预计收盘；浅蓝范围是波动参考带，并非置信区间。</p>')
            image_inserted = True
        elif line == '---':
            close_list()
            out.append('<hr style="border:none;border-top:1px solid #E7EDF5;margin:24px 0;"/>')
        else:
            close_list()
            style = 'font-size:15px;margin:10px 0;'
            if line.startswith('*QQQM Investment Assistant'):
                style = 'font-size:12px;color:#8190A4;margin-top:24px;'
            out.append(f'<p style="{style}">{markdown_inline_to_html(line)}</p>')
    close_list()
    out.append('</article></body></html>')
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 QQQM 下一交易日观察、图表与公众号排版稿")
    parser.add_argument("--target-date", required=True, help="要观察的美股交易日，例如 2026-10-19")
    parser.add_argument("--context-file", default=str(Path(__file__).resolve().parents[1] / "market_context.json"), help="与目标日期对应的盘前新闻 JSON")
    args = parser.parse_args()
    try:
        target_date = date.fromisoformat(args.target_date)
        summary = build_forecast(target_date)
        context = load_market_context(Path(args.context_file), target_date)
        if context:
            expected_date = context.get("expected_latest_market_date")
            expected_close = context.get("expected_latest_close")
            actual_close = float(summary["latest_close"])
            if expected_date and summary["latest_market_date"] != expected_date:
                print(f"[警告] 新闻背景要求行情截至 {expected_date}，但本地数据截至 {summary['latest_market_date']}。本期不会混用不匹配的背景。", file=sys.stderr)
                context = None
            elif expected_close is not None and abs(actual_close - float(expected_close)) > 0.05:
                print(f"[警告] 新闻背景引用的收盘价为 ${float(expected_close):.2f}，本地数据为 ${actual_close:.2f}。请核对数据源，本期不会自动加入这段背景。", file=sys.stderr)
                context = None
    except Exception as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1

    try:
        apply_intraday_forecast(summary, context)
    except Exception as exc:
        print(f"[单日预测失败] {exc}", file=sys.stderr)
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = target_date.strftime("%Y%m%d")
    md_path = REPORTS_DIR / f"QQQM_forecast_{stamp}.md"
    html_path = REPORTS_DIR / f"QQQM_forecast_{stamp}_wechat.html"
    json_path = REPORTS_DIR / f"QQQM_forecast_{stamp}.json"
    csv_path = REPORTS_DIR / f"QQQM_walk_forward_{stamp}.csv"
    path_csv_path = REPORTS_DIR / f"QQQM_intraday_path_{stamp}.csv"
    chart_path = REPORTS_DIR / f"QQQM_forecast_{stamp}.png"

    try:
        make_chart(summary, chart_path)
    except Exception as exc:
        print(f"[绘图失败] {exc}", file=sys.stderr)
        return 1

    markdown = make_markdown(summary, context, chart_path.name)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_publish_html(markdown, chart_path, summary, context), encoding="utf-8")
    json_safe = dict(summary)
    rows = json_safe.pop("walk_forward_results", [])
    json_path.write_text(json.dumps(json_safe, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["intraday_forecast"]["path"]).to_csv(path_csv_path, index=False, encoding="utf-8-sig")

    print("\n[完成] 已生成以下文件：")
    print(f"公众号排版稿（推荐打开后复制到公众号）：{html_path}")
    print(f"公众号 Markdown 备份：{md_path}")
    print(f"单日预测曲线图：{chart_path}")
    print(f"单日预测快照（明日复盘用）：{json_path}")
    print(f"盘中模拟路径 CSV：{path_csv_path}")
    print(f"滚动验证明细：{csv_path}")
    f = summary["intraday_forecast"]
    print(f"行情截止日期：{summary['latest_market_date']} | 目标交易日：{summary['target_date']}")
    print(f"预测开盘：{money(float(f['estimated_open']))} | 盘中低/高参考：{money(float(f['estimated_low_reference']))} / {money(float(f['estimated_high_reference']))}")
    print(f"预测收盘：{money(float(f['estimated_close']))} (较前收 {pct(float(f['estimated_close_return_from_previous_pct']))})")
    print(f"日线模型：{summary['model_used']}")
    if summary["ridge_mae_pct_points"] is not None:
        print(f"滚动验证 MAE | Ridge={summary['ridge_mae_pct_points']:.3f}% | 持平基准={summary['baseline_mae_pct_points']:.3f}%")
    if context is None:
        print("[提示] 未加入目标交易日匹配的新闻。当前开盘与收盘情景将只使用历史模型/中性新闻信号，请更新 market_context.json。")
    print("提示：HTML 文件包含内嵌单日模拟曲线。盘中节点是情景锚点，不是实际分钟行情；明天同步真实行情后可运行 evaluate_intraday_forecast.py 复盘。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
