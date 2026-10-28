from __future__ import annotations

import math
from typing import Any

import pandas as pd


def add_indicators(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()

    df = history.copy()
    close = df["close"].astype(float)
    df["daily_return"] = close.pct_change()
    df["sma_20"] = close.rolling(20, min_periods=20).mean()
    df["sma_50"] = close.rolling(50, min_periods=50).mean()
    df["sma_200"] = close.rolling(200, min_periods=200).mean()
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["volatility_20d_annualized"] = (
        df["daily_return"].rolling(20, min_periods=20).std() * math.sqrt(252)
    )

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100.0

    rolling_high = close.rolling(252, min_periods=1).max()
    df["drawdown_from_1y_high"] = close / rolling_high - 1
    return df


def _number(value: Any, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def build_summary(ticker: str, history: pd.DataFrame) -> dict[str, Any]:
    df = add_indicators(history)
    if df.empty:
        raise ValueError(f"数据库中没有 {ticker} 的行情，请先执行 sync。")

    latest = df.iloc[-1]
    latest_date = df.index[-1].date().isoformat()
    recent = []
    for date, row in df.tail(10).iterrows():
        recent.append(
            {
                "date": date.date().isoformat(),
                "open": _number(row.get("open"), 4),
                "high": _number(row.get("high"), 4),
                "low": _number(row.get("low"), 4),
                "close": _number(row.get("close"), 4),
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else None,
            }
        )

    sources = sorted({str(s) for s in df["source"].dropna().unique()}) if "source" in df.columns else []
    metrics = {
        "close": _number(latest.get("close")),
        "daily_return_pct": _number(latest.get("daily_return") * 100, 2),
        "return_5d_pct": _number(latest.get("return_5d") * 100, 2),
        "return_20d_pct": _number(latest.get("return_20d") * 100, 2),
        "sma_20": _number(latest.get("sma_20")),
        "sma_50": _number(latest.get("sma_50")),
        "sma_200": _number(latest.get("sma_200")),
        "rsi_14": _number(latest.get("rsi_14"), 2),
        "volatility_20d_annualized_pct": _number(
            latest.get("volatility_20d_annualized") * 100, 2
        ),
        "drawdown_from_1y_high_pct": _number(
            latest.get("drawdown_from_1y_high") * 100, 2
        ),
    }
    return {
        "ticker": ticker.upper(),
        "latest_date": latest_date,
        "number_of_daily_rows": len(df),
        "data_sources": sources,
        "metrics": metrics,
        "recent_daily_bars": recent,
        "data_notes": [
            "行情来自当前配置的数据供应商；日线可能延迟、缺失或被后续修订。",
            "本报告第一版尚未接入实时新闻、FRED 宏观数据或美联储公告。",
            "这里的日收益率基于未复权 Close，不包含分红再投资收益；Adj Close 的处理需按具体数据源确认。",
        ],
    }
