from __future__ import annotations

"""Build a point-in-time-aware market/news/macro context snapshot.

This module is intentionally a *context/feature preview*, not a training pipeline.
The current database's macro_observations table lacks release timestamps and data
vintages. Such values are useful for current analysis but must not be naively used
as historical training features because that can introduce look-ahead leakage.
"""

import json
import math
import sqlite3
from bisect import bisect_left, bisect_right
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import pstdev
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "market_data.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "context_features"
NEW_YORK = ZoneInfo("America/New_York")

# Use conservative, explainable keyword groups as transparent metadata features.
# These are not a sentiment model and must not be interpreted as causal signals.
KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "hawkish_keyword_count": (
        "hawkish", "rate hike", "rate hikes", "raise rates", "higher for longer",
        "monetary tightening", "tightening policy", "tighten policy",
    ),
    "dovish_keyword_count": (
        "dovish", "rate cut", "rate cuts", "cut rates", "easing policy",
        "monetary easing", "lower rates",
    ),
    "inflation_keyword_count": (
        "inflation", "consumer price", "consumer prices", "cpi", "price pressures",
    ),
    "labor_keyword_count": (
        "employment", "payroll", "payrolls", "unemployment", "labor market",
        "labour market", "nonfarm",
    ),
    "growth_risk_keyword_count": (
        "recession", "economic slowdown", "slowing growth", "growth outlook",
        "gross domestic product", "gdp",
    ),
}

# These are common FRED/BLS IDs in this project. Other IDs remain visible in the
# database audit output, but are not silently assigned a meaning here.
MACRO_SERIES_LABELS = {
    "EFFR": "Effective Federal Funds Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DFII10": "10-Year Treasury Inflation-Indexed Security Rate",
    "CPIAUCSL": "CPI All Urban Consumers, Seasonally Adjusted",
    "CUSR0000SA0": "CPI All Urban Consumers, Seasonally Adjusted (BLS)",
    "LNS14000000": "Unemployment Rate",
    "CES0000000001": "Total Nonfarm Employment",
}


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Database ISO timestamps should include offsets. If absent, assume UTC
        # conservatively rather than using the workstation's local timezone.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table_name}")')}


def _load_market_rows(conn: sqlite3.Connection, ticker: str) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "market_daily")
    if not columns:
        raise RuntimeError("数据库中找不到 market_daily 表。")
    required = {"ticker", "trade_date", "close"}
    missing = required - columns
    if missing:
        raise RuntimeError(f"market_daily 缺少必要字段：{sorted(missing)}")
    price_col = "adj_close" if "adj_close" in columns else "close"
    volume_expr = "volume" if "volume" in columns else "NULL AS volume"
    sql = f"""
        SELECT trade_date, close, {price_col} AS price, {volume_expr}
        FROM market_daily
        WHERE ticker = ?
        ORDER BY trade_date
    """
    rows = []
    for row in conn.execute(sql, (ticker,)):
        trade_date, close, price, volume = row
        if not trade_date or price is None:
            continue
        rows.append(
            {
                "trade_date": str(trade_date)[:10],
                "close": float(close) if close is not None else float(price),
                "price": float(price),
                "volume": int(volume) if volume is not None else None,
            }
        )
    if not rows:
        raise RuntimeError(f"market_daily 中没有 {ticker} 的可用价格记录。")
    return rows


def _assign_news_to_session(published_at: datetime, trading_dates: list[str]) -> str | None:
    """Assign news to the first session whose close had not already passed.

    Published before 16:00 New York time on a trading date is assigned to that
    date. At/after 16:00 it is assigned to the next available trading date.
    Weekend/holiday publications are assigned to the next date in market_daily.
    """
    if not trading_dates:
        return None
    local = published_at.astimezone(NEW_YORK)
    local_day = local.date().isoformat()
    idx = bisect_left(trading_dates, local_day)
    is_trading_date = idx < len(trading_dates) and trading_dates[idx] == local_day
    before_close = local.timetz().replace(tzinfo=None) < time(16, 0)
    if is_trading_date and before_close:
        return local_day
    next_idx = bisect_right(trading_dates, local_day)
    if next_idx >= len(trading_dates):
        return None
    return trading_dates[next_idx]


def _load_news(conn: sqlite3.Connection, trading_dates: list[str]) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "news_items")
    if not columns:
        return []
    selected = [
        name for name in (
            "news_id", "source_name", "category", "title", "summary", "url",
            "published_at", "fetched_at",
        ) if name in columns
    ]
    if "published_at" not in selected:
        return []
    sql = f'SELECT {", ".join(selected)} FROM news_items WHERE published_at IS NOT NULL'
    index = {name: i for i, name in enumerate(selected)}
    items: list[dict[str, Any]] = []
    for row in conn.execute(sql):
        published = _parse_datetime(row[index["published_at"]])
        if published is None:
            continue
        record = {name: row[i] for name, i in index.items()}
        record["published_at_utc"] = published.astimezone(timezone.utc).isoformat()
        record["available_from_session"] = _assign_news_to_session(published, trading_dates)
        record["search_text"] = f"{record.get('title') or ''} {record.get('summary') or ''}".lower()
        items.append(record)
    items.sort(key=lambda item: item["published_at_utc"], reverse=True)
    return items


def _news_features(items: list[dict[str, Any]], as_of_date: str) -> dict[str, Any]:
    cutoff = date.fromisoformat(as_of_date)
    start_30d = (cutoff - timedelta(days=30)).isoformat()
    usable = [
        item for item in items
        if item.get("available_from_session")
        and item["available_from_session"] <= as_of_date
        and item["available_from_session"] >= start_30d
    ]
    counts_by_category: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    for item in usable:
        category = str(item.get("category") or "unknown")
        source = str(item.get("source_name") or "unknown")
        counts_by_category[category] = counts_by_category.get(category, 0) + 1
        counts_by_source[source] = counts_by_source.get(source, 0) + 1

    keyword_counts: dict[str, int] = {}
    for feature_name, phrases in KEYWORD_GROUPS.items():
        keyword_counts[feature_name] = sum(
            1 for item in usable
            if any(phrase in item["search_text"] for phrase in phrases)
        )

    recent_items = []
    for item in usable[:8]:
        recent_items.append({
            "title": str(item.get("title") or ""),
            "source": str(item.get("source_name") or ""),
            "category": str(item.get("category") or ""),
            "published_at": str(item.get("published_at") or ""),
            "available_from_session": item.get("available_from_session"),
            "url": str(item.get("url") or ""),
        })

    return {
        "window_start_date": start_30d,
        "window_end_date": as_of_date,
        "news_count_30_calendar_days": len(usable),
        "counts_by_category": counts_by_category,
        "counts_by_source": counts_by_source,
        "keyword_counts": keyword_counts,
        "recent_items": recent_items,
        "limitations": [
            "关键词计数仅用于透明、可解释的上下文摘要，不等于经过验证的情绪预测模型。",
            "新闻总量有限且历史覆盖可能不完整；当前不自动将这些计数加入训练集。",
        ],
    }


def _load_macro_snapshot(conn: sqlite3.Connection, as_of_date: str) -> dict[str, Any]:
    columns = _table_columns(conn, "macro_observations")
    if not columns:
        return {
            "available": False,
            "items": [],
            "historical_backtest_eligible": False,
            "reason": "数据库中没有 macro_observations 表。",
        }
    required = {"series_id", "observation_date", "value"}
    if not required.issubset(columns):
        return {
            "available": False,
            "items": [],
            "historical_backtest_eligible": False,
            "reason": f"macro_observations 缺少字段：{sorted(required - columns)}",
        }

    rows = conn.execute(
        """
        SELECT series_id, observation_date, value, retrieved_at
        FROM macro_observations
        WHERE observation_date <= ?
        ORDER BY series_id, observation_date
        """,
        (as_of_date,),
    ).fetchall()
    by_series: dict[str, list[dict[str, Any]]] = {}
    for series_id, observation_date, value, retrieved_at in rows:
        if value is None:
            continue
        key = str(series_id)
        by_series.setdefault(key, []).append({
            "observation_date": str(observation_date),
            "value": float(value),
            "retrieved_at": str(retrieved_at) if retrieved_at is not None else None,
        })

    items = []
    preferred = set(MACRO_SERIES_LABELS)
    series_ids = sorted(by_series, key=lambda x: (x not in preferred, x))
    for series_id in series_ids:
        observations = by_series[series_id]
        latest = observations[-1]
        previous = observations[-2] if len(observations) >= 2 else None
        items.append({
            "series_id": series_id,
            "label": MACRO_SERIES_LABELS.get(series_id, series_id),
            "latest_observation_date": latest["observation_date"],
            "latest_value": latest["value"],
            "previous_observation_date": previous["observation_date"] if previous else None,
            "previous_value": previous["value"] if previous else None,
            "change_from_previous_observation": (
                latest["value"] - previous["value"] if previous else None
            ),
            "retrieved_at": latest["retrieved_at"],
        })

    return {
        "available": bool(items),
        "as_of_observation_date": as_of_date,
        "items": items,
        "historical_backtest_eligible": False,
        "reason": (
            "表中只有 observation_date 与 retrieved_at，没有原始发布日期/可用时间和历史数据版本。"
            "retrieved_at 是系统抓取时间，不等于该数据当时首次公开的时间；因此这些历史宏观值暂不用于训练或历史回测。"
        ),
    }


def build_market_context_snapshot(
    db_path: str | Path = DEFAULT_DB_PATH,
    ticker: str = "QQQM",
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Create an explanatory market context snapshot without modifying the DB."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到行情数据库：{path}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        market_rows = _load_market_rows(conn, ticker)
        trading_dates = [row["trade_date"] for row in market_rows]
        effective_date = as_of_date or trading_dates[-1]
        if effective_date not in trading_dates:
            available_dates = [day for day in trading_dates if day <= effective_date]
            if not available_dates:
                raise ValueError(f"指定日期 {effective_date} 之前没有 {ticker} 行情。")
            effective_date = available_dates[-1]
        rows = [row for row in market_rows if row["trade_date"] <= effective_date]
        if not rows:
            raise ValueError(f"{ticker} 在 {effective_date} 之前没有可用行情。")

        prices = [float(row["price"]) for row in rows]
        latest = rows[-1]
        previous_return = None
        if len(prices) >= 2 and prices[-2] != 0:
            previous_return = (prices[-1] / prices[-2] - 1) * 100

        def period_return(sessions: int) -> float | None:
            if len(prices) <= sessions:
                return None
            base = prices[-sessions - 1]
            return (prices[-1] / base - 1) * 100 if base else None

        daily_returns = [
            prices[i] / prices[i - 1] - 1
            for i in range(1, len(prices))
            if prices[i - 1] != 0
        ]
        last_20_returns = daily_returns[-20:]
        volatility_annualized_pct = (
            pstdev(last_20_returns) * math.sqrt(252) * 100
            if len(last_20_returns) >= 2 else None
        )
        sma20 = sum(prices[-20:]) / min(20, len(prices))
        sma20_gap_pct = (prices[-1] / sma20 - 1) * 100 if sma20 else None

        all_news = _load_news(conn, trading_dates)
        news = _news_features(all_news, effective_date)
        macro = _load_macro_snapshot(conn, effective_date)

        news_table_count = conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0] if _table_exists(conn, "news_items") else 0
        macro_table_count = conn.execute("SELECT COUNT(*) FROM macro_observations").fetchone()[0] if _table_exists(conn, "macro_observations") else 0

        return {
            "ticker": ticker,
            "as_of_date": effective_date,
            "latest_price": latest["price"],
            "price_field": "adj_close" if "adj_close" in _table_columns(conn, "market_daily") else "close",
            "market": {
                "latest_trade_date": latest["trade_date"],
                "latest_close": latest["close"],
                "latest_adjusted_price": latest["price"],
                "return_1_session_pct": previous_return,
                "return_5_sessions_pct": period_return(5),
                "return_20_sessions_pct": period_return(20),
                "annualized_volatility_20_session_pct": volatility_annualized_pct,
                "price_vs_sma20_pct": sma20_gap_pct,
                "sessions_available": len(rows),
            },
            "news": {
                "rows_in_database": news_table_count,
                **news,
            },
            "macro": {
                "rows_in_database": macro_table_count,
                **macro,
            },
            "modeling_policy": {
                "macro_features_allowed_in_historical_backtest": False,
                "news_features_allowed_in_training_by_default": False,
                "reason": (
                    "先用于当前情境展示与 AI 解释。数据库新闻仅 40 条左右，历史覆盖有限；"
                    "宏观表没有真实发布时间/历史版本字段。完成时间点可用性核验后，再做严格样本外特征实验。"
                ),
            },
        }
    finally:
        conn.close()


def write_context_snapshot(
    db_path: str | Path = DEFAULT_DB_PATH,
    ticker: str = "QQQM",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write a timestamped snapshot JSON and a stable latest JSON; no DB writes."""
    data = build_market_context_snapshot(db_path=db_path, ticker=ticker)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dated_path = out_dir / f"{ticker}_context_snapshot_{stamp}.json"
    latest_path = out_dir / f"{ticker}_context_snapshot_latest.json"
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    dated_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return latest_path
