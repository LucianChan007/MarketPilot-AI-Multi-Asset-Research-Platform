from __future__ import annotations

import csv
import io
import json
import os
import re
from datetime import date, datetime, time, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from app.db import initialize_database, upsert_history


def _period_start(period: str) -> str:
    """Convert a small yfinance-like period string to an ISO start date."""
    value = period.strip().lower()
    today = date.today()
    if value == "max":
        # EODHD free plans only provide one year. Paid plans may provide more.
        return "2000-01-01"
    match = re.fullmatch(r"(\d+)(d|mo|y)", value)
    if not match:
        raise ValueError("period 仅支持例如 30d、6mo、1y、5y 或 max。")
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        start = today - timedelta(days=amount)
    elif unit == "mo":
        start = today - timedelta(days=round(amount * 30.4375))
    else:
        start = today - timedelta(days=round(amount * 365.2425))
    return start.isoformat()


def _fetch_eodhd(ticker: str, period: str) -> pd.DataFrame:
    api_key = os.getenv("EODHD_API_KEY", "").strip()
    if not api_key or api_key == "replace_with_your_eodhd_api_key":
        raise RuntimeError(
            "未配置 EODHD_API_KEY。请在项目根目录的 .env 文件中填写 EODHD API Token。"
        )

    symbol = ticker.upper()
    if not symbol.endswith(".US"):
        symbol = f"{symbol}.US"
    params = {
        "api_token": api_key,
        "fmt": "json",
        "from": _period_start(period),
        "to": date.today().isoformat(),
        "period": "d",
    }
    url = f"https://eodhd.com/api/eod/{symbol}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "QQQM-Investment-Assistant/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"EODHD HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 EODHD：{exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("EODHD 返回的内容不是有效 JSON。") from exc

    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("error") or payload.get("detail") or str(payload)
        raise RuntimeError(f"EODHD 返回错误：{message}")
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(
            f"EODHD 没有返回 {symbol} 的日线数据。请检查标的代码、API Token 和套餐权限。"
        )

    df = pd.DataFrame(payload)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"EODHD 返回字段不完整：{list(df.columns)}")

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "Adj Close",
            "volume": "Volume",
        }
    )
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]


def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    instrument = yf.Ticker(ticker)
    frame = instrument.history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=True,
        repair=True,
        raise_errors=True,
    )
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo Finance 没有返回日线数据。")
    return frame


def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame:
    """Fetch public daily OHLCV CSV from Stooq as a no-key final fallback."""
    symbol = ticker.strip().lower()
    if symbol.endswith(".us"):
        symbol = symbol[:-3]
    if not re.fullmatch(r"[a-z0-9.^=_-]{1,25}", symbol):
        raise ValueError(f"标的代码不适用于 Stooq：{ticker}")
    start = _period_start(period).replace("-", "")
    end = datetime.now(ZoneInfo("America/New_York")).date().strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={symbol}.us&i=d&d1={start}&d2={end}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 QQQM-Investment-Assistant/1.0"})
    try:
        with urlopen(request, timeout=25) as response:
            payload = response.read().decode("utf-8-sig", errors="replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Stooq CSV 请求失败：{exc}") from exc
    if not payload.strip() or "No data" in payload[:100]:
        raise RuntimeError("Stooq 没有返回日线数据。")
    try:
        frame = pd.read_csv(io.StringIO(payload))
    except Exception as exc:
        raise RuntimeError(f"Stooq CSV 无法解析：{exc}") from exc
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns) or frame.empty:
        raise RuntimeError(f"Stooq CSV 字段不完整或无数据：{list(frame.columns)}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame["Adj Close"] = frame["Close"]
    frame["Dividends"] = 0.0
    frame["Stock Splits"] = 0.0
    if frame.empty:
        raise RuntimeError("Stooq 数据没有可用价格行。")
    return frame[["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]


def _latest_completed_nasdaq_session() -> date | None:
    """Get the last completed NASDAQ session in New York time, when calendar data exists."""
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    cutoff = now_ny.date() if now_ny.time().replace(tzinfo=None) >= time(16, 30) else now_ny.date() - timedelta(days=1)
    try:
        import pandas_market_calendars as mcal
        calendar = mcal.get_calendar("NASDAQ")
        schedule = calendar.schedule(start_date=(cutoff - timedelta(days=14)).isoformat(), end_date=cutoff.isoformat())
        if schedule.empty:
            return None
        return pd.Timestamp(schedule.index[-1]).date()
    except ImportError:
        # Avoid rejecting data based on a weekday-only holiday guess.
        return None
    except Exception as exc:
        print(f"[行情日期校验提示] 无法读取 NASDAQ 日历，跳过严格日期校验：{exc}")
        return None


def _validate_market_freshness(ticker: str, history: pd.DataFrame, source: str) -> None:
    if history.empty:
        raise RuntimeError(f"{ticker} 没有返回行情。请检查网络、标的代码、数据源权限或请求限制。")
    # The index is sorted by every provider adapter above; normalize once more for safety.
    latest_value = pd.to_datetime(history.index, errors="coerce").max()
    if pd.isna(latest_value):
        raise RuntimeError(f"{source} 返回的行情日期无法解析。")
    latest_date = pd.Timestamp(latest_value).date()
    expected_date = _latest_completed_nasdaq_session()
    if expected_date and latest_date < expected_date:
        raise RuntimeError(
            f"{source} 返回的最新日期是 {latest_date.isoformat()}，但按 NASDAQ 交易日历，"
            f"最近一个已完成交易日应为 {expected_date.isoformat()}。已拒绝把旧数据标记为更新成功。"
        )


def sync_ticker(ticker: str, period: str = "1y") -> int:
    """Fetch daily OHLCV; try multiple sources and reject stale successful responses."""
    provider = os.getenv("MARKET_DATA_PROVIDER", "eodhd").strip().lower()
    ticker = ticker.upper()
    print(f"[行情同步] 首选数据源={provider}，正在下载 {ticker} 的 {period} 日线数据……")

    attempts: list[tuple[str, Any]] = []
    if provider == "eodhd":
        attempts = [("eodhd", _fetch_eodhd), ("yfinance_fallback", _fetch_yfinance), ("stooq_fallback", _fetch_stooq)]
    elif provider == "yfinance":
        attempts = [("yfinance", _fetch_yfinance), ("stooq_fallback", _fetch_stooq)]
    elif provider == "stooq":
        attempts = [("stooq", _fetch_stooq)]
    else:
        raise ValueError("MARKET_DATA_PROVIDER 仅支持 eodhd、yfinance 或 stooq。")

    failures: list[str] = []
    history: pd.DataFrame | None = None
    source = ""
    for source_name, fetcher in attempts:
        try:
            candidate = fetcher(ticker, period)
            if candidate is None or candidate.empty:
                raise RuntimeError("数据源返回空数据。")
            candidate = candidate.sort_index()
            _validate_market_freshness(ticker, candidate, source_name)
            history, source = candidate, source_name
            break
        except Exception as exc:
            failures.append(f"{source_name}: {type(exc).__name__}: {exc}")
            print(f"[行情源失败] {source_name}：{type(exc).__name__}: {exc}")
            if source_name != attempts[-1][0]:
                print("[行情备用源] 尝试下一个数据源……")

    if history is None:
        raise RuntimeError(f"{ticker} 行情同步失败，所有数据源均未提供可验证的最新数据。" + "；".join(failures))

    initialize_database()
    saved = upsert_history(ticker, history, source=source)
    latest_date = pd.to_datetime(history.index, errors="coerce").max().date().isoformat()
    print(f"[完成] {ticker}: 本次写入/更新 {saved} 条记录，最新行情日期 {latest_date}，来源 {source}")
    print("[校验] 已检查数据日期，不会将低于最近已完成 NASDAQ 交易日的响应视为同步成功。")
    if provider == "eodhd" and period.strip().lower() not in {"1y", "12mo", "365d"}:
        print("[提示] EODHD 免费方案历史范围可能受限；如请求很长历史，请检查套餐权限与实际返回日期。")
    return saved
