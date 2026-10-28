from __future__ import annotations

"""Multi-asset intraday research engine (V2).

- Keeps multi-asset rows isolated by symbol/feed.
- Downloads public Yahoo Finance chart data on explicit user action.
- Stores immutable forecasts by symbol/as-of/target date.
- Settles only forecasts whose target session is complete and whose source matches.
- Uses a transparent NumPy ridge model and rolling out-of-sample diagnostics.
- News/LLM explanations are context only and are never silently treated as price predictions.
"""

import json
import io
import math
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import date, datetime, time as daytime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

CONFIG_PATH = ROOT / "config" / "instruments.json"
DB_PATH = ROOT / "data" / "multi_asset_research.db"
REPORT_ROOT = ROOT / "reports" / "multi_asset"
ARCHIVE_ROOT = REPORT_ROOT / "forecast_archive"
NEWS_ROOT = REPORT_ROOT / "news_analysis"
NY = ZoneInfo("America/New_York")
YAHOO_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
TIME_SLOTS = ["09:30"] + [
    (datetime.combine(date(2000, 1, 1), daytime(9, 30)) + timedelta(minutes=15 * i)).strftime("%H:%M")
    for i in range(1, 27)
]

DEFAULT_INSTRUMENTS = [
    {"symbol": "QQQM", "name": "Invesco Nasdaq 100 ETF", "type": "etf", "enabled": True, "description": "纳斯达克100指数ETF，四资产平行研究之一"},
    {"symbol": "AAPL", "name": "Apple Inc.", "type": "stock", "enabled": True, "description": "苹果公司"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "stock", "enabled": True, "description": "微软公司"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "stock", "enabled": True, "description": "英伟达公司"},
]


@contextmanager
def connect_db(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def init_db(path: Path = DB_PATH) -> None:
    with connect_db(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS asset_daily (
              symbol TEXT NOT NULL, trade_date TEXT NOT NULL, feed TEXT NOT NULL,
              open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
              adj_close REAL, volume REAL DEFAULT 0, fetched_at TEXT NOT NULL,
              PRIMARY KEY(symbol, trade_date, feed)
            );
            CREATE INDEX IF NOT EXISTS ix_asset_daily_date ON asset_daily(symbol, trade_date);
            CREATE TABLE IF NOT EXISTS asset_intraday_bars (
              symbol TEXT NOT NULL, feed TEXT NOT NULL, timeframe TEXT NOT NULL,
              timestamp_utc TEXT NOT NULL, timestamp_ny TEXT NOT NULL, trade_date_ny TEXT NOT NULL,
              session_time_ny TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL,
              low REAL NOT NULL, close REAL NOT NULL, volume REAL DEFAULT 0, fetched_at TEXT NOT NULL,
              PRIMARY KEY(symbol, feed, timeframe, timestamp_utc)
            );
            CREATE INDEX IF NOT EXISTS ix_asset_intraday_date ON asset_intraday_bars(symbol, feed, trade_date_ny);
            CREATE TABLE IF NOT EXISTS asset_news (
              symbol TEXT NOT NULL, url TEXT NOT NULL, title TEXT NOT NULL, summary TEXT DEFAULT '',
              source TEXT DEFAULT '', category TEXT DEFAULT '', published_at TEXT DEFAULT '',
              fetched_at TEXT NOT NULL, ai_summary TEXT DEFAULT '', ai_sentiment TEXT DEFAULT '',
              ai_confidence REAL, ai_model TEXT DEFAULT '',
              PRIMARY KEY(symbol, url)
            );
            CREATE INDEX IF NOT EXISTS ix_asset_news_date ON asset_news(symbol, published_at);
            CREATE TABLE IF NOT EXISTS asset_evaluations (
              symbol TEXT NOT NULL, target_date TEXT NOT NULL, forecast_file TEXT NOT NULL,
              actual_feed TEXT NOT NULL, bar_count INTEGER NOT NULL, mae_usd REAL,
              mae_bps REAL, close_error_pct REAL, baseline_mae_bps REAL,
              relative_improvement_pct REAL, evaluated_at TEXT NOT NULL, status TEXT NOT NULL,
              PRIMARY KEY(symbol, target_date, forecast_file)
            );
            CREATE TABLE IF NOT EXISTS asset_sync_runs (
              run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL,
              finished_at TEXT NOT NULL, symbol TEXT NOT NULL, success INTEGER NOT NULL,
              daily_latest_date TEXT, intraday_latest_complete_date TEXT,
              daily_rows INTEGER DEFAULT 0, intraday_rows INTEGER DEFAULT 0,
              news_rows INTEGER DEFAULT 0, source TEXT DEFAULT 'yahoo', error TEXT DEFAULT ''
            );
            """
        )


def load_instruments(config_path: Path = CONFIG_PATH, enabled_only: bool = True) -> list[dict[str, Any]]:
    cfg = _read_json(config_path) or {}
    rows = cfg.get("instruments")
    if not isinstance(rows, list) or not rows:
        rows = DEFAULT_INSTRUMENTS
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", symbol) or symbol in seen:
            continue
        seen.add(symbol)
        item = {
            "symbol": symbol,
            "name": str(row.get("name") or symbol).strip()[:120],
            "type": str(row.get("type") or "stock").lower(),
            "enabled": bool(row.get("enabled", True)),
            "description": str(row.get("description") or "").strip()[:250],
        }
        if not enabled_only or item["enabled"]:
            result.append(item)
    return result


def add_instrument(symbol: str, name: str | None = None, instrument_type: str = "stock", config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    symbol = str(symbol).strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", symbol):
        raise ValueError("股票代码格式无效，只允许 1–15 位英文字母、数字、点、等号或连字符。")
    cfg = _read_json(config_path) or {"schema_version": 1, "default_feed": "yahoo", "timeframe": "15m", "instruments": []}
    instruments = cfg.get("instruments") if isinstance(cfg.get("instruments"), list) else []
    for row in instruments:
        if str(row.get("symbol", "")).upper() == symbol:
            row["enabled"] = True
            _write_json(config_path, cfg)
            return row
    item = {"symbol": symbol, "name": str(name or symbol).strip()[:120], "type": instrument_type if instrument_type in {"stock", "etf"} else "stock", "enabled": True, "description": "用户添加的研究标的"}
    instruments.append(item)
    cfg["instruments"] = instruments
    _write_json(config_path, cfg)
    return item


def set_instrument_enabled(symbol: str, enabled: bool, config_path: Path = CONFIG_PATH) -> None:
    cfg = _read_json(config_path) or {}
    rows = cfg.get("instruments") or []
    found = False
    for row in rows:
        if str(row.get("symbol", "")).upper() == str(symbol).upper():
            row["enabled"] = bool(enabled)
            found = True
    if not found:
        raise KeyError(f"配置里找不到标的 {symbol}")
    cfg["instruments"] = rows
    _write_json(config_path, cfg)


def _fetch_chart_payload(symbol: str, period: str, interval: str) -> dict[str, Any]:
    errors: list[str] = []
    for host in YAHOO_HOSTS:
        try:
            response = requests.get(
                f"{host}/v8/finance/chart/{quote_plus(symbol, safe='.^=-')}",
                params={"range": period, "interval": interval, "events": "div,splits"},
                headers={"User-Agent": "Mozilla/5.0 QQQMResearchV2/1.0", "Accept": "application/json,*/*"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            chart = payload.get("chart") if isinstance(payload, dict) else None
            rows = chart.get("result") if isinstance(chart, dict) else None
            if not rows or not isinstance(rows[0], dict):
                error = chart.get("error") if isinstance(chart, dict) else None
                raise RuntimeError(str(error or "数据源未返回行情数据"))
            return rows[0]
        except Exception as exc:
            errors.append(f"{host}: {type(exc).__name__}: {str(exc)[:200]}")
    raise RuntimeError("Yahoo Finance 公开图表接口请求失败：" + "；".join(errors))


def _payload_frame(payload: dict[str, Any], symbol: str, interval: str) -> pd.DataFrame:
    timestamps = payload.get("timestamp") or []
    indicators = payload.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0] or {}
    if not timestamps:
        return pd.DataFrame()
    n = len(timestamps)
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce"),
        "open": quote.get("open") or [None] * n,
        "high": quote.get("high") or [None] * n,
        "low": quote.get("low") or [None] * n,
        "close": quote.get("close") or [None] * n,
        "volume": quote.get("volume") or [0] * n,
    })
    if interval == "1d":
        adj = indicators.get("adjclose") or []
        values = ((adj[0] or {}).get("adjclose") if adj else None)
        frame["adj_close"] = values if values and len(values) == n else frame["close"]
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if interval == "1d":
        frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc", "open", "high", "low", "close"]).copy()
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame["timestamp_ny"] = frame["timestamp_utc"].dt.tz_convert(NY)
    if interval == "1d":
        # Yahoo daily candle timestamps denote the exchange date in UTC; converting
        # midnight UTC to New York first can shift the key to the prior calendar day.
        frame["trade_date_ny"] = frame["timestamp_utc"].dt.strftime("%Y-%m-%d")
        frame["session_time_ny"] = "00:00"
    else:
        frame["trade_date_ny"] = frame["timestamp_ny"].dt.strftime("%Y-%m-%d")
        frame["session_time_ny"] = frame["timestamp_ny"].dt.strftime("%H:%M")
        frame = frame[(frame["session_time_ny"] >= "09:30") & (frame["session_time_ny"] < "16:00")]
    return frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _fetch_yfinance_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch Yahoo data through yfinance as a fallback to the direct chart endpoint.

    yfinance handles Yahoo's cookie/crumb flow, which can behave differently from a
    bare requests call. It is still Yahoo data, not an independent data vendor.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("未安装 yfinance 备用客户端；请运行 Prepare_MarketPilot_Environment.bat") from exc
    try:
        frame = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
            repair=True,
            timeout=15,
            raise_errors=True,
        )
    except TypeError:
        # Compatibility path for older yfinance versions, though requirements pin a recent version.
        frame = yf.Ticker(symbol).history(
            period=period, interval=interval, auto_adjust=False, actions=False,
            prepost=False, timeout=15,
        )
    if frame is None or frame.empty:
        raise RuntimeError("yfinance 未返回有效行情行")
    return frame


def _yfinance_daily_frame(symbol: str, period: str = "1y") -> pd.DataFrame:
    history = _fetch_yfinance_history(symbol, period, "1d")
    idx = pd.DatetimeIndex(pd.to_datetime(history.index, errors="coerce"))
    if idx.tz is not None:
        idx_local = idx.tz_convert(NY)
    else:
        idx_local = idx
    close = pd.to_numeric(history.get("Close"), errors="coerce")
    adj = pd.to_numeric(history.get("Adj Close", history.get("Close")), errors="coerce")
    volume = pd.to_numeric(history.get("Volume", pd.Series(0, index=history.index)), errors="coerce")
    frame = pd.DataFrame({
        "trade_date_ny": [stamp.strftime("%Y-%m-%d") if not pd.isna(stamp) else None for stamp in idx_local],
        "open": pd.to_numeric(history.get("Open"), errors="coerce").to_numpy(),
        "high": pd.to_numeric(history.get("High"), errors="coerce").to_numpy(),
        "low": pd.to_numeric(history.get("Low"), errors="coerce").to_numpy(),
        "close": close.to_numpy(),
        "adj_close": adj.to_numpy(),
        "volume": volume.fillna(0).to_numpy(),
    })
    frame = frame.dropna(subset=["trade_date_ny", "open", "high", "low", "close"])
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame = frame.sort_values("trade_date_ny").drop_duplicates("trade_date_ny", keep="last").reset_index(drop=True)
    return frame


def _yfinance_intraday_frame(symbol: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    history = _fetch_yfinance_history(symbol, period, interval)
    idx = pd.DatetimeIndex(pd.to_datetime(history.index, errors="coerce"))
    if idx.tz is None:
        idx_local = idx.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    else:
        idx_local = idx.tz_convert(NY)
    frame = pd.DataFrame({
        "timestamp_ny": idx_local,
        "open": pd.to_numeric(history.get("Open"), errors="coerce").to_numpy(),
        "high": pd.to_numeric(history.get("High"), errors="coerce").to_numpy(),
        "low": pd.to_numeric(history.get("Low"), errors="coerce").to_numpy(),
        "close": pd.to_numeric(history.get("Close"), errors="coerce").to_numpy(),
        "volume": pd.to_numeric(history.get("Volume", pd.Series(0, index=history.index)), errors="coerce").fillna(0).to_numpy(),
    })
    frame = frame.dropna(subset=["timestamp_ny", "open", "high", "low", "close"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_ny"], utc=True, errors="coerce")
    frame["trade_date_ny"] = frame["timestamp_ny"].dt.strftime("%Y-%m-%d")
    frame["session_time_ny"] = frame["timestamp_ny"].dt.strftime("%H:%M")
    frame = frame[(frame["session_time_ny"] >= "09:30") & (frame["session_time_ny"] < "16:00")]
    return frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def fetch_daily(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily bars via Yahoo chart API, then yfinance if direct requests fail."""
    errors: list[str] = []
    try:
        frame = _payload_frame(_fetch_chart_payload(symbol, period, "1d"), symbol, "1d")
        if frame.empty:
            raise RuntimeError("Yahoo chart API 返回空日线")
        frame.attrs["fetch_method"] = "yahoo_chart_api"
        frame.attrs["fetch_warnings"] = []
        return frame
    except Exception as exc:
        errors.append(f"chart_api: {type(exc).__name__}: {str(exc)[:180]}")
    try:
        frame = _yfinance_daily_frame(symbol, period)
        if frame.empty:
            raise RuntimeError("yfinance 返回空日线")
        frame.attrs["fetch_method"] = "yfinance_fallback"
        frame.attrs["fetch_warnings"] = errors.copy()
        return frame
    except Exception as exc:
        errors.append(f"yfinance: {type(exc).__name__}: {str(exc)[:180]}")
    raise RuntimeError("Yahoo 日线获取失败，已尝试直接图表接口和 yfinance 备用路径；" + "；".join(errors))


def fetch_intraday(symbol: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    """Fetch regular-session intraday bars via Yahoo chart API, then yfinance fallback."""
    errors: list[str] = []
    try:
        frame = _payload_frame(_fetch_chart_payload(symbol, period, interval), symbol, interval)
        if frame.empty:
            raise RuntimeError(f"{symbol} 的公开十五分钟行情为空")
        frame.attrs["fetch_method"] = "yahoo_chart_api"
        frame.attrs["fetch_warnings"] = []
        return frame
    except Exception as exc:
        errors.append(f"chart_api: {type(exc).__name__}: {str(exc)[:180]}")
    try:
        frame = _yfinance_intraday_frame(symbol, period, interval)
        if frame.empty:
            raise RuntimeError(f"yfinance 未返回 {symbol} 有效分时")
        frame.attrs["fetch_method"] = "yfinance_fallback"
        frame.attrs["fetch_warnings"] = errors.copy()
        return frame
    except Exception as exc:
        errors.append(f"yfinance: {type(exc).__name__}: {str(exc)[:180]}")
    raise RuntimeError("Yahoo 分时获取失败，已尝试直接图表接口和 yfinance 备用路径；" + "；".join(errors))


def fetch_stooq_daily(symbol: str, period_days: int = 370) -> pd.DataFrame:
    """Fetch a public daily-only fallback. These rows are NOT eligible to pair with Yahoo intraday forecasts."""
    ticker = str(symbol).strip().lower()
    if ticker.endswith(".us"):
        ticker = ticker[:-3]
    if not re.fullmatch(r"[a-z0-9.^=_-]{1,25}", ticker):
        raise ValueError(f"标的代码不适用于 Stooq 日线备用源：{symbol}")
    if ticker.startswith("^"):
        raise ValueError("Stooq 日线备用源暂不支持指数符号")
    today = datetime.now(NY).date()
    start = today - timedelta(days=max(30, min(int(period_days), 3650)))
    url = f"https://stooq.com/q/d/l/?s={quote_plus(ticker + '.us')}&i=d&d1={start:%Y%m%d}&d2={today:%Y%m%d}"
    try:
        response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0 MarketPilotAI/2.2", "Accept": "text/csv,*/*"})
        response.raise_for_status()
        payload = response.text
    except Exception as exc:
        raise RuntimeError(f"Stooq 日线备用请求失败（{type(exc).__name__}）") from exc
    if not payload.strip() or "No data" in payload[:120]:
        raise RuntimeError("Stooq 没有返回日线数据")
    try:
        raw = pd.read_csv(io.StringIO(payload))
    except Exception as exc:
        raise RuntimeError("Stooq 返回内容无法解析为 CSV") from exc
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if raw.empty or not required.issubset(raw.columns):
        raise RuntimeError("Stooq CSV 字段不完整或没有数据行")
    frame = pd.DataFrame({
        "trade_date_ny": pd.to_datetime(raw["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "open": pd.to_numeric(raw["Open"], errors="coerce"),
        "high": pd.to_numeric(raw["High"], errors="coerce"),
        "low": pd.to_numeric(raw["Low"], errors="coerce"),
        "close": pd.to_numeric(raw["Close"], errors="coerce"),
        "adj_close": pd.to_numeric(raw["Close"], errors="coerce"),
        "volume": pd.to_numeric(raw["Volume"], errors="coerce").fillna(0),
    })
    frame = frame.dropna(subset=["trade_date_ny", "open", "high", "low", "close"])
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame = frame.sort_values("trade_date_ny").drop_duplicates("trade_date_ny", keep="last").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Stooq 日线没有有效价格行")
    frame.attrs["fetch_method"] = "stooq_daily_fallback"
    return frame

def _eodhd_token() -> str | None:
    for key in ("EODHD_API_KEY", "EODHD_API_TOKEN", "EODHD_TOKEN", "EODHD_APIKEY"):
        value = os.getenv(key, "").strip()
        if value and not value.lower().startswith(("replace", "your_", "paste_")):
            return value
    return None


def fetch_eodhd_daily(symbol: str) -> pd.DataFrame:
    """Optional secondary provider; it is used only when a real EODHD key exists."""
    if not _eodhd_token():
        raise RuntimeError("未配置 EODHD API Key，跳过该备用日线源。")
    try:
        from app.market_data import _fetch_eodhd
        history = _fetch_eodhd(symbol.upper(), "1y")
    except Exception as exc:
        # Do not include the exception text: some HTTP libraries echo credential-bearing URLs.
        raise RuntimeError(f"EODHD {symbol} 日线获取失败（{type(exc).__name__}）。请检查密钥、权限和配额。") from exc
    if history is None or history.empty:
        raise RuntimeError(f"EODHD 没有返回 {symbol} 日线。")
    idx = pd.to_datetime(history.index, errors="coerce")
    frame = pd.DataFrame({
        "trade_date_ny": idx.strftime("%Y-%m-%d"),
        "open": pd.to_numeric(history["Open"], errors="coerce").to_numpy(),
        "high": pd.to_numeric(history["High"], errors="coerce").to_numpy(),
        "low": pd.to_numeric(history["Low"], errors="coerce").to_numpy(),
        "close": pd.to_numeric(history["Close"], errors="coerce").to_numpy(),
        "adj_close": pd.to_numeric(history.get("Adj Close", history["Close"]), errors="coerce").to_numpy(),
        "volume": pd.to_numeric(history.get("Volume", pd.Series(0, index=history.index)), errors="coerce").fillna(0).to_numpy(),
    })
    frame = frame.dropna(subset=["trade_date_ny", "open", "high", "low", "close"])
    if frame.empty:
        raise RuntimeError(f"EODHD {symbol} 日线字段不完整。")
    return frame.reset_index(drop=True)


def fetch_eodhd_intraday(symbol: str, period_days: int = 60) -> pd.DataFrame:
    """Fetch EODHD 5-minute bars and aggregate real groups of three into 15-minute bars."""
    token = _eodhd_token()
    if not token:
        raise RuntimeError("未配置 EODHD API Key，跳过该备用分时源。")
    if not isinstance(period_days, int) or isinstance(period_days, bool) or not 2 <= period_days <= 600:
        raise ValueError("period_days 必须为 2 到 600 之间的整数天数。")
    now = datetime.now(timezone.utc)
    params = {"api_token": token, "interval": "5m", "from": int((now - timedelta(days=period_days)).timestamp()), "to": int(now.timestamp()), "fmt": "json"}
    ticker = symbol.upper() if symbol.upper().endswith(".US") else f"{symbol.upper()}.US"
    try:
        response = requests.get(f"https://eodhd.com/api/intraday/{quote_plus(ticker, safe='.')}", params=params, timeout=25, headers={"User-Agent": "QQQMResearchV2/1.0"})
        if response.status_code != 200:
            raise RuntimeError(f"EODHD HTTP {response.status_code}；请检查密钥、权限或配额。")
        payload = response.json()
    except Exception as exc:
        # The response URL may contain api_token, so never print exception text here.
        raise RuntimeError(f"EODHD {symbol} 分时请求失败（{type(exc).__name__}）。请检查密钥、权限和配额。") from exc
    if isinstance(payload, dict) or not isinstance(payload, list) or not payload:
        raise RuntimeError(f"EODHD 没有返回 {symbol} 的五分钟行情数组。")
    frame = pd.DataFrame(payload)
    if not {"datetime", "open", "high", "low", "close"}.issubset(frame.columns):
        raise RuntimeError("EODHD 分时字段缺失。")
    parsed_utc = []
    for raw in frame["datetime"]:
        try:
            stamp = pd.Timestamp(raw)
            if pd.isna(stamp):
                parsed_utc.append(pd.NaT)
            elif stamp.tzinfo is None:
                parsed_utc.append(stamp.tz_localize(NY, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC"))
            else:
                parsed_utc.append(stamp.tz_convert("UTC"))
        except (TypeError, ValueError, OverflowError):
            parsed_utc.append(pd.NaT)
    frame["timestamp_utc"] = pd.to_datetime(parsed_utc, utc=True, errors="coerce")
    frame["timestamp_ny"] = frame["timestamp_utc"].dt.tz_convert(NY)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in frame:
            frame[col] = 0
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp_ny", "open", "high", "low", "close"])
    clock = frame["timestamp_ny"].dt.strftime("%H:%M")
    frame = frame[(clock >= "09:30") & (clock < "16:00")].copy()
    frame["trade_date_ny"] = frame["timestamp_ny"].dt.strftime("%Y-%m-%d")
    minute = frame["timestamp_ny"].dt.hour * 60 + frame["timestamp_ny"].dt.minute
    frame["slot_minute"] = 570 + ((minute - 570) // 15) * 15
    frame["session_time_ny"] = frame["slot_minute"].map(lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}")
    grouped = []
    for (day, slot), group in frame.sort_values("timestamp_ny").groupby(["trade_date_ny", "session_time_ny"], sort=True):
        if len(group) < 3:
            continue
        first, last = group.iloc[0], group.iloc[-1]
        local_stamp = datetime.combine(date.fromisoformat(str(day)), daytime.fromisoformat(str(slot)), tzinfo=NY)
        grouped.append({"timestamp_utc": pd.Timestamp(local_stamp).tz_convert("UTC"), "timestamp_ny": pd.Timestamp(local_stamp), "trade_date_ny": str(day), "session_time_ny": str(slot), "open": float(first["open"]), "high": float(group["high"].max()), "low": float(group["low"].min()), "close": float(last["close"]), "volume": float(group["volume"].fillna(0).sum())})
    result = pd.DataFrame(grouped)
    if result.empty:
        raise RuntimeError(f"EODHD {symbol} 没有足够五分钟记录组成完整十五分钟 bar。")
    return result.reset_index(drop=True)


def _is_complete_session(day: str, frame: pd.DataFrame, now_ny: datetime | None = None) -> bool:
    now_ny = now_ny or datetime.now(timezone.utc).astimezone(NY)
    try:
        parsed_day = date.fromisoformat(str(day)[:10])
    except ValueError:
        return False
    if parsed_day > now_ny.date():
        return False
    if parsed_day == now_ny.date() and now_ny.time().replace(tzinfo=None) < daytime(16, 15):
        return False
    if frame.empty or "session_time_ny" not in frame:
        return False
    times = set(frame["session_time_ny"].astype(str))
    return len(times) >= 24 and "09:30" in times and max(times) >= "15:45"


def save_daily(symbol: str, frame: pd.DataFrame, db_path: Path = DB_PATH, feed: str = "yahoo") -> int:
    if frame is None or frame.empty:
        return 0
    fetched = _utc_now()
    rows = []
    for _, row in frame.iterrows():
        rows.append((symbol.upper(), str(row["trade_date_ny"]), feed, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("adj_close", row["close"])), float(row.get("volume", 0) or 0), fetched))
    init_db(db_path)
    with connect_db(db_path) as conn:
        conn.executemany("""INSERT INTO asset_daily(symbol,trade_date,feed,open,high,low,close,adj_close,volume,fetched_at)
        VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,trade_date,feed) DO UPDATE SET
        open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,adj_close=excluded.adj_close,volume=excluded.volume,fetched_at=excluded.fetched_at""", rows)
    return len(rows)


def save_intraday(symbol: str, frame: pd.DataFrame, db_path: Path = DB_PATH, feed: str = "yahoo") -> int:
    if frame is None or frame.empty:
        return 0
    fetched = _utc_now()
    rows = []
    for _, row in frame.iterrows():
        ts_utc = pd.Timestamp(row["timestamp_utc"]).tz_convert("UTC").isoformat(timespec="seconds").replace("+00:00", "Z")
        ts_ny = pd.Timestamp(row["timestamp_ny"]).isoformat(timespec="seconds")
        rows.append((symbol.upper(), feed, "15m", ts_utc, ts_ny, str(row["trade_date_ny"]), str(row["session_time_ny"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0) or 0), fetched))
    init_db(db_path)
    with connect_db(db_path) as conn:
        conn.executemany("""INSERT INTO asset_intraday_bars(symbol,feed,timeframe,timestamp_utc,timestamp_ny,trade_date_ny,session_time_ny,open,high,low,close,volume,fetched_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,feed,timeframe,timestamp_utc) DO UPDATE SET
        timestamp_ny=excluded.timestamp_ny,trade_date_ny=excluded.trade_date_ny,session_time_ny=excluded.session_time_ny,
        open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume,fetched_at=excluded.fetched_at""", rows)
    return len(rows)


def _daily_rows(symbol: str, db_path: Path = DB_PATH, feed: str | None = None) -> pd.DataFrame:
    init_db(db_path)
    with connect_db(db_path) as conn:
        sql = "SELECT trade_date,open,high,low,close,adj_close,volume,feed FROM asset_daily WHERE symbol=?"
        params: tuple[Any, ...] = (symbol.upper(),)
        if feed:
            sql += " AND lower(feed)=?"
            params = (symbol.upper(), feed.lower())
        sql += " ORDER BY trade_date"
        frame = pd.read_sql_query(sql, conn, params=params)
    if frame.empty:
        return frame
    frame["trade_date"] = frame["trade_date"].astype(str)
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.dropna(subset=["trade_date", "open", "high", "low", "close"]).drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)


def _intraday_rows(symbol: str, db_path: Path = DB_PATH, feed: str = "yahoo") -> pd.DataFrame:
    init_db(db_path)
    with connect_db(db_path) as conn:
        frame = pd.read_sql_query("""SELECT timestamp_utc,timestamp_ny,trade_date_ny,session_time_ny,open,high,low,close,volume,feed FROM asset_intraday_bars WHERE symbol=? AND feed=? AND timeframe='15m' ORDER BY timestamp_utc""", conn, params=(symbol.upper(), feed.lower()))
    if frame.empty:
        return frame
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["timestamp_ny"] = frame["timestamp_utc"].dt.tz_convert(NY)
    for c in ("open", "high", "low", "close", "volume"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc", "open", "high", "low", "close"])
    frame["trade_date_ny"] = frame["timestamp_ny"].dt.strftime("%Y-%m-%d")
    frame["session_time_ny"] = frame["timestamp_ny"].dt.strftime("%H:%M")
    return frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def complete_sessions(symbol: str, db_path: Path = DB_PATH, feed: str = "yahoo", now_ny: datetime | None = None) -> list[dict[str, Any]]:
    frame = _intraday_rows(symbol, db_path, feed)
    sessions = []
    if frame.empty:
        return sessions
    for day, group in frame.groupby("trade_date_ny", sort=True):
        group = group.sort_values("timestamp_ny").drop_duplicates("session_time_ny", keep="last").reset_index(drop=True)
        if not _is_complete_session(str(day), group, now_ny):
            continue
        first = group[group["session_time_ny"] == "09:30"]
        if first.empty:
            continue
        sessions.append({"date": str(day), "bars": group, "open": float(first.iloc[0]["open"]), "high": float(group["high"].max()), "low": float(group["low"].min()), "close": float(group.iloc[-1]["close"]), "feed": feed})
    return sessions


def latest_dates(symbol: str, db_path: Path = DB_PATH) -> dict[str, Any]:
    candidates = []
    requested = os.getenv("MULTI_ASSET_MARKET_DATA_PROVIDER", os.getenv("MARKET_DATA_PROVIDER", "yahoo")).strip().lower()
    preference = [requested, "eodhd", "yahoo", "stooq"]
    preference = list(dict.fromkeys(x for x in preference if x in {"eodhd", "yahoo", "stooq"}))
    for feed in preference:
        daily = _daily_rows(symbol, db_path, feed)
        sessions = complete_sessions(symbol, db_path, feed) if feed in {"eodhd", "yahoo"} else []
        daily_latest = str(daily["trade_date"].max()) if not daily.empty else None
        intraday_latest = sessions[-1]["date"] if sessions else None
        matches = bool(daily_latest and intraday_latest and daily_latest == intraday_latest)
        candidates.append(((int(matches), max(daily_latest or "", intraday_latest or ""), -preference.index(feed)), feed, daily, sessions, daily_latest, intraday_latest))
    if not candidates:
        return {"symbol": symbol.upper(), "daily_latest_date": None, "intraday_latest_complete_date": None, "daily_rows": 0, "complete_intraday_sessions": 0, "feed": "unknown", "dates_aligned": False, "latest_daily_any_feed_date": None, "latest_daily_any_feed_source": None}
    _, feed, daily, sessions, daily_latest, intraday_latest = max(candidates, key=lambda item: item[0])
    daily_candidates = [(item[4], item[1], len(item[2])) for item in candidates if item[4]]
    latest_daily_any_feed_date, latest_daily_any_feed_source, latest_daily_any_feed_rows = max(daily_candidates, key=lambda item: (item[0], -preference.index(item[1]))) if daily_candidates else (None, None, 0)
    daily_fetched_at = None
    intraday_fetched_at = None
    try:
        with connect_db(db_path) as conn:
            if daily_latest:
                row = conn.execute("SELECT MAX(fetched_at) FROM asset_daily WHERE symbol=? AND feed=?", (symbol.upper(), feed)).fetchone()
                daily_fetched_at = row[0] if row else None
            if intraday_latest and feed in {"eodhd", "yahoo"}:
                row = conn.execute("SELECT MAX(fetched_at) FROM asset_intraday_bars WHERE symbol=? AND feed=? AND timeframe='15m'", (symbol.upper(), feed)).fetchone()
                intraday_fetched_at = row[0] if row else None
    except sqlite3.Error:
        pass
    return {
        "symbol": symbol.upper(), "daily_latest_date": daily_latest,
        "intraday_latest_complete_date": intraday_latest, "daily_rows": int(len(daily)),
        "complete_intraday_sessions": len(sessions), "feed": feed,
        "dates_aligned": bool(daily_latest and daily_latest == intraday_latest),
        "latest_daily_any_feed_date": latest_daily_any_feed_date,
        "latest_daily_any_feed_source": latest_daily_any_feed_source,
        "latest_daily_any_feed_rows": int(latest_daily_any_feed_rows),
        "daily_fetched_at": daily_fetched_at, "intraday_fetched_at": intraday_fetched_at,
        "provider_candidates": [{"feed": item[1], "daily_latest_date": item[4], "intraday_latest_complete_date": item[5], "dates_aligned": bool(item[4] and item[4] == item[5])} for item in candidates],
    }


def _fetch_rss(url: str, source: str, category: str) -> list[dict[str, str]]:
    response = requests.get(url, timeout=18, headers={"User-Agent": "Mozilla/5.0 QQQMResearchV2/1.0", "Accept": "application/rss+xml,application/xml,text/xml,*/*"})
    response.raise_for_status()
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError("新闻 RSS 格式无法解析") from exc
    rows = []
    for item in root.findall(".//item")[:30]:
        title = (item.findtext("title") or "").strip()
        url_value = (item.findtext("link") or "").strip()
        summary = re.sub(r"\s+", " ", (item.findtext("description") or "")).strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title or not url_value:
            continue
        try:
            published = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        except (TypeError, ValueError, OverflowError):
            published = ""
        rows.append({"title": title, "url": url_value, "summary": summary[:2000], "published_at": published, "source": source, "category": category})
    return rows


def sync_news(symbols: Iterable[str], config_path: Path = CONFIG_PATH, db_path: Path = DB_PATH) -> dict[str, Any]:
    names = {row["symbol"]: row["name"] for row in load_instruments(config_path, enabled_only=False)}
    status: dict[str, Any] = {"success": True, "symbols": {}, "errors": []}
    init_db(db_path)
    fed_items: list[dict[str, str]] = []
    try:
        fed_items = _fetch_rss("https://www.federalreserve.gov/feeds/press_monetary.xml", "Federal Reserve", "monetary_policy")
    except Exception as exc:
        status["errors"].append(f"美联储 RSS 同步失败：{type(exc).__name__}: {str(exc)[:250]}")
    all_symbols = [str(s).upper() for s in symbols]
    for symbol in all_symbols:
        items = [dict(row, symbol=symbol) for row in fed_items]
        query = f'"{names.get(symbol, symbol)}" OR {symbol} stock'
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            items.extend(dict(row, symbol=symbol) for row in _fetch_rss(url, "Google News RSS", "company_news"))
        except Exception as exc:
            status["errors"].append(f"{symbol} 新闻 RSS 同步失败：{type(exc).__name__}: {str(exc)[:250]}")
        rows_to_insert = []
        for item in items:
            rows_to_insert.append((symbol, item["url"], item["title"], item["summary"], item["source"], item["category"], item["published_at"], _utc_now()))
        with connect_db(db_path) as conn:
            conn.executemany("""INSERT INTO asset_news(symbol,url,title,summary,source,category,published_at,fetched_at)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol,url) DO UPDATE SET
                title=excluded.title,summary=excluded.summary,source=excluded.source,category=excluded.category,
                published_at=CASE WHEN excluded.published_at<>'' THEN excluded.published_at ELSE asset_news.published_at END,
                fetched_at=excluded.fetched_at""", rows_to_insert)
        status["symbols"][symbol] = {"news_rows_received": len(items), "news_rows_saved": len(rows_to_insert)}
    status["success"] = not bool(status["errors"])
    status["finished_at"] = _utc_now()
    return status


def list_news(symbol: str, limit: int = 20, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect_db(db_path) as conn:
        rows = conn.execute("SELECT symbol,title,summary,source,category,published_at,url,ai_summary,ai_sentiment,ai_confidence FROM asset_news WHERE symbol=? ORDER BY CASE WHEN published_at='' THEN 1 ELSE 0 END,published_at DESC LIMIT ?", (symbol.upper(), max(1, min(int(limit), 100)))).fetchall()
    return [dict(zip(["symbol", "title", "summary", "source", "category", "published_at", "url", "ai_summary", "ai_sentiment", "ai_confidence"], row)) for row in rows]


def _deepseek_api_key() -> str | None:
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY"):
        value = os.getenv(key, "").strip()
        if value and not value.lower().startswith(("your_", "replace", "paste_")):
            return value
    env_path = ROOT / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name.strip() in {"DEEPSEEK_API_KEY", "DEEPSEEK_KEY"}:
                value = value.strip().strip("\"'")
                if value and not value.lower().startswith(("your_", "replace", "paste_")):
                    return value
    except OSError:
        pass
    return None


def analyze_news_with_deepseek(symbol: str, db_path: Path = DB_PATH, max_items: int = 8) -> dict[str, Any]:
    """Optional LLM summary. Output is narrative context, never direct forecast input."""
    api_key = _deepseek_api_key()
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY。请在项目根目录 .env 中配置后再使用 AI 新闻分析。")
    items = list_news(symbol, max_items, db_path)
    if not items:
        raise RuntimeError(f"{symbol} 暂无可用新闻记录，请先同步新闻。")
    prompt_items = [{"published_at": x["published_at"], "source": x["source"], "title": x["title"], "summary": x["summary"][:400], "url": x["url"]} for x in items]
    system_prompt = (
        "你是金融研究助理。使用中文分析新闻，只做事实摘要与不确定性说明。"
        "必须输出合法 JSON，不可给出保证性涨跌预测或买卖建议。不要把发布时间晚于当前研究时间的信息冒充为此前已知。"
        "字段为 overall_summary, events(array of {title, event_type, affected_entity, possible_direction, confidence, rationale, risk}), caveats。"
        "possible_direction 只能是 positive, negative, mixed, unclear；confidence 为 0 到 1。"
        "新闻解读不会直接作为价格预测值。"
    )
    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "temperature": 0.1,
        "max_tokens": 1400,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps({"symbol": symbol.upper(), "news": prompt_items}, ensure_ascii=False)}],
    }
    response = requests.post(
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=45,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"DeepSeek 请求失败，HTTP {response.status_code}。请检查密钥、额度和模型配置；未打印密钥内容。")
    body = response.json()
    text = body["choices"][0]["message"]["content"]
    parsed = json.loads(text)
    output = {
        "symbol": symbol.upper(), "generated_at": _utc_now(), "model": payload["model"],
        "overall_summary": str(parsed.get("overall_summary") or "暂无摘要"),
        "events": parsed.get("events") if isinstance(parsed.get("events"), list) else [],
        "caveats": parsed.get("caveats") or [],
        "source_articles": [{"title": x["title"], "source": x["source"], "published_at": x["published_at"], "url": x["url"]} for x in items],
        "usage": body.get("usage", {}),
        "notice": "AI 解读仅为新闻背景，不会直接决定模型价格路径，不构成投资建议。",
    }
    path = NEWS_ROOT / f"{symbol.upper()}_latest.json"
    _write_json(path, output)
    return output


def _next_session_date(last_complete_date: str) -> str:
    try:
        from app.services.market_calendar import is_nasdaq_session, next_nasdaq_session
        now_ny = datetime.now(timezone.utc).astimezone(NY)
        today = now_ny.date()
        latest = date.fromisoformat(last_complete_date)
        # Before today's open, today can be the target if last complete data is yesterday.
        if is_nasdaq_session(today) and now_ny.time().replace(tzinfo=None) < daytime(9, 30) and latest < today:
            return today.isoformat()
        # During a live session, only predict the next full session, never today's already-opened session.
        if is_nasdaq_session(today) and daytime(9, 30) <= now_ny.time().replace(tzinfo=None) < daytime(16, 15):
            return next_nasdaq_session(today).isoformat()
        base = max(today, latest) if is_nasdaq_session(today) and now_ny.time().replace(tzinfo=None) >= daytime(16, 15) else latest
        return next_nasdaq_session(base).isoformat()
    except Exception:
        latest = date.fromisoformat(last_complete_date)
        candidate = latest + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate.isoformat()


def _context_features(daily: pd.DataFrame, session_date: str) -> np.ndarray | None:
    hist = daily[daily["trade_date"] < session_date].sort_values("trade_date")
    if len(hist) < 11:
        return None
    closes = hist["close"].astype(float).to_numpy()
    opens = hist["open"].astype(float).to_numpy()
    highs = hist["high"].astype(float).to_numpy()
    lows = hist["low"].astype(float).to_numpy()
    volumes = hist["volume"].fillna(0).astype(float).to_numpy()
    rets = closes[1:] / closes[:-1] - 1.0
    last_ret = float(rets[-1])
    ret5 = float(closes[-1] / closes[-6] - 1.0)
    ret10 = float(closes[-1] / closes[-11] - 1.0)
    vol10 = float(np.std(rets[-10:], ddof=1)) if len(rets[-10:]) > 1 else 0.0
    prev_range = float((highs[-1] - lows[-1]) / closes[-1])
    prev_oc = float(closes[-1] / opens[-1] - 1.0)
    mean_vol = float(np.mean(volumes[-10:])) if np.mean(volumes[-10:]) > 0 else float(volumes[-1] or 1.0)
    volume_ratio = float(volumes[-1] / mean_vol - 1.0) if mean_vol > 0 else 0.0
    return np.array([last_ret, ret5, ret10, vol10, prev_range, prev_oc, float(np.clip(volume_ratio, -3, 5))], dtype=float)


def _bar_path(session: dict[str, Any]) -> list[float]:
    bars = session["bars"].sort_values("session_time_ny")
    first = bars[bars["session_time_ny"] == "09:30"]
    if first.empty:
        return []
    return [float(first.iloc[0]["open"])] + pd.to_numeric(bars["close"], errors="coerce").astype(float).tolist()


def _design(context: np.ndarray, slot_index: int) -> np.ndarray:
    t = float(slot_index) / 26.0
    return np.array([
        1.0, t, t * t, math.sin(math.pi * t), math.cos(math.pi * t),
        float(context[0]), float(context[1]), float(context[2]), float(context[3]),
        float(context[4]), float(context[5]), float(context[6]),
        float(context[0]) * t, float(context[1]) * t, float(context[3]) * t,
    ], dtype=float)


def _build_train_rows(symbol: str, cutoff_date: str, db_path: Path = DB_PATH, feed: str = "yahoo") -> tuple[np.ndarray, np.ndarray, list[str]]:
    daily = _daily_rows(symbol, db_path, feed)
    sessions = [s for s in complete_sessions(symbol, db_path, feed) if s["date"] < cutoff_date]
    xs, ys, sample_dates = [], [], []
    for session in sessions:
        context = _context_features(daily, session["date"])
        previous = daily[daily["trade_date"] < session["date"]].sort_values("trade_date")
        path = _bar_path(session)
        if context is None or previous.empty or len(path) < 25:
            continue
        prev_close = float(previous.iloc[-1]["close"])
        if prev_close <= 0:
            continue
        usable = min(27, len(path))
        for idx in range(usable):
            price = float(path[idx])
            if not np.isfinite(price) or price <= 0:
                continue
            xs.append(_design(context, idx))
            ys.append(price / prev_close - 1.0)
            sample_dates.append(session["date"])
    if not xs:
        return np.empty((0, 15)), np.empty((0,)), []
    return np.vstack(xs), np.asarray(ys, dtype=float), sample_dates


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 0.8) -> dict[str, Any]:
    if x.ndim != 2 or len(x) < max(80, x.shape[1] * 3):
        raise ValueError(f"训练样本不足：目前 {len(x)} 行，至少需要 {max(80, x.shape[1] * 3)} 行。")
    means = x[:, 1:].mean(axis=0)
    scales = x[:, 1:].std(axis=0)
    scales[scales < 1e-8] = 1.0
    z = x.copy()
    z[:, 1:] = (z[:, 1:] - means) / scales
    penalty = np.eye(z.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(z.T @ z + penalty, z.T @ y, rcond=None)[0]
    fitted = z @ beta
    residuals = y - fitted
    return {"means": means, "scales": scales, "beta": beta, "rmse": float(np.sqrt(np.mean(residuals ** 2))), "residuals": residuals, "train_rows": int(len(x)), "alpha": float(alpha)}


def _predict_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = x.copy()
    z[:, 1:] = (z[:, 1:] - model["means"]) / model["scales"]
    return z @ model["beta"]


def _baseline_path(symbol: str, target_date: str, db_path: Path = DB_PATH, feed: str = "yahoo", lookback: int = 20) -> np.ndarray:
    daily = _daily_rows(symbol, db_path, feed)
    sessions = [s for s in complete_sessions(symbol, db_path, feed) if s["date"] < target_date][-lookback:]
    paths = []
    for session in sessions:
        context = _context_features(daily, session["date"])
        prev = daily[daily["trade_date"] < session["date"]].sort_values("trade_date")
        path = _bar_path(session)
        if context is None or prev.empty or len(path) < 25:
            continue
        prev_close = float(prev.iloc[-1]["close"])
        if prev_close <= 0:
            continue
        norm = np.array(path[:27], dtype=float) / prev_close - 1.0
        if len(norm) == 27:
            paths.append(norm)
    if not paths:
        return np.full(27, np.nan)
    return np.nanmedian(np.vstack(paths), axis=0)


def walk_forward_metrics(symbol: str, db_path: Path = DB_PATH, feed: str = "yahoo", holdout_sessions: int = 5) -> dict[str, Any]:
    daily = _daily_rows(symbol, db_path, feed)
    sessions = complete_sessions(symbol, db_path, feed)
    usable = []
    for session in sessions:
        context = _context_features(daily, session["date"])
        prev = daily[daily["trade_date"] < session["date"]].sort_values("trade_date")
        path = _bar_path(session)
        if context is not None and not prev.empty and len(path) >= 25:
            usable.append(session)
    if len(usable) < 20:
        return {"available": False, "reason": f"完整且有足够历史日线特征的分时样本只有 {len(usable)} 个，至少需要 20 个交易日。", "usable_sessions": len(usable)}
    start = max(15, len(usable) - max(1, int(holdout_sessions)))
    model_errors: list[float] = []
    baseline_errors: list[float] = []
    direction_hits = 0
    direction_count = 0
    detail = []
    for idx in range(start, len(usable)):
        test = usable[idx]
        cutoff = test["date"]
        x_train, y_train, dates_train = _build_train_rows(symbol, cutoff, db_path, feed)
        if len(x_train) < 80:
            continue
        try:
            model = _fit_ridge(x_train, y_train)
        except ValueError:
            continue
        context = _context_features(daily, cutoff)
        prior = daily[daily["trade_date"] < cutoff].sort_values("trade_date")
        actual = _bar_path(test)
        if context is None or prior.empty or len(actual) < 27:
            continue
        prev_close = float(prior.iloc[-1]["close"])
        predicted_return = _predict_ridge(model, np.vstack([_design(context, i) for i in range(27)]))
        pred_prices = prev_close * (1.0 + predicted_return)
        actual_prices = np.asarray(actual[:27], dtype=float)
        actual_return = actual_prices / prev_close - 1.0
        baseline_return = _baseline_path(symbol, cutoff, db_path, feed)
        if not np.isfinite(baseline_return).all():
            baseline_return = np.zeros(27)
        err = np.abs(pred_prices / prev_close - 1.0 - actual_return) * 10000.0
        base_err = np.abs(baseline_return - actual_return) * 10000.0
        model_errors.extend(err.tolist())
        baseline_errors.extend(base_err.tolist())
        direction_hits += int((predicted_return[-1] >= 0) == (actual_return[-1] >= 0))
        direction_count += 1
        detail.append({"target_date": cutoff, "bar_count": 27, "ridge_mae_bps": float(np.mean(err)), "baseline_mae_bps": float(np.mean(base_err)), "predicted_close_return_pct": float(predicted_return[-1] * 100), "actual_close_return_pct": float(actual_return[-1] * 100)})
    if not model_errors:
        return {"available": False, "reason": "滚动验证没有生成有效的样本外预测。", "usable_sessions": len(usable)}
    model_mae = float(np.mean(model_errors))
    base_mae = float(np.mean(baseline_errors))
    improvement = ((base_mae - model_mae) / base_mae * 100.0) if base_mae > 0 else None
    return {
        "available": True, "method": "rolling-origin walk-forward; each target session excluded from training",
        "usable_sessions": len(usable), "test_sessions": len(detail),
        "ridge_mae_bps": round(model_mae, 3), "ridge_rmse_bps": round(float(np.sqrt(np.mean(np.square(model_errors)))), 3),
        "historical_profile_mae_bps": round(base_mae, 3), "improvement_pct_vs_profile": round(improvement, 3) if improvement is not None else None,
        "close_direction_accuracy": round(direction_hits / direction_count, 4) if direction_count else None,
        "detail": detail,
        "note": "滚动样本外结果仅是历史评估；样本量少、行情数据源变化或市场制度变化都会影响结论。不会自动晋级生产模型。",
    }


def _find_existing_forecast(symbol: str, target_date: str) -> Path | None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    matches = list(ARCHIVE_ROOT.glob(f"{symbol.upper()}_asof_*_target_{target_date}.json"))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def generate_forecast(symbol: str, db_path: Path = DB_PATH, feed: str = "yahoo") -> dict[str, Any]:
    symbol = symbol.upper()
    daily = _daily_rows(symbol, db_path, feed)
    sessions = complete_sessions(symbol, db_path, feed)
    if daily.empty or not sessions:
        raise RuntimeError(f"{symbol} 缺少日线或完整十五分钟数据，不能生成下一交易日情景。")
    latest_session = sessions[-1]
    latest_daily = str(daily["trade_date"].max())
    if latest_session["date"] != latest_daily:
        raise RuntimeError(f"{symbol} 的完整十五分钟数据截至 {latest_session['date']}，日线截至 {latest_daily}，日期不一致。为避免过期训练，停止生成。")
    target = _next_session_date(latest_session["date"])
    existing = _find_existing_forecast(symbol, target)
    if existing:
        archived = _read_json(existing)
        if archived and archived.get("symbol") == symbol and archived.get("target_date") == target and archived.get("immutable_snapshot"):
            latest_path = REPORT_ROOT / symbol / "forecast_latest.json"
            _write_json(latest_path, archived)
            chart_path = REPORT_ROOT / symbol / f"forecast_{target}.png"
            if not chart_path.is_file():
                _plot_forecast(archived, chart_path)
            return {"success": True, "symbol": symbol, "target_date": target, "as_of_date": latest_session["date"], "forecast_file": str(existing), "reused_immutable_snapshot": True, "forecast": archived}
    x, y, sample_dates = _build_train_rows(symbol, target, db_path, feed)
    if len(x) < 80:
        raise RuntimeError(f"{symbol} 训练样本只有 {len(x)} 条，少于最低要求 80 条，暂不生成预测。")
    model = _fit_ridge(x, y)
    context = _context_features(daily, target)
    prior = daily[daily["trade_date"] <= latest_session["date"]].sort_values("trade_date")
    if context is None or prior.empty:
        raise RuntimeError(f"{symbol} 的历史日线特征不足。")
    previous_close = float(prior.iloc[-1]["close"])
    predicted_returns = _predict_ridge(model, np.vstack([_design(context, i) for i in range(27)]))
    prices = previous_close * (1.0 + predicted_returns)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise RuntimeError(f"{symbol} 模型输出了非有限或非正价格，已停止保存。")
    backtest = walk_forward_metrics(symbol, db_path, feed)
    residuals = np.asarray(model["residuals"], dtype=float)
    # This is a historical residual reference band, not a probabilistic guarantee.
    margin_return = max(float(np.quantile(np.abs(residuals), 0.90)) if len(residuals) else 0.005, 0.003)
    curve = []
    for idx, slot in enumerate(TIME_SLOTS):
        predicted = float(prices[idx])
        margin = max(predicted * margin_return, previous_close * 0.003)
        curve.append({"time_ny": slot, "predicted_price": round(predicted, 4), "median_price": round(predicted, 4), "low_price": round(max(0.01, predicted - margin), 4), "high_price": round(predicted + margin, 4), "return_from_previous_close_pct": round(float(predicted_returns[idx] * 100.0), 5)})
    instrument = next((x for x in load_instruments(enabled_only=False) if x["symbol"] == symbol), {"symbol": symbol, "name": symbol, "type": "stock"})
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    prediction = {
        "schema_version": 2, "symbol": symbol, "instrument_name": instrument.get("name", symbol), "instrument_type": instrument.get("type", "stock"),
        "feed": feed, "timeframe": "15m", "model_family": "ridge_intraday_v2", "model_version": "ridge-multi-asset-v2.1",
        "model_status": "candidate_not_promoted", "created_at_utc": _utc_now(), "created_at_new_york": now_ny.isoformat(timespec="seconds"),
        "as_of_date": latest_session["date"], "target_date": target, "forecast_date_weekday_estimate": target,
        "previous_close": previous_close, "predicted_open": curve[0]["predicted_price"], "predicted_close": curve[-1]["predicted_price"],
        "curve": curve, "feature_as_of_date": latest_session["date"], "training_sessions": len(set(sample_dates)), "training_rows": int(len(x)),
        "backtest": backtest, "residual_band_note": "上下界根据历史训练残差生成，只是误差参考范围，不是统计置信区间或涨跌概率。",
        "news_policy": "新闻与 AI 分析仅作为解释背景，未直接用于本版价格主路径；避免未经验证的新闻信号造成伪精度。",
        "immutable_snapshot": True,
        "audit": {"uses_future_data": False, "training_target_dates_strictly_before_forecast": True, "same_symbol_same_feed_only": True},
    }
    path = ARCHIVE_ROOT / f"{symbol}_asof_{latest_session['date']}_target_{target}.json"
    # Immutable: never overwrite an existing forecast generated for the same symbol/as-of/target.
    if path.exists():
        return {"success": True, "symbol": symbol, "target_date": target, "as_of_date": latest_session["date"], "forecast_file": str(path), "reused_immutable_snapshot": True, "forecast": _read_json(path) or prediction}
    _write_json(path, prediction)
    latest_path = REPORT_ROOT / symbol / "forecast_latest.json"
    _write_json(latest_path, prediction)
    _plot_forecast(prediction, REPORT_ROOT / symbol / f"forecast_{target}.png")
    return {"success": True, "symbol": symbol, "target_date": target, "as_of_date": latest_session["date"], "forecast_file": str(path), "reused_immutable_snapshot": False, "forecast": prediction}


def _plot_forecast(forecast: dict[str, Any], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.2))
    curve = forecast.get("curve") or []
    xs = list(range(len(curve)))
    mids = [float(p["predicted_price"]) for p in curve]
    lows = [float(p["low_price"]) for p in curve]
    highs = [float(p["high_price"]) for p in curve]
    ax.fill_between(xs, lows, highs, alpha=.2, label="历史训练残差参考范围")
    ax.plot(xs, mids, linewidth=2.5, label="Ridge 机器学习候选路径")
    ticks = sorted(set([0, 4, 8, 12, 16, 20, 26]))
    ax.set_xticks(ticks, [curve[i]["time_ny"] for i in ticks])
    ax.set_title(f"{forecast['symbol']} 下一交易日走势情景：{forecast['target_date']}\n基准 {forecast['as_of_date']} · 模型 {forecast['model_version']}")
    ax.set_xlabel("纽约时间")
    ax.set_ylabel("价格（美元）")
    ax.grid(alpha=.25)
    ax.legend(loc="best")
    fig.text(.01, .01, "研究情景，不是确定价格；候选模型未自动晋级，不构成投资建议。", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _valid_forecast_for_target(path: Path, target_date: str) -> dict[str, Any] | None:
    forecast = _read_json(path)
    if not forecast or str(forecast.get("target_date") or "") != target_date:
        return None
    if not forecast.get("immutable_snapshot") or not forecast.get("curve"):
        return None
    asof = str(forecast.get("as_of_date") or "")
    if not asof or asof >= target_date:
        return None
    raw_created = str(forecast.get("created_at_utc") or "")
    try:
        created = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
        market_open_utc = datetime.combine(date.fromisoformat(target_date), daytime(9, 30), tzinfo=NY).astimezone(timezone.utc)
        if created.tzinfo is None or created.astimezone(timezone.utc) >= market_open_utc:
            return None
    except (ValueError, TypeError):
        return None
    return forecast


def settle_due_forecasts(symbol: str, db_path: Path = DB_PATH, feed: str = "yahoo") -> list[dict[str, Any]]:
    symbol = symbol.upper()
    sessions = {item["date"]: item for item in complete_sessions(symbol, db_path, feed)}
    if not sessions:
        return []
    results = []
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in sorted(ARCHIVE_ROOT.glob(f"{symbol}_asof_*_target_*.json")):
        forecast_raw = _read_json(path)
        target = str((forecast_raw or {}).get("target_date") or "")
        if not target or target not in sessions:
            continue
        try:
            with connect_db(db_path) as conn:
                existed = conn.execute("SELECT 1 FROM asset_evaluations WHERE symbol=? AND target_date=? AND forecast_file=?", (symbol, target, str(path.resolve()))).fetchone()
            if existed:
                continue
            forecast = _valid_forecast_for_target(path, target)
            if not forecast:
                results.append({"symbol": symbol, "target_date": target, "status": "skipped_non_prospective_or_invalid_archive", "forecast_file": str(path)})
                continue
            forecast_feed = str(forecast.get("feed") or "").lower()
            if forecast_feed != str(feed).lower():
                results.append({"symbol": symbol, "target_date": target, "status": "skipped_feed_mismatch", "forecast_feed": forecast_feed or "unknown", "actual_feed": str(feed).lower(), "forecast_file": str(path)})
                continue
            actual = sessions[target]
            points = forecast["curve"]
            pred_map = {str(p.get("time_ny")): float(p.get("predicted_price", p.get("median_price"))) for p in points if p.get("time_ny") and p.get("predicted_price", p.get("median_price")) is not None}
            actual_bars = actual["bars"].sort_values("session_time_ny")
            actual_times = ["09:30"] + [(datetime.strptime(str(t), "%H:%M") + timedelta(minutes=15)).strftime("%H:%M") for t in actual_bars["session_time_ny"].astype(str)]
            actual_prices = [float(actual_bars[actual_bars["session_time_ny"] == "09:30"].iloc[0]["open"])] + actual_bars["close"].astype(float).tolist()
            paired = [(float(pred_map[t]), float(y)) for t, y in zip(actual_times, actual_prices) if t in pred_map and np.isfinite(float(y)) and float(y) > 0]
            if len(paired) < 20:
                results.append({"symbol": symbol, "target_date": target, "status": "waiting_incomplete_actual_alignment", "matched_points": len(paired)})
                continue
            prev_close = float(forecast.get("previous_close") or 0)
            if prev_close <= 0:
                continue
            mae_usd = float(np.mean([abs(p - a) for p, a in paired]))
            mae_bps = float(np.mean([abs(p / prev_close - a / prev_close) * 10000 for p, a in paired]))
            actual_close = actual_prices[-1]
            predicted_close = pred_map.get("16:00", paired[-1][0])
            close_error_pct = (predicted_close / actual_close - 1) * 100 if actual_close > 0 else None
            baseline_path = _baseline_path(symbol, target, db_path, feed)
            baseline_errors = []
            for idx, actual_price in enumerate(actual_prices[:len(baseline_path)]):
                if np.isfinite(baseline_path[idx]) and actual_price > 0:
                    baseline_errors.append(abs(baseline_path[idx] - (actual_price / prev_close - 1.0)) * 10000)
            baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else None
            improvement = ((baseline_mae - mae_bps) / baseline_mae * 100) if baseline_mae and baseline_mae > 0 else None
            record = {
                "symbol": symbol, "target_date": target, "forecast_file": str(path.resolve()), "actual_feed": feed,
                "bar_count": len(paired), "mae_usd": round(mae_usd, 5), "mae_bps": round(mae_bps, 3),
                "close_error_pct": round(float(close_error_pct), 4) if close_error_pct is not None else None,
                "baseline_mae_bps": round(baseline_mae, 3) if baseline_mae is not None else None,
                "relative_improvement_pct": round(float(improvement), 3) if improvement is not None else None,
                "evaluated_at": _utc_now(), "status": "evaluated",
            }
            with connect_db(db_path) as conn:
                conn.execute("""INSERT OR IGNORE INTO asset_evaluations(symbol,target_date,forecast_file,actual_feed,bar_count,mae_usd,mae_bps,close_error_pct,baseline_mae_bps,relative_improvement_pct,evaluated_at,status)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", (symbol,target,str(path.resolve()),feed,len(paired),record["mae_usd"],record["mae_bps"],record["close_error_pct"],record["baseline_mae_bps"],record["relative_improvement_pct"],record["evaluated_at"],record["status"]))
            _plot_review(forecast, actual, REPORT_ROOT / symbol / f"review_{target}.png", record)
            results.append(record)
        except Exception as exc:
            results.append({"symbol": symbol, "target_date": target, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:350]}"})
    return results


def _plot_review(forecast: dict[str, Any], session: dict[str, Any], out_path: Path, metrics: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bars = session["bars"].sort_values("session_time_ny")
    times = ["09:30"] + [(datetime.strptime(str(t), "%H:%M") + timedelta(minutes=15)).strftime("%H:%M") for t in bars["session_time_ny"].astype(str)]
    actual = [float(bars[bars["session_time_ny"] == "09:30"].iloc[0]["open"])] + bars["close"].astype(float).tolist()
    pred_by_time = {str(p.get("time_ny")): float(p.get("predicted_price", p.get("median_price"))) for p in forecast["curve"] if p.get("time_ny")}
    matched = [(i, pred_by_time[t]) for i, t in enumerate(times) if t in pred_by_time]
    fig, ax = plt.subplots(figsize=(12.4, 6.7))
    ax.plot(range(len(actual)), actual, linewidth=2.7, label="真实十五分钟走势")
    if len(matched) >= 20:
        ax.plot([i for i, _ in matched], [p for _, p in matched], linestyle="--", linewidth=2.4, label="目标日开盘前已保存的 Ridge 预测")
    ticks = sorted(set([0, 4, 8, 12, 16, 20, 26]))
    ax.set_xticks(ticks, [times[i] if i < len(times) else "16:00" for i in ticks])
    ax.set_title(f"{forecast['symbol']} 历史预测复盘 {session['date']}\n同一预测档案与真实行情，MAE={metrics.get('mae_usd')} 美元")
    ax.set_xlabel("纽约时间")
    ax.set_ylabel("价格（美元）")
    ax.grid(alpha=.25)
    ax.legend(loc="best")
    fig.text(.01, .01, f"原始预测文件：{Path(metrics['forecast_file']).name}。本图仅评价事前快照，不做事后补画。", fontsize=8.8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def get_evaluations(symbol: str | None = None, limit: int = 100, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect_db(db_path) as conn:
        if symbol:
            rows = conn.execute("SELECT symbol,target_date,forecast_file,actual_feed,bar_count,mae_usd,mae_bps,close_error_pct,baseline_mae_bps,relative_improvement_pct,evaluated_at,status FROM asset_evaluations WHERE symbol=? ORDER BY target_date DESC LIMIT ?", (symbol.upper(), limit)).fetchall()
        else:
            rows = conn.execute("SELECT symbol,target_date,forecast_file,actual_feed,bar_count,mae_usd,mae_bps,close_error_pct,baseline_mae_bps,relative_improvement_pct,evaluated_at,status FROM asset_evaluations ORDER BY target_date DESC,symbol LIMIT ?", (limit,)).fetchall()
    keys = ["symbol", "target_date", "forecast_file", "actual_feed", "bar_count", "mae_usd", "mae_bps", "close_error_pct", "baseline_mae_bps", "relative_improvement_pct", "evaluated_at", "status"]
    return [dict(zip(keys, row)) for row in rows]


def _expected_completed_date() -> str | None:
    try:
        from app.services.system_health import expected_last_completed_session
        return expected_last_completed_session(datetime.now(NY))
    except Exception:
        return None


def _provider_order() -> list[str]:
    requested = os.getenv("MULTI_ASSET_MARKET_DATA_PROVIDER", os.getenv("MARKET_DATA_PROVIDER", "yahoo")).strip().lower()
    if requested not in {"yahoo", "eodhd"}:
        requested = "yahoo"
    candidates = [requested]
    if _eodhd_token():
        candidates.append("eodhd")
    candidates.append("yahoo")
    return list(dict.fromkeys(candidates))


def _fetch_provider_pair(symbol: str, provider: str, db_path: Path) -> dict[str, Any]:
    """Try both daily and intraday from one vendor; never blend a cross-vendor pair."""
    result: dict[str, Any] = {"feed": provider, "daily_rows_updated": 0, "intraday_rows_updated": 0, "errors": [], "warnings": []}
    try:
        daily = fetch_eodhd_daily(symbol) if provider == "eodhd" else fetch_daily(symbol)
        result["daily_method"] = daily.attrs.get("fetch_method", "eodhd_api" if provider == "eodhd" else "yahoo_chart_api")
        result["warnings"].extend(daily.attrs.get("fetch_warnings", []))
        now_ny = datetime.now(timezone.utc).astimezone(NY)
        if now_ny.time().replace(tzinfo=None) < daytime(16, 15):
            daily = daily[daily["trade_date_ny"] < now_ny.date().isoformat()]
        result["daily_rows_updated"] = save_daily(symbol, daily, db_path, provider)
    except Exception as exc:
        result["errors"].append(f"{provider} 日线：{type(exc).__name__}: {str(exc)[:360]}")
    try:
        intraday = fetch_eodhd_intraday(symbol) if provider == "eodhd" else fetch_intraday(symbol)
        result["intraday_method"] = intraday.attrs.get("fetch_method", "eodhd_api" if provider == "eodhd" else "yahoo_chart_api")
        result["warnings"].extend(intraday.attrs.get("fetch_warnings", []))
        now_ny = datetime.now(timezone.utc).astimezone(NY)
        if now_ny.time().replace(tzinfo=None) < daytime(16, 15):
            intraday = intraday[intraday["trade_date_ny"] < now_ny.date().isoformat()]
        result["intraday_rows_updated"] = save_intraday(symbol, intraday, db_path, provider)
    except Exception as exc:
        result["errors"].append(f"{provider} 十五分钟分时：{type(exc).__name__}: {str(exc)[:360]}")
    daily_saved = _daily_rows(symbol, db_path, provider)
    sessions = complete_sessions(symbol, db_path, provider)
    result["daily_latest_date"] = str(daily_saved["trade_date"].max()) if not daily_saved.empty else None
    result["intraday_latest_complete_date"] = sessions[-1]["date"] if sessions else None
    result["dates_aligned"] = bool(result["daily_latest_date"] and result["daily_latest_date"] == result["intraday_latest_complete_date"])
    return result


def _refresh_stooq_daily_if_needed(symbol: str, db_path: Path, expected_date: str | None) -> dict[str, Any] | None:
    """Add a daily-only display fallback if the regular provider returned stale/missing daily bars.

    Stooq daily bars are stored under their own feed and are never mixed with Yahoo/EODHD
    intraday data for forecasting or forecast evaluation.
    """
    current = latest_dates(symbol, db_path)
    latest = current.get("latest_daily_any_feed_date")
    if latest and (not expected_date or latest >= expected_date):
        return None
    attempt: dict[str, Any] = {"feed": "stooq", "daily_method": "stooq_daily_fallback", "daily_rows_updated": 0, "intraday_rows_updated": 0, "errors": [], "warnings": ["Stooq 只提供日线备用数据，不参与十五分钟预测。"]}
    try:
        daily = fetch_stooq_daily(symbol)
        now_ny = datetime.now(timezone.utc).astimezone(NY)
        if now_ny.time().replace(tzinfo=None) < daytime(16, 15):
            daily = daily[daily["trade_date_ny"] < now_ny.date().isoformat()]
        attempt["daily_rows_updated"] = save_daily(symbol, daily, db_path, "stooq")
        saved = _daily_rows(symbol, db_path, "stooq")
        attempt["daily_latest_date"] = str(saved["trade_date"].max()) if not saved.empty else None
        if expected_date and attempt["daily_latest_date"] and attempt["daily_latest_date"] < expected_date:
            attempt["errors"].append(f"Stooq 最新日线截至 {attempt['daily_latest_date']}，仍早于应有交易日 {expected_date}。")
    except Exception as exc:
        attempt["errors"].append(f"Stooq 日线备用源：{type(exc).__name__}: {str(exc)[:300]}")
    return attempt


def sync_market_data(symbols: Iterable[str] | None = None, db_path: Path = DB_PATH) -> dict[str, Any]:
    """Refresh market data only; do not sync news, train, settle, or forecast."""
    selected = [str(s).upper().strip() for s in (symbols or [row["symbol"] for row in load_instruments()])]
    configured = {row["symbol"] for row in load_instruments(enabled_only=False)}
    selected = list(dict.fromkeys(s for s in selected if s in configured))
    if not selected:
        raise ValueError("没有选择已配置的有效标的。")
    init_db(db_path)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    expected = _expected_completed_date()
    status: dict[str, Any] = {
        "version": "2.2", "workflow": "market_data_sync_only", "started_at": _utc_now(),
        "expected_last_completed_session": expected, "selected_symbols": selected,
        "symbols": {}, "overall_success": True, "forecast_ready_all": True,
        "configured_provider_order": _provider_order(),
        "note": "此操作只刷新并校验行情，不运行预测或新闻分析。Stooq 仅用于日线展示，预测仍要求同源日线与分时。",
    }
    for symbol in selected:
        started = _utc_now()
        attempts: list[dict[str, Any]] = []
        for provider in _provider_order():
            if provider == "eodhd" and not _eodhd_token():
                continue
            attempt = _fetch_provider_pair(symbol, provider, db_path)
            attempts.append(attempt)
            fresh_enough = bool(attempt.get("intraday_latest_complete_date") and (not expected or attempt["intraday_latest_complete_date"] >= expected))
            if attempt.get("dates_aligned") and fresh_enough:
                break
        stooq_attempt = _refresh_stooq_daily_if_needed(symbol, db_path, expected)
        if stooq_attempt:
            attempts.append(stooq_attempt)
        dates = latest_dates(symbol, db_path)
        daily_date = dates.get("latest_daily_any_feed_date")
        daily_fresh = bool(daily_date and (not expected or daily_date >= expected))
        intraday_fresh = bool(dates.get("intraday_latest_complete_date") and (not expected or dates["intraday_latest_complete_date"] >= expected))
        forecast_ready = bool(dates.get("dates_aligned") and intraday_fresh and dates.get("feed") in {"yahoo", "eodhd"})
        errors = [str(err) for att in attempts for err in att.get("errors", [])]
        warnings = [str(msg) for att in attempts for msg in att.get("warnings", [])]
        if not daily_fresh:
            errors.append(f"最新可用日线仍未达到应有交易日：当前 {daily_date or '缺失'}；预期 {expected or '未知'}。")
        if not intraday_fresh:
            warnings.append(f"同源完整十五分钟行情当前截至 {dates.get('intraday_latest_complete_date') or '缺失'}；预测模块将暂停使用过期数据。")
        item = {
            "symbol": symbol, **dates, "provider_attempts": attempts, "errors": errors,
            "warnings": list(dict.fromkeys(warnings)), "daily_fresh": daily_fresh,
            "intraday_fresh": intraday_fresh, "forecast_ready": forecast_ready,
            "data_refreshed": any(int(att.get("daily_rows_updated", 0)) > 0 or int(att.get("intraday_rows_updated", 0)) > 0 for att in attempts),
            "success": daily_fresh, "started_at": started, "finished_at": _utc_now(),
        }
        status["symbols"][symbol] = item
        status["overall_success"] = bool(status["overall_success"] and daily_fresh)
        status["forecast_ready_all"] = bool(status["forecast_ready_all"] and forecast_ready)
        with connect_db(db_path) as conn:
            conn.execute("INSERT INTO asset_sync_runs(started_at,finished_at,symbol,success,daily_latest_date,intraday_latest_complete_date,daily_rows,intraday_rows,news_rows,source,error) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                item["started_at"], item["finished_at"], symbol, int(item["success"]),
                dates.get("daily_latest_date"), dates.get("intraday_latest_complete_date"),
                sum(int(att.get("daily_rows_updated", 0)) for att in attempts),
                sum(int(att.get("intraday_rows_updated", 0)) for att in attempts), 0,
                dates.get("feed") or dates.get("latest_daily_any_feed_source") or "unknown", "; ".join(errors),
            ))
    status["finished_at"] = _utc_now()
    status["status_file"] = str(REPORT_ROOT / "latest_status.json")
    _write_json(REPORT_ROOT / "latest_status.json", status)
    return status


def run_cycle(symbols: Iterable[str] | None = None, refresh: bool = True, sync_news_flag: bool = True, db_path: Path = DB_PATH) -> dict[str, Any]:
    selected = [s.upper() for s in symbols] if symbols else [row["symbol"] for row in load_instruments()]
    configured = {row["symbol"]: row for row in load_instruments(enabled_only=False)}
    selected = list(dict.fromkeys(s for s in selected if s in configured))
    if not selected:
        raise ValueError("没有选择已配置的有效标的。")
    init_db(db_path)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "version": "2.2", "workflow": "full_research_cycle", "started_at": _utc_now(), "selected_symbols": selected,
        "symbols": {}, "overall_success": True, "configured_provider_order": _provider_order(),
        "notes": [
            "每个标的独立存储，模型按标的分别训练。",
            "每次预测保存为不可变快照，收盘后只结算事前生成的预测。",
            "Yahoo 先尝试公开图表接口，再尝试 yfinance cookie/crumb 路径；如配置 EODHD Key 则可切换/回退。模型只使用单一来源的分时与同源日线。Stooq 仅作日线显示备用源，不参与预测。",
            "新闻与大模型解读只用于背景，不直接决定价格主路径。",
        ],
    }
    if sync_news_flag:
        try:
            status["news_sync"] = sync_news(selected, db_path=db_path)
        except Exception as exc:
            status["news_sync"] = {"success": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}

    expected = _expected_completed_date()
    status["expected_last_completed_session"] = expected
    for symbol in selected:
        item: dict[str, Any] = {"symbol": symbol, "started_at": _utc_now(), "errors": [], "source": "unknown"}
        provider_attempts: list[dict[str, Any]] = []
        if refresh:
            for provider in _provider_order():
                if provider == "eodhd" and not _eodhd_token():
                    continue
                attempt = _fetch_provider_pair(symbol, provider, db_path)
                provider_attempts.append(attempt)
                # Prefer a source whose daily and intraday rows line up to the most
                # recent fully closed exchange session. If calendar lookup is unavailable,
                # aligned data is still required but the limitation is reported.
                fresh_enough = bool(attempt.get("intraday_latest_complete_date") and (not expected or attempt["intraday_latest_complete_date"] >= expected))
                if attempt.get("dates_aligned") and fresh_enough:
                    break
            stooq_attempt = _refresh_stooq_daily_if_needed(symbol, db_path, expected)
            if stooq_attempt:
                provider_attempts.append(stooq_attempt)
        dates = latest_dates(symbol, db_path)
        item.update(dates)
        item["provider_attempts"] = provider_attempts
        item["source"] = dates.get("feed") or "unknown"
        item["using_cached_data"] = not refresh
        item["errors"] = [error for attempt in provider_attempts for error in attempt.get("errors", [])]
        if expected and dates.get("intraday_latest_complete_date") and dates["intraday_latest_complete_date"] < expected:
            item["errors"].append(f"最近完整分时数据截至 {dates['intraday_latest_complete_date']}，按交易日历应至少为 {expected}；不会用过期数据生成新预测。")
        if not dates.get("dates_aligned"):
            item["forecast_error"] = f"同源日线日期 {dates.get('daily_latest_date') or '缺失'} 与完整十五分钟分时日期 {dates.get('intraday_latest_complete_date') or '缺失'} 不一致；为避免源混用或日期错配，停止生成。"
            item["errors"].append(item["forecast_error"])
        elif expected and dates.get("intraday_latest_complete_date") and dates["intraday_latest_complete_date"] < expected:
            item["forecast_error"] = item["errors"][-1]
        else:
            feed = str(dates.get("feed") or "yahoo")
            item["settlements"] = settle_due_forecasts(symbol, db_path, feed)
            try:
                item["new_forecast"] = generate_forecast(symbol, db_path, feed)
            except Exception as exc:
                item["forecast_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                item["errors"].append(item["forecast_error"])
        item.setdefault("settlements", [])
        item["evaluations"] = get_evaluations(symbol, 10, db_path)
        item["success"] = bool(item.get("dates_aligned") and item.get("new_forecast") and (not expected or str(item.get("intraday_latest_complete_date") or "") >= expected))
        item["finished_at"] = _utc_now()
        status["symbols"][symbol] = item
        with connect_db(db_path) as conn:
            conn.execute("INSERT INTO asset_sync_runs(started_at,finished_at,symbol,success,daily_latest_date,intraday_latest_complete_date,daily_rows,intraday_rows,news_rows,source,error) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                item["started_at"], item["finished_at"], symbol, int(item["success"]),
                item.get("daily_latest_date"), item.get("intraday_latest_complete_date"),
                sum(int(attempt.get("daily_rows_updated", 0)) for attempt in provider_attempts),
                sum(int(attempt.get("intraday_rows_updated", 0)) for attempt in provider_attempts),
                (status.get("news_sync", {}).get("symbols", {}).get(symbol, {}) or {}).get("news_rows_saved", 0),
                item["source"], "; ".join(item["errors"]),
            ))
        if not item["success"]:
            status["overall_success"] = False
    status["finished_at"] = _utc_now()
    status["status_file"] = str(REPORT_ROOT / "latest_status.json")
    _write_json(REPORT_ROOT / "latest_status.json", status)
    return status

