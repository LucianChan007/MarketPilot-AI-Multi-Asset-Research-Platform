from __future__ import annotations
"""No-key Yahoo Finance chart-data client for the video content workflow.

This is deliberately separate from the project's Alpaca research pipeline. Yahoo
bars are stored with feed='yahoo' and never mislabeled as Alpaca/IEX data.
"""

import json
import os
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market_data.db"
NY = ZoneInfo("America/New_York")
YAHOO_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
EODHD_INTRADAY_URL = "https://eodhd.com/api/intraday/QQQM.US"

# Load local credentials if present; only the process environment is modified.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


def _eodhd_token() -> str | None:
    for name in ("EODHD_API_KEY", "EODHD_API_TOKEN", "EODHD_TOKEN", "EODHD_APIKEY"):
        value = os.getenv(name, "").strip()
        if value and not value.startswith("replace_with_"):
            return value
    return None


def _fetch_chart_payload(symbol: str, period: str, interval: str) -> dict[str, Any]:
    errors: list[str] = []
    for host in YAHOO_HOSTS:
        url = f"{host}/v8/finance/chart/{symbol}"
        try:
            response = requests.get(
                url,
                params={"range": period, "interval": interval, "events": "div,splits"},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 Chrome/128.0 Safari/537.36",
                    "Accept": "application/json,text/plain,*/*",
                },
                timeout=22,
            )
            response.raise_for_status()
            payload = response.json()
            chart = payload.get("chart") if isinstance(payload, dict) else None
            result = chart.get("result") if isinstance(chart, dict) else None
            if not result or not isinstance(result[0], dict):
                err = chart.get("error") if isinstance(chart, dict) else None
                raise RuntimeError(str(err or "行情服务没有返回数据"))
            return result[0]
        except Exception as exc:  # try secondary public host too
            errors.append(f"{host}: {type(exc).__name__}: {exc}")
    raise RuntimeError("公开行情源请求失败；" + "；".join(errors)[-1200:])


def fetch_eodhd_intraday(period_days: int = 60) -> pd.DataFrame:
    """Fetch 5-minute bars from the user's already-configured EODHD account and aggregate to 15m.

    Any token stays in the request parameters only. Error messages intentionally do
    not include request URLs or response bodies that could accidentally expose it.
    """
    token = _eodhd_token()
    if not token:
        raise RuntimeError("本机未配置 EODHD 行情密钥")
    now = datetime.now(timezone.utc)
    params = {
        "api_token": token,
        "interval": "5m",
        "from": int((now - timedelta(days=period_days)).timestamp()),
        "to": int(now.timestamp()),
        "fmt": "json",
    }
    try:
        response = requests.get(EODHD_INTRADAY_URL, params=params, timeout=25,
                                headers={"User-Agent": "QQQM-Video-Research/1.0"})
    except requests.RequestException as exc:
        raise RuntimeError(f"EODHD 分时网络请求未成功（{type(exc).__name__}）") from exc
    if response.status_code != 200:
        raise RuntimeError(f"EODHD 分时接口返回 HTTP {response.status_code}；可能是账户权限、限流或网络问题")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("EODHD 分时接口返回的不是有效数据格式") from exc
    if isinstance(payload, dict):
        # Do not print response content because a provider may echo URL parameters.
        raise RuntimeError("EODHD 分时接口未返回行情数组；请检查套餐权限和密钥状态")
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("EODHD 没有返回可用的五分钟行情")
    frame = pd.DataFrame(payload)
    required = {"datetime", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        raise RuntimeError("EODHD 分时数据字段不完整，无法安全生成曲线")
    stamps = pd.to_datetime(frame["datetime"], errors="coerce")
    if getattr(stamps.dt, "tz", None) is None:
        stamps = stamps.dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    else:
        stamps = stamps.dt.tz_convert(NY)
    frame["timestamp_ny"] = stamps
    frame["timestamp_utc"] = stamps.dt.tz_convert("UTC")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in frame:
            frame[col] = 0
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp_ny", "open", "high", "low", "close"])
    frame = frame[(frame["timestamp_ny"].dt.strftime("%H:%M") >= "09:30") &
                  (frame["timestamp_ny"].dt.strftime("%H:%M") < "16:00")].copy()
    frame["trade_date_ny"] = frame["timestamp_ny"].dt.strftime("%Y-%m-%d")
    if frame.empty:
        raise RuntimeError("EODHD 返回数据中没有正常交易时段的行情")
    minutes = frame["timestamp_ny"].dt.hour * 60 + frame["timestamp_ny"].dt.minute
    frame["slot_minute"] = 570 + ((minutes - 570) // 15) * 15
    frame["session_time_ny"] = frame["slot_minute"].map(lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}")
    frame = frame.sort_values("timestamp_ny")
    grouped = []
    for (day, slot), group in frame.groupby(["trade_date_ny", "session_time_ny"], sort=True):
        # Keep only complete 15-minute buckets made of at least three 5-minute bars.
        if len(group) < 3:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        local_stamp = datetime.combine(date.fromisoformat(str(day)), time.fromisoformat(str(slot))).replace(tzinfo=NY)
        grouped.append({
            "timestamp_ny": pd.Timestamp(local_stamp),
            "timestamp_utc": pd.Timestamp(local_stamp).tz_convert("UTC"),
            "trade_date_ny": str(day), "session_time_ny": str(slot),
            "open": float(first["open"]), "high": float(group["high"].max()),
            "low": float(group["low"].min()), "close": float(last["close"]),
            "volume": int(group["volume"].fillna(0).sum()),
            "feed": "eodhd", "timeframe": "15Min",
        })
    result = pd.DataFrame(grouped)
    if result.empty:
        raise RuntimeError("EODHD 返回了五分钟记录，但无法组成完整的十五分钟曲线")
    return result.reset_index(drop=True)


def fetch_eodhd_daily() -> pd.DataFrame:
    """Use the project's existing EODHD daily client as a backup to the public endpoint."""
    token = _eodhd_token()
    if not token:
        raise RuntimeError("本机未配置 EODHD 行情密钥")
    previous_key = os.environ.get("EODHD_API_KEY")
    os.environ["EODHD_API_KEY"] = token
    try:
        from app.market_data import _fetch_eodhd
        history = _fetch_eodhd("QQQM", "1y")
    except Exception as exc:
        # Avoid forwarding request URLs or other details that could contain credentials.
        raise RuntimeError(f"EODHD 日线接口未成功返回数据（{type(exc).__name__}）") from exc
    finally:
        if previous_key is None:
            os.environ.pop("EODHD_API_KEY", None)
        else:
            os.environ["EODHD_API_KEY"] = previous_key
    if history is None or history.empty:
        raise RuntimeError("EODHD 没有返回可用日线")
    frame = pd.DataFrame({
        "trade_date_ny": pd.to_datetime(history.index, errors="coerce").strftime("%Y-%m-%d"),
        "open": pd.to_numeric(history["Open"], errors="coerce").to_numpy(),
        "high": pd.to_numeric(history["High"], errors="coerce").to_numpy(),
        "low": pd.to_numeric(history["Low"], errors="coerce").to_numpy(),
        "close": pd.to_numeric(history["Close"], errors="coerce").to_numpy(),
        "adj_close": pd.to_numeric(history.get("Adj Close", history["Close"]), errors="coerce").to_numpy(),
        "volume": pd.to_numeric(history.get("Volume", 0), errors="coerce").fillna(0).to_numpy()
            if hasattr(history.get("Volume", 0), "fillna") else 0,
    })
    return frame.dropna(subset=["trade_date_ny", "open", "high", "low", "close"]).reset_index(drop=True)


def _payload_to_frame(payload: dict[str, Any], *, daily: bool) -> pd.DataFrame:
    timestamps = payload.get("timestamp") or []
    indicators = payload.get("indicators") or {}
    quote_list = indicators.get("quote") or []
    if not timestamps or not quote_list:
        return pd.DataFrame()
    quote = quote_list[0] or {}
    adj_list = indicators.get("adjclose") or []
    adjclose = (adj_list[0] or {}).get("adjclose", []) if adj_list else []
    length = len(timestamps)
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": (quote.get("open") or [None] * length),
        "high": (quote.get("high") or [None] * length),
        "low": (quote.get("low") or [None] * length),
        "close": (quote.get("close") or [None] * length),
        "volume": (quote.get("volume") or [0] * length),
    })
    if daily:
        frame["adj_close"] = adjclose if len(adjclose) == length else frame["close"]
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if daily:
        frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame["timestamp_ny"] = frame["timestamp_utc"].dt.tz_convert(NY)
    frame["trade_date_ny"] = frame["timestamp_ny"].dt.strftime("%Y-%m-%d")
    frame["session_time_ny"] = frame["timestamp_ny"].dt.strftime("%H:%M")
    frame = frame.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    return frame.reset_index(drop=True)


def fetch_intraday(symbol: str = "QQQM", period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    payload = _fetch_chart_payload(symbol, period, interval)
    frame = _payload_to_frame(payload, daily=False)
    # Keep regular-session bars only, rather than mixing in premarket/after-hours.
    frame = frame[(frame["session_time_ny"] >= "09:30") & (frame["session_time_ny"] < "16:00")]
    if frame.empty:
        raise RuntimeError("公开行情源返回了数据，但没有找到正常交易时段的分时行情。")
    return frame


def fetch_daily(symbol: str = "QQQM", period: str = "1y") -> pd.DataFrame:
    payload = _fetch_chart_payload(symbol, period, "1d")
    frame = _payload_to_frame(payload, daily=True)
    if frame.empty:
        raise RuntimeError("公开行情源没有返回有效日线行情。")
    return frame


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS intraday_bars (
        symbol TEXT NOT NULL, feed TEXT NOT NULL, timeframe TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL, trade_date_ny TEXT NOT NULL, timestamp_ny TEXT NOT NULL,
        session_time_ny TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
        close REAL NOT NULL, volume INTEGER NOT NULL DEFAULT 0, trade_count INTEGER,
        vwap REAL, fetched_at_utc TEXT NOT NULL,
        PRIMARY KEY (symbol, feed, timeframe, timestamp_utc))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_intraday_symbol_date ON intraday_bars(symbol, feed, timeframe, trade_date_ny)")


def save_intraday(frame: pd.DataFrame, db_path: Path = DB_PATH, feed: str = "yahoo") -> int:
    if frame.empty:
        return 0
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = []
    for _, row in frame.iterrows():
        stamp = pd.Timestamp(row["timestamp_utc"]).tz_convert("UTC").isoformat(timespec="seconds").replace("+00:00", "Z")
        ny_stamp = pd.Timestamp(row["timestamp_ny"]).isoformat(timespec="seconds")
        row_feed = str(row.get("feed") or feed)
        rows.append((
            "QQQM", row_feed, "15Min", stamp, str(row["trade_date_ny"]), ny_stamp,
            str(row["session_time_ny"]), float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]), int(row["volume"] or 0), None, None, fetched,
        ))
    with sqlite3.connect(db_path, timeout=30) as conn:
        _ensure_tables(conn)
        conn.executemany("""INSERT INTO intraday_bars (
            symbol,feed,timeframe,timestamp_utc,trade_date_ny,timestamp_ny,session_time_ny,
            open,high,low,close,volume,trade_count,vwap,fetched_at_utc
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol,feed,timeframe,timestamp_utc) DO UPDATE SET
            trade_date_ny=excluded.trade_date_ny,timestamp_ny=excluded.timestamp_ny,
            session_time_ny=excluded.session_time_ny,open=excluded.open,high=excluded.high,
            low=excluded.low,close=excluded.close,volume=excluded.volume,fetched_at_utc=excluded.fetched_at_utc""", rows)
    return len(rows)


def save_daily(frame: pd.DataFrame, db_path: Path = DB_PATH, source: str = "yahoo_chart_api") -> int:
    if frame.empty:
        return 0
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for _, row in frame.iterrows():
        rows.append((
            "QQQM", str(row["trade_date_ny"]), float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]), float(row.get("adj_close", row["close"])),
            int(row.get("volume", 0) or 0), 0.0, 0.0, source, fetched,
        ))
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS market_daily (
            ticker TEXT NOT NULL, trade_date TEXT NOT NULL, open REAL, high REAL, low REAL,
            close REAL, adj_close REAL, volume INTEGER, dividends REAL DEFAULT 0,
            stock_splits REAL DEFAULT 0, source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL, PRIMARY KEY(ticker,trade_date))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_daily_date ON market_daily(trade_date)")
        conn.executemany("""INSERT INTO market_daily (
            ticker,trade_date,open,high,low,close,adj_close,volume,dividends,stock_splits,source,fetched_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(ticker,trade_date) DO UPDATE SET open=excluded.open,high=excluded.high,
        low=excluded.low,close=excluded.close,adj_close=excluded.adj_close,volume=excluded.volume,
        source=excluded.source,fetched_at=excluded.fetched_at""", rows)
    return len(rows)


def save_latest_completed_intraday_as_daily(frame: pd.DataFrame, db_path: Path = DB_PATH, source: str = "yahoo_intraday_derived") -> str | None:
    """Backfill today's settled daily OHLC from a complete intraday session only if daily data lag."""
    if frame.empty:
        return None
    work = frame.copy().sort_values("timestamp_ny")
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    rows = []
    for day, group in work.groupby("trade_date_ny", sort=True):
        group = group.sort_values("timestamp_ny")
        if str(day) > now_ny.date().isoformat():
            continue
        if str(day) == now_ny.date().isoformat() and now_ny.time().replace(tzinfo=None).hour < 16:
            continue
        if str(group["session_time_ny"].max()) < "15:45" or len(group) < 24:
            continue
        opening = group[group["session_time_ny"] == "09:30"]
        if opening.empty:
            continue
        rows.append({"trade_date_ny": str(day), "open": float(opening.iloc[0]["open"]),
                     "high": float(group["high"].max()), "low": float(group["low"].min()),
                     "close": float(group.iloc[-1]["close"]), "adj_close": float(group.iloc[-1]["close"]),
                     "volume": int(pd.to_numeric(group["volume"], errors="coerce").fillna(0).sum())})
    if not rows:
        return None
    latest = rows[-1]
    with sqlite3.connect(db_path) as conn:
        current = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
    existing = str(current[0])[:10] if current and current[0] else ""
    if latest["trade_date_ny"] <= existing:
        return existing or None
    save_daily(pd.DataFrame([latest]), db_path)
    # Mark the derived daily row explicitly so users can distinguish it from the daily provider response.
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE market_daily SET source=? WHERE UPPER(ticker)='QQQM' AND trade_date=?",
                     (source, latest["trade_date_ny"]))
    return latest["trade_date_ny"]


def load_intraday_from_db(db_path: Path = DB_PATH, feed: str | None = None) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='intraday_bars'").fetchone()
            if not exists:
                return pd.DataFrame()
            sql = """SELECT timestamp_utc,trade_date_ny,session_time_ny,
                open,high,low,close,volume,feed,timeframe FROM intraday_bars
                WHERE UPPER(symbol)='QQQM' AND lower(timeframe)='15min' AND lower(session_time_ny)>='09:30' AND lower(session_time_ny)<'16:00'"""
            params: tuple[Any, ...] = ()
            if feed:
                sql += " AND lower(feed)=?"
                params = (feed.lower(),)
            sql += " ORDER BY timestamp_utc"
            frame = pd.read_sql_query(sql, conn, params=params)
        if frame.empty:
            return frame
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp_utc"])
        frame["timestamp_ny"] = frame["timestamp_utc"].dt.tz_convert(NY)
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        # Only use one 15-minute source for a given timestamp: prefer Yahoo rows,
        # then choose a single available alternative to avoid duplicate plotting.
        frame = frame.sort_values(["timestamp_utc", "feed"], key=lambda s: s.map(lambda x: {"eodhd": 0, "yahoo": 1}.get(str(x).lower(), 2)) if s.name == "feed" else s)
        frame = frame.drop_duplicates("timestamp_utc", keep="first")
        frame["trade_date_ny"] = frame["timestamp_ny"].dt.strftime("%Y-%m-%d")
        frame["session_time_ny"] = frame["timestamp_ny"].dt.strftime("%H:%M")
        return frame.reset_index(drop=True)
    except (sqlite3.Error, ValueError, KeyError):
        return pd.DataFrame()


def refresh_market_data(db_path: Path = DB_PATH) -> dict[str, Any]:
    """Try the user's configured EODHD account first, then Yahoo's no-key public feed."""
    status: dict[str, Any] = {
        "source": "优先尝试已配置的 EODHD，再回退到免密公开行情",
        "daily_rows": 0, "intraday_rows": 0, "errors": [], "attempt_notes": [],
    }
    daily_updated = False
    try:
        daily = fetch_daily("QQQM", "1y")
        now_ny = datetime.now(timezone.utc).astimezone(NY)
        if now_ny.time().replace(tzinfo=None).hour < 16 or (now_ny.time().replace(tzinfo=None).hour == 16 and now_ny.time().minute < 15):
            daily = daily[daily["trade_date_ny"] < now_ny.date().isoformat()]
        status["daily_rows"] = save_daily(daily, db_path)
        status["daily_latest_date"] = str(daily["trade_date_ny"].max()) if not daily.empty else None
        status["daily_provider"] = "公开免密行情"
        daily_updated = not daily.empty
    except Exception as exc:
        status["attempt_notes"].append(f"公开日线源未成功：{type(exc).__name__}")

    if not daily_updated and _eodhd_token():
        try:
            daily = fetch_eodhd_daily()
            now_ny = datetime.now(timezone.utc).astimezone(NY)
            if now_ny.time().replace(tzinfo=None).hour < 16 or (now_ny.time().replace(tzinfo=None).hour == 16 and now_ny.time().minute < 15):
                daily = daily[daily["trade_date_ny"] < now_ny.date().isoformat()]
            status["daily_rows"] = save_daily(daily, db_path, source="eodhd_chart_api")
            status["daily_latest_date"] = str(daily["trade_date_ny"].max()) if not daily.empty else None
            status["daily_provider"] = "EODHD 备用日线"
            daily_updated = not daily.empty
        except Exception as exc:
            status["attempt_notes"].append(f"备用日线源未成功：{type(exc).__name__}")

    intraday: pd.DataFrame = pd.DataFrame()
    provider_errors: list[str] = []
    if _eodhd_token():
        try:
            intraday = fetch_eodhd_intraday("60d")
            now_ny = datetime.now(timezone.utc).astimezone(NY)
            if now_ny.time().replace(tzinfo=None).hour < 16 or (now_ny.time().replace(tzinfo=None).hour == 16 and now_ny.time().minute < 15):
                intraday = intraday[intraday["trade_date_ny"] < now_ny.date().isoformat()]
            if not intraday.empty:
                status["intraday_provider"] = "EODHD"
                status["intraday_rows"] = save_intraday(intraday, db_path, feed="eodhd")
        except Exception as exc:
            status["attempt_notes"].append(f"备用分时源未成功：{type(exc).__name__}")

    if intraday.empty:
        try:
            intraday = fetch_intraday("QQQM", "60d", "15m")
            now_ny = datetime.now(timezone.utc).astimezone(NY)
            if now_ny.time().replace(tzinfo=None).hour < 16 or (now_ny.time().replace(tzinfo=None).hour == 16 and now_ny.time().minute < 15):
                intraday = intraday[intraday["trade_date_ny"] < now_ny.date().isoformat()]
            if not intraday.empty:
                intraday["feed"] = "yahoo"
                status["intraday_provider"] = "雅虎财经公开接口"
                status["intraday_rows"] = save_intraday(intraday, db_path, feed="yahoo")
        except Exception as exc:
            provider_errors.append(f"免密公开分时源未成功：{type(exc).__name__}")
    if intraday.empty:
        status["errors"].extend(status["attempt_notes"] + provider_errors or ["所有可用分时行情源均未返回数据"])
    elif provider_errors:
        status["attempt_notes"].extend(provider_errors)

    if not intraday.empty:
        try:
            derived_source = "eodhd_intraday_derived" if str(status.get("intraday_provider")) == "EODHD" else "yahoo_intraday_derived"
            derived_day = save_latest_completed_intraday_as_daily(intraday, db_path, source=derived_source)
            if derived_day:
                status["daily_latest_date"] = derived_day
                status["daily_derived_from_intraday"] = True
        except Exception as exc:
            status["errors"].append(f"完整分时转日线失败：{type(exc).__name__}")

    fallback = load_intraday_from_db(db_path)
    status["fresh_intraday_frame"] = intraday
    status["intraday_frame"] = intraday if not intraday.empty else fallback
    if not intraday.empty:
        provider = str(status.get("intraday_provider") or "")
        provider_label = "已配置行情源" if provider == "EODHD" else "免密公开行情"
        status["data_source_used"] = f"公开联网行情（{provider_label}）"
    elif not fallback.empty:
        status["data_source_used"] = "本地历史缓存"
    else:
        status["data_source_used"] = "无可用行情"
    if status["intraday_frame"] is not None and not status["intraday_frame"].empty:
        status["intraday_latest_date"] = str(status["intraday_frame"]["trade_date_ny"].max())
    return status

