from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "market_data.db"
ENV_PATH = ROOT / ".env"
ALPACA_BASE_URL = "https://data.alpaca.markets/v2/stocks"
NEW_YORK = ZoneInfo("America/New_York")

BAR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS intraday_bars (
    symbol TEXT NOT NULL,
    feed TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    trade_date_ny TEXT NOT NULL,
    timestamp_ny TEXT NOT NULL,
    session_time_ny TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL DEFAULT 0,
    trade_count INTEGER,
    vwap REAL,
    fetched_at_utc TEXT NOT NULL,
    PRIMARY KEY (symbol, feed, timeframe, timestamp_utc)
)
"""


def load_env_values(path: Path = ENV_PATH) -> dict[str, str]:
    """Read simple KEY=VALUE entries without displaying their values."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    return values


def get_alpaca_credentials(env_path: Path = ENV_PATH) -> tuple[str, str]:
    env_values = load_env_values(env_path)
    key = os.environ.get("ALPACA_API_KEY_ID") or env_values.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY") or env_values.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "未找到 Alpaca API 凭据。请在项目根目录 .env 中设置 "
            "ALPACA_API_KEY_ID 和 ALPACA_API_SECRET_KEY。不要把密钥发给任何人。"
        )
    return key.strip(), secret.strip()


def parse_bar_timestamp(timestamp: str) -> tuple[str, str, str]:
    """Return normalized UTC, New York timestamp and New York session date."""
    normalized = timestamp.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        raise ValueError(f"K线时间缺少时区信息: {timestamp}")
    dt_utc = dt.astimezone(timezone.utc)
    dt_ny = dt_utc.astimezone(NEW_YORK)
    timestamp_utc = dt_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
    timestamp_ny = dt_ny.isoformat(timespec="seconds")
    return timestamp_utc, timestamp_ny, dt_ny.date().isoformat()


def _request_page(
    *, symbol: str, timeframe: str, start: str, end: str,
    feed: str, page_token: str | None, key: str, secret: str,
    timeout: int = 30,
) -> dict[str, Any]:
    params: dict[str, str] = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": "10000",
        "feed": feed,
        "adjustment": "raw",
        "sort": "asc",
    }
    if page_token:
        params["page_token"] = page_token
    url = f"{ALPACA_BASE_URL}/{symbol}/bars?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
            "User-Agent": "QQQM-Investment-Assistant-intraday-data/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Alpaca 请求失败：{type(exc).__name__}: {exc}") from exc

    if status != 200:
        try:
            payload = json.loads(body)
            message = payload.get("message") or payload.get("error") or body[:800]
        except json.JSONDecodeError:
            message = body[:800]
        if status == 401:
            raise RuntimeError("Alpaca 返回 401：请检查 API Key/Secret 是否正确。")
        if status == 403:
            raise RuntimeError(f"Alpaca 返回 403：账户或 feed 权限不足。详情：{message}")
        if status == 429:
            raise RuntimeError("Alpaca 返回 429：触发频率限制，请稍后再试。")
        raise RuntimeError(f"Alpaca HTTP {status}：{message}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Alpaca 返回内容不是有效 JSON。") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("bars", []), list):
        raise RuntimeError("Alpaca 返回结构不符合预期。")
    return payload


def fetch_intraday_bars(
    *, symbol: str = "QQQM", timeframe: str = "5Min", feed: str = "iex",
    lookback_days: int = 90, end_utc: datetime | None = None,
    env_path: Path = ENV_PATH,
) -> list[dict[str, Any]]:
    """Fetch historical intraday bars, following all Alpaca pagination tokens.

    Prices stay in Alpaca's raw intraday price basis. The feed is always recorded
    with the bars so they are never silently mixed with EODHD daily prices.
    """
    if lookback_days < 2 or lookback_days > 3650:
        raise ValueError("lookback_days 必须在 2 到 3650 天之间。")
    if not symbol.strip() or not timeframe.strip():
        raise ValueError("symbol 和 timeframe 不能为空。")
    if feed not in {"iex", "sip"}:
        raise ValueError("feed 仅允许 iex 或 sip；免费 Basic 原型通常使用 iex。")

    key, secret = get_alpaca_credentials(env_path)
    end_dt = end_utc or datetime.now(timezone.utc)
    if end_dt.tzinfo is None:
        raise ValueError("end_utc 必须包含时区。")
    end_dt = end_dt.astimezone(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    start = start_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    end = end_dt.isoformat(timespec="seconds").replace("+00:00", "Z")

    bars: list[dict[str, Any]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    pages = 0
    while True:
        pages += 1
        if pages > 50:
            raise RuntimeError("分页次数超过安全上限 50；已停止，避免异常循环。")
        payload = _request_page(
            symbol=symbol.upper(), timeframe=timeframe, start=start, end=end,
            feed=feed, page_token=page_token, key=key, secret=secret,
        )
        rows = payload.get("bars", [])
        bars.extend(row for row in rows if isinstance(row, dict))
        next_token = payload.get("next_page_token")
        if not next_token:
            break
        next_token = str(next_token)
        if next_token in seen_tokens:
            raise RuntimeError("Alpaca 返回重复 page_token，已停止防止无限循环。")
        seen_tokens.add(next_token)
        page_token = next_token

    # Deduplicate timestamps in memory, keeping the last version returned.
    by_timestamp: dict[str, dict[str, Any]] = {}
    for bar in bars:
        timestamp = bar.get("t")
        if timestamp:
            by_timestamp[str(timestamp)] = bar
    return [by_timestamp[key] for key in sorted(by_timestamp)]


def ensure_intraday_table(conn: sqlite3.Connection) -> None:
    conn.execute(BAR_TABLE_SQL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_intraday_symbol_date "
        "ON intraday_bars(symbol, feed, timeframe, trade_date_ny)"
    )
    conn.commit()


def save_intraday_bars(
    bars: Iterable[dict[str, Any]], *, symbol: str, feed: str,
    timeframe: str, db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Idempotently upsert bars into a dedicated table in the local DB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Any, ...]] = []
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    for bar in bars:
        timestamp = str(bar.get("t") or "")
        if not timestamp:
            continue
        timestamp_utc, timestamp_ny, trade_date_ny = parse_bar_timestamp(timestamp)
        session_time_ny = timestamp_ny[11:16]
        try:
            open_ = float(bar["o"])
            high = float(bar["h"])
            low = float(bar["l"])
            close = float(bar["c"])
            volume = int(float(bar.get("v", 0) or 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"K线缺少有效 OHLCV 字段：{bar}") from exc
        if min(open_, high, low, close) <= 0 or low > min(open_, close) or high < max(open_, close):
            raise ValueError(f"K线 OHLC 不合理：{bar}")
        trade_count_val = bar.get("n")
        try:
            trade_count = int(trade_count_val) if trade_count_val is not None else None
        except (TypeError, ValueError):
            trade_count = None
        vwap_val = bar.get("vw")
        try:
            vwap = float(vwap_val) if vwap_val is not None else None
        except (TypeError, ValueError):
            vwap = None
        rows.append((
            symbol.upper(), feed, timeframe, timestamp_utc, trade_date_ny,
            timestamp_ny, session_time_ny, open_, high, low, close, volume,
            trade_count, vwap, fetched_at,
        ))

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        ensure_intraday_table(conn)
        before = conn.total_changes
        conn.executemany(
            """INSERT INTO intraday_bars (
                symbol, feed, timeframe, timestamp_utc, trade_date_ny,
                timestamp_ny, session_time_ny, open, high, low, close, volume,
                trade_count, vwap, fetched_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, feed, timeframe, timestamp_utc) DO UPDATE SET
                trade_date_ny=excluded.trade_date_ny,
                timestamp_ny=excluded.timestamp_ny,
                session_time_ny=excluded.session_time_ny,
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume,
                trade_count=excluded.trade_count, vwap=excluded.vwap,
                fetched_at_utc=excluded.fetched_at_utc""",
            rows,
        )
        conn.commit()
        changed = conn.total_changes - before
        total = conn.execute(
            "SELECT COUNT(*) FROM intraday_bars WHERE symbol=? AND feed=? AND timeframe=?",
            (symbol.upper(), feed, timeframe),
        ).fetchone()[0]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT trade_date_ny) FROM intraday_bars "
            "WHERE symbol=? AND feed=? AND timeframe=? "
            "AND session_time_ny >= '09:30' AND session_time_ny < '16:00'",
            (symbol.upper(), feed, timeframe),
        ).fetchone()[0]
        return {"received": len(rows), "written_or_updated": changed, "total_rows": total, "sessions_with_regular_bars": sessions}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def collect_intraday_history(
    *, symbol: str = "QQQM", timeframe: str = "5Min", feed: str = "iex",
    lookback_days: int = 90, db_path: Path = DEFAULT_DB_PATH,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    bars = fetch_intraday_bars(
        symbol=symbol, timeframe=timeframe, feed=feed,
        lookback_days=lookback_days, env_path=env_path,
    )
    if not bars:
        raise RuntimeError(
            "API 请求成功但没有返回 K 线。检查日期区间、标的、feed 与数据权限；未修改数据库。"
        )
    saved = save_intraday_bars(
        bars, symbol=symbol, feed=feed, timeframe=timeframe, db_path=db_path,
    )
    timestamps = [str(b["t"]) for b in bars if b.get("t")]
    return {
        "symbol": symbol.upper(),
        "feed": feed,
        "timeframe": timeframe,
        "lookback_days": lookback_days,
        "raw_prices": True,
        "timestamps_are_stored_in_utc": True,
        "session_dates_use": "America/New_York",
        "received_bars": len(bars),
        "first_timestamp_utc": min(timestamps) if timestamps else None,
        "last_timestamp_utc": max(timestamps) if timestamps else None,
        **saved,
        "db_path": str(db_path.resolve()),
        "note": "IEX 是单一交易所 feed。缺失的 5 分钟时间段不会被伪造为真实 K 线；请先查看覆盖率审计再训练或绘制预测路径。",
    }
