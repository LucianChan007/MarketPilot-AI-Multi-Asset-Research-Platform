"""Market clock for QQQM Investment Assistant.

Uses an exchange calendar when pandas_market_calendars is installed so that
weekends, US market holidays, and early closes are handled. If the dependency
is missing, it falls back to a clearly marked weekday/time estimate.

The displayed local time defaults to Hong Kong time. Override with
QQQM_LOCAL_TIMEZONE, e.g. QQQM_LOCAL_TIMEZONE=Asia/Tokyo.
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

NEW_YORK_TZ = ZoneInfo("America/New_York")
DEFAULT_LOCAL_TIMEZONE = "Asia/Hong_Kong"


def _get_local_zone() -> tuple[ZoneInfo, str]:
    timezone_name = os.getenv("QQQM_LOCAL_TIMEZONE", DEFAULT_LOCAL_TIMEZONE).strip()
    try:
        return ZoneInfo(timezone_name), timezone_name
    except ZoneInfoNotFoundError:
        return ZoneInfo(DEFAULT_LOCAL_TIMEZONE), DEFAULT_LOCAL_TIMEZONE


def _format_duration(delta: timedelta | None) -> str | None:
    if delta is None:
        return None
    total_seconds = max(0, int(delta.total_seconds()))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}天 {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@lru_cache(maxsize=32)
def _calendar_sessions(start_day: date, end_day: date) -> tuple[tuple[date, datetime, datetime], ...] | None:
    """Cache exchange sessions by date so a one-second UI clock stays light."""
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        return None

    calendar = mcal.get_calendar("NASDAQ")
    schedule = calendar.schedule(
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
    )
    sessions: list[tuple[date, datetime, datetime]] = []
    for session_label, row in schedule.iterrows():
        session_date = session_label.date() if hasattr(session_label, "date") else session_label
        market_open = row["market_open"].to_pydatetime().astimezone(NEW_YORK_TZ)
        market_close = row["market_close"].to_pydatetime().astimezone(NEW_YORK_TZ)
        sessions.append((session_date, market_open, market_close))
    return tuple(sessions)


def _schedule_with_calendar(now_ny: datetime) -> dict[str, Any] | None:
    """Return calendar-based market status, or None if calendar is unavailable."""
    today = now_ny.date()
    sessions = _calendar_sessions(today, today + timedelta(days=31))
    if sessions is None:
        return None

    today_open: datetime | None = None
    today_close: datetime | None = None
    upcoming_opens: list[datetime] = []
    for session_date, market_open, market_close in sessions:
        if session_date == today:
            today_open = market_open
            today_close = market_close
        if market_open > now_ny:
            upcoming_opens.append(market_open)

    regular_open = today_open is not None and today_close is not None and today_open <= now_ny < today_close
    before_regular_open = today_open is not None and now_ny < today_open
    after_regular_close = today_close is not None and now_ny >= today_close

    next_open = upcoming_opens[0] if upcoming_opens else None
    if regular_open:
        status = "🟢 正常交易中"
        countdown_target = today_close
        countdown_label = "距离常规收盘"
        is_regular_open = True
    elif before_regular_open and now_ny.time() >= time(4, 0):
        status = "🟡 盘前时段（常规交易未开市）"
        countdown_target = today_open
        countdown_label = "距离常规开盘"
        is_regular_open = False
    elif after_regular_close and now_ny.time() < time(20, 0):
        status = "🟠 盘后时段（常规交易已收市）"
        countdown_target = next_open
        countdown_label = "距离下次常规开盘"
        is_regular_open = False
    else:
        is_holiday_or_weekend = today_open is None
        status = "🔴 休市（非交易日）" if is_holiday_or_weekend else "🔴 常规交易休市"
        countdown_target = next_open
        countdown_label = "距离下次常规开盘"
        is_regular_open = False

    countdown = countdown_target - now_ny if countdown_target else None
    local_zone, local_zone_name = _get_local_zone()
    return {
        "local_time": now_ny.astimezone(local_zone).strftime("%Y-%m-%d %H:%M:%S"),
        "local_timezone": local_zone_name,
        "new_york_time": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "new_york_date": now_ny.date().isoformat(),
        "market_status": status,
        "is_open": is_regular_open,
        "countdown_label": countdown_label,
        "countdown": _format_duration(countdown),
        "next_regular_open": next_open.strftime("%Y-%m-%d %H:%M:%S %Z") if next_open else None,
        "today_regular_open": today_open.strftime("%Y-%m-%d %H:%M:%S %Z") if today_open else None,
        "today_regular_close": today_close.strftime("%Y-%m-%d %H:%M:%S %Z") if today_close else None,
        "calendar_source": "NASDAQ 交易日历（pandas_market_calendars）",
        "calendar_is_authoritative": True,
        "calendar_note": "按交易日历与美东时间计算；不是券商逐笔实时状态 API。",
    }


def _fallback_weekday_estimate(now_ny: datetime) -> dict[str, Any]:
    """Fallback when the exchange-calendar package is unavailable."""
    current_minutes = now_ny.hour * 60 + now_ny.minute
    open_minutes = 9 * 60 + 30
    close_minutes = 16 * 60
    premarket_minutes = 4 * 60
    after_hours_minutes = 20 * 60
    is_weekday = now_ny.weekday() < 5

    target: datetime | None = None
    if is_weekday and open_minutes <= current_minutes < close_minutes:
        status = "🟢 正常交易中（近似判断）"
        is_open = True
        target = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        countdown_label = "距离常规收盘"
    elif is_weekday and premarket_minutes <= current_minutes < open_minutes:
        status = "🟡 盘前时段（近似判断）"
        is_open = False
        target = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        countdown_label = "距离常规开盘"
    elif is_weekday and close_minutes <= current_minutes < after_hours_minutes:
        status = "🟠 盘后时段（近似判断）"
        is_open = False
        next_day = now_ny.date() + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        target = datetime.combine(next_day, time(9, 30), tzinfo=NEW_YORK_TZ)
        countdown_label = "距离下次常规开盘"
    else:
        status = "🔴 休市（近似判断，未检查节假日）"
        is_open = False
        next_day = now_ny.date() + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        target = datetime.combine(next_day, time(9, 30), tzinfo=NEW_YORK_TZ)
        countdown_label = "距离下次常规开盘"

    local_zone, local_zone_name = _get_local_zone()
    return {
        "local_time": now_ny.astimezone(local_zone).strftime("%Y-%m-%d %H:%M:%S"),
        "local_timezone": local_zone_name,
        "new_york_time": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "new_york_date": now_ny.date().isoformat(),
        "market_status": status,
        "is_open": is_open,
        "countdown_label": countdown_label,
        "countdown": _format_duration(target - now_ny if target else None),
        "next_regular_open": target.strftime("%Y-%m-%d %H:%M:%S %Z") if target and not is_open else None,
        "today_regular_open": None,
        "today_regular_close": None,
        "calendar_source": "工作日/时段近似规则（未加载交易日历）",
        "calendar_is_authoritative": False,
        "calendar_note": "此模式不会识别美股节假日或提前收盘。安装 pandas-market-calendars 后可启用交易日历判断。",
    }


def get_market_clock(now: datetime | None = None) -> dict[str, Any]:
    """Get Hong Kong/local time, New York time and regular-session status.

    Args:
        now: Optional test timestamp. Naive timestamps are interpreted as New York time.
    """
    if now is None:
        now_ny = datetime.now(NEW_YORK_TZ)
    elif now.tzinfo is None:
        now_ny = now.replace(tzinfo=NEW_YORK_TZ)
    else:
        now_ny = now.astimezone(NEW_YORK_TZ)

    calendar_result = _schedule_with_calendar(now_ny)
    if calendar_result is not None:
        return calendar_result
    return _fallback_weekday_estimate(now_ny)


# Backwards-compatible function name used in earlier dashboard drafts.
def get_market_time() -> dict[str, Any]:
    return get_market_clock()
