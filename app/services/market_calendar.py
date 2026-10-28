"""NASDAQ session calendar helpers for QQQM forecasting.

Uses pandas-market-calendars when installed. Forecast date calculation can fall
back to weekdays for interactive research, but scheduled automation requires the
exchange calendar and refuses to create forecasts on an uncertain session date.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any


def _load_calendar() -> Any:
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise ImportError(
            "缺少 pandas-market-calendars。请运行："
            "uv pip install --python .\\.venv\\Scripts\\python.exe pandas-market-calendars"
        ) from exc
    return mcal.get_calendar("NASDAQ")


def market_calendar_available() -> bool:
    try:
        _load_calendar()
        return True
    except ImportError:
        return False


def market_calendar_label() -> str:
    return "NASDAQ 官方交易日历（pandas-market-calendars）" if market_calendar_available() else "仅按周末估算（节假日可能不准确）"


def is_nasdaq_session(day: date, calendar: Any | None = None) -> bool:
    """Return whether `day` is a NASDAQ session; requires the calendar package by default."""
    cal = calendar if calendar is not None else _load_calendar()
    schedule = cal.schedule(start_date=day.isoformat(), end_date=day.isoformat())
    return not bool(schedule.empty)


def next_nasdaq_session(after_date: date, calendar: Any | None = None) -> date:
    """Return the next scheduled NASDAQ session after `after_date`.

    If the calendar dependency is not installed, a weekday-only fallback is used
    for interactive forecasts and must be disclosed in the forecast caveats.
    """
    try:
        cal = calendar if calendar is not None else _load_calendar()
    except ImportError:
        candidate = after_date + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

    start = (after_date + timedelta(days=1)).isoformat()
    end = (after_date + timedelta(days=31)).isoformat()
    schedule = cal.schedule(start_date=start, end_date=end)
    if bool(schedule.empty):
        raise RuntimeError(f"未来 31 天内找不到 NASDAQ 交易日：{after_date.isoformat()} 之后。")
    first_session = schedule.index[0]
    # pandas-market-calendars returns a pandas.Timestamp in the schedule index.
    # Timestamp is also an instance of datetime/date, so check datetime first;
    # otherwise isoformat() becomes "YYYY-MM-DDT00:00:00" and date.fromisoformat()
    # fails later in the forecast pipeline.
    if isinstance(first_session, datetime):
        return first_session.date()
    if isinstance(first_session, date):
        return first_session
    # Compatibility with timestamp-like index objects without importing pandas.
    if hasattr(first_session, "date"):
        result = first_session.date()
        if isinstance(result, datetime):
            return result.date()
        if isinstance(result, date):
            return result
    return date.fromisoformat(str(first_session)[:10])
