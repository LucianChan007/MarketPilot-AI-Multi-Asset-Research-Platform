from __future__ import annotations

"""Read-only health checks for local QQQM data and workflows.

This module never contacts providers and never exposes credential values.
It distinguishes cached data dates from successful refresh timestamps.
"""

import json
import os
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
DISPLAY_TZ = ZoneInfo("Asia/Hong_Kong")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_env_values(path: Path) -> dict[str, str]:
    """Parse simple .env entries for status checks only. Never return these to the UI."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            values[key.strip()] = value
    return values


def credential_status(root: Path) -> dict[str, Any]:
    values = read_env_values(root / ".env")

    def configured(*names: str) -> bool:
        for name in names:
            value = os.environ.get(name) or values.get(name) or ""
            value = value.strip()
            if value and not value.lower().startswith(("replace_with_", "your_", "填入", "你的")):
                return True
        return False

    alpaca_key = configured("ALPACA_API_KEY_ID")
    alpaca_secret = configured("ALPACA_API_SECRET_KEY")
    return {
        "alpaca_configured": alpaca_key and alpaca_secret,
        "alpaca_key_present": alpaca_key,
        "alpaca_secret_present": alpaca_secret,
        "eodhd_configured": configured("EODHD_API_KEY", "EODHD_API_TOKEN", "EODHD_TOKEN", "EODHD_APIKEY"),
        "deepseek_configured": configured("DEEPSEEK_API_KEY"),
        "env_file_exists": (root / ".env").exists(),
    }


def expected_last_completed_session(now_ny: datetime | None = None) -> str | None:
    """Return the latest NASDAQ session whose regular close is in the past."""
    now = now_ny or datetime.now(NEW_YORK)
    if now.tzinfo is None:
        now = now.replace(tzinfo=NEW_YORK)
    else:
        now = now.astimezone(NEW_YORK)
    try:
        import pandas_market_calendars as mcal

        start = (now.date() - timedelta(days=14)).isoformat()
        schedule = mcal.get_calendar("NASDAQ").schedule(start_date=start, end_date=now.date().isoformat())
        completed: list[str] = []
        for session_day, row in schedule.iterrows():
            close = row["market_close"]
            if close.tzinfo is None:
                close = close.tz_localize("UTC")
            close_ny = close.to_pydatetime().astimezone(NEW_YORK)
            if close_ny <= now:
                completed.append(str(session_day.date()))
        if completed:
            return completed[-1]
        # If called before the current day's open, the calendar range should still
        # include prior sessions, but extend it for unusual holiday stretches.
        schedule = mcal.get_calendar("NASDAQ").schedule(
            start_date=(now.date() - timedelta(days=45)).isoformat(),
            end_date=now.date().isoformat(),
        )
        for session_day, row in schedule.iloc[::-1].iterrows():
            close = row["market_close"]
            if close.tzinfo is None:
                close = close.tz_localize("UTC")
            if close.to_pydatetime().astimezone(NEW_YORK) <= now:
                return str(session_day.date())
        return None
    except Exception:
        # Conservative weekday-only fallback. The UI labels it as an estimate.
        candidate = now.date()
        if now.time().replace(tzinfo=None) < datetime.strptime("16:15", "%H:%M").time():
            candidate -= timedelta(days=1)
        for _ in range(10):
            if candidate.weekday() < 5:
                return candidate.isoformat()
            candidate -= timedelta(days=1)
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _date_lag(latest: str | None, expected: str | None) -> int | None:
    if not latest or not expected or latest >= expected:
        return 0 if latest and expected and latest >= expected else None
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("NASDAQ")
        schedule = cal.schedule(start_date=latest, end_date=expected)
        # The latest date itself is already represented in latest; count newer sessions.
        return max(0, len([idx for idx in schedule.index if str(idx.date()) > latest]))
    except Exception:
        try:
            days = (date.fromisoformat(expected) - date.fromisoformat(latest)).days
            return max(1, sum(1 for step in range(1, days + 1) if (date.fromisoformat(latest) + timedelta(days=step)).weekday() < 5))
        except ValueError:
            return None


def inspect_project(root: Path, now_ny: datetime | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    db_path = root / "data" / "market_data.db"
    expected_day = expected_last_completed_session(now_ny)
    result: dict[str, Any] = {
        "checked_at_local": datetime.now(DISPLAY_TZ).isoformat(timespec="seconds"),
        "expected_last_completed_session": expected_day,
        "calendar_source": "NASDAQ 官方交易日历" if _calendar_available() else "工作日估算（假期可能有误差）",
        "credentials": credential_status(root),
        "database_exists": db_path.exists(),
        "daily": {}, "intraday": {}, "news": {}, "macro": {},
        "video": {}, "forecast": {}, "model_evaluation": {},
        "intraday_cycle": _read_json(root / "reports" / "intraday_learning_cycle" / "latest_status.json"),
        "daily_workflow": _read_json(root / "reports" / "unified_daily_workflow" / "latest_status.json"),
    }
    if db_path.exists():
        try:
            with closing(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=4)) as conn:
                conn.row_factory = sqlite3.Row
                if _table_exists(conn, "market_daily"):
                    row = conn.execute(
                        "SELECT trade_date, close, source, fetched_at FROM market_daily WHERE UPPER(ticker)='QQQM' ORDER BY trade_date DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        latest = str(row["trade_date"])[:10]
                        result["daily"] = {
                            "latest_date": latest, "latest_close": row["close"],
                            "source": row["source"], "fetched_at": row["fetched_at"],
                            "lag_sessions": _date_lag(latest, expected_day),
                        }
                if _table_exists(conn, "intraday_bars"):
                    rows = conn.execute(
                        """SELECT trade_date_ny, feed, COUNT(DISTINCT session_time_ny) AS slots,
                                  MIN(session_time_ny) AS first_slot, MAX(session_time_ny) AS last_slot,
                                  MAX(fetched_at_utc) AS fetched_at
                           FROM intraday_bars WHERE UPPER(symbol)='QQQM' AND LOWER(timeframe)='15min'
                           AND session_time_ny >= '09:30' AND session_time_ny < '16:00'
                           GROUP BY trade_date_ny, feed ORDER BY trade_date_ny DESC"""
                    ).fetchall()
                    all_latest = conn.execute(
                        "SELECT MAX(trade_date_ny) FROM intraday_bars WHERE UPPER(symbol)='QQQM' AND LOWER(timeframe)='15min'"
                    ).fetchone()[0]
                    complete = next((r for r in rows if int(r["slots"] or 0) >= 24 and str(r["first_slot"]) == "09:30" and str(r["last_slot"]) >= "15:45"), None)
                    if complete:
                        latest = str(complete["trade_date_ny"])[:10]
                        result["intraday"] = {
                            "latest_complete_date": latest, "latest_any_date": str(all_latest)[:10] if all_latest else None,
                            "feed": complete["feed"], "slots": int(complete["slots"] or 0),
                            "first_slot": complete["first_slot"], "last_slot": complete["last_slot"],
                            "fetched_at": complete["fetched_at"], "lag_sessions": _date_lag(latest, expected_day),
                        }
                    else:
                        result["intraday"] = {"latest_complete_date": None, "latest_any_date": str(all_latest)[:10] if all_latest else None}
                if _table_exists(conn, "news_items"):
                    row = conn.execute(
                        "SELECT MAX(substr(published_at,1,10)) AS latest_published_date, MAX(fetched_at) AS latest_fetched_at, COUNT(*) AS count FROM news_items"
                    ).fetchone()
                    result["news"] = dict(row) if row else {}
                if _table_exists(conn, "feed_sync_state"):
                    sync_rows = conn.execute(
                        "SELECT source_id, source_name, last_attempt_at, last_success_at, last_item_count, last_error FROM feed_sync_state ORDER BY source_id"
                    ).fetchall()
                    result["news"]["source_sync"] = [dict(r) for r in sync_rows]
                if _table_exists(conn, "macro_observations"):
                    row = conn.execute("SELECT MAX(observation_date) AS latest_observation_date, MAX(retrieved_at) AS latest_retrieved_at, COUNT(*) AS count FROM macro_observations").fetchone()
                    result["macro"] = dict(row) if row else {}
                if _table_exists(conn, "forecast_feedback"):
                    row = conn.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN status='evaluated' THEN 1 ELSE 0 END) AS evaluated, SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending FROM forecast_feedback WHERE UPPER(ticker)='QQQM'").fetchone()
                    result["forecast"]["feedback"] = dict(row) if row else {}
        except (sqlite3.Error, OSError) as exc:
            result["database_error"] = f"{type(exc).__name__}: {exc}"
    video_dirs = sorted((root / "reports" / "video_scripts").glob("*/"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if video_dirs:
        newest = video_dirs[0]
        archive = newest / "每日行情视频脚本与两张配图.zip"
        text_file = newest / "每日视频口播稿_可直接照念.txt"
        result["video"] = {
            "folder": newest.name,
            "generated_at": datetime.fromtimestamp((archive if archive.exists() else newest).stat().st_mtime, DISPLAY_TZ).isoformat(timespec="seconds"),
            "has_archive": archive.exists(), "has_voiceover": text_file.exists(),
        }
    for key, filename in (
        ("video_forecast", "QQQM_video_forecast_latest.json"),
        ("intraday_forecast", "QQQM_intraday_forecast_latest.json"),
    ):
        data = _read_json(root / "reports" / "intraday" / filename)
        if data:
            result["forecast"][key] = {
                "as_of_date": data.get("as_of_date"), "target_date": data.get("forecast_date_weekday_estimate"),
                "available": data.get("available", True), "generated_at": data.get("generated_at_utc") or data.get("generated_at_local"),
                "model": (data.get("main_path") or {}).get("model_name") or data.get("curve_method"),
            }
    scorecard = _read_json(root / "reports" / "model_lab" / "QQQM_model_evaluation_scorecard_latest.json")
    if scorecard:
        live = scorecard.get("live_shadow") or {}
        history = scorecard.get("history") or {}
        result["model_evaluation"] = {
            "created_at": scorecard.get("created_at_local"),
            "data_latest_date": history.get("combined_latest_date"),
            "rolling_test_samples": ((scorecard.get("walk_forward") or {}).get("metadata") or {}).get("out_of_sample_samples"),
            "live_evaluated_samples": live.get("evaluated_samples", 0),
            "live_pending_samples": live.get("pending_samples", 0),
            "gate_status": ((scorecard.get("overall_gate") or {}).get("status")),
            "gate_reason": ((scorecard.get("live_shadow") or {}).get("reason")),
        }
    return result


def _calendar_available() -> bool:
    try:
        import pandas_market_calendars  # noqa: F401
        return True
    except ImportError:
        return False


def format_time(value: Any) -> str:
    if not value:
        return "暂无记录"
    try:
        raw = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M HKT")
    except (ValueError, TypeError):
        return str(value)


def status_label(latest_date: str | None, expected_date: str | None) -> tuple[str, str]:
    if not latest_date:
        return "缺数据", "本地没有可用记录"
    if not expected_date:
        return "日期待核对", f"本地最新记录 {latest_date}，无法确认应有的最近交易日"
    if latest_date >= expected_date:
        return "已到最近交易日", f"本地最新记录 {latest_date}"
    return "有滞后", f"本地最新记录 {latest_date}；最近完整交易日应为 {expected_date}"
