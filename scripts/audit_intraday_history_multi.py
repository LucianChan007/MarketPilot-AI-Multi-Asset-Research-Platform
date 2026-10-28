from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market_data.db"
REPORT_DIR = ROOT / "reports" / "intraday"
NY = ZoneInfo("America/New_York")

TIMEFRAME_CONFIG = {
    "1Min": {"minutes": 1},
    "5Min": {"minutes": 5},
    "15Min": {"minutes": 15},
    "30Min": {"minutes": 30},
    "1Hour": {"minutes": 60},
}


def audit_timeframe(conn: sqlite3.Connection, symbol: str, feed: str, timeframe: str) -> dict:
    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"不支持的 timeframe: {timeframe}。可选：{', '.join(TIMEFRAME_CONFIG)}")

    minutes = TIMEFRAME_CONFIG[timeframe]["minutes"]
    expected_bars = 390 // minutes
    last_expected_start_minutes = 16 * 60 - minutes
    last_expected_time = f"{last_expected_start_minutes // 60:02d}:{last_expected_start_minutes % 60:02d}"

    rows = conn.execute(
        """SELECT trade_date_ny, session_time_ny, timestamp_utc, open, high, low, close, volume
           FROM intraday_bars
           WHERE UPPER(symbol)=UPPER(?) AND feed=? AND timeframe=?
             AND session_time_ny >= '09:30' AND session_time_ny < '16:00'
           ORDER BY trade_date_ny, session_time_ny""",
        (symbol, feed, timeframe),
    ).fetchall()

    by_day: dict[str, list[sqlite3.Row | tuple]] = defaultdict(list)
    for row in rows:
        by_day[row[0]].append(row)

    today_ny = datetime.now(NY).date().isoformat()
    daily: list[dict] = []
    for trade_date, bars in sorted(by_day.items()):
        times = sorted({bar[1] for bar in bars})
        last_time = times[-1]
        # Heuristic: an older session with the last expected bucket is treated as completed.
        # Holiday early closes can require explicit exchange-calendar handling.
        is_complete = trade_date < today_ny and last_time >= last_expected_time
        coverage = 100.0 * len(times) / expected_bars if expected_bars else 0.0
        daily.append({
            "trade_date_ny": trade_date,
            "bars_in_regular_session": len(times),
            "expected_bars": expected_bars,
            "coverage_pct": round(coverage, 2),
            "first_bar_ny": times[0],
            "last_bar_ny": last_time,
            "completed_session_heuristic": is_complete,
            "feed": feed,
            "timeframe": timeframe,
        })

    completed = [day for day in daily if day["completed_session_heuristic"]]
    completed_coverages = [day["coverage_pct"] for day in completed]
    completed_counts = [day["bars_in_regular_session"] for day in completed]
    median_coverage = median(completed_coverages) if completed_coverages else None

    if len(completed) >= 30 and median_coverage is not None and median_coverage >= 80:
        readiness = "enough_for_initial_prototype_backtest"
        message = "历史覆盖达到初步原型回测门槛；仍需样本外验证，不代表预测可靠。"
    elif len(completed) >= 20 and median_coverage is not None and median_coverage >= 45:
        readiness = "prototype_only_sparse_iex"
        message = "存在一定历史覆盖，但缺失时间段较多；只能用于谨慎的原型实验，不应展示为完整全市场轨迹。"
    else:
        readiness = "insufficient_intraday_coverage"
        message = "完整交易日数量或数据覆盖不足；不要训练或发布貌似精确的盘中曲线。"

    return {
        "symbol": symbol.upper(),
        "feed": feed,
        "timeframe": timeframe,
        "database_path": str(DB_PATH.resolve()),
        "current_new_york_date": today_ny,
        "expected_regular_session_bars": expected_bars,
        "last_expected_bar_start_ny": last_expected_time,
        "bars_in_regular_session_total": sum(day["bars_in_regular_session"] for day in daily),
        "dates_with_regular_session_bars": len(daily),
        "completed_sessions_by_heuristic": len(completed),
        "median_bars_per_completed_session": median(completed_counts) if completed_counts else None,
        "median_coverage_pct_completed_sessions": round(median_coverage, 2) if median_coverage is not None else None,
        "mean_coverage_pct_completed_sessions": round(mean(completed_coverages), 2) if completed_coverages else None,
        "readiness": readiness,
        "readiness_message": message,
        "important_caveat": (
            "Alpaca IEX 是单一交易所 feed，不覆盖全美市场；不存在的 bar 不会被补造。"
            "完整交易日按最后 bar 起始时间做启发式判断；早收市假日可能被误判，正式回测前应使用交易所日历。"
        ),
        "daily_coverage": daily,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="审计 Alpaca 盘中行情覆盖率，支持多种时间粒度。")
    parser.add_argument("--symbol", default="QQQM")
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--timeframes", nargs="+", default=["5Min", "15Min"], choices=list(TIMEFRAME_CONFIG))
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"找不到数据库：{db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='intraday_bars'"
        ).fetchone()
        if not exists:
            print("尚未找到 intraday_bars 表。请先运行 scripts/sync_intraday_history.py。")
            return 2
        conn.row_factory = sqlite3.Row
        results = []
        for timeframe in args.timeframes:
            result = audit_timeframe(conn, args.symbol, args.feed, timeframe)
            results.append(result)
    finally:
        conn.close()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for result in results:
        timeframe = result["timeframe"]
        json_path = REPORT_DIR / f"{args.symbol.upper()}_{timeframe}_audit_latest.json"
        csv_path = REPORT_DIR / f"{args.symbol.upper()}_{timeframe}_daily_coverage_latest.csv"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        fields = [
            "trade_date_ny", "bars_in_regular_session", "expected_bars", "coverage_pct",
            "first_bar_ny", "last_bar_ny", "completed_session_heuristic", "feed", "timeframe",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["daily_coverage"])

        concise = {key: value for key, value in result.items() if key != "daily_coverage"}
        print(f"\n========== {args.symbol.upper()} {timeframe} 覆盖率审计 ==========")
        print(json.dumps(concise, ensure_ascii=False, indent=2))
        print("最近 10 个交易日：")
        for day in result["daily_coverage"][-10:]:
            print(
                f"{day['trade_date_ny']} {day['bars_in_regular_session']}/{day['expected_bars']} "
                f"({day['coverage_pct']}%) last={day['last_bar_ny']} "
                f"complete={day['completed_session_heuristic']}"
            )
        print(f"JSON：{json_path}")
        print(f"CSV：{csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
