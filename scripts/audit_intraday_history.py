from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market_data.db"
REPORT_DIR = ROOT / "reports" / "intraday"
NY = ZoneInfo("America/New_York")
EXPECTED_REGULAR_SESSION_BARS = 78  # 09:30–16:00 ET, five-minute buckets


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    if not DB_PATH.exists():
        print(f"找不到数据库：{DB_PATH}")
        return 1
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='intraday_bars'"
        ).fetchone()
        if not exists:
            print("尚未找到 intraday_bars 表。请先运行 scripts/sync_intraday_history.py。")
            return 2
        rows = conn.execute(
            """SELECT symbol, feed, timeframe, trade_date_ny, session_time_ny,
                      timestamp_utc, open, high, low, close, volume
               FROM intraday_bars
               WHERE symbol='QQQM' AND feed='iex' AND timeframe='5Min'
                 AND session_time_ny >= '09:30' AND session_time_ny < '16:00'
               ORDER BY trade_date_ny, session_time_ny"""
        ).fetchall()
    finally:
        conn.close()

    by_day: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_day[row["trade_date_ny"]].append(row)

    today_ny = datetime.now(NY).date().isoformat()
    daily: list[dict] = []
    for trade_date, bars in sorted(by_day.items()):
        # A session is considered completed only when it is not today's NY date
        # and the last observed 5-minute bucket starts at 15:55 ET.
        last_time = max(b["session_time_ny"] for b in bars)
        is_complete = trade_date < today_ny and last_time >= "15:55"
        unique_times = len({b["session_time_ny"] for b in bars})
        daily.append({
            "trade_date_ny": trade_date,
            "bars_in_regular_session": unique_times,
            "expected_5min_bars": EXPECTED_REGULAR_SESSION_BARS,
            "coverage_pct": round(100 * unique_times / EXPECTED_REGULAR_SESSION_BARS, 2),
            "first_bar_ny": min(b["session_time_ny"] for b in bars),
            "last_bar_ny": last_time,
            "completed_session_heuristic": is_complete,
            "feed": "iex",
        })

    complete = [d for d in daily if d["completed_session_heuristic"]]
    coverages = [d["coverage_pct"] for d in complete]
    counts = [d["bars_in_regular_session"] for d in complete]
    total_bar_count = sum(d["bars_in_regular_session"] for d in daily)

    # Fewer than 20 complete sessions or low median bar coverage is not enough
    # evidence for a credible curve model; use data only for prototype/audit.
    if len(complete) >= 30 and median(coverages) >= 65:
        readiness = "enough_for_initial_prototype_backtest"
        message = "可开展初步的 15 分钟路径基线回测；仍需严格样本外评估，不代表预测可靠。"
    elif len(complete) >= 20 and median(coverages) >= 45:
        readiness = "prototype_only_sparse_iex"
        message = "有一定历史覆盖，但 IEX 缺失时间段较多；先测试 15 分钟粒度，不应声称完整全市场轨迹。"
    else:
        readiness = "insufficient_intraday_coverage"
        message = "完整交易日数量或 IEX 覆盖率不足；先扩大历史范围或考虑更完整的数据源，不要训练/发布貌似精确的盘中曲线。"

    result = {
        "symbol": "QQQM",
        "feed": "iex",
        "timeframe": "5Min",
        "database_path": str(DB_PATH.resolve()),
        "current_new_york_date": today_ny,
        "expected_regular_session_bars": EXPECTED_REGULAR_SESSION_BARS,
        "bars_in_regular_session_total": total_bar_count,
        "dates_with_regular_session_bars": len(daily),
        "completed_sessions_by_heuristic": len(complete),
        "median_bars_per_completed_session": median(counts) if counts else None,
        "median_coverage_pct_completed_sessions": median(coverages) if coverages else None,
        "mean_coverage_pct_completed_sessions": round(mean(coverages), 2) if coverages else None,
        "readiness": readiness,
        "readiness_message": message,
        "important_caveat": "IEX 是单一交易所数据，并不覆盖全美市场；缺失 bar 不能伪造或直接当作无价格变化。完整交易日用末根 bar 时间的启发式判断，早收市假日可能被判为非完整。",
        "daily_coverage": daily,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "QQQM_intraday_data_audit_latest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import csv
    with (REPORT_DIR / "QQQM_intraday_daily_coverage_latest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["trade_date_ny", "bars_in_regular_session", "expected_5min_bars", "coverage_pct", "first_bar_ny", "last_bar_ny", "completed_session_heuristic", "feed"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(daily)

    print("========== QQQM 盘中历史数据质量审计 ==========")
    print(json.dumps({k: v for k, v in result.items() if k != "daily_coverage"}, ensure_ascii=False, indent=2))
    print("最近 10 个日期的覆盖情况：")
    for row in daily[-10:]:
        print(f"{row['trade_date_ny']} bars={row['bars_in_regular_session']}/78 coverage={row['coverage_pct']}% last={row['last_bar_ny']} complete={row['completed_session_heuristic']}")
    print(f"JSON 报告：{REPORT_DIR / 'QQQM_intraday_data_audit_latest.json'}")
    print(f"CSV 报告：{REPORT_DIR / 'QQQM_intraday_daily_coverage_latest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
