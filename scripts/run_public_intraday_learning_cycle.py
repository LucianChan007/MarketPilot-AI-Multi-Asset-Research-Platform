from __future__ import annotations
"""Public-data intraday learning cycle (no Alpaca credentials required).

Refreshes a single public 15-minute data feed, settles eligible previous archived
forecasts against the same feed, trains/evaluates a walk-forward intraday candidate,
and writes a pre-session forecast archive only when data and target dates align.
"""
import json
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.intraday_forecast import choose_intraday_feed, run_intraday_prototype  # noqa: E402
from app.services.system_health import expected_last_completed_session  # noqa: E402
from scripts.generate_video_script import _next_target_day  # noqa: E402
from scripts.build_intraday_learning_dataset import build_dataset  # noqa: E402
from scripts.video_market_data import DB_PATH, refresh_market_data  # noqa: E402

NY = ZoneInfo("America/New_York")
REPORT = ROOT / "reports" / "public_intraday_learning_cycle"
ARCHIVE = ROOT / "reports" / "intraday" / "archive"
MODEL_REPORT = ROOT / "reports" / "model_lab"


def _latest_daily_date() -> str | None:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
        return str(row[0])[:10] if row and row[0] else None
    except sqlite3.Error:
        return None


def _run() -> dict:
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    events: list[dict] = []
    refresh = refresh_market_data(DB_PATH)
    events.append({
        "stage": "刷新公开日线与十五分钟分时",
        "daily_latest_date": refresh.get("daily_latest_date"),
        "intraday_latest_date": refresh.get("intraday_latest_date"),
        "daily_provider": refresh.get("daily_provider"),
        "intraday_provider": refresh.get("intraday_provider"),
        "errors": refresh.get("errors", []),
        "attempt_notes": refresh.get("attempt_notes", []),
    })
    try:
        from app.news_data import sync_news
        news_status = sync_news()
        events.append({"stage": "刷新官方新闻", "success": not bool(news_status.get("failed")), "summary": {k: v for k, v in news_status.items() if k not in {"items", "raw", "content"}}})
    except Exception as exc:
        events.append({"stage": "刷新官方新闻", "success": False, "error": f"{type(exc).__name__}: {exc}"})

    now_ny = datetime.now(NY)
    expected = expected_last_completed_session(now_ny)
    latest_daily = _latest_daily_date()
    if not expected:
        raise RuntimeError("无法通过 NASDAQ 交易日历确认最近完整交易日；为防止错标目标日期，已停止生成。")
    if not latest_daily or latest_daily < expected:
        raise RuntimeError(f"公开日线截至 {latest_daily or '缺失'}，最近完整交易日应为 {expected}。请检查数据源后重试。")

    feed, sessions = choose_intraday_feed(
        db_path=DB_PATH, symbol="QQQM", timeframe="15Min",
        expected_latest_date=expected, min_training_sessions=25,
    )
    eligible = [s for s in sessions if s.get("path_eligible")]
    latest_intraday = max((str(s["trade_date"]) for s in eligible), default=None)
    if latest_intraday != expected:
        raise RuntimeError(f"所选分时源 {feed} 最新完整交易日为 {latest_intraday}，预期是 {expected}。不会用旧曲线预测下一交易日。")
    if len(eligible) < 25:
        raise RuntimeError(f"完整分时样本只有 {len(eligible)} 个交易日，少于最初滚动训练要求 25 天。")

    # Score all earlier immutable forecasts first, with actuals from the same feed.
    before = build_dataset(DB_PATH, ARCHIVE, MODEL_REPORT)
    requested_target = _next_target_day(expected, now_ny)
    model_result = run_intraday_prototype(
        db_path=DB_PATH, report_dir=ROOT / "reports" / "intraday",
        symbol="QQQM", feed=feed, timeframe="15Min", min_training_sessions=20,
        forecast_target_date=requested_target,
    )
    forecast = model_result["forecast"]
    main_path = forecast.get("main_path") or {}
    model_family = str(main_path.get("model_family") or "")
    target = str(forecast.get("forecast_date_weekday_estimate") or "")
    as_of = str(forecast.get("as_of_date") or "")
    if as_of != expected:
        raise RuntimeError(f"模型预测基准日 {as_of or '缺失'} 与最近完整交易日 {expected} 不一致。")
    if not target or target <= as_of:
        raise RuntimeError(f"模型预测目标日 {target or '缺失'} 不是晚于基准日 {as_of} 的未来交易日。")
    if target != requested_target:
        raise RuntimeError(f"模型目标日 {target} 与按美东时段确定的下一目标日 {requested_target} 不一致。")
    if "ridge" not in model_family.lower() or not isinstance(main_path.get("price_by_time"), dict) or len(main_path.get("price_by_time", {})) < 18:
        raise RuntimeError("历史样本暂未生成合格的岭回归盘中主路径；本次不会把统计模板标记成机器学习预测。")
    if forecast.get("capture_mode") != "live_prediction" or not forecast.get("pre_session_snapshot"):
        raise RuntimeError("当前运行时目标交易日已经开盘，不能将本次结果记为事前预测。请在下一交易日美东开盘前运行，或收盘后更新完整数据再生成下一交易日预测。")

    after = build_dataset(DB_PATH, ARCHIVE, MODEL_REPORT)
    status = {
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_success": True,
        "mode": "public_feed_intraday_ml",
        "latest_completed_market_date": expected,
        "latest_daily_date": latest_daily,
        "latest_complete_intraday_date": latest_intraday,
        "feed": feed,
        "completed_training_sessions": len(eligible),
        "forecast_as_of_date": as_of,
        "forecast_target_date": target,
        "model_family": model_family,
        "model_training_performed": True,
        "model_promotion_performed": False,
        "forecast_json": model_result.get("forecast_json"),
        "forecast_archive_json": model_result.get("archive_json"),
        "walk_forward_backtest": model_result.get("backtest_summary"),
        "learning_dataset_before": before,
        "learning_dataset_after": after,
        "refresh_errors": refresh.get("errors", []),
        "attempt_notes": refresh.get("attempt_notes", []),
        "news_refresh_failed": any(e.get("stage") == "刷新官方新闻" and not e.get("success", True) for e in events),
        "events": events,
        "note": "本流程使用公开分时源对历史交易日进行滚动训练与复盘，不需要 Alpaca。每个预测只与相同 feed 的实际行情结算。模型不会因单次运行自动晋级。",
    }
    return status


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        status = _run()
        rc = 0
    except Exception as exc:
        status = {
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "overall_success": False,
            "mode": "public_feed_intraday_ml",
            "model_training_performed": False,
            "model_promotion_performed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc()[-5000:],
            "note": "失败时不会把过期/统计模板曲线标记成新的机器学习预测。",
        }
        rc = 1
    (REPORT / "latest_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
