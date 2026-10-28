from __future__ import annotations

"""Run sync -> settle old forecasts -> create a new forecast when safe."""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "intraday_learning_cycle"
DB_PATH = ROOT / "data" / "market_data.db"
FORECAST_PATH = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
NEW_YORK = ZoneInfo("America/New_York")

from app.services.intraday_cycle_guard import evaluate_catchup_gate  # noqa: E402
from app.services.market_calendar import is_nasdaq_session, next_nasdaq_session  # noqa: E402


def _run_stage(name: str, args: list[str], timeout_seconds: int = 900) -> dict[str, Any]:
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(args, cwd=str(ROOT), env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds, check=False)
        return {
            "stage": name, "command": args, "started_at": started,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "success": result.returncode == 0, "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - t0, 2),
            "stdout_tail": (result.stdout or "")[-5000:], "stderr_tail": (result.stderr or "")[-3000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "stage": name, "command": args, "started_at": started,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "success": False, "returncode": None, "duration_seconds": round(time.monotonic() - t0, 2),
            "error": f"运行超过 {timeout_seconds} 秒，已停止。",
            "stdout_tail": str(exc.stdout or "")[-3000:], "stderr_tail": str(exc.stderr or "")[-2000:],
        }
    except OSError as exc:
        return {
            "stage": name, "command": args, "started_at": started,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "success": False, "returncode": None, "duration_seconds": round(time.monotonic() - t0, 2),
            "error": f"无法启动子进程：{type(exc).__name__}: {exc}",
        }


def _latest_intraday_date() -> str | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='intraday_bars'"
        ).fetchone()
        if not has_table:
            return None
        row = conn.execute(
            """SELECT MAX(trade_date_ny) FROM intraday_bars
               WHERE UPPER(symbol)='QQQM' AND LOWER(feed)='iex' AND LOWER(timeframe)='15min'"""
        ).fetchone()
        return str(row[0])[:10] if row and row[0] else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _latest_forecast_as_of_date() -> str | None:
    try:
        data = json.loads(FORECAST_PATH.read_text(encoding="utf-8"))
        value = data.get("as_of_date")
        return str(value)[:10] if value else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _write_status(status: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "latest_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    history_dir = REPORT_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    (history_dir / f"QQQM_intraday_learning_cycle_{stamp}.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_status(stages: list[dict[str, Any]], run_mode: str) -> dict[str, Any]:
    failed = [record["stage"] for record in stages if not record.get("success")]
    return {
        "started_at": stages[0]["started_at"] if stages else datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_success": not failed, "skipped": False, "skip_reason": None,
        "failed_stages": failed, "stages": stages,
        "run_mode": run_mode,
        "purpose": "存档前瞻预测、收盘后结算实际路径，建立结构化学习数据集。",
        "model_training_performed": False, "model_promotion_performed": False,
    }


def _guard_after_sync() -> tuple[bool, str | None]:
    now_ny = datetime.now(timezone.utc).astimezone(NEW_YORK)
    latest_date = _latest_intraday_date()
    latest_as_of = _latest_forecast_as_of_date()
    if not latest_date:
        return False, "行情表中没有找到 QQQM IEX 15 分钟数据；跳过补跑。"
    try:
        data_day = date.fromisoformat(latest_date[:10])
        current_session = is_nasdaq_session(now_ny.date())
        data_session = is_nasdaq_session(data_day)
        next_session = next_nasdaq_session(data_day)
    except ImportError as exc:
        return False, f"缺少交易日历依赖，拒绝根据不完整日历生成预测：{exc}"
    except (TypeError, ValueError, RuntimeError) as exc:
        return False, f"无法确认交易日历或行情日期，安全跳过：{type(exc).__name__}: {exc}"
    return evaluate_catchup_gate(
        now_ny=now_ny,
        current_day_is_session=current_session,
        latest_data_date=latest_date,
        latest_forecast_as_of_date=latest_as_of,
        latest_data_day_is_session=data_session,
        next_session_date=next_session,
    )


def _cycle_lock():
    """Acquire a portable exclusive lock file; return file path or None if active."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock = REPORT_DIR / "cycle.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
            if age > 3 * 60 * 60:
                lock.unlink(missing_ok=True)
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                return None
        except (OSError, FileExistsError):
            return None
    os.write(fd, f"pid={os.getpid()} started={datetime.now().astimezone().isoformat()}".encode("utf-8"))
    os.close(fd)
    return lock


def _main_impl(run_mode: str) -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    stages: list[dict[str, Any]] = []
    commands = [
        ("同步最新 15 分钟行情", [python, "-m", "scripts.sync_intraday_history", "--symbol", "QQQM", "--timeframe", "15Min", "--feed", "iex", "--lookback-days", "90"], 1200),
        ("结算已有预测并导出学习数据集", [python, "-m", "scripts.build_intraday_learning_dataset"], 300),
        ("归档现有预测并检查实际走势", [python, "-m", "scripts.plot_intraday_forecast"], 300),
        ("生成下一交易日预测及候选完整路径", [python, "-m", "scripts.run_intraday_path_backtest"], 900),
        ("重画归档预测并叠加已完成交易日实际走势", [python, "-m", "scripts.plot_intraday_forecast", "--all-archives"], 600),
        ("刷新预测误差与机器学习候选数据集", [python, "-m", "scripts.build_intraday_learning_dataset"], 300),
    ]
    print("========== QQQM 盘中预测与复盘流程 ==========")
    print(f"运行方式：{run_mode}")
    for index, (name, command, timeout) in enumerate(commands):
        print(f"\n--- {name} ---")
        record = _run_stage(name, command, timeout)
        stages.append(record)
        print(f"状态：{'成功' if record['success'] else '失败'}；耗时 {record.get('duration_seconds')} 秒")
        if record.get("stdout_tail"):
            print(record["stdout_tail"][-2400:])
        if record.get("stderr_tail"):
            print("stderr:", record["stderr_tail"][-1400:])
        if not record["success"] and name in {"同步最新 15 分钟行情", "生成下一交易日预测及候选完整路径"}:
            break
        if index == 0:
            proceed, reason = _guard_after_sync()
            print(f"\n日期与重复运行检查：{reason}")
            if not proceed:
                status = _base_status(stages, run_mode)
                missing_calendar = bool(reason and reason.startswith("缺少交易日历依赖"))
                status["overall_success"] = not missing_calendar and all(s.get("success") for s in stages)
                status["failed_stages"] = ["交易日历资格检查"] if missing_calendar else []
                status["skipped"] = not missing_calendar
                status["skip_reason"] = reason
                status["latest_intraday_data_date"] = _latest_intraday_date()
                status["latest_forecast_as_of_date"] = _latest_forecast_as_of_date()
                status["new_york_time"] = datetime.now(timezone.utc).astimezone(NEW_YORK).isoformat(timespec="seconds")
                _write_status(status)
                print("\n========== 流程状态 ==========")
                print(json.dumps({k: v for k, v in status.items() if k != "stages"}, ensure_ascii=False, indent=2))
                print(f"详细状态：{REPORT_DIR / 'latest_status.json'}")
                return 0 if status["overall_success"] else 1

    status = _base_status(stages, run_mode)
    status["latest_intraday_data_date"] = _latest_intraday_date()
    status["latest_forecast_as_of_date"] = _latest_forecast_as_of_date()
    status["new_york_time"] = datetime.now(timezone.utc).astimezone(NEW_YORK).isoformat(timespec="seconds")
    _write_status(status)
    print("\n========== 流程状态 ==========")
    print(json.dumps({k: v for k, v in status.items() if k != "stages"}, ensure_ascii=False, indent=2))
    print(f"详细状态：{REPORT_DIR / 'latest_status.json'}")
    return 0 if status["overall_success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync, evaluate, and generate QQQM intraday forecast when safe.")
    parser.add_argument("--catch-up-on-open", action="store_true", help="Run a safe catch-up check when opening the project; skip if already done or market is active.")
    args = parser.parse_args()
    run_mode = "catch_up_on_open" if args.catch_up_on_open else "scheduled_or_manual"
    lock = _cycle_lock()
    if lock is None:
        print("检测到另一个盘中流程正在运行，本次跳过，避免重复同步或生成预测。")
        return 0
    try:
        return _main_impl(run_mode)
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
