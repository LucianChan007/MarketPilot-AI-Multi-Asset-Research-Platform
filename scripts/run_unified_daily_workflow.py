from __future__ import annotations

"""Run the project's existing daily automation and intraday-learning cycle.

This is an orchestrator, not a new predictor: it reuses the established modules,
then records workflow status. Video scripts are generated separately from verified
forecast snapshots; stale forecasts are never presented as tomorrow predictions.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports" / "unified_daily_workflow"
DB_PATH = ROOT / "data" / "market_data.db"
FORECAST_PATH = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"


def latest_market_date() -> str | None:
    if not DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
        return str(row[0])[:10] if row and row[0] else None
    except sqlite3.Error:
        return None


def latest_forecast_target() -> str | None:
    try:
        payload = json.loads(FORECAST_PATH.read_text(encoding="utf-8-sig"))
        value = payload.get("forecast_date_weekday_estimate")
        return str(value)[:10] if value else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def run_stage(name: str, command: list[str], timeout: int = 1200) -> dict[str, Any]:
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout, check=False)
        return {
            "name": name, "command": command, "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "started_at": started, "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - t0, 2),
            "stdout": (completed.stdout or "")[-6000:], "stderr": (completed.stderr or "")[-3500:],
        }
    except subprocess.TimeoutExpired as exc:
        return {"name": name, "command": command, "success": False, "returncode": None,
                "started_at": started, "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_seconds": round(time.monotonic() - t0, 2), "error": f"超过 {timeout} 秒，已停止等待。",
                "stdout": str(exc.stdout or "")[-3000:], "stderr": str(exc.stderr or "")[-2000:]}
    except OSError as exc:
        return {"name": name, "command": command, "success": False, "returncode": None,
                "started_at": started, "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_seconds": round(time.monotonic() - t0, 2), "error": f"无法启动：{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="统一运行 QQQM 行情同步、预测反馈与盘中路径学习。")
    parser.add_argument("--skip-news", action="store_true", help="跳过新闻同步与新闻分类")
    parser.add_argument("--skip-macro", action="store_true", help="跳过 FRED 宏观同步")
    parser.add_argument("--skip-intraday", action="store_true", help="只运行现有日线自动化，不运行盘中路径流程")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []
    automation = [sys.executable, "-m", "app.daily_automation", "--skip-ai"]
    if args.skip_news:
        automation.append("--skip-news")
    if args.skip_macro:
        automation.append("--skip-macro")
    stages.append(run_stage("行情、新闻与宏观同步；结算日线预测反馈", automation, timeout=1800))

    if not args.skip_intraday:
        stages.append(run_stage("同步盘中 K 线、复盘旧预测并生成下一交易日路径", [
            sys.executable, "-m", "scripts.run_intraday_learning_cycle"
        ], timeout=2400))

    target = latest_forecast_target()
    actual_date = latest_market_date()
    status: dict[str, Any] = {
        "started_at": stages[0]["started_at"],
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "latest_market_date": actual_date,
        "latest_intraday_forecast_target": target,
        "stages": stages,
        "notes": [
            "打开 Dashboard 不会自动预测，只有主动运行此流程才会同步或生成结果。",
            "视频脚本由独立页面生成；预测目标日期不新于真实行情日期时，不会把旧曲线当成下一交易日预测。",
            "新闻与宏观同步可能受 API 权限、网络、配额影响；请检查每个步骤的状态。",
        ],
    }

    if target and actual_date and target <= actual_date:
        status["forecast_skip_reason"] = (
            f"当前预测目标日期 {target} 不晚于行情最新日期 {actual_date}；"
            "未导出新预测或视频脚本，避免把过期曲线当成下一交易日预测。"
        )

    status["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    status["overall_success"] = all(bool(item.get("success")) for item in stages)
    status["partial_failures"] = [item["name"] for item in stages if not item.get("success")]
    path = REPORT_DIR / "latest_status.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    (REPORT_DIR / f"workflow_{stamp}.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== QQQM 统一每日工作流 ==========")
    for stage in stages:
        print(f"{'成功' if stage.get('success') else '失败'}：{stage['name']} ({stage.get('duration_seconds', '—')} 秒)")
        if stage.get("stdout"):
            print(stage["stdout"][-1800:])
        if stage.get("stderr"):
            print("提示/错误：", stage["stderr"][-1400:])
    if status.get("forecast_skip_reason"):
        print("新预测未生成：" + status["forecast_skip_reason"])
    print("统一状态文件：" + str(path))
    return 0 if status["overall_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
