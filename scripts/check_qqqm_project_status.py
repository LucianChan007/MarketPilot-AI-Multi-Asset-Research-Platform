from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = [
    "QQQM Investment Assistant - Daily Data Sync",
    "QQQM Investment Assistant - Intraday Learning Cycle",
]

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

def main() -> int:
    print("========== QQQM 项目状态检查 ==========")
    required = [
        ROOT / "app" / "intraday_forecast.py",
        ROOT / "scripts" / "run_intraday_learning_cycle.py",
        ROOT / "scripts" / "sync_intraday_history.py",
        ROOT / "scripts" / "build_intraday_learning_dataset.py",
        ROOT / "scripts" / "plot_intraday_forecast.py",
        ROOT / "gui" / "dashboard.py",
        ROOT / "data" / "market_data.db",
    ]
    missing = []
    for path in required:
        ok = path.exists()
        print(f"{'[正常]' if ok else '[缺少]'} {path.relative_to(ROOT)}")
        if not ok:
            missing.append(path)

    forecast_path = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
    forecast = read_json(forecast_path)
    if forecast:
        main_path = forecast.get("main_path") or {}
        print("\n最新预测：")
        print(f"  行情截止日期：{forecast.get('as_of_date', '未知')}")
        print(f"  目标交易日：{forecast.get('forecast_date_weekday_estimate', '未知')}")
        print(f"  主路径模型：{main_path.get('model_name') or main_path.get('model_family') or '未记录'}")
        print(f"  独立于 P10/P50/P90：{'是' if main_path.get('independent_of_quantile_curves') else '否或回退中'}")
        print(f"  新闻实验调整：{'已应用' if main_path.get('news_adjustment_applied') else '未应用'}")
    else:
        print("\n[提醒] 尚未读取到有效的最新盘中预测 JSON。")

    cycle_path = ROOT / "reports" / "intraday_learning_cycle" / "latest_status.json"
    cycle = read_json(cycle_path)
    if cycle:
        print("\n最近一次盘中自动流程：")
        print(f"  结束时间：{cycle.get('finished_at', '未知')}")
        print(f"  流程成功：{cycle.get('overall_success', False)}")
        print(f"  本次安全跳过：{cycle.get('skipped', False)}")
        if cycle.get("skip_reason"):
            print(f"  跳过原因：{cycle['skip_reason']}")
        if cycle.get("failed_stages"):
            print(f"  失败环节：{', '.join(cycle['failed_stages'])}")
    else:
        print("\n最近一次自动流程状态：尚无记录。")

    if sys.platform == "win32":
        print("\nWindows 计划任务：")
        for task_name in TASK_NAMES:
            result = subprocess.run(
                ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            if result.returncode == 0:
                print(f"[已注册] {task_name}")
                lines = [line.strip() for line in result.stdout.splitlines() if any(k in line.lower() for k in ("next run time", "last run time", "last result", "下次运行时间", "上次运行时间", "上次结果"))]
                for line in lines:
                    print(f"  {line}")
            else:
                print(f"[未查询到] {task_name}")

    print("\n说明：此检查只读文件和任务状态，不会触发同步、生成预测或训练模型。")
    return 1 if missing else 0

if __name__ == "__main__":
    raise SystemExit(main())
