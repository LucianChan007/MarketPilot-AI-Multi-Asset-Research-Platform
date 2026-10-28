from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.intraday_forecast import run_intraday_prototype  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    try:
        result = run_intraday_prototype()
    except Exception as exc:
        print(f"盘中预测原型运行失败：{type(exc).__name__}: {exc}")
        return 1

    print("========== QQQM 下一交易日盘中主路径模型 V2 ==========")
    forecast = result["forecast"]
    main_path = forecast.get("main_path") or {}
    if main_path.get("model_family") == "ridge_regression_per_time_slot":
        print("主预测路径：扩展窗口岭回归独立生成，不由 P10/P50/P90 拼接。")
    else:
        print("注意：Ridge 主路径样本不足，当前退回历史候选代表路径；请在报告中查看 fallback_reason。")
    print("P10/P50/P90 是逐时间点的历史分位情景，不是未来概率保证。")
    print("当前新闻只可能通过单独标记的封顶实验叠加影响路径；它尚未作为历史训练特征。")

    print("\n滚动回测指标：")
    print(json.dumps(result["backtest_summary"], ensure_ascii=False, indent=2))
    print("\n最新预测摘要：")
    print(json.dumps({
        "as_of_date": forecast["as_of_date"],
        "forecast_date_weekday_estimate": forecast["forecast_date_weekday_estimate"],
        "previous_session_close": forecast["previous_session_close"],
        "curve_method": forecast.get("curve_method"),
        "conditional_model_available": forecast.get("conditional_model_available"),
        "estimated_open": forecast["estimated_open"],
        "estimated_close_from_path": forecast["estimated_close_from_path"],
        "eligible_training_sessions": forecast["eligible_training_sessions"],
        "similar_session_count": forecast.get("similar_session_count"),
        "candidate_paths_saved": len(forecast.get("candidate_paths", [])),
        "market_only_main_path": forecast.get("market_only_main_path"),
        "main_path": forecast.get("main_path"),
        "historical_cluster_path": forecast.get("historical_cluster_path"),
        "path_selection_diagnostics": forecast.get("path_selection_diagnostics"),
        "ridge_walk_forward_path_mae_bps": (forecast.get("backtest_summary") or {}).get("ridge_ml_path_mae_bps"),
        "news_context_summary": {
            "available": (forecast.get("news_context") or {}).get("available"),
            "included_article_count": (forecast.get("news_context") or {}).get("included_article_count"),
            "reason": (forecast.get("news_context") or {}).get("reason"),
        },
        "curve_points": len(forecast["curve"]),
        "data_quality": forecast["data_quality"],
        "caveats": forecast["caveats"],
    }, ensure_ascii=False, indent=2))
    print(f"\n预测 JSON：{result['forecast_json']}")
    print(f"不可变前瞻预测归档：{result.get('archive_json')}")
    print(f"逐日回测 CSV：{result['daily_backtest_csv']}")
    print(f"预测曲线 CSV：{result['curve_csv']}")
    print("后续可执行：python -m scripts.plot_intraday_forecast；收盘后执行 python -m scripts.build_intraday_learning_dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
