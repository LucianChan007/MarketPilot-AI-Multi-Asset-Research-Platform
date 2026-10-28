from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.next_day_model import run_backtest_and_forecast


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QQQM 下一交易日方向预测探索基线 + expanding walk-forward 样本外回测"
    )
    parser.add_argument(
        "--csv",
        default=str(ROOT / "data" / "QQQM_tiingo_history.csv"),
        help="Tiingo 风格的历史行情 CSV 路径",
    )
    parser.add_argument(
        "--db",
        default=str(ROOT / "data" / "market_data.db"),
        help="用于读取比训练 CSV 更新的 QQQM 最新行情特征的 SQLite 数据库",
    )
    args = parser.parse_args()

    try:
        result = run_backtest_and_forecast(args.csv, args.db)
    except Exception as exc:
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_dir = ROOT / "reports" / "forecast"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    prediction_path = output_dir / f"QQQM_walk_forward_predictions_{stamp}.csv"
    summary_path = output_dir / f"QQQM_forecast_summary_{stamp}.json"
    latest_path = output_dir / "latest_forecast.json"

    result.predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    payload = {
        "backtest": result.summary,
        "forecast": result.forecast,
        "generated_at_local": datetime.now().astimezone().isoformat(),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    model_metrics = result.summary["model_metrics"]
    majority_metrics = result.summary["historical_majority_baseline_metrics"]
    momentum_metrics = result.summary["previous_day_momentum_baseline_metrics"]
    forecast = result.forecast

    print("\n========== QQQM 下一交易日方向预测（探索性基线） ==========")
    print(f"数据截止日：{forecast['as_of_date']}")
    print(f"最新调整收盘价：${forecast['latest_adjusted_close']:.4f}")
    print(f"特征来源：{forecast['feature_source']}")
    print(f"训练标签已知至：{forecast['training_labels_known_through']}")
    print(f"方向标签：{forecast['directional_label']}")
    print(f"模型原始上涨分数（未经校准）：{forecast['next_session_directional_score_up_raw_uncalibrated']:.1%}")
    print("\n========== 样本外 walk-forward 评估 ==========")
    print(f"预测测试样本：{result.summary['test_sample_count']}")
    print(f"测试区间：{result.summary['test_start_date']} 至 {result.summary['test_end_date']}")
    print("模型：", json.dumps(model_metrics, ensure_ascii=False))
    print("历史多数类基线：", json.dumps(majority_metrics, ensure_ascii=False))
    print("前一交易日方向基线：", json.dumps(momentum_metrics, ensure_ascii=False))
    print(f"模型准确率是否高于多数类基线：{result.summary['model_beats_majority_on_accuracy']}")
    print("\n重要限制：原始模型分数未校准，不代表可靠真实概率，也不构成买卖建议。")
    print("本版本没有使用新闻特征；新闻融合属于后续阶段。")
    print(f"\n预测结果 CSV：{prediction_path}")
    print(f"摘要 JSON：{summary_path}")
    print(f"最新预测 JSON：{latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
