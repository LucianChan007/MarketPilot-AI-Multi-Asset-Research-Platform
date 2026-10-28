from __future__ import annotations

"""Chronological Champion/Challenger research gate for QQQM return forecasts.

This script ONLY evaluates saved historical out-of-sample predictions. It does
not change the production forecaster, trade positions, or database. It chooses
a challenger on an earlier development slice and evaluates that fixed choice on
a later holdout slice to reduce selection bias.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = ROOT / "reports" / "forecast"
OUTPUT_DIR = ROOT / "reports" / "model_lab"

MODEL_COLUMNS = {
    "zero_return_baseline": "zero_return_baseline_predicted_return_pct",
    "expanding_mean_baseline": "expanding_mean_baseline_predicted_return_pct",
    "rolling_20d_mean_baseline": "rolling_20d_mean_baseline_predicted_return_pct",
    "ridge_full": "ridge_full_predicted_return_pct",
    "ridge_returns_only": "ridge_returns_only_predicted_return_pct",
}
INCUMBENT = "expanding_mean_baseline"
CHALLENGERS = [name for name in MODEL_COLUMNS if name != INCUMBENT]


def _metrics(actual_pct: np.ndarray, predicted_pct: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual_pct, dtype=float)
    predicted = np.asarray(predicted_pct, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return {"n": 0, "mae_bps": None, "rmse_bps": None, "direction_accuracy": None}

    # Inputs are in percentage points: 1 percentage point equals 100 bps.
    diff_pct_points = predicted - actual
    return {
        "n": int(len(actual)),
        "mae_bps": float(np.mean(np.abs(diff_pct_points)) * 100),
        "rmse_bps": float(np.sqrt(np.mean(diff_pct_points**2)) * 100),
        "direction_accuracy": float(np.mean((predicted > 0) == (actual > 0))),
        "mean_actual_return_pct": float(np.mean(actual)),
        "mean_predicted_return_pct": float(np.mean(predicted)),
    }


def _latest_predictions_file() -> Path:
    files = list(FORECAST_DIR.glob("QQQM_return_forecast_predictions_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"找不到历史收益预测 CSV：{FORECAST_DIR / 'QQQM_return_forecast_predictions_*.csv'}。"
            "请先运行 scripts/run_next_day_return_backtest.py。"
        )
    return max(files, key=lambda path: (path.stat().st_mtime, path.name))


def run_tournament(predictions_path: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    frame = pd.read_csv(predictions_path)
    required = {"target_date", "actual_return_pct", *MODEL_COLUMNS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"预测 CSV 缺少字段：{missing}")

    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    frame = frame.dropna(subset=["target_date", "actual_return_pct"]).sort_values("target_date").reset_index(drop=True)
    for column in MODEL_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(MODEL_COLUMNS.values())).reset_index(drop=True)

    if len(frame) < 150:
        raise ValueError(f"有效样本只有 {len(frame)} 条，少于 150 条；暂不进行候选模型筛选。")

    # Chronological split. Earlier segment chooses a challenger; later segment
    # is used once to compare the preselected challenger with the incumbent.
    split_index = int(len(frame) * 0.70)
    development = frame.iloc[:split_index].copy()
    holdout = frame.iloc[split_index:].copy()
    actual_dev = development["actual_return_pct"].to_numpy(dtype=float)

    dev_scores: dict[str, dict[str, Any]] = {}
    for model_name in CHALLENGERS:
        dev_scores[model_name] = _metrics(actual_dev, development[MODEL_COLUMNS[model_name]].to_numpy(dtype=float))
    chosen_challenger = min(
        CHALLENGERS,
        key=lambda name: (
            float("inf") if dev_scores[name]["mae_bps"] is None else dev_scores[name]["mae_bps"],
            float("inf") if dev_scores[name]["rmse_bps"] is None else dev_scores[name]["rmse_bps"],
        ),
    )

    actual_holdout = holdout["actual_return_pct"].to_numpy(dtype=float)
    incumbent_metrics = _metrics(actual_holdout, holdout[MODEL_COLUMNS[INCUMBENT]].to_numpy(dtype=float))
    challenger_metrics = _metrics(actual_holdout, holdout[MODEL_COLUMNS[chosen_challenger]].to_numpy(dtype=float))

    # Compare challenger vs incumbent on holdout month-by-month. A small number
    # of months is explicitly treated as insufficient for promoting a model.
    holdout["target_month"] = holdout["target_date"].dt.to_period("M").astype(str)
    monthly_rows: list[dict[str, Any]] = []
    challenger_month_wins = 0
    monthly_count = 0
    for month, group in holdout.groupby("target_month", sort=True):
        month_actual = group["actual_return_pct"].to_numpy(dtype=float)
        base_metric = _metrics(month_actual, group[MODEL_COLUMNS[INCUMBENT]].to_numpy(dtype=float))
        test_metric = _metrics(month_actual, group[MODEL_COLUMNS[chosen_challenger]].to_numpy(dtype=float))
        won = (
            base_metric["mae_bps"] is not None
            and test_metric["mae_bps"] is not None
            and test_metric["mae_bps"] < base_metric["mae_bps"]
        )
        monthly_count += 1
        challenger_month_wins += int(won)
        monthly_rows.append({
            "month": month,
            "n": int(len(group)),
            "incumbent_mae_bps": base_metric["mae_bps"],
            "challenger_mae_bps": test_metric["mae_bps"],
            "challenger_won_month": bool(won),
        })

    incumbent_mae = incumbent_metrics["mae_bps"]
    challenger_mae = challenger_metrics["mae_bps"]
    improvement_pct = None
    if incumbent_mae and challenger_mae is not None:
        improvement_pct = float((incumbent_mae - challenger_mae) / incumbent_mae * 100)

    win_fraction = challenger_month_wins / monthly_count if monthly_count else 0.0
    gate_reasons: list[str] = []
    if len(holdout) < 100:
        gate_reasons.append("最终时间留出集少于 100 条样本")
    if monthly_count < 6:
        gate_reasons.append("最终留出集覆盖月份少于 6 个月")
    if improvement_pct is None or improvement_pct < 5.0:
        gate_reasons.append("留出集 MAE 未改善至少 5%")
    if win_fraction < 0.60:
        gate_reasons.append("挑战模型赢得的月份比例低于 60%")

    # Even passing this historical gate is not enough to automatically replace
    # production: it still needs a prospective shadow test on real future dates.
    candidate_gate_passed = not gate_reasons
    recommendation = (
        "进入未来实时影子验证；本脚本不会自动替换生产模型。"
        if candidate_gate_passed
        else "保留当前参考模型，不晋级候选模型；继续收集样本并复测。"
    )

    now = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = output_dir / f"QQQM_champion_challenger_monthly_{now}.csv"
    monthly_latest_path = output_dir / "QQQM_champion_challenger_monthly_latest.csv"
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly_rows).to_csv(monthly_latest_path, index=False, encoding="utf-8-sig")

    holdout_metrics = {
        model: _metrics(actual_holdout, holdout[column].to_numpy(dtype=float))
        for model, column in MODEL_COLUMNS.items()
    }
    report: dict[str, Any] = {
        "ticker": "QQQM",
        "created_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_predictions_file": str(predictions_path.resolve()),
        "source_sample_count": int(len(frame)),
        "development": {
            "n": int(len(development)),
            "start_target_date": development["target_date"].iloc[0].date().isoformat(),
            "end_target_date": development["target_date"].iloc[-1].date().isoformat(),
            "challenger_scores": dev_scores,
            "selected_challenger": chosen_challenger,
            "selection_rule": "lowest development-slice MAE; RMSE breaks ties",
        },
        "holdout": {
            "n": int(len(holdout)),
            "start_target_date": holdout["target_date"].iloc[0].date().isoformat(),
            "end_target_date": holdout["target_date"].iloc[-1].date().isoformat(),
            "metrics_by_model": holdout_metrics,
            "incumbent": INCUMBENT,
            "selected_challenger": chosen_challenger,
            "incumbent_metrics": incumbent_metrics,
            "challenger_metrics": challenger_metrics,
            "challenger_mae_improvement_pct": improvement_pct,
            "holdout_months": monthly_count,
            "challenger_month_wins": challenger_month_wins,
            "challenger_month_win_fraction": win_fraction,
        },
        "promotion_gate": {
            "historical_gate_passed": candidate_gate_passed,
            "reasons_not_passed": gate_reasons,
            "recommendation": recommendation,
            "automatic_production_replacement": False,
            "minimum_live_shadow_predictions_before_review": 30,
        },
        "limitations": [
            "此实验只使用已有历史行情收益预测，不使用新闻/宏观特征。",
            "历史留出集仍是有限历史时期；结果不保证未来表现。",
            "即使历史门槛通过，也只允许进入未来影子验证，不自动改变生产预测。",
            "MAE、RMSE 和方向准确率均不代表扣除交易成本后的策略盈利能力。",
        ],
        "monthly_report_path": str(monthly_path.resolve()),
    }
    report_path = output_dir / f"QQQM_champion_challenger_{now}.json"
    latest_path = output_dir / "QQQM_champion_challenger_latest.json"
    for path in (report_path, latest_path):
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    try:
        predictions_path = _latest_predictions_file()
        report = run_tournament(predictions_path)
    except Exception as exc:
        print(f"模型竞技场未完成：{exc}")
        return 1

    development = report["development"]
    holdout = report["holdout"]
    gate = report["promotion_gate"]
    print("========== QQQM Champion-Challenger 模型实验 ==========")
    print(f"输入预测文件：{report['source_predictions_file']}")
    print(f"总样本：{report['source_sample_count']}")
    print(
        f"开发集：{development['n']} 条，"
        f"{development['start_target_date']} 至 {development['end_target_date']}"
    )
    print(f"开发集选出的挑战者：{development['selected_challenger']}")
    print(
        f"最终留出集：{holdout['n']} 条，"
        f"{holdout['start_target_date']} 至 {holdout['end_target_date']}"
    )
    print("\n最终留出集指标（MAE/RMSE 越低越好）：")
    for model_name, metrics in holdout["metrics_by_model"].items():
        print(
            f"  {model_name:28s} n={metrics['n']:3d} "
            f"MAE={metrics['mae_bps']:.2f} bps "
            f"RMSE={metrics['rmse_bps']:.2f} bps "
            f"方向准确率={metrics['direction_accuracy']:.3f}"
        )
    print("\n挑战者与参考模型比较：")
    print(f"  参考模型：{holdout['incumbent']}")
    print(f"  挑战者：{holdout['selected_challenger']}")
    print(f"  MAE 改善：{holdout['challenger_mae_improvement_pct']}")
    print(
        f"  月度胜出：{holdout['challenger_month_wins']}/"
        f"{holdout['holdout_months']} 个月"
    )
    print(f"\n历史晋级门槛：{'通过' if gate['historical_gate_passed'] else '未通过'}")
    print(f"建议：{gate['recommendation']}")
    if gate["reasons_not_passed"]:
        print("未通过原因：")
        for reason in gate["reasons_not_passed"]:
            print(f"  - {reason}")
    print("\n本脚本只生成实验报告，不会修改生产模型、数据库或交易。")
    print(f"报告：{OUTPUT_DIR / 'QQQM_champion_challenger_latest.json'}")
    print(f"月度报告：{OUTPUT_DIR / 'QQQM_champion_challenger_monthly_latest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
