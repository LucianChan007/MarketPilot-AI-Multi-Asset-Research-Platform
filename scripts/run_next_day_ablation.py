from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.next_day_model import (  # noqa: E402
    FEATURE_COLUMNS,
    NumpyLogisticRegression,
    build_feature_frame,
    load_history,
)

FEATURE_GROUPS: dict[str, list[str]] = {
    "full_current": list(FEATURE_COLUMNS),
    "without_volume": [c for c in FEATURE_COLUMNS if c != "volume_change_1d"],
    "momentum_trend": [
        "ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d",
        "close_to_sma5", "close_to_sma10", "close_to_sma20", "close_to_sma50",
    ],
    "returns_only": ["ret_1d", "ret_2d", "ret_5d", "ret_10d", "ret_20d"],
    "risk_context": [
        "volatility_5d", "volatility_20d", "rsi_14", "drawdown_20d",
        "intraday_range_pct", "overnight_gap_pct",
    ],
}


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    n = int(len(actual))
    if n == 0:
        return {"n": 0}

    true_up = int(np.sum((actual == 1) & (predicted == 1)))
    true_down = int(np.sum((actual == 0) & (predicted == 0)))
    false_up = int(np.sum((actual == 0) & (predicted == 1)))
    missed_up = int(np.sum((actual == 1) & (predicted == 0)))
    actual_up_count = int(np.sum(actual == 1))
    actual_down_count = int(np.sum(actual == 0))

    recall_up = true_up / actual_up_count if actual_up_count else float("nan")
    recall_down = true_down / actual_down_count if actual_down_count else float("nan")
    precision_up = true_up / (true_up + false_up) if true_up + false_up else float("nan")
    result: dict[str, Any] = {
        "n": n,
        "actual_up_rate": float(np.mean(actual)),
        "predicted_up_rate": float(np.mean(predicted)),
        "accuracy": float(np.mean(actual == predicted)),
        "balanced_accuracy": float(np.nanmean([recall_up, recall_down])),
        "recall_up": float(recall_up),
        "recall_down": float(recall_down),
        "precision_up": float(precision_up),
        "true_up": true_up,
        "true_down": true_down,
        "false_up": false_up,
        "missed_up": missed_up,
    }
    if probabilities is not None:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
        result["mean_probability_up"] = float(np.mean(p))
        result["brier_score"] = float(np.mean((p - actual) ** 2))
        result["log_loss"] = float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))
    else:
        result["mean_probability_up"] = None
        result["brier_score"] = None
        result["log_loss"] = None
    return result


def run_ablation(
    csv_path: Path,
    train_fraction: float = 0.70,
    minimum_train_rows: int = 300,
    refit_every: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = load_history(csv_path)
    engineered = build_feature_frame(history)

    required = list(FEATURE_COLUMNS) + ["target_up", "target_return", "target_date"]
    labeled = engineered.dropna(subset=required).copy().reset_index(drop=True)
    if len(labeled) < minimum_train_rows + 50:
        raise ValueError(
            f"有效样本只有 {len(labeled)} 条；至少需要 {minimum_train_rows + 50} 条。"
        )

    first_test = max(int(len(labeled) * train_fraction), minimum_train_rows)
    if first_test >= len(labeled) - 10:
        raise ValueError("样本外测试数据不足，请增加历史数据或调整参数。")

    test = labeled.iloc[first_test:].reset_index(drop=True)
    y_all = labeled["target_up"].to_numpy(dtype=int)
    ret_all = labeled["target_return"].to_numpy(dtype=float)
    feature_all = labeled[FEATURE_COLUMNS].to_numpy(dtype=float)
    records: list[dict[str, Any]] = []

    for block_start in range(first_test, len(labeled), refit_every):
        block_end = min(block_start + refit_every, len(labeled))
        y_train = y_all[:block_start]
        majority_probability = float(np.mean(y_train))
        majority_prediction = int(majority_probability >= 0.5)

        block_rows = range(block_start, block_end)
        for model_name, columns in FEATURE_GROUPS.items():
            indices = [FEATURE_COLUMNS.index(column) for column in columns]
            x_train = feature_all[:block_start, indices]
            x_test = feature_all[block_start:block_end, :][:, indices]
            model = NumpyLogisticRegression().fit(x_train, y_train)
            probs = model.predict_proba_up(x_test)

            for offset, probability in enumerate(probs):
                row_index = block_start + offset
                source = labeled.iloc[row_index]
                records.append({
                    "model_name": model_name,
                    "feature_date": str(source["date"]),
                    "target_date": str(source["target_date"]),
                    "actual_return_pct": float(ret_all[row_index] * 100),
                    "actual_up": int(y_all[row_index]),
                    "model_probability_up_raw": float(probability),
                    "model_prediction": int(probability >= 0.5),
                    "majority_baseline_prediction": majority_prediction,
                    "majority_baseline_probability_up": majority_probability,
                    "previous_day_momentum_prediction": int(float(source["ret_1d"]) > 0),
                    "always_up_prediction": 1,
                })

    predictions = pd.DataFrame(records)
    summary_rows: list[dict[str, Any]] = []
    for model_name, frame in predictions.groupby("model_name", sort=False):
        actual = frame["actual_up"].to_numpy(dtype=int)
        metrics = _metrics(
            actual,
            frame["model_prediction"].to_numpy(dtype=int),
            frame["model_probability_up_raw"].to_numpy(dtype=float),
        )
        summary_rows.append({"experiment": model_name, "features": ",".join(FEATURE_GROUPS[model_name]), **metrics})

    # Baselines are computed once from the identical test dates represented in each run.
    full = predictions[predictions["model_name"] == "full_current"].copy()
    actual = full["actual_up"].to_numpy(dtype=int)
    summary_rows.extend([
        {
            "experiment": "baseline_majority_probability",
            "features": "training prevalence at each refit",
            **_metrics(
                actual,
                full["majority_baseline_prediction"].to_numpy(dtype=int),
                full["majority_baseline_probability_up"].to_numpy(dtype=float),
            ),
        },
        {
            "experiment": "baseline_previous_day_momentum",
            "features": "predict up when previous session return > 0",
            **_metrics(actual, full["previous_day_momentum_prediction"].to_numpy(dtype=int)),
        },
        {
            "experiment": "baseline_always_up",
            "features": "always predict up; classification metrics only",
            **_metrics(actual, full["always_up_prediction"].to_numpy(dtype=int)),
        },
    ])
    summary = pd.DataFrame(summary_rows)

    monthly_rows: list[dict[str, Any]] = []
    predictions["target_month"] = pd.to_datetime(predictions["target_date"]).dt.to_period("M").astype(str)
    for (model_name, month), frame in predictions.groupby(["model_name", "target_month"], sort=True):
        m = _metrics(
            frame["actual_up"].to_numpy(dtype=int),
            frame["model_prediction"].to_numpy(dtype=int),
            frame["model_probability_up_raw"].to_numpy(dtype=float),
        )
        monthly_rows.append({"experiment": model_name, "month": month, **m})
    monthly = pd.DataFrame(monthly_rows)
    return summary, predictions.drop(columns=["target_month"]), monthly


def main() -> int:
    parser = argparse.ArgumentParser(description="QQQM 下一交易日模型特征消融实验")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "QQQM_tiingo_history.csv",
        help="Tiingo 历史行情 CSV 路径",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--minimum-train-rows", type=int, default=300)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "forecast")
    args = parser.parse_args()

    try:
        summary, predictions, monthly = run_ablation(
            args.csv,
            train_fraction=args.train_fraction,
            minimum_train_rows=args.minimum_train_rows,
            refit_every=args.refit_every,
        )
    except Exception as exc:
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"QQQM_feature_ablation_summary_{stamp}.csv"
    predictions_path = args.output_dir / f"QQQM_feature_ablation_predictions_{stamp}.csv"
    monthly_path = args.output_dir / f"QQQM_feature_ablation_monthly_{stamp}.csv"
    metadata_path = args.output_dir / f"QQQM_feature_ablation_{stamp}.json"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    metadata = {
        "csv_path": str(args.csv.resolve()),
        "sample_count": int(predictions["target_date"].nunique()),
        "target_date_start": str(predictions["target_date"].min()),
        "target_date_end": str(predictions["target_date"].max()),
        "train_fraction": args.train_fraction,
        "minimum_train_rows": args.minimum_train_rows,
        "refit_every": args.refit_every,
        "classification_threshold": 0.5,
        "feature_groups": FEATURE_GROUPS,
        "warning": (
            "这是使用同一历史样本进行的探索性特征消融比较。不要仅凭其中最佳结果宣称有预测优势，"
            "也不要直接将原始模型分数作为真实概率或买卖信号。"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    display_columns = [
        "experiment", "n", "actual_up_rate", "predicted_up_rate", "accuracy",
        "balanced_accuracy", "recall_up", "recall_down", "brier_score", "log_loss",
    ]
    shown = summary[[c for c in display_columns if c in summary.columns]].copy()
    for column in shown.columns:
        if column not in {"experiment", "n"}:
            shown[column] = pd.to_numeric(shown[column], errors="coerce").round(4)
    print("========== QQQM 特征消融实验 ==========")
    print(f"样本外目标日期：{metadata['target_date_start']} 至 {metadata['target_date_end']}")
    print(f"样本数：{metadata['sample_count']}；固定分类阈值：0.50；每 {args.refit_every} 条重训")
    print("\n整体指标（只用于探索，不能直接据此挑选最终模型）：")
    print(shown.to_string(index=False))
    print("\n最近 12 个月的月度指标已保存，可用来观察表现是否只来自个别月份。")
    print(f"汇总：{summary_path}")
    print(f"逐日预测：{predictions_path}")
    print(f"月度指标：{monthly_path}")
    print(f"实验元数据：{metadata_path}")
    print("注意：本脚本不改动原模型、不修改数据库，也不搜索最佳阈值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
