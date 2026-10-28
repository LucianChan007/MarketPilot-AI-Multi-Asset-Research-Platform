from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "reports" / "forecast"
REQUIRED_COLUMNS = {
    "target_date",
    "actual_up",
    "model_probability_up_raw",
    "model_prediction",
    "majority_baseline_prediction",
    "majority_baseline_probability_up",
    "previous_day_momentum_prediction",
}


def safe_log_loss(actual: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(actual, dtype=int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(actual: np.ndarray, predicted: np.ndarray, probability: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(actual, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    if len(y) == 0:
        return {"n": 0}

    recalls: list[float] = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == label)))

    result: dict[str, Any] = {
        "n": int(len(y)),
        "actual_up_rate": float(np.mean(y)),
        "predicted_up_rate": float(np.mean(pred)),
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "true_up": int(np.sum((y == 1) & (pred == 1))),
        "true_down_or_flat": int(np.sum((y == 0) & (pred == 0))),
        "false_up": int(np.sum((y == 0) & (pred == 1))),
        "missed_up": int(np.sum((y == 1) & (pred == 0))),
    }
    if probability is not None:
        p = np.asarray(probability, dtype=float)
        result["brier_score"] = float(np.mean((p - y) ** 2))
        result["log_loss"] = safe_log_loss(y, p)
        result["mean_probability_up"] = float(np.mean(p))
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def build_diagnostics(csv_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"预测 CSV 缺少必要字段：{', '.join(missing)}")
    if frame.empty:
        raise ValueError("预测 CSV 没有记录。")

    for column in REQUIRED_COLUMNS - {"target_date"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    frame = frame.dropna(subset=["target_date", "actual_up", "model_probability_up_raw"])
    if frame.empty:
        raise ValueError("CSV 中没有有效的日期、真实方向和模型分数。")

    # Preserve recorded class predictions so the diagnostic reflects the model's original decision rule.
    actual = frame["actual_up"].astype(int).to_numpy()
    model_pred = frame["model_prediction"].fillna(
        (frame["model_probability_up_raw"] >= 0.5).astype(int)
    ).astype(int).to_numpy()
    model_prob = frame["model_probability_up_raw"].astype(float).to_numpy()
    majority_pred = frame["majority_baseline_prediction"].astype(int).to_numpy()
    majority_prob = frame["majority_baseline_probability_up"].astype(float).to_numpy()
    momentum_pred = frame["previous_day_momentum_prediction"].astype(int).to_numpy()

    overall = {
        "model": metrics(actual, model_pred, model_prob),
        "historical_majority_baseline": metrics(actual, majority_pred, majority_prob),
        "previous_day_momentum_baseline": metrics(actual, momentum_pred),
    }

    frame["month"] = frame["target_date"].dt.to_period("M").astype(str)
    monthly_rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        y = group["actual_up"].astype(int).to_numpy()
        p = group["model_probability_up_raw"].astype(float).to_numpy()
        pred = group["model_prediction"].astype(int).to_numpy()
        majority_p = group["majority_baseline_probability_up"].astype(float).to_numpy()
        majority_pred_month = group["majority_baseline_prediction"].astype(int).to_numpy()
        momentum_pred_month = group["previous_day_momentum_prediction"].astype(int).to_numpy()
        model_m = metrics(y, pred, p)
        majority_m = metrics(y, majority_pred_month, majority_p)
        momentum_m = metrics(y, momentum_pred_month)
        monthly_rows.append({
            "month": month,
            "n": len(group),
            "actual_up_rate": model_m["actual_up_rate"],
            "model_predicted_up_rate": model_m["predicted_up_rate"],
            "model_accuracy": model_m["accuracy"],
            "model_balanced_accuracy": model_m["balanced_accuracy"],
            "model_brier_score": model_m["brier_score"],
            "model_log_loss": model_m["log_loss"],
            "majority_accuracy": majority_m["accuracy"],
            "momentum_accuracy": momentum_m["accuracy"],
        })

    # Calibration diagnostics only. These bins do not calibrate the model.
    bins = [0.0, 0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.70, 1.000001]
    labels = ["<40%", "40-45%", "45-50%", "50-52.5%", "52.5-55%", "55-57.5%", "57.5-60%", "60-70%", ">=70%"]
    bin_series = pd.cut(frame["model_probability_up_raw"], bins=bins, labels=labels, right=False, include_lowest=True)
    calibration_rows: list[dict[str, Any]] = []
    for label in labels:
        mask = bin_series == label
        if not mask.any():
            continue
        group = frame.loc[mask]
        calibration_rows.append({
            "score_bin": label,
            "n": int(len(group)),
            "mean_raw_score": float(group["model_probability_up_raw"].mean()),
            "observed_up_rate": float(group["actual_up"].mean()),
            "interpretation": "观察性分箱检查；样本少时波动很大，不能据此认定校准有效。",
        })

    p = model_prob
    quantiles = {f"p{q}": float(np.quantile(p, q / 100)) for q in [0, 10, 25, 50, 75, 90, 100]}
    predicted_up_rate = float(np.mean(model_pred))
    actual_up_rate = float(np.mean(actual))

    return {
        "source_file": csv_path.name,
        "n": int(len(frame)),
        "date_start": str(frame["target_date"].min().date()),
        "date_end": str(frame["target_date"].max().date()),
        "overall": overall,
        "model_probability_score_distribution": quantiles,
        "actual_up_rate": actual_up_rate,
        "model_predicted_up_rate": predicted_up_rate,
        "monthly": monthly_rows,
        "calibration_bins": calibration_rows,
        "diagnostic_notes": [
            "当前模型原始分数未经校准，不应称为真实上涨概率。",
            "若模型预测上涨占比接近 100%，需检查分类阈值是否导致模型几乎总预测上涨。",
            "月度样本少时指标波动很大；这些诊断用于发现问题，不构成模型有效性证明。",
            "不要根据同一段测试集反复调整阈值后再报告该测试集成绩；阈值和模型选择必须在训练/验证区间完成。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断 QQQM 下一交易日方向模型的历史样本外结果。")
    parser.add_argument("--csv", type=Path, default=None, help="指定预测 CSV；默认选择 reports/forecast 中最新文件。")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(
            DEFAULT_REPORT_DIR.glob("QQQM_walk_forward_predictions_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(f"没有找到预测文件：{DEFAULT_REPORT_DIR}")
        csv_path = candidates[0]
    elif not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()

    result = build_diagnostics(csv_path)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"QQQM_model_diagnostics_{stamp}.json"
    monthly_path = out_dir / f"QQQM_monthly_diagnostics_{stamp}.csv"
    calibration_path = out_dir / f"QQQM_calibration_diagnostics_{stamp}.csv"

    json_path.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(result["monthly"]).to_csv(monthly_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(result["calibration_bins"]).to_csv(calibration_path, index=False, encoding="utf-8-sig")

    print("========== QQQM 模型诊断 ==========")
    print(f"输入文件：{csv_path}")
    print(f"样本：{result['n']}，区间：{result['date_start']} 至 {result['date_end']}")
    print(f"真实上涨占比：{result['actual_up_rate']:.1%}")
    print(f"模型预测上涨占比：{result['model_predicted_up_rate']:.1%}")
    print(f"模型原始分数分位数：{json.dumps(result['model_probability_score_distribution'], ensure_ascii=False)}")
    print("\n整体模型与基线指标：")
    for name, values in result["overall"].items():
        print(f"{name}: {json.dumps(json_safe(values), ensure_ascii=False)}")
    print("\n最近 12 个月指标：")
    monthly_df = pd.DataFrame(result["monthly"])
    print(monthly_df.tail(12).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n原始分数分箱观察：")
    calibration_df = pd.DataFrame(result["calibration_bins"])
    if calibration_df.empty:
        print("没有可用分箱。")
    else:
        print(calibration_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n注意：分箱结果不是校准模型，也不能把原始分数称为真实概率。")
    print(f"JSON 诊断：{json_path}")
    print(f"月度诊断：{monthly_path}")
    print(f"分箱诊断：{calibration_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
