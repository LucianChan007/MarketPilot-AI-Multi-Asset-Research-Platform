from __future__ import annotations

"""Run a reproducible, local-only QQQM model evaluation and promotion-gate suite.

The script extends the historical daily price CSV with newer rows already in the
local SQLite market database, runs an expanding-window walk-forward benchmark,
then runs a chronological champion/challenger holdout tournament. It also reports
prospective live-shadow feedback from forecast_feedback. It never changes the
production forecaster, places trades, or promotes a model automatically.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_next_day_return_backtest import run_backtest  # noqa: E402
from scripts.run_champion_challenger import run_tournament  # noqa: E402

DEFAULT_HISTORY = ROOT / "data" / "QQQM_tiingo_history.csv"
DEFAULT_DB = ROOT / "data" / "market_data.db"
OUTPUT_DIR = ROOT / "reports" / "model_lab"


def _lookup_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().casefold(): str(column) for column in frame.columns}


def _numeric_col(frame: pd.DataFrame, lookup: dict[str, str], names: tuple[str, ...]) -> pd.Series:
    for name in names:
        col = lookup.get(name.casefold())
        if col is not None:
            return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _normalize_base_history(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"历史行情文件为空：{path}")
    lookup = _lookup_columns(raw)
    date_col = lookup.get("date") or lookup.get("trade_date") or lookup.get("datetime")
    if date_col is None:
        raise ValueError("历史行情 CSV 缺少 date/trade_date/datetime 日期列。")
    dates = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")

    close = _numeric_col(raw, lookup, ("close", "adjclose", "adj_close"))
    adj_close = _numeric_col(raw, lookup, ("adjclose", "adj_close"))
    adj_close = adj_close.where(adj_close.notna(), close)
    open_price = _numeric_col(raw, lookup, ("open", "adjopen", "adj_open"))
    high = _numeric_col(raw, lookup, ("high", "adjhigh", "adj_high"))
    low = _numeric_col(raw, lookup, ("low", "adjlow", "adj_low"))
    adj_open = _numeric_col(raw, lookup, ("adjopen", "adj_open"))
    adj_high = _numeric_col(raw, lookup, ("adjhigh", "adj_high"))
    adj_low = _numeric_col(raw, lookup, ("adjlow", "adj_low"))
    volume = _numeric_col(raw, lookup, ("volume", "adjvolume", "adj_volume"))
    adj_volume = _numeric_col(raw, lookup, ("adjvolume", "adj_volume"))

    frame = pd.DataFrame({
        "date": dates,
        "close": close,
        "adjClose": adj_close,
        "open": open_price,
        "high": high,
        "low": low,
        "adjOpen": adj_open.where(adj_open.notna(), open_price),
        "adjHigh": adj_high.where(adj_high.notna(), high),
        "adjLow": adj_low.where(adj_low.notna(), low),
        "volume": volume,
        "adjVolume": adj_volume.where(adj_volume.notna(), volume),
    })
    frame = frame.dropna(subset=["date", "adjClose"])
    frame = frame[frame["adjClose"] > 0]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(frame) < 350:
        raise ValueError(f"历史行情只有 {len(frame)} 条有效记录，至少需要 350 条才能进行时序评估。")
    return frame


def prepare_extended_history(
    base_csv: str | Path = DEFAULT_HISTORY,
    db_path: str | Path = DEFAULT_DB,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Merge local daily DB rows strictly newer than the historical CSV cutoff."""
    base_path, database_path = Path(base_csv), Path(db_path)
    if not base_path.exists():
        raise FileNotFoundError(f"历史行情文件不存在：{base_path}")
    base = _normalize_base_history(base_path)
    base_end = str(base["date"].max())
    appended = pd.DataFrame(columns=base.columns)
    db_latest: str | None = None
    rows: list[tuple[Any, ...]] = []
    table = None

    if database_path.exists():
        conn = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_daily'"
            ).fetchone()
            if table:
                rows = conn.execute(
                    """SELECT trade_date, open, high, low, close, adj_close, volume
                       FROM market_daily WHERE UPPER(ticker)='QQQM' AND trade_date > ?
                       ORDER BY trade_date""",
                    (base_end,),
                ).fetchall()
                latest = conn.execute(
                    "SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'"
                ).fetchone()
                db_latest = str(latest[0])[:10] if latest and latest[0] else None
        finally:
            conn.close()

        append_rows: list[dict[str, Any]] = []
        for day, open_raw, high_raw, low_raw, close_raw, adj_close_raw, volume in rows if table else []:
            try:
                close_val = float(close_raw)
                adj_close_val = float(adj_close_raw) if adj_close_raw is not None else close_val
            except (TypeError, ValueError):
                continue
            if close_val <= 0 or adj_close_val <= 0:
                continue
            scale = adj_close_val / close_val
            append_rows.append({
                "date": str(day)[:10],
                "close": close_val,
                "adjClose": adj_close_val,
                "open": float(open_raw) if open_raw is not None else np.nan,
                "high": float(high_raw) if high_raw is not None else np.nan,
                "low": float(low_raw) if low_raw is not None else np.nan,
                "adjOpen": float(open_raw) * scale if open_raw is not None else np.nan,
                "adjHigh": float(high_raw) * scale if high_raw is not None else np.nan,
                "adjLow": float(low_raw) * scale if low_raw is not None else np.nan,
                "volume": float(volume) if volume is not None else np.nan,
                "adjVolume": float(volume) if volume is not None else np.nan,
            })
        if append_rows:
            appended = pd.DataFrame(append_rows, columns=base.columns)

    combined = pd.concat([base, appended], ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    latest_out = output_csv or (OUTPUT_DIR / "QQQM_model_evaluation_history_latest.csv")
    latest_out = Path(latest_out)
    latest_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(latest_out, index=False, encoding="utf-8-sig")
    return {
        "base_csv": str(base_path.resolve()),
        "database": str(database_path.resolve()),
        "base_latest_date": base_end,
        "database_latest_date": db_latest,
        "appended_database_rows": int(len(appended)),
        "combined_rows": int(len(combined)),
        "combined_latest_date": str(combined["date"].max()),
        "combined_history_csv": str(latest_out.resolve()),
    }


def summarize_live_shadow(db_path: str | Path = DEFAULT_DB, minimum_samples: int = 30) -> dict[str, Any]:
    """Summarize prospective forecast feedback; too few samples cannot pass the gate."""
    path = Path(db_path)
    if not path.exists():
        return {"evaluated_samples": 0, "pending_samples": 0, "minimum_samples_required": minimum_samples,
                "gate_passed": False, "reason": "本地行情数据库不存在。", "models": {}}
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='forecast_feedback'"
        ).fetchone()
        if not has_table:
            return {"evaluated_samples": 0, "pending_samples": 0, "minimum_samples_required": minimum_samples,
                    "gate_passed": False, "reason": "没有实时预测反馈表。", "models": {}}
        evaluated = conn.execute(
            "SELECT * FROM forecast_feedback WHERE UPPER(ticker)='QQQM' AND status='evaluated' ORDER BY actual_target_date"
        ).fetchall()
        pending = conn.execute(
            "SELECT COUNT(*) FROM forecast_feedback WHERE UPPER(ticker)='QQQM' AND status='pending'"
        ).fetchone()[0]
    finally:
        conn.close()

    column_map = {
        "零收益基准": ("error_zero_return_bps", "direction_correct_zero_return"),
        "历史均值基准": ("error_historical_mean_bps", "direction_correct_historical_mean"),
        "完整特征 Ridge": ("error_ridge_full_bps", "direction_correct_ridge_full"),
        "收益率特征 Ridge": ("error_ridge_returns_only_bps", "direction_correct_ridge_returns_only"),
    }
    models: dict[str, Any] = {}
    for label, (error_col, direction_col) in column_map.items():
        errors = [abs(float(row[error_col])) for row in evaluated if row[error_col] is not None]
        dirs = [int(row[direction_col]) for row in evaluated if row[direction_col] is not None]
        models[label] = {
            "n": len(errors),
            "mae_bps": float(np.mean(errors)) if errors else None,
            "rmse_bps": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "direction_accuracy": float(np.mean(dirs)) if dirs else None,
        }

    n = min(models["历史均值基准"]["n"], models["完整特征 Ridge"]["n"], models["收益率特征 Ridge"]["n"])
    baseline_mae = models["历史均值基准"]["mae_bps"]
    candidates = {name: models[name]["mae_bps"] for name in ("完整特征 Ridge", "收益率特征 Ridge")}
    candidate_improvements = {
        name: ((baseline_mae - mae) / baseline_mae * 100 if baseline_mae not in (None, 0) and mae is not None else None)
        for name, mae in candidates.items()
    }
    best_name = max(
        (name for name, improvement in candidate_improvements.items() if improvement is not None),
        key=lambda name: candidate_improvements[name],
        default=None,
    )
    best_improvement = candidate_improvements.get(best_name) if best_name else None
    gate_passed = bool(n >= minimum_samples and best_improvement is not None and best_improvement >= 5.0)
    reason = (
        "达到最小前瞻样本数，且 Ridge 相对历史均值基准的 MAE 改善至少 5%；仍需人工复核，不自动替换。"
        if gate_passed else
        f"实时前瞻样本不足或改善未达到 5%；当前可比已结算样本 {n} 条，至少需要 {minimum_samples} 条。"
    )
    return {
        "evaluated_samples": len(evaluated),
        "comparable_samples": n,
        "pending_samples": int(pending),
        "minimum_samples_required": minimum_samples,
        "models": models,
        "candidate_improvement_vs_historical_mean_pct": candidate_improvements,
        "best_candidate": best_name,
        "best_candidate_improvement_pct": best_improvement,
        "gate_passed": gate_passed,
        "reason": reason,
        "automatic_production_replacement": False,
        "first_evaluated_target_date": str(evaluated[0]["actual_target_date"]) if evaluated else None,
        "last_evaluated_target_date": str(evaluated[-1]["actual_target_date"]) if evaluated else None,
    }


def _portable_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return path.name


def run_suite(
    base_csv: str | Path = DEFAULT_HISTORY,
    db_path: str | Path = DEFAULT_DB,
    output_dir: str | Path = OUTPUT_DIR,
    minimum_live_samples: int = 30,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_info = prepare_extended_history(base_csv, db_path, out_dir / "QQQM_model_evaluation_history_latest.csv")

    summary, predictions, intervals, metadata, monthly = run_backtest(
        Path(history_info["combined_history_csv"]),
        train_fraction=0.70,
        minimum_train_rows=300,
        refit_every=20,
        ridge_alpha=10.0,
    )
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    predictions_path = out_dir / f"QQQM_model_evaluation_predictions_{stamp}.csv"
    summary_path = out_dir / f"QQQM_model_evaluation_metrics_{stamp}.csv"
    monthly_path = out_dir / f"QQQM_model_evaluation_monthly_{stamp}.csv"
    interval_path = out_dir / f"QQQM_model_evaluation_intervals_{stamp}.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    intervals.to_csv(interval_path, index=False, encoding="utf-8-sig")

    tournament = run_tournament(predictions_path, out_dir)
    # Keep saved reports portable between the build environment and the user's PC.
    tournament["source_predictions_file"] = _portable_path(tournament.get("source_predictions_file"))
    tournament["monthly_report_path"] = _portable_path(tournament.get("monthly_report_path"))
    # Refresh the user-facing latest tournament report with portable relative paths too.
    (out_dir / "QQQM_champion_challenger_latest.json").write_text(
        json.dumps(tournament, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_info["base_csv"] = _portable_path(history_info.get("base_csv"))
    history_info["database"] = _portable_path(history_info.get("database"))
    history_info["combined_history_csv"] = _portable_path(history_info.get("combined_history_csv"))
    metadata["csv_path"] = _portable_path(metadata.get("csv_path"))
    live_shadow = summarize_live_shadow(db_path, minimum_samples=minimum_live_samples)
    overall = {
        "historical_gate_passed": bool(tournament["promotion_gate"]["historical_gate_passed"]),
        "live_shadow_gate_passed": bool(live_shadow.get("gate_passed")),
        "production_model_changed": False,
        "production_forecast_changed": False,
        "status": "可进入人工复核" if tournament["promotion_gate"]["historical_gate_passed"] and live_shadow.get("gate_passed")
                 else "继续影子评估，不晋级",
        "reason": "历史留出集和真实前瞻反馈必须同时通过；本流程永不自动替换生产模型。",
    }
    report = {
        "ticker": "QQQM",
        "created_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "history": history_info,
        "walk_forward": {
            "metadata": metadata,
            "metrics": summary.to_dict(orient="records"),
            "monthly_metrics_csv": _portable_path(monthly_path),
            "interval_metrics": intervals.to_dict(orient="records"),
            "predictions_csv": _portable_path(predictions_path),
        },
        "historical_champion_challenger": tournament,
        "live_shadow": live_shadow,
        "overall_gate": overall,
        "guardrails": [
            "只补入本地 SQLite 中晚于训练历史 CSV 截止日的日线，不发起外部网络请求。",
            "开发集选模型，较晚的时间留出集只用于一次比较。",
            "至少 30 条实时前瞻可比样本，且 MAE 改善至少 5%，才可能通过前瞻门槛。",
            "即使两道门槛均通过，也只标记为人工复核，不自动替换生产模型。",
            "每日方向预测、下一交易日收益预测和十五分钟路径预测属于不同任务，不能混用准确率。",
        ],
    }
    latest_path = out_dir / "QQQM_model_evaluation_scorecard_latest.json"
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="QQQM 模型评估与晋级门槛总流程（仅评估，不改生产模型）")
    parser.add_argument("--csv", type=Path, default=DEFAULT_HISTORY, help="历史日线 CSV")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="本地行情与预测反馈 SQLite")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="评估报告目录")
    parser.add_argument("--minimum-live-samples", type=int, default=30, help="实时影子晋级最小样本数")
    args = parser.parse_args()

    try:
        report = run_suite(args.csv, args.db, args.output_dir, args.minimum_live_samples)
    except Exception as exc:
        print(f"模型评估流程失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    history = report["history"]
    walk = report["walk_forward"]["metadata"]
    tournament = report["historical_champion_challenger"]
    live = report["live_shadow"]
    print("========== QQQM 模型评估与晋级门槛 ==========")
    print(f"历史数据截止日：原 CSV {history['base_latest_date']}；合并后 {history['combined_latest_date']}")
    print(f"从本地数据库新增记录：{history['appended_database_rows']} 条")
    print(f"滚动验证区间：{walk['test_start']} 至 {walk['test_end']}，样本 {walk['out_of_sample_samples']} 条")
    print("\n滚动验证指标（MAE/RMSE 越低越好；单位 bps）：")
    print(pd.DataFrame(report["walk_forward"]["metrics"])[["experiment", "n", "mae_bps", "rmse_bps", "sign_accuracy"]].to_string(index=False))
    holdout = tournament["holdout"]
    gate = tournament["promotion_gate"]
    print("\n历史时间留出集：")
    print(f"参考模型：{holdout['incumbent']}；候选模型：{holdout['selected_challenger']}")
    print(f"候选相对参考模型 MAE 改善：{holdout['challenger_mae_improvement_pct']}")
    print(f"历史晋级门槛：{'通过' if gate['historical_gate_passed'] else '未通过'}")
    print("\n真实前瞻影子验证：")
    print(f"已结算样本：{live['evaluated_samples']}；可比样本：{live.get('comparable_samples', 0)}；待验证：{live['pending_samples']}")
    print(f"前瞻门槛：{'通过' if live.get('gate_passed') else '未通过'}；{live['reason']}")
    print(f"综合状态：{report['overall_gate']['status']}")
    print("生产模型是否更改：否。此流程只评估、留档，不会更改日常预测模型，也不会下单。")
    print(f"成绩单：{OUTPUT_DIR / 'QQQM_model_evaluation_scorecard_latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
