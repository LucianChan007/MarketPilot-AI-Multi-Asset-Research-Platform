from __future__ import annotations

"""Build a leakage-aware training dataset from point-in-time forecast snapshots.

Only snapshots captured live at forecast time and already paired with a realized
next-session return are eligible. Reconstructed snapshots and news/macro fields
are deliberately excluded from the candidate training matrix.
"""

import csv
import json
import math
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "market_data.db"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "model_lab"

FEATURE_COLUMNS = (
    "market_return_1_session_pct",
    "market_return_5_sessions_pct",
    "market_return_20_sessions_pct",
    "market_annualized_volatility_20_session_pct",
    "market_price_vs_sma20_pct",
)
OUTPUT_COLUMNS = (
    "ticker",
    "as_of_date",
    "target_date",
    "current_price",
    "actual_adjusted_close",
    "actual_return_pct",
    *FEATURE_COLUMNS,
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value)[:10])
        return True
    except (TypeError, ValueError):
        return False


def _json_object(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_training_dataset(
    db_path: str | Path = DEFAULT_DB_PATH,
    ticker: str = "QQQM",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return eligible rows and a detailed readiness/audit report."""
    db_path = Path(db_path).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        required_tables = {"forecast_feedback", "forecast_context_snapshots"}
        missing_tables = [name for name in required_tables if not _table_exists(conn, name)]
        if missing_tables:
            raise RuntimeError(
                "缺少必要的数据表：" + ", ".join(sorted(missing_tables))
                + "。请先运行每日预测周期并初始化上下文归档。"
            )

        ticker = ticker.strip().upper()
        feedback_count = conn.execute(
            "SELECT COUNT(*) FROM forecast_feedback WHERE UPPER(ticker)=?",
            (ticker,),
        ).fetchone()[0]
        evaluated_count = conn.execute(
            "SELECT COUNT(*) FROM forecast_feedback WHERE UPPER(ticker)=? AND status='evaluated'",
            (ticker,),
        ).fetchone()[0]
        pending_count = conn.execute(
            "SELECT COUNT(*) FROM forecast_feedback WHERE UPPER(ticker)=? AND status='pending'",
            (ticker,),
        ).fetchone()[0]
        capture_rows = conn.execute(
            "SELECT capture_mode, COUNT(*) AS n FROM forecast_context_snapshots "
            "WHERE UPPER(ticker)=? GROUP BY capture_mode",
            (ticker,),
        ).fetchall()
        capture_modes = {str(row["capture_mode"]): int(row["n"]) for row in capture_rows}
        snapshots_total = sum(capture_modes.values())

        joined = conn.execute(
            """
            SELECT
                f.id AS feedback_id,
                f.ticker,
                f.as_of_date,
                f.status,
                f.current_price,
                f.actual_target_date,
                f.actual_adjusted_close,
                f.actual_return_pct,
                s.id AS snapshot_id,
                s.capture_mode,
                s.context_schema_version,
                s.feature_vector_json,
                s.training_eligibility_json
            FROM forecast_feedback AS f
            JOIN forecast_context_snapshots AS s
              ON s.forecast_feedback_id = f.id
             AND UPPER(s.ticker) = UPPER(f.ticker)
             AND s.as_of_date = f.as_of_date
            WHERE UPPER(f.ticker)=?
            ORDER BY f.as_of_date, f.id
            """,
            (ticker,),
        ).fetchall()

        reasons = {
            "reconstructed_snapshot": 0,
            "not_evaluated_yet": 0,
            "not_point_in_time_eligible": 0,
            "invalid_target_alignment": 0,
            "missing_feature_values": 0,
            "eligible": 0,
        }
        dataset: list[dict[str, Any]] = []
        for row in joined:
            if row["capture_mode"] != "live_prediction":
                reasons["reconstructed_snapshot"] += 1
                continue
            if row["status"] != "evaluated":
                reasons["not_evaluated_yet"] += 1
                continue
            eligibility = _json_object(row["training_eligibility_json"])
            market_eligibility = eligibility.get("market")
            market_eligibility = market_eligibility if isinstance(market_eligibility, dict) else {}
            if not (
                market_eligibility.get("eligible_for_candidate_training") is True
                and market_eligibility.get("point_in_time_captured") is True
            ):
                reasons["not_point_in_time_eligible"] += 1
                continue

            as_of_date = str(row["as_of_date"] or "")[:10]
            target_date = str(row["actual_target_date"] or "")[:10]
            current_price = _finite_number(row["current_price"])
            actual_close = _finite_number(row["actual_adjusted_close"])
            actual_return = _finite_number(row["actual_return_pct"])
            if (
                not _valid_date(as_of_date)
                or not _valid_date(target_date)
                or target_date <= as_of_date
                or current_price is None or current_price <= 0
                or actual_close is None or actual_close <= 0
                or actual_return is None
            ):
                reasons["invalid_target_alignment"] += 1
                continue

            vector = _json_object(row["feature_vector_json"])
            values = {name: _finite_number(vector.get(name)) for name in FEATURE_COLUMNS}
            if any(value is None for value in values.values()):
                reasons["missing_feature_values"] += 1
                continue

            record: dict[str, Any] = {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "target_date": target_date,
                "current_price": current_price,
                "actual_adjusted_close": actual_close,
                "actual_return_pct": actual_return,
            }
            record.update(values)
            dataset.append(record)
            reasons["eligible"] += 1

        eligible_n = len(dataset)
        if eligible_n < 30:
            readiness = "collecting_only"
            readiness_message = "已验证的实时特征样本少于 30 条；只归档，不训练或比较新闻增强模型。"
        elif eligible_n < 100:
            readiness = "exploratory_only"
            readiness_message = "样本允许进行初步实验，但不允许自动晋级模型。"
        else:
            readiness = "shadow_experiment_eligible"
            readiness_message = "可以运行候选模型影子实验；仍需严格样本外比较，不自动替换生产模型。"

        report = {
            "ticker": ticker,
            "database_path": str(db_path),
            "forecast_records_total": int(feedback_count),
            "forecast_records_evaluated": int(evaluated_count),
            "forecast_records_pending": int(pending_count),
            "context_snapshots_total": int(snapshots_total),
            "context_snapshots_by_capture_mode": capture_modes,
            "joined_forecast_snapshot_rows": len(joined),
            "eligible_training_rows": eligible_n,
            "feature_columns": list(FEATURE_COLUMNS),
            "target_column": "actual_return_pct",
            "exclusion_counts": reasons,
            "readiness": readiness,
            "readiness_message": readiness_message,
            "news_features_in_training": False,
            "macro_features_in_training": False,
            "reconstructed_snapshots_in_training": False,
            "promotion_enabled": False,
        }
        return dataset, report
    finally:
        conn.close()


def export_training_dataset(
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ticker: str = "QQQM",
) -> dict[str, Any]:
    """Write a dataset CSV and audit JSON; safe even when there are zero rows."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, report = build_training_dataset(db_path=db_path, ticker=ticker)
    csv_path = output_dir / f"{ticker.upper()}_context_training_dataset_latest.csv"
    json_path = output_dir / f"{ticker.upper()}_context_training_readiness_latest.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report["dataset_csv"] = str(csv_path)
    report["readiness_json"] = str(json_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
