"""Plain-language daily forecast review helpers for the QQQM dashboard.

This module deliberately keeps human feedback separate from machine-learning
training data. A feedback row is an observation, not a validated label.
"""
from __future__ import annotations

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEGACY_FEEDBACK_RELATIVE_PATH = Path("reports") / "model_lab" / "QQQM_intraday_human_feedback.csv"
USER_FEEDBACK_RELATIVE_DIR = Path("reports") / "model_lab" / "users"
FEEDBACK_FIELDS = [
    "feedback_id",
    "user_id",
    "reviewed_at_utc",
    "target_date",
    "as_of_date",
    "forecast_generated_at_utc",
    "forecast_model_name",
    "forecast_model_family",
    "summary_target_date",
    "actual_points_found",
    "expected_points",
    "predicted_open_price",
    "actual_open_price",
    "open_error_usd_actual_minus_predicted",
    "predicted_close_price",
    "actual_close_price",
    "close_error_usd_actual_minus_predicted",
    "main_path_mae_pct",
    "p50_path_mae_pct",
    "user_overall_match",
    "user_biggest_miss",
    "user_news_helpfulness",
    "user_notes",
]


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _target_date(forecast: dict[str, Any] | None) -> str:
    if not isinstance(forecast, dict):
        return ""
    return str(forecast.get("forecast_date_weekday_estimate") or forecast.get("target_date") or "")


def _forecast_main_path(forecast: dict[str, Any]) -> dict[str, Any]:
    path = forecast.get("main_path")
    return path if isinstance(path, dict) else {}


def build_daily_review_context(
    forecast: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a date-aligned review context, refusing to join stale actuals.

    The `summary_latest` file can still refer to the prior forecast after a new
    forecast has been generated. Actual results are only attached when target
    dates match exactly.
    """
    forecast = forecast if isinstance(forecast, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    target = _target_date(forecast)
    summary_target = str(summary.get("target_date") or "")
    matched = bool(target and summary_target and target == summary_target)
    matched_summary = summary if matched else {}
    main = _forecast_main_path(forecast)

    points = _integer(matched_summary.get("actual_points_found"), 0)
    expected = len(forecast.get("curve") or [])
    if expected <= 0:
        expected = 26

    predicted_open = _number(main.get("predicted_open_price"))
    predicted_close = _number(main.get("predicted_close_price"))
    actual_open = _number(matched_summary.get("actual_open"))
    actual_close = _number(matched_summary.get("actual_close_near_last_bar"))

    main_mae_bps = _number(matched_summary.get("main_path_curve_mae_bps_vs_actual"))
    p50_mae_bps = _number(matched_summary.get("p50_curve_mae_bps_vs_actual"))
    # 100 basis points = 1 percent.
    main_mae_pct = main_mae_bps / 100.0 if main_mae_bps is not None else None
    p50_mae_pct = p50_mae_bps / 100.0 if p50_mae_bps is not None else None

    open_error = actual_open - predicted_open if actual_open is not None and predicted_open is not None else None
    close_error = actual_close - predicted_close if actual_close is not None and predicted_close is not None else None

    if not matched:
        status = "等待目标交易日的真实行情"
        explanation = (
            f"当前展示的是 {target or '待确认日期'} 的预测。最新复盘文件对应 {summary_target or '尚无结算日期'}，"
            "日期不一致，因此不会把旧交易日的真实价格拿来评价这次新预测。"
        )
    elif points <= 0:
        status = "尚未匹配到实际盘中价格"
        explanation = "目标日期已经匹配，但目前没有可用于比较的真实 15 分钟价格。"
    elif points < max(1, int(expected * 0.75)):
        status = "实际行情还不完整"
        explanation = f"目前匹配到 {points}/{expected} 个预期盘中价格点。数据较少时，先不要急着判断模型好坏。"
    else:
        status = "已有实际行情，可开始复盘"
        explanation = f"目标日期匹配成功，已找到 {points}/{expected} 个盘中价格点。误差仍只是这一天的结果，不能代表长期稳定表现。"

    # Preserve a prior completed summary for a separate, clearly-labelled note.
    previous_completed = None
    if summary_target and summary_target != target and _integer(summary.get("actual_points_found"), 0) > 0:
        previous_completed = {
            "target_date": summary_target,
            "actual_points_found": _integer(summary.get("actual_points_found"), 0),
            "main_path_curve_mae_bps_vs_actual": _number(summary.get("main_path_curve_mae_bps_vs_actual")),
            "p50_curve_mae_bps_vs_actual": _number(summary.get("p50_curve_mae_bps_vs_actual")),
            "actual_open": _number(summary.get("actual_open")),
            "actual_close_near_last_bar": _number(summary.get("actual_close_near_last_bar")),
        }

    return {
        "target_date": target,
        "as_of_date": str(forecast.get("as_of_date") or ""),
        "forecast_generated_at_utc": str(forecast.get("generated_at_utc") or ""),
        "summary_target_date": summary_target,
        "summary_matches_target": matched,
        "matched_summary": matched_summary,
        "status": status,
        "explanation": explanation,
        "actual_points_found": points,
        "expected_points": expected,
        "predicted_open_price": predicted_open,
        "actual_open_price": actual_open,
        "open_error_usd_actual_minus_predicted": open_error,
        "predicted_close_price": predicted_close,
        "actual_close_price": actual_close,
        "close_error_usd_actual_minus_predicted": close_error,
        "main_path_mae_pct": main_mae_pct,
        "p50_path_mae_pct": p50_mae_pct,
        "forecast_model_name": str(_forecast_main_path(forecast).get("model_name") or "未记录"),
        "forecast_model_family": str(_forecast_main_path(forecast).get("model_family") or "未记录"),
        "previous_completed_summary": previous_completed,
    }


def _feedback_output_path(root: str | Path, user_id: int | str | None) -> Path:
    """Return a feedback CSV path isolated to the authenticated account."""
    root_path = Path(root)
    if user_id is None:
        # Compatibility for existing local tests or explicit legacy maintenance.
        return root_path / LEGACY_FEEDBACK_RELATIVE_PATH
    try:
        normalized_user_id = int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id must be a positive integer") from exc
    if normalized_user_id <= 0:
        raise ValueError("user_id must be a positive integer")
    return root_path / USER_FEEDBACK_RELATIVE_DIR / f"user_{normalized_user_id}" / "QQQM_intraday_human_feedback.csv"


def save_human_feedback(
    root: str | Path,
    context: dict[str, Any],
    answers: dict[str, Any],
    user_id: int | str | None = None,
) -> Path:
    """Append one review row. When user_id is supplied, save only in that user's directory; never auto-train from feedback."""
    root_path = Path(root)
    output = _feedback_output_path(root_path, user_id)
    output.parent.mkdir(parents=True, exist_ok=True)

    row = {field: "" for field in FEEDBACK_FIELDS}
    row.update({
        "feedback_id": str(uuid.uuid4()),
        "user_id": str(user_id) if user_id is not None else "",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_date": context.get("target_date", ""),
        "as_of_date": context.get("as_of_date", ""),
        "forecast_generated_at_utc": context.get("forecast_generated_at_utc", ""),
        "forecast_model_name": context.get("forecast_model_name", ""),
        "forecast_model_family": context.get("forecast_model_family", ""),
        "summary_target_date": context.get("summary_target_date", ""),
        "actual_points_found": context.get("actual_points_found", 0),
        "expected_points": context.get("expected_points", 0),
        "predicted_open_price": context.get("predicted_open_price", ""),
        "actual_open_price": context.get("actual_open_price", ""),
        "open_error_usd_actual_minus_predicted": context.get("open_error_usd_actual_minus_predicted", ""),
        "predicted_close_price": context.get("predicted_close_price", ""),
        "actual_close_price": context.get("actual_close_price", ""),
        "close_error_usd_actual_minus_predicted": context.get("close_error_usd_actual_minus_predicted", ""),
        "main_path_mae_pct": context.get("main_path_mae_pct", ""),
        "p50_path_mae_pct": context.get("p50_path_mae_pct", ""),
        "user_overall_match": answers.get("overall_match", ""),
        "user_biggest_miss": answers.get("biggest_miss", ""),
        "user_news_helpfulness": answers.get("news_helpfulness", ""),
        "user_notes": str(answers.get("notes", "") or "").strip()[:4000],
    })

    should_write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS, extrasaction="ignore")
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)
    return output


def load_recent_human_feedback(
    root: str | Path, limit: int = 5, user_id: int | str | None = None
) -> list[dict[str, str]]:
    path = _feedback_output_path(root, user_id)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []
    return list(reversed(rows[-max(1, int(limit)):]))
