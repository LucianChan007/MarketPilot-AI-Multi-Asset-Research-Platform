from __future__ import annotations

"""Build an honest, point-in-time-aligned intraday forecast learning dataset.

PNG charts are linked for human review; they are not silently treated as model features.
Only prospective, versioned forecast snapshots with later completed actuals are eligible.
"""

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "market_data.db"
DEFAULT_ARCHIVE = ROOT / "reports" / "intraday" / "archive"
DEFAULT_REPORT = ROOT / "reports" / "model_lab"
EXPECTED_TIMES = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
    "11:00", "11:15", "11:30", "11:45", "12:00", "12:15",
    "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
    "14:00", "14:15", "14:30", "14:45", "15:00", "15:15",
    "15:30", "15:45",
]
REQUIRED_SNAPSHOT_VERSION = "intraday-learning-snapshot-v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return value


def _load_actual_sessions(db_path: Path, ticker: str = "QQQM", feed: str = "iex", timeframe: str = "15Min") -> dict[str, dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"找不到行情数据库：{db_path}")
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "intraday_bars" not in tables:
            raise RuntimeError("数据库里没有 intraday_bars 表。")
        columns = {row[1] for row in connection.execute('PRAGMA table_info("intraday_bars")')}
        required = {"symbol", "feed", "timeframe", "trade_date_ny", "session_time_ny", "open", "close"}
        if not required.issubset(columns):
            raise RuntimeError(f"intraday_bars 缺少字段：{sorted(required - columns)}")
        rows = connection.execute(
            """SELECT trade_date_ny, session_time_ny, open, close
               FROM intraday_bars
               WHERE UPPER(symbol)=? AND LOWER(feed)=? AND timeframe=?
               ORDER BY trade_date_ny, session_time_ny""",
            (ticker.upper(), feed.lower(), timeframe),
        ).fetchall()
    finally:
        connection.close()

    sessions: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        try:
            slot = str(row["session_time_ny"])[:5]
            open_price = float(row["open"])
            close_price = float(row["close"])
        except (TypeError, ValueError):
            continue
        if slot not in EXPECTED_TIMES or open_price <= 0 or close_price <= 0:
            continue
        sessions[str(row["trade_date_ny"])][slot] = {
            "open": open_price,
            "close": close_price,
        }

    result: dict[str, dict[str, Any]] = {}
    for session_date, bars in sessions.items():
        # Do not label a session complete until the final regular-session bar is present.
        if "09:30" not in bars or "15:45" not in bars or len(bars) < 24:
            continue
        open_price = float(bars["09:30"]["open"])
        close_price = float(bars["15:45"]["close"])
        if open_price <= 0 or close_price <= 0:
            continue
        actual_returns = {
            slot: (float(bars[slot]["close"]) / open_price - 1.0) * 100.0
            for slot in EXPECTED_TIMES if slot in bars
        }
        result[session_date] = {
            "bars": bars,
            "open_price": open_price,
            "close_price": close_price,
            "coverage_count": len(bars),
            "actual_returns_from_open_pct": actual_returns,
        }
    return result


def _price_map_from_forecast_path(path_obj: Any) -> dict[str, float]:
    if not isinstance(path_obj, dict):
        return {}
    map_obj = path_obj.get("price_by_time")
    if isinstance(map_obj, dict):
        output: dict[str, float] = {}
        for slot, value in map_obj.items():
            try:
                number = float(value)
                if math.isfinite(number) and number > 0:
                    output[str(slot)] = number
            except (TypeError, ValueError):
                pass
        if output:
            return output
    points = path_obj.get("path", [])
    if isinstance(points, list):
        output = {}
        for point in points:
            if not isinstance(point, dict):
                continue
            try:
                output[str(point["time_ny"])] = float(point.get("predicted_price", point.get("price")))
            except (KeyError, TypeError, ValueError):
                continue
        return output
    return {}


def _curve_price_map(forecast: dict[str, Any], field: str) -> dict[str, float]:
    result: dict[str, float] = {}
    curve = forecast.get("curve", [])
    aliases = {
        "p10": "low_price",
        "p50": "median_price",
        "p90": "high_price",
    }
    for point in curve if isinstance(curve, list) else []:
        if not isinstance(point, dict):
            continue
        try:
            result[str(point["time_ny"])] = float(point[aliases[field]])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _path_mae_bps(predicted: dict[str, float], actual_bars: dict[str, Any]) -> float | None:
    errors: list[float] = []
    for slot, price in predicted.items():
        if slot not in actual_bars:
            continue
        actual = float(actual_bars[slot]["close"])
        if actual > 0:
            errors.append(abs(price - actual) / actual * 10000.0)
    return round(statistics.mean(errors), 3) if errors else None


def _forecast_was_pre_session(forecast: dict[str, Any], target_date: str) -> bool:
    """Return True only when the whole-session forecast was available before the NY open."""
    if forecast.get("capture_mode") == "late_or_stale_forecast":
        return False
    if forecast.get("capture_mode") == "live_prediction" and forecast.get("pre_session_snapshot") is False:
        return False
    try:
        generated = datetime.fromisoformat(str(forecast.get("generated_at_utc", "")).replace("Z", "+00:00"))
        if generated.tzinfo is None:
            return False
        generated_ny = generated.astimezone(ZoneInfo("America/New_York"))
        target = date.fromisoformat(target_date)
        return generated_ny.date() < target or (
            generated_ny.date() == target and generated_ny.time().replace(tzinfo=None) < datetime.strptime("09:30", "%H:%M").time()
        )
    except (ValueError, TypeError):
        return False


def _flatten_news(news_context: dict[str, Any] | None) -> dict[str, Any]:
    context = news_context if isinstance(news_context, dict) else {}
    rows: dict[str, Any] = {
        "news_context_available": bool(context.get("available", False)),
        "news_included_article_count": int(context.get("included_article_count") or 0),
        "news_mean_impact_score": context.get("mean_impact_score"),
        "news_mean_relevance_score": context.get("mean_relevance_score"),
        "news_mean_confidence": context.get("mean_confidence"),
    }
    event_counts = context.get("event_type_counts") or {}
    bias_counts = context.get("nasdaq_bias_counts") or {}
    known_event_types = ["monetary_policy", "inflation", "employment", "earnings", "growth", "geopolitical", "other", "unknown"]
    known_biases = ["bullish", "bearish", "neutral", "mixed", "unknown"]
    for name in known_event_types:
        rows[f"news_event_{name}_count"] = int(event_counts.get(name, 0) or 0)
    for name in known_biases:
        rows[f"news_bias_{name}_count"] = int(bias_counts.get(name, 0) or 0)
    return rows


FEEDBACK_FIELDS = [
    "target_date", "as_of_date", "chart_path", "review_status",
    "human_expected_direction", "human_expected_path_shape", "human_note",
    "predicted_error_reason", "proposed_change", "approved_for_experiment",
]


def _sync_human_feedback_template(archive_files: list[Path], report_dir: Path) -> Path:
    """Add one review row per target date without overwriting existing human annotations."""
    feedback_path = report_dir / "QQQM_intraday_human_feedback.csv"
    existing: dict[str, dict[str, str]] = {}
    if feedback_path.exists():
        try:
            with feedback_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    target = str(row.get("target_date", "")).strip()
                    if target:
                        existing[target] = {key: str(row.get(key, "") or "") for key in FEEDBACK_FIELDS}
        except (OSError, csv.Error):
            # Do not overwrite a damaged file; let the caller fix it explicitly.
            raise RuntimeError(f"无法读取人工反馈表：{feedback_path}。为避免覆盖人工意见，已停止。")

    forecast_meta: dict[str, str] = {}
    for path in archive_files:
        try:
            forecast = _read_json(path)
        except Exception:
            continue
        target = str(forecast.get("forecast_date_weekday_estimate", "")).strip()
        if not target:
            continue
        versioned = bool(forecast.get("forecast_snapshot_version"))
        previous = forecast_meta.get(target, "")
        # prefer the versioned prospective archive when both a legacy file and new snapshot exist
        if not previous or (versioned and "[versioned]" not in previous):
            chart_path = report_dir.parent / "intraday" / f"QQQM_intraday_forecast_vs_actual_{target}.png"
            meta = {
                "target_date": target,
                "as_of_date": str(forecast.get("as_of_date", "")),
                "chart_path": str(chart_path.resolve()),
            }
            forecast_meta[target] = json.dumps(meta, ensure_ascii=False) + (" [versioned]" if versioned else "")

    for target, encoded in forecast_meta.items():
        meta_text = encoded.replace(" [versioned]", "")
        meta = json.loads(meta_text)
        if target not in existing:
            existing[target] = {
                **meta,
                "review_status": "not_reviewed",
                "human_expected_direction": "",
                "human_expected_path_shape": "",
                "human_note": "",
                "predicted_error_reason": "",
                "proposed_change": "",
                "approved_for_experiment": "false",
            }
        else:
            # Fill only blank metadata; never clear or rewrite a user's existing notes.
            for field in ("as_of_date", "chart_path"):
                if not existing[target].get(field):
                    existing[target][field] = str(meta.get(field, ""))
    report_dir.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS)
        writer.writeheader()
        for target in sorted(existing):
            writer.writerow({field: existing[target].get(field, "") for field in FEEDBACK_FIELDS})
    return feedback_path


def _load_human_feedback(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            target = str(row.get("target_date", "")).strip()
            if target:
                result[target] = {key: str(row.get(key, "") or "") for key in FEEDBACK_FIELDS}
    return result


def build_dataset(db_path: Path = DEFAULT_DB, archive_dir: Path = DEFAULT_ARCHIVE, report_dir: Path = DEFAULT_REPORT) -> dict[str, Any]:
    # Keep source feeds separate: a forecast must be scored against actual bars from
    # the same provider that generated it, never against a silently mixed price series.
    actual_sessions_by_feed = {
        feed: _load_actual_sessions(db_path, feed=feed)
        for feed in ("iex", "yahoo", "eodhd")
    }
    archive_files = sorted(archive_dir.glob("QQQM_intraday_forecast_*.json")) if archive_dir.exists() else []
    report_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = _sync_human_feedback_template(archive_files, report_dir)
    human_feedback = _load_human_feedback(feedback_path)

    training_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    excluded_reasons: dict[str, int] = defaultdict(int)
    forecasts_seen = 0
    forecasts_with_actuals = 0
    waiting_for_actuals = 0

    for forecast_file in archive_files:
        try:
            forecast = _read_json(forecast_file)
        except (OSError, json.JSONDecodeError, ValueError):
            excluded_reasons["invalid_json"] += 1
            continue
        forecasts_seen += 1
        target_date = str(forecast.get("forecast_date_weekday_estimate") or "")
        as_of_date = str(forecast.get("as_of_date") or "")
        if not target_date or not as_of_date:
            excluded_reasons["missing_dates"] += 1
            continue
        forecast_feed = str(forecast.get("feed") or "iex").lower()
        actual_sessions = actual_sessions_by_feed.get(forecast_feed, actual_sessions_by_feed["iex"])
        actual = actual_sessions.get(target_date)
        if actual is None:
            waiting_for_actuals += 1
            continue
        if not _forecast_was_pre_session(forecast, target_date):
            excluded_reasons["forecast_not_available_before_target_session"] += 1
            continue
        forecasts_with_actuals += 1
        previous_close = forecast.get("previous_session_close")
        try:
            previous_close = float(previous_close)
        except (TypeError, ValueError):
            previous_close = 0.0
        actual_open = float(actual["open_price"])
        actual_close = float(actual["close_price"])
        actual_bars = actual["bars"]

        main_path = forecast.get("main_path")
        market_only_path = forecast.get("market_only_main_path")
        predicted_main_prices = _price_map_from_forecast_path(main_path)
        predicted_market_prices = _price_map_from_forecast_path(market_only_path)
        predicted_p50_prices = _curve_price_map(forecast, "p50")
        predicted_open_main = None
        predicted_open_market = None
        if isinstance(main_path, dict):
            predicted_open_main = main_path.get("predicted_open_price")
        if isinstance(market_only_path, dict):
            predicted_open_market = market_only_path.get("predicted_open_price")
        predicted_open_p50 = (forecast.get("estimated_open") or {}).get("p50_price")
        actual_gap_pct = (actual_open / previous_close - 1.0) * 100.0 if previous_close > 0 else None
        actual_intraday_return_pct = (actual_close / actual_open - 1.0) * 100.0 if actual_open > 0 else None
        actual_close_return_prev_pct = (actual_close / previous_close - 1.0) * 100.0 if previous_close > 0 else None

        def open_error(pred: Any) -> float | None:
            try:
                p = float(pred)
                return round(abs(p - actual_open) / actual_open * 10000.0, 3) if actual_open > 0 else None
            except (TypeError, ValueError):
                return None

        annotation = human_feedback.get(target_date, {})
        eval_row = {
            "target_date": target_date,
            "annotation_review_status": annotation.get("review_status", "not_reviewed"),
            "annotation_human_expected_direction": annotation.get("human_expected_direction", ""),
            "annotation_human_expected_path_shape": annotation.get("human_expected_path_shape", ""),
            "annotation_human_note": annotation.get("human_note", ""),
            "annotation_predicted_error_reason": annotation.get("predicted_error_reason", ""),
            "annotation_proposed_change": annotation.get("proposed_change", ""),
            "annotation_approved_for_experiment": annotation.get("approved_for_experiment", "false"),
            "as_of_date": as_of_date,
            "forecast_generated_at_utc": forecast.get("generated_at_utc"),
            "forecast_snapshot_version": forecast.get("forecast_snapshot_version", "legacy_or_unknown"),
            "capture_mode": forecast.get("capture_mode", "legacy_or_unknown"),
            "actual_bars_observed": int(actual["coverage_count"]),
            "actual_open": round(actual_open, 4),
            "actual_close_1545": round(actual_close, 4),
            "actual_gap_pct": round(actual_gap_pct, 5) if actual_gap_pct is not None else None,
            "actual_intraday_return_pct": round(actual_intraday_return_pct, 5) if actual_intraday_return_pct is not None else None,
            "actual_close_return_from_previous_close_pct": round(actual_close_return_prev_pct, 5) if actual_close_return_prev_pct is not None else None,
            "main_path_open_error_bps": open_error(predicted_open_main),
            "market_only_path_open_error_bps": open_error(predicted_open_market),
            "p50_open_error_bps": open_error(predicted_open_p50),
            "main_path_curve_mae_bps": _path_mae_bps(predicted_main_prices, actual_bars),
            "market_only_path_curve_mae_bps": _path_mae_bps(predicted_market_prices, actual_bars),
            "p50_curve_mae_bps": _path_mae_bps(predicted_p50_prices, actual_bars),
            "main_path_neighbor_date": (main_path or {}).get("selected_neighbor_date") if isinstance(main_path, dict) else None,
            "main_path_cluster_size": (main_path or {}).get("cluster_member_count") if isinstance(main_path, dict) else None,
            "main_path_cluster_weight_share_not_probability": (main_path or {}).get("cluster_weight_share_not_calibrated_probability") if isinstance(main_path, dict) else None,
            "news_adjustment_applied": bool((forecast.get("path_selection_diagnostics") or {}).get("news_adjustment_applied", False)),
            "news_signal": ((forecast.get("path_selection_diagnostics") or {}).get("news_signal") or {}).get("signal"),
            "chart_path": str((report_dir.parent / "intraday" / f"QQQM_intraday_forecast_vs_actual_{target_date}.png").resolve()),
        }
        evaluation_rows.append(eval_row)

        snapshot_version = forecast.get("forecast_snapshot_version")
        capture_mode = forecast.get("capture_mode")
        feature_values = forecast.get("market_state_features")
        news_context = forecast.get("news_context")
        if snapshot_version != REQUIRED_SNAPSHOT_VERSION:
            excluded_reasons["legacy_or_unversioned_forecast"] += 1
            continue
        if capture_mode != "live_prediction":
            excluded_reasons["not_live_prediction_snapshot"] += 1
            continue
        if not isinstance(feature_values, dict) or not isinstance(news_context, dict):
            excluded_reasons["missing_point_in_time_features"] += 1
            continue
        try:
            generated_at = datetime.fromisoformat(str(forecast.get("generated_at_utc", "")).replace("Z", "+00:00"))
            forecast_generated_date = generated_at.astimezone(ZoneInfo("America/New_York")).date()
        except ValueError:
            excluded_reasons["missing_or_invalid_generated_at"] += 1
            continue
        if forecast_generated_date > date.fromisoformat(target_date):
            excluded_reasons["forecast_timestamp_after_target_date"] += 1
            continue

        annotation = human_feedback.get(target_date, {})
        row: dict[str, Any] = {
            "sample_id": f"QQQM:{target_date}:{forecast.get('generated_at_utc', '')}",
            "ticker": forecast.get("ticker", "QQQM"),
            "as_of_date": as_of_date,
            "target_date": target_date,
            "forecast_generated_at_utc": forecast.get("generated_at_utc"),
            "forecast_snapshot_version": snapshot_version,
            "capture_mode": capture_mode,
            "feed": forecast.get("feed", "iex"),
            "timeframe": forecast.get("timeframe", "15Min"),
            "previous_session_close": round(previous_close, 6) if previous_close > 0 else None,
            "forecast_open_main_path": predicted_open_main,
            "forecast_open_market_only_path": predicted_open_market,
            "forecast_open_p50": predicted_open_p50,
            "forecast_close_main_path": (main_path or {}).get("predicted_close_price") if isinstance(main_path, dict) else None,
            "forecast_close_market_only_path": (market_only_path or {}).get("predicted_close_price") if isinstance(market_only_path, dict) else None,
            "main_path_selected_neighbor_date": (main_path or {}).get("selected_neighbor_date") if isinstance(main_path, dict) else None,
            "main_path_cluster_size": (main_path or {}).get("cluster_member_count") if isinstance(main_path, dict) else None,
            "main_path_cluster_weight_share_not_probability": (main_path or {}).get("cluster_weight_share_not_calibrated_probability") if isinstance(main_path, dict) else None,
            "news_adjustment_applied": bool((forecast.get("path_selection_diagnostics") or {}).get("news_adjustment_applied", False)),
            "news_signal": ((forecast.get("path_selection_diagnostics") or {}).get("news_signal") or {}).get("signal"),
            "actual_open": round(actual_open, 6),
            "actual_close_1545": round(actual_close, 6),
            "target_actual_gap_pct": round(actual_gap_pct, 6) if actual_gap_pct is not None else None,
            "target_actual_intraday_return_pct": round(actual_intraday_return_pct, 6) if actual_intraday_return_pct is not None else None,
            "target_actual_close_return_from_previous_close_pct": round(actual_close_return_prev_pct, 6) if actual_close_return_prev_pct is not None else None,
            "main_path_open_error_bps": open_error(predicted_open_main),
            "market_only_path_open_error_bps": open_error(predicted_open_market),
            "p50_open_error_bps": open_error(predicted_open_p50),
            "main_path_curve_mae_bps": _path_mae_bps(predicted_main_prices, actual_bars),
            "market_only_path_curve_mae_bps": _path_mae_bps(predicted_market_prices, actual_bars),
            "p50_curve_mae_bps": _path_mae_bps(predicted_p50_prices, actual_bars),
            "chart_path_for_human_review_only": eval_row["chart_path"],
            "png_is_model_input": False,
            "annotation_review_status": annotation.get("review_status", "not_reviewed"),
            "annotation_human_expected_direction": annotation.get("human_expected_direction", ""),
            "annotation_human_expected_path_shape": annotation.get("human_expected_path_shape", ""),
            "annotation_human_note": annotation.get("human_note", ""),
            "annotation_predicted_error_reason": annotation.get("predicted_error_reason", ""),
            "annotation_proposed_change": annotation.get("proposed_change", ""),
            "annotation_approved_for_experiment": annotation.get("approved_for_experiment", "false"),
            "human_annotations_are_model_features": False,
        }
        for key, value in feature_values.items():
            row[f"feature_{key}"] = value
        row.update({f"feature_{key}": value for key, value in _flatten_news(news_context).items()})
        for slot in EXPECTED_TIMES:
            row[f"target_actual_return_{slot.replace(':', '')}_from_open_pct"] = actual["actual_returns_from_open_pct"].get(slot)
            row[f"feature_p50_price_{slot.replace(':', '')}"] = predicted_p50_prices.get(slot)
            row[f"feature_main_path_price_{slot.replace(':', '')}"] = predicted_main_prices.get(slot)
            row[f"feature_market_only_path_price_{slot.replace(':', '')}"] = predicted_market_prices.get(slot)
        training_rows.append(row)

    evaluation_path = report_dir / "QQQM_intraday_path_evaluations_latest.csv"
    dataset_path = report_dir / "QQQM_intraday_path_training_dataset_latest.csv"
    readiness_path = report_dir / "QQQM_intraday_path_training_readiness_latest.json"
    _write_csv(evaluation_path, evaluation_rows)
    _write_csv(dataset_path, training_rows)

    readiness = {
        "ticker": "QQQM",
        "archive_dir": str(archive_dir.resolve()),
        "database_path": str(db_path.resolve()),
        "archive_forecasts_seen": forecasts_seen,
        "forecasts_with_completed_actual_session": forecasts_with_actuals,
        "forecasts_waiting_for_actual_data": waiting_for_actuals,
        "eligible_live_prediction_training_rows": len(training_rows),
        "evaluation_rows_including_legacy_forecasts": len(evaluation_rows),
        "excluded_or_not_ready_counts": dict(sorted(excluded_reasons.items())),
        "feature_policy": "使用预测时点已留存的结构化市场特征和新闻特征；图片只存路径供人工复盘，不作为默认模型输入。",
        "label_policy": "实际 15 分钟收盘路径、实际开盘跳空和实际收盘收益；仅 target_date 已完成且具备至少 20 根 bar 并含 09:30/15:45 锚点时标记。",
        "historical_data_policy": "没有点时特征快照的旧预测可参与回测评估，但不进入监督训练数据集。",
        "news_features_ready_for_offline_experiment": len(training_rows) >= 30,
        "model_training_enabled": False,
        "model_promotion_enabled": False,
        "readiness": "collecting_only" if len(training_rows) < 30 else "eligible_for_offline_experiment_only",
        "minimum_prospective_rows_for_first_experiment": 30,
        "warning": "达到 30 条只代表可以开始离线实验；仍需时间顺序验证、基线比较和更多样本，不等于可以自动晋级模型。",
        "dataset_csv": str(dataset_path.resolve()),
        "evaluation_csv": str(evaluation_path.resolve()),
        "human_feedback_template_csv": str(feedback_path.resolve()),
    }
    readiness_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    return readiness


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="把前瞻盘中预测与后续实际走势对齐，生成可审计的学习数据集。")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 行情数据库")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE), help="不可变预测归档目录")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT), help="学习数据集和评估输出目录")
    args = parser.parse_args()
    try:
        readiness = build_dataset(Path(args.db), Path(args.archive_dir), Path(args.report_dir))
    except Exception as exc:
        print(f"生成盘中学习数据集失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("========== QQQM 盘中路径学习数据集审计 ==========")
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    print("说明：此脚本只对齐数据并输出评估/训练候选集，不训练或自动替换任何模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
