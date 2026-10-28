from __future__ import annotations

"""Immutable point-in-time context snapshots linked to saved forecasts.

This module archives context known when a forecast is generated. It does not
train a model. Market features can be considered for training; news features are
stored for later validation but remain disabled by default due limited history;
macro observations are for current explanation only until release/vintage timing
is available.
"""

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.market_context_features import build_market_context_snapshot
from app.news_event_context import build_point_in_time_news_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "market_data.db"
CONTEXT_SCHEMA_VERSION = "market_context_features_v2_news_events"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_iso_date(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是 YYYY-MM-DD 日期字符串")
    text = value.strip()[:10]
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field} 必须是 YYYY-MM-DD 日期字符串：{value!r}") from exc
    return text


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"数据库不存在：{path}")
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_context_archive_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_context_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_feedback_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                captured_at_utc TEXT NOT NULL,
                capture_mode TEXT NOT NULL,
                context_schema_version TEXT NOT NULL,
                feature_vector_json TEXT NOT NULL,
                context_json TEXT NOT NULL,
                training_eligibility_json TEXT NOT NULL,
                context_sha256 TEXT NOT NULL,
                UNIQUE (ticker, as_of_date)
            )
            """
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_context_snapshots_feedback_id
               ON forecast_context_snapshots(forecast_feedback_id)"""
        )
        connection.commit()
    finally:
        connection.close()


def build_context_feature_vector(context: Mapping[str, Any]) -> dict[str, Any]:
    """Build a stable, transparent feature vector from a context snapshot.

    Macro values are intentionally not included in the ML vector because the
    current database does not preserve release timestamps or data vintages.
    News counts are preserved but marked ineligible by default pending enough
    point-in-time history and validation.
    """
    market = context.get("market") if isinstance(context.get("market"), Mapping) else {}
    news = context.get("news") if isinstance(context.get("news"), Mapping) else {}
    keywords = news.get("keyword_counts") if isinstance(news.get("keyword_counts"), Mapping) else {}
    event_context = context.get("news_event_context") if isinstance(context.get("news_event_context"), Mapping) else {}
    bias_counts = event_context.get("nasdaq_bias_counts") if isinstance(event_context.get("nasdaq_bias_counts"), Mapping) else {}

    def safe(value: Any) -> float | int | None:
        number = _as_float(value)
        if number is None:
            return None
        if float(number).is_integer():
            return int(number)
        return number

    return {
        "as_of_date": context.get("as_of_date"),
        "price": safe(context.get("latest_price")),
        "market_return_1_session_pct": safe(market.get("return_1_session_pct")),
        "market_return_5_sessions_pct": safe(market.get("return_5_sessions_pct")),
        "market_return_20_sessions_pct": safe(market.get("return_20_sessions_pct")),
        "market_annualized_volatility_20_session_pct": safe(
            market.get("annualized_volatility_20_session_pct")
        ),
        "market_price_vs_sma20_pct": safe(market.get("price_vs_sma20_pct")),
        "market_sessions_available": safe(market.get("sessions_available")),
        "news_count_30_calendar_days": safe(news.get("news_count_30_calendar_days")),
        "news_hawkish_keyword_count": safe(keywords.get("hawkish_keyword_count")),
        "news_dovish_keyword_count": safe(keywords.get("dovish_keyword_count")),
        "news_inflation_keyword_count": safe(keywords.get("inflation_keyword_count")),
        "news_labor_keyword_count": safe(keywords.get("labor_keyword_count")),
        "news_growth_risk_keyword_count": safe(keywords.get("growth_risk_keyword_count")),
        "news_event_classified_count_30d": safe(event_context.get("included_article_count")),
        "news_event_bullish_count_30d": safe(bias_counts.get("bullish")),
        "news_event_bearish_count_30d": safe(bias_counts.get("bearish")),
        "news_event_neutral_count_30d": safe(bias_counts.get("neutral")),
        "news_event_mixed_count_30d": safe(bias_counts.get("mixed")),
        "news_event_unknown_count_30d": safe(bias_counts.get("unknown")),
        "news_event_mean_impact_score_30d": safe(event_context.get("mean_impact_score")),
        "news_event_mean_impact_score_high_confidence_relevant_30d": safe(
            event_context.get("mean_impact_score_high_confidence_relevant")
        ),
        "news_event_mean_relevance_score_30d": safe(event_context.get("mean_relevance_score")),
        "news_event_mean_confidence_30d": safe(event_context.get("mean_confidence")),
    }


def _training_eligibility(capture_mode: str) -> dict[str, Any]:
    point_in_time_captured = capture_mode == "live_prediction"
    return {
        "market": {
            "eligible_for_candidate_training": True,
            "point_in_time_captured": point_in_time_captured,
            "reason": (
                "由 as_of_date 当日及之前的行情计算；在训练前仍需与预测目标日期严格对齐。"
            ),
        },
        "news": {
            "eligible_for_candidate_training": False,
            "point_in_time_captured": point_in_time_captured,
            "reason": (
                "新闻特征已保存，但目前历史覆盖有限；必须先验证发布时间、抓取时间与预测时刻的可用性，"
                "并积累足够前瞻归档后才允许进入候选模型。"
            ),
        },
        "macro": {
            "eligible_for_candidate_training": False,
            "point_in_time_captured": point_in_time_captured,
            "reason": (
                "宏观表尚无原始发布日期/可用时间和历史数据版本；当前只用于解释，不进入历史训练。"
            ),
        },
        "capture_mode": capture_mode,
        "model_training_enabled_by_this_archive": False,
    }


def archive_forecast_context(
    payload: Mapping[str, Any],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_file: str | Path | None = None,
    capture_mode: str = "live_prediction",
    context_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one immutable context snapshot for a recorded forecast.

    capture_mode='live_prediction' means this function runs immediately in the
    same forecast cycle. 'reconstructed' means a missing snapshot is being
    created for an older forecast; news and macro features remain ineligible.
    Repeated calls for an already archived ticker/date never overwrite it.
    """
    ticker = str(payload.get("ticker") or "QQQM").strip().upper()
    if not ticker:
        raise ValueError("预测 ticker 不能为空")
    as_of_date = _validate_iso_date(payload.get("as_of_date"), "as_of_date")
    forecast_price = _as_float(payload.get("latest_adjusted_close"))
    if forecast_price is None or forecast_price <= 0:
        raise ValueError("latest_adjusted_close 必须是正数")
    if capture_mode not in {"live_prediction", "reconstructed"}:
        raise ValueError("capture_mode 只允许 live_prediction 或 reconstructed")

    initialize_context_archive_db(db_path)
    connection = _connect(db_path)
    try:
        existing = connection.execute(
            "SELECT id, forecast_feedback_id, capture_mode, context_sha256 "
            "FROM forecast_context_snapshots WHERE ticker=? AND as_of_date=?",
            (ticker, as_of_date),
        ).fetchone()
        if existing:
            return {
                "inserted": False,
                "message": "context already archived; original snapshot retained",
                "id": int(existing["id"]),
                "forecast_feedback_id": int(existing["forecast_feedback_id"]),
                "ticker": ticker,
                "as_of_date": as_of_date,
                "capture_mode": existing["capture_mode"],
                "context_sha256": existing["context_sha256"],
            }

        if not _table_exists(connection, "forecast_feedback"):
            raise RuntimeError("数据库中没有 forecast_feedback 表；请先运行每日预测周期。")
        feedback = connection.execute(
            "SELECT id, current_price, status FROM forecast_feedback "
            "WHERE UPPER(ticker)=? AND as_of_date=?",
            (ticker, as_of_date),
        ).fetchone()
        if feedback is None:
            raise RuntimeError(
                f"找不到 {ticker} 在 {as_of_date} 的 forecast_feedback 记录；不能归档未保存的预测。"
            )
        if abs(float(feedback["current_price"]) - forecast_price) > max(0.05, forecast_price * 0.0005):
            raise ValueError("预测 JSON 价格与 forecast_feedback.current_price 不一致，拒绝归档。")
    finally:
        connection.close()

    snapshot = dict(context_snapshot) if context_snapshot is not None else build_market_context_snapshot(
        db_path=db_path,
        ticker=ticker,
        as_of_date=as_of_date,
    )
    snapshot_date = _validate_iso_date(snapshot.get("as_of_date"), "context.as_of_date")
    if snapshot_date != as_of_date:
        raise ValueError(
            f"上下文日期 {snapshot_date} 与预测日期 {as_of_date} 不一致，拒绝归档。"
        )
    context_price = _as_float(snapshot.get("latest_price"))
    if context_price is None or abs(context_price - forecast_price) > max(0.05, forecast_price * 0.0005):
        raise ValueError("上下文价格与预测价格不一致，拒绝归档。")

    # Attach structured news event features that were genuinely available by the
    # forecast generation timestamp. This preserves provenance without enabling
    # these features for model training prematurely.
    snapshot = dict(snapshot)
    forecast_cutoff = payload.get("generated_at_local")
    snapshot["news_event_context"] = build_point_in_time_news_context(
        db_path=db_path,
        forecast_cutoff=str(forecast_cutoff) if forecast_cutoff else None,
        window_days=30,
    )

    features = build_context_feature_vector(snapshot)
    eligibility = _training_eligibility(capture_mode)
    context_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    feature_json = json.dumps(features, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    eligibility_json = json.dumps(eligibility, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(context_json.encode("utf-8")).hexdigest()

    connection = _connect(db_path)
    try:
        # A concurrent process may have inserted the row after the first check.
        insert_cursor = connection.execute(
            """INSERT OR IGNORE INTO forecast_context_snapshots (
                forecast_feedback_id, ticker, as_of_date, captured_at_utc,
                capture_mode, context_schema_version, feature_vector_json,
                context_json, training_eligibility_json, context_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(feedback["id"]), ticker, as_of_date, _utc_now(), capture_mode,
                CONTEXT_SCHEMA_VERSION, feature_json, context_json, eligibility_json, digest,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT id, forecast_feedback_id, capture_mode, context_sha256 "
            "FROM forecast_context_snapshots WHERE ticker=? AND as_of_date=?",
            (ticker, as_of_date),
        ).fetchone()
        if row is None:
            raise RuntimeError("上下文快照写入后仍无法读取。")
        was_inserted = insert_cursor.rowcount == 1
        return {
            "inserted": was_inserted,
            "message": "context archived" if was_inserted else "context already archived; original snapshot retained",
            "id": int(row["id"]),
            "forecast_feedback_id": int(row["forecast_feedback_id"]),
            "ticker": ticker,
            "as_of_date": as_of_date,
            "capture_mode": row["capture_mode"],
            "context_sha256": row["context_sha256"],
            "source_file": str(Path(source_file).resolve()) if source_file else None,
        }
    finally:
        connection.close()


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_context_archive_summary(db_path: str | Path = DEFAULT_DB_PATH, ticker: str = "QQQM") -> dict[str, Any]:
    """Return archive counts and latest dates for monitoring/debugging."""
    if not Path(db_path).exists():
        return {"ticker": ticker.upper(), "snapshots": 0, "latest_as_of_date": None}
    connection = _connect(db_path)
    try:
        if not _table_exists(connection, "forecast_context_snapshots"):
            return {"ticker": ticker.upper(), "snapshots": 0, "latest_as_of_date": None}
        row = connection.execute(
            "SELECT COUNT(*) AS n, MAX(as_of_date) AS latest FROM forecast_context_snapshots WHERE UPPER(ticker)=?",
            (ticker.upper(),),
        ).fetchone()
        modes = connection.execute(
            "SELECT capture_mode, COUNT(*) AS n FROM forecast_context_snapshots WHERE UPPER(ticker)=? GROUP BY capture_mode",
            (ticker.upper(),),
        ).fetchall()
        return {
            "ticker": ticker.upper(),
            "snapshots": int(row["n"]),
            "latest_as_of_date": row["latest"],
            "capture_modes": {item["capture_mode"]: int(item["n"]) for item in modes},
        }
    finally:
        connection.close()
