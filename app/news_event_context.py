from __future__ import annotations

"""Build point-in-time news-event aggregates for a forecast snapshot.

Only event classifications available at the forecast timestamp are included.
This module is for immutable snapshotting and current-context reporting. It does
not enable news features for historical model training by itself.
"""

import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "market_data.db"
FEATURE_VERSION = "news-event-v1"


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # News event published/classified timestamps should be timezone-aware.
        # Naive values are ambiguous and therefore excluded instead of guessed.
        return None
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def build_point_in_time_news_context(
    db_path: str | Path = DEFAULT_DB_PATH,
    forecast_cutoff: str | None = None,
    window_days: int = 30,
    feature_version: str = FEATURE_VERSION,
    max_items: int = 20,
) -> dict[str, Any]:
    """Summarize only news published AND classified by a forecast cutoff.

    forecast_cutoff must be an ISO timestamp with timezone. If unavailable,
    return no usable records rather than accidentally leak later classifications.
    """
    cutoff = _parse_datetime(forecast_cutoff)
    base: dict[str, Any] = {
        "feature_version": feature_version,
        "window_days": int(window_days),
        "forecast_cutoff_utc": cutoff.isoformat(timespec="seconds") if cutoff else None,
        "available": False,
        "included_article_count": 0,
        "event_type_counts": {},
        "nasdaq_bias_counts": {},
        "mean_impact_score": None,
        "mean_relevance_score": None,
        "mean_confidence": None,
        "items": [],
        "excluded_counts": {
            "published_after_cutoff": 0,
            "classified_after_cutoff": 0,
            "outside_window": 0,
            "missing_or_invalid_timestamp": 0,
        },
        "eligible_for_historical_training": False,
        "training_note": (
            "仅保存预测时点可用的结构化新闻摘要；进入机器学习训练前仍需足够的前瞻样本、"
            "时序回测与数据来源审计。"
        ),
    }
    if cutoff is None:
        base["reason"] = "缺少有效且带时区的预测时间，拒绝纳入新闻事件，避免时间泄漏。"
        return base

    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        base["reason"] = f"数据库不存在：{path}"
        return base

    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "news_event_features" not in tables:
            base["reason"] = "尚无 news_event_features 表；本次快照不含结构化新闻。"
            return base

        feature_columns = {
            row[1] for row in connection.execute('PRAGMA table_info("news_event_features")')
        }
        required = {
            "news_id", "feature_version", "source_name", "published_at", "classified_at_utc",
            "event_type", "nasdaq_bias", "impact_score", "relevance_score", "confidence",
            "time_horizon", "facts_summary", "classifier_model",
        }
        if not required.issubset(feature_columns):
            base["reason"] = "news_event_features 表结构不完整，未纳入新闻事件。"
            return base

        if "news_items" in tables:
            news_cols = {
                row[1] for row in connection.execute('PRAGMA table_info("news_items")')
            }
            title_expr = "n.title" if "title" in news_cols else "'' AS title"
            url_expr = "n.url" if "url" in news_cols else "'' AS url"
            sql = f"""
                SELECT f.news_id, f.source_name, f.published_at, f.classified_at_utc,
                       f.classifier_model, f.event_type, f.nasdaq_bias, f.impact_score,
                       f.relevance_score, f.confidence, f.time_horizon, f.facts_summary,
                       {title_expr}, {url_expr}
                FROM news_event_features f
                LEFT JOIN news_items n ON n.news_id = f.news_id
                WHERE f.feature_version = ?
                ORDER BY COALESCE(f.published_at, '') DESC
            """
        else:
            sql = """
                SELECT news_id, source_name, published_at, classified_at_utc,
                       classifier_model, event_type, nasdaq_bias, impact_score,
                       relevance_score, confidence, time_horizon, facts_summary,
                       '' AS title, '' AS url
                FROM news_event_features
                WHERE feature_version = ?
                ORDER BY COALESCE(published_at, '') DESC
            """
        rows = connection.execute(sql, (feature_version,)).fetchall()
    except sqlite3.Error as exc:
        base["reason"] = f"读取新闻事件失败：{type(exc).__name__}: {exc}"
        return base
    finally:
        connection.close()

    window_start = cutoff - timedelta(days=max(1, int(window_days)))
    included: list[dict[str, Any]] = []
    excluded = Counter()
    for row in rows:
        published = _parse_datetime(row["published_at"])
        classified = _parse_datetime(row["classified_at_utc"])
        if published is None or classified is None:
            excluded["missing_or_invalid_timestamp"] += 1
            continue
        if published > cutoff:
            excluded["published_after_cutoff"] += 1
            continue
        if classified > cutoff:
            excluded["classified_after_cutoff"] += 1
            continue
        if published < window_start:
            excluded["outside_window"] += 1
            continue
        included.append({
            "news_id": str(row["news_id"]),
            "title": str(row["title"] or ""),
            "source_name": str(row["source_name"] or ""),
            "published_at": str(row["published_at"] or ""),
            "classified_at_utc": str(row["classified_at_utc"] or ""),
            "classifier_model": str(row["classifier_model"] or ""),
            "event_type": str(row["event_type"] or "unknown"),
            "nasdaq_bias": str(row["nasdaq_bias"] or "unknown"),
            "impact_score": _safe_float(row["impact_score"]),
            "relevance_score": _safe_float(row["relevance_score"]),
            "confidence": _safe_float(row["confidence"]),
            "time_horizon": str(row["time_horizon"] or "unknown"),
            "facts_summary": str(row["facts_summary"] or "")[:500],
            "url": str(row["url"] or ""),
        })

    event_counts = Counter(item["event_type"] for item in included)
    bias_counts = Counter(item["nasdaq_bias"] for item in included)
    impacts = [item["impact_score"] for item in included if item["impact_score"] is not None]
    relevant_impacts = [
        item["impact_score"] for item in included
        if item["impact_score"] is not None
        and (item["relevance_score"] or 0) >= 0.5
        and (item["confidence"] or 0) >= 0.5
    ]
    relevances = [item["relevance_score"] for item in included if item["relevance_score"] is not None]
    confidences = [item["confidence"] for item in included if item["confidence"] is not None]

    base.update({
        "available": True,
        "window_start_utc": window_start.isoformat(timespec="seconds"),
        "included_article_count": len(included),
        "event_type_counts": dict(sorted(event_counts.items())),
        "nasdaq_bias_counts": dict(sorted(bias_counts.items())),
        "mean_impact_score": _mean(impacts),
        "mean_impact_score_high_confidence_relevant": _mean(relevant_impacts),
        "mean_relevance_score": _mean(relevances),
        "mean_confidence": _mean(confidences),
        "items": included[: max(0, int(max_items))],
        "excluded_counts": {key: int(excluded.get(key, 0)) for key in base["excluded_counts"]},
        "reason": "仅纳入预测截止时间之前已发布且已完成分类的新闻；此摘要当前只归档，不自动进入模型训练。",
    })
    return base
