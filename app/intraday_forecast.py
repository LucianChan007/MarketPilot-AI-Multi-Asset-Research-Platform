from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any
from zoneinfo import ZoneInfo

from app.services.market_calendar import market_calendar_label, next_nasdaq_session

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "market_data.db"
REPORT_DIR = ROOT / "reports" / "intraday"
NEW_YORK = ZoneInfo("America/New_York")
EXPECTED_SLOTS = [
    (datetime.combine(date(2000, 1, 1), time(9, 30)) + timedelta(minutes=15 * i)).strftime("%H:%M")
    for i in range(26)
]
STATE_FEATURES = (
    "last_close_return_pct",
    "return_5_sessions_pct",
    "return_20_sessions_pct",
    "volatility_20_sessions_pct",
    "last_intraday_return_pct",
    "last_intraday_range_pct",
    "last_session_coverage_pct",
)


def _parse_date_value(value: Any) -> date:
    """Normalize date, datetime, pandas Timestamp, and ISO date/datetime strings.

    Exchange calendar libraries may return a pandas.Timestamp like
    2026-08-19T00:00:00. Forecast dates are trading-day labels and must be
    normalized to a plain date before comparisons, archives, or filenames.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("预测日期为空，无法判断快照是否在开盘前生成。")
    try:
        # Supports YYYY-MM-DD and ISO datetime strings with timezone/time suffix.
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"无法识别预测日期：{text!r}") from exc


def _quantile(values: list[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return clean[low]
    fraction = position - low
    return clean[low] * (1 - fraction) + clean[high] * fraction


def _next_weekday(d: date) -> date:
    """Compatibility name; now prefers the NASDAQ exchange calendar."""
    return next_nasdaq_session(d)


def load_intraday_sessions(
    db_path: Path = DEFAULT_DB_PATH,
    symbol: str = "QQQM",
    feed: str = "iex",
    timeframe: str = "15Min",
    min_coverage: float = 0.70,
    now_ny: datetime | None = None,
) -> list[dict[str, Any]]:
    """Load completed NY regular sessions; missing bars are never synthesized."""
    if not db_path.exists():
        raise FileNotFoundError(f"找不到行情数据库：{db_path}")
    now_ny = now_ny or datetime.now(timezone.utc).astimezone(NEW_YORK)
    if now_ny.tzinfo is None:
        raise ValueError("now_ny 必须带有时区信息。")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='intraday_bars'"
        ).fetchone()
        if not table:
            raise RuntimeError("未找到 intraday_bars 表。请先同步 Alpaca 15Min 行情。")
        rows = conn.execute(
            """SELECT trade_date_ny, session_time_ny, open, high, low, close, volume
               FROM intraday_bars
               WHERE symbol=? AND feed=? AND timeframe=?
                 AND session_time_ny >= '09:30' AND session_time_ny < '16:00'
               ORDER BY trade_date_ny, session_time_ny""",
            (symbol.upper(), feed.lower(), timeframe),
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["trade_date_ny"])][str(row["session_time_ny"])[:5]] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"] or 0),
        }

    sessions: list[dict[str, Any]] = []
    for session_date, bars in sorted(grouped.items()):
        sd = date.fromisoformat(session_date)
        is_closed = sd < now_ny.date() or (
            sd == now_ny.date() and now_ny.time().replace(tzinfo=None) >= time(16, 15)
        )
        if not is_closed or "09:30" not in bars or "15:45" not in bars:
            continue
        slot_count = sum(1 for slot in EXPECTED_SLOTS if slot in bars)
        coverage = slot_count / len(EXPECTED_SLOTS)
        open_price = bars["09:30"]["open"]
        close_price = bars["15:45"]["close"]
        if open_price <= 0 or close_price <= 0:
            continue
        path_returns = {
            slot: (bar["close"] / open_price - 1.0) * 100.0
            for slot, bar in bars.items()
            if slot in EXPECTED_SLOTS and open_price > 0 and bar["close"] > 0
        }
        all_highs = [v["high"] for v in bars.values()]
        all_lows = [v["low"] for v in bars.values()]
        sessions.append({
            "trade_date": session_date,
            "bars": bars,
            "slot_count": slot_count,
            "coverage_pct": round(coverage * 100.0, 2),
            "open_price": open_price,
            "close_price": close_price,
            "high_price": max(all_highs),
            "low_price": min(all_lows),
            "path_returns_pct": path_returns,
            "path_eligible": coverage >= min_coverage,
            "feed": feed.lower(),
            "timeframe": timeframe,
        })
    return sessions


def choose_intraday_feed(
    db_path: Path = DEFAULT_DB_PATH,
    symbol: str = "QQQM",
    timeframe: str = "15Min",
    expected_latest_date: str | None = None,
    min_training_sessions: int = 25,
) -> tuple[str, list[dict[str, Any]]]:
    """Choose one internally consistent intraday source with enough completed sessions.

    Public Yahoo/EODHD bars are eligible for the research pipeline when a complete
    source-specific history exists. The model never blends different feeds in one
    training/evaluation sequence; each archived forecast carries its source label.
    """
    candidates: list[tuple[tuple[int, int, int], str, list[dict[str, Any]]]] = []
    for feed in ("eodhd", "yahoo", "iex"):
        try:
            sessions = load_intraday_sessions(db_path, symbol, feed, timeframe, min_coverage=0.70)
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            continue
        eligible = [row for row in sessions if row.get("path_eligible")]
        if expected_latest_date and (not eligible or max(str(r["trade_date"]) for r in eligible) != expected_latest_date):
            continue
        if len(eligible) < min_training_sessions:
            continue
        latest = max(str(r["trade_date"]) for r in eligible)
        # Most recent full session first, then longer training history; stable source
        # preference breaks ties without combining price feeds.
        source_rank = {"eodhd": 3, "yahoo": 2, "iex": 1}[feed]
        candidates.append(((int(bool(expected_latest_date and latest == expected_latest_date)), len(eligible), source_rank), feed, sessions))
    if not candidates:
        raise RuntimeError(
            "没有找到与最近完整交易日对齐、且至少包含 25 个完整交易日的单一分时数据源。"
            "请在数据诊断页检查公开分时刷新结果；不要混用不同来源拼接训练集。"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, feed, sessions = candidates[0]
    return feed, sessions


def _overnight_gaps(history: list[dict[str, Any]]) -> list[float]:
    gaps: list[float] = []
    for previous, current in zip(history, history[1:]):
        prev_date = date.fromisoformat(previous["trade_date"])
        cur_date = date.fromisoformat(current["trade_date"])
        calendar_gap = (cur_date - prev_date).days
        if calendar_gap < 1 or calendar_gap > 4:
            continue
        prev_close = float(previous["close_price"])
        cur_open = float(current["open_price"])
        if prev_close > 0 and cur_open > 0:
            gaps.append((cur_open / prev_close - 1.0) * 100.0)
    return gaps


def _build_path_profile(history: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    eligible = [s for s in history if s["path_eligible"]]
    if not eligible:
        return {}
    minimum_slot_samples = max(8, int(math.ceil(len(eligible) * 0.50)))
    result: dict[str, dict[str, float | int]] = {}
    for slot in EXPECTED_SLOTS:
        values = [float(s["path_returns_pct"][slot]) for s in eligible if slot in s["path_returns_pct"]]
        if len(values) < minimum_slot_samples:
            continue
        result[slot] = {
            "p10_return_pct": float(_quantile(values, 0.10)),
            "p50_return_pct": float(_quantile(values, 0.50)),
            "p90_return_pct": float(_quantile(values, 0.90)),
            "sample_count": len(values),
        }
    return result


def _state_features(history: list[dict[str, Any]]) -> list[float]:
    """Features available after the latest completed session; no target-day data."""
    if not history:
        raise ValueError("构造市场状态需要至少一个已完成交易日。")
    closes = [float(s["close_price"]) for s in history if float(s["close_price"]) > 0]
    if not closes:
        raise ValueError("历史收盘价格无效。")
    latest = history[-1]
    last_ret = (closes[-1] / closes[-2] - 1.0) * 100.0 if len(closes) >= 2 else 0.0
    ret5 = (closes[-1] / closes[-6] - 1.0) * 100.0 if len(closes) >= 6 else last_ret
    ret20 = (closes[-1] / closes[-21] - 1.0) * 100.0 if len(closes) >= 21 else ret5
    daily_returns = [((b / a) - 1.0) * 100.0 for a, b in zip(closes[-21:-1], closes[-20:]) if a > 0]
    vol = pstdev(daily_returns) * math.sqrt(252) if len(daily_returns) >= 2 else 0.0
    intraday_ret = (float(latest["close_price"]) / float(latest["open_price"]) - 1.0) * 100.0
    intraday_range = (float(latest["high_price"]) - float(latest["low_price"])) / float(latest["open_price"]) * 100.0
    return [last_ret, ret5, ret20, vol, intraday_ret, intraday_range, float(latest["coverage_pct"])]


def _training_examples(history: list[dict[str, Any]], min_training_sessions: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx, target in enumerate(history):
        if idx < min_training_sessions or not target["path_eligible"]:
            continue
        prior = history[:idx]
        try:
            features = _state_features(prior)
        except ValueError:
            continue
        prior_close = float(prior[-1]["close_price"])
        gap = (float(target["open_price"]) / prior_close - 1.0) * 100.0 if prior_close > 0 else None
        examples.append({"features": features, "target": target, "gap_pct": gap})
    return examples


def _knn_weights(training_x: list[list[float]], query_x: list[float], k: int | None = None) -> list[tuple[int, float]]:
    if not training_x:
        return []
    n_features = len(query_x)
    means = [mean(row[j] for row in training_x) for j in range(n_features)]
    scales = []
    for j in range(n_features):
        sd = pstdev(row[j] for row in training_x)
        scales.append(sd if sd > 1e-9 else 1.0)
    distances: list[tuple[int, float]] = []
    for idx, row in enumerate(training_x):
        distance = math.sqrt(sum(((row[j] - query_x[j]) / scales[j]) ** 2 for j in range(n_features)) / n_features)
        distances.append((idx, distance))
    n = len(distances)
    count = k or max(8, min(15, int(math.ceil(math.sqrt(n) * 2))))
    count = min(count, n)
    nearest = sorted(distances, key=lambda x: x[1])[:count]
    return [(idx, 1.0 / (distance + 0.15)) for idx, distance in nearest]


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    total_weight = sum(w for _, w in values)
    if total_weight <= 0:
        return mean(v for v, _ in values)
    return sum(v * w for v, w in values) / total_weight


def _conditional_profile(history: list[dict[str, Any]], min_training_sessions: int) -> tuple[dict[str, dict[str, float | int]], list[dict[str, Any]], list[tuple[float, float]]]:
    examples = _training_examples(history, min_training_sessions)
    if len(examples) < 12:
        raise RuntimeError(f"相似历史样本只有 {len(examples)} 条，至少需要 12 条；当前仅输出历史基线。")
    query = _state_features(history)
    weights = _knn_weights([e["features"] for e in examples], query)
    ranked = [{**examples[idx], "weight": weight} for idx, weight in weights]
    profile: dict[str, dict[str, float | int]] = {}
    for slot in EXPECTED_SLOTS:
        slot_neighbors = [e for e in ranked if slot in e["target"]["path_returns_pct"]]
        if len(slot_neighbors) < 6:
            continue
        vals = [float(e["target"]["path_returns_pct"][slot]) for e in slot_neighbors]
        weighted = _weighted_mean([(float(e["target"]["path_returns_pct"][slot]), float(e["weight"])) for e in slot_neighbors])
        profile[slot] = {
            "p10_return_pct": float(_quantile(vals, 0.10)),
            # Keep the percentile label statistically literal: P50 is the median.
            # The neighbour-weighted mean is saved separately for diagnostics only.
            "p50_return_pct": float(_quantile(vals, 0.50)),
            "weighted_mean_return_pct": float(weighted if weighted is not None else _quantile(vals, 0.50)),
            "p90_return_pct": float(_quantile(vals, 0.90)),
            "sample_count": len(slot_neighbors),
        }
    gap_values = [(float(e["gap_pct"]), float(e["weight"])) for e in ranked if e["gap_pct"] is not None]
    if not profile:
        raise RuntimeError("相似历史行情无法构建足够的盘中时间点。")
    return profile, ranked, gap_values



def _news_direction_signal(news_context: dict[str, Any] | None) -> dict[str, Any]:
    """Create a cautious, explicitly experimental direction signal from point-in-time classified news."""
    context = news_context if isinstance(news_context, dict) else {}
    items = context.get("items", [])
    if not isinstance(items, list):
        items = []
    qualified: list[tuple[float, float]] = []
    direction_counts = {"bullish": 0, "bearish": 0}
    for item in items:
        if not isinstance(item, dict):
            continue
        bias = str(item.get("nasdaq_bias", "unknown")).lower()
        if bias not in {"bullish", "bearish"}:
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
            relevance = float(item.get("relevance_score") or 0.0)
            impact = abs(float(item.get("impact_score") or 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < 0.55 or relevance < 0.55:
            continue
        # If the classifier emitted a direction but zero impact, retain only a small directional weight.
        magnitude = max(min(impact / 2.0, 1.0), 0.25)
        direction = 1.0 if bias == "bullish" else -1.0
        weight = confidence * relevance
        qualified.append((direction * magnitude, weight))
        direction_counts[bias] += 1
    total_weight = sum(weight for _, weight in qualified)
    signal = sum(value * weight for value, weight in qualified) / total_weight if total_weight > 0 else 0.0
    mean_confidence = (
        sum(float(item.get("confidence") or 0.0) for item in items if isinstance(item, dict)) / len(items)
        if items else 0.0
    )
    # Require at least two classified, relevant and reasonably confident directional articles.
    # One headline must not swing the entire intraday curve.
    enough_evidence = len(qualified) >= 2 and abs(signal) >= 0.15
    return {
        "signal": round(max(-1.0, min(1.0, signal)), 4),
        "qualified_directional_articles": len(qualified),
        "direction_counts": direction_counts,
        "mean_confidence_all_included": round(mean_confidence, 4),
        "adjustment_eligible": bool(enough_evidence),
        "method": "heuristic_directional_news_signal_v1",
        "note": (
            "仅是实验性候选路径重加权信号，不是收益概率或因果结论；至少需要两条高相关、高置信方向性新闻。"
        ),
    }


def _candidate_records(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for entry in ranked:
        target = entry.get("target") or {}
        path = target.get("path_returns_pct") or {}
        valid_path = {
            str(slot): round(float(value), 6)
            for slot, value in path.items()
            if str(slot) in EXPECTED_SLOTS and value is not None and math.isfinite(float(value))
        }
        if len(valid_path) < 20:
            continue
        weight = max(float(entry.get("weight", 0.0)), 0.0)
        gap = entry.get("gap_pct")
        if gap is None:
            continue
        final_return = valid_path.get("15:45")
        if final_return is None:
            continue
        raw.append({
            "trade_date": str(target.get("trade_date", "")),
            "similarity_weight": weight,
            "normalized_similarity_weight": 0.0,
            "gap_pct": round(float(gap), 6),
            "close_return_from_open_pct": round(float(final_return), 6),
            "path_returns_pct": valid_path,
            "coverage_pct": target.get("coverage_pct"),
            "observed_slots": len(valid_path),
        })
    total = sum(item["similarity_weight"] for item in raw)
    for item in raw:
        item["normalized_similarity_weight"] = round(
            item["similarity_weight"] / total if total > 0 else 1.0 / max(len(raw), 1), 8
        )
    return raw


def _path_rmse_pct(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa = a.get("path_returns_pct", {})
    pb = b.get("path_returns_pct", {})
    common = [slot for slot in EXPECTED_SLOTS if slot in pa and slot in pb]
    if len(common) < 15:
        return float("inf")
    return math.sqrt(sum((float(pa[slot]) - float(pb[slot])) ** 2 for slot in common) / len(common))


def _select_cluster_path(
    candidates: list[dict[str, Any]],
    previous_close: float,
    weight_key: str = "similarity_weight",
) -> dict[str, Any] | None:
    """Select a medoid from the highest-weight cluster of candidate complete paths.

    The cluster mass is an empirical weight share, not a calibrated probability.
    """
    if not candidates or previous_close <= 0:
        return None
    if len(candidates) == 1:
        distances = [[0.0]]
    else:
        distances = [[0.0 for _ in candidates] for _ in candidates]
        pairwise: list[float] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                distance = _path_rmse_pct(candidates[i], candidates[j])
                if not math.isfinite(distance):
                    distance = 1e6
                distances[i][j] = distance
                distances[j][i] = distance
                pairwise.append(distance)
        radius = max(0.08, float(_quantile(pairwise, 0.25) or 0.08))
    if len(candidates) == 1:
        radius = 0.08
    total_weight = sum(max(float(c.get(weight_key, 0.0)), 0.0) for c in candidates)
    if total_weight <= 0:
        effective_weights = [1.0 for _ in candidates]
        total_weight = float(len(candidates))
    else:
        effective_weights = [max(float(c.get(weight_key, 0.0)), 0.0) for c in candidates]

    clusters: list[tuple[float, int, float, list[int]]] = []
    for center in range(len(candidates)):
        members = [j for j in range(len(candidates)) if distances[center][j] <= radius]
        cluster_weight = sum(effective_weights[j] for j in members)
        mean_distance = (
            sum(distances[center][j] for j in members) / len(members) if members else float("inf")
        )
        clusters.append((cluster_weight, len(members), -mean_distance, members))
    best_cluster = max(clusters, key=lambda item: (item[0], item[1], item[2]))
    members = best_cluster[3]
    medoid = min(
        members,
        key=lambda i: sum(effective_weights[j] * distances[i][j] for j in members),
    )
    selected = candidates[medoid]
    predicted_open = previous_close * (1.0 + float(selected["gap_pct"]) / 100.0)
    path: list[dict[str, Any]] = []
    for slot in EXPECTED_SLOTS:
        if slot not in selected["path_returns_pct"]:
            continue
        ret = float(selected["path_returns_pct"][slot])
        path.append({
            "time_ny": slot,
            "return_from_open_pct": round(ret, 5),
            "predicted_price": round(predicted_open * (1.0 + ret / 100.0), 4),
        })
    member_weight = sum(effective_weights[j] for j in members)
    cluster_share = member_weight / total_weight if total_weight > 0 else 0.0
    last_price = path[-1]["predicted_price"] if path else predicted_open
    return {
        "selected_neighbor_date": selected["trade_date"],
        "selection_method": "weighted_medoid_of_highest_mass_path_cluster",
        "cluster_member_dates": [candidates[j]["trade_date"] for j in members],
        "cluster_member_count": len(members),
        "candidate_count": len(candidates),
        "cluster_weight_share_not_calibrated_probability": round(cluster_share, 6),
        "cluster_radius_rmse_pct_points": round(radius, 6),
        "predicted_open_price": round(predicted_open, 4),
        "predicted_close_price": round(last_price, 4),
        "selected_neighbor_gap_pct": float(selected["gap_pct"]),
        "selected_neighbor_close_return_from_open_pct": float(selected["close_return_from_open_pct"]),
        "path": path,
        "price_by_time": {point["time_ny"]: point["predicted_price"] for point in path},
        "explanation": (
            "选择相似历史候选中权重总量最高的路径簇，再取簇内加权 medoid 作为一条完整候选路径；"
            "簇权重占比不是经校准的未来发生概率。"
        ),
    }


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Small, dependency-free Gaussian elimination with partial pivoting."""
    n = len(vector)
    if n == 0 or len(matrix) != n:
        return None
    augmented = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def _ridge_predict(
    rows_x: list[list[float]], rows_y: list[float], query_x: list[float], alpha: float = 8.0
) -> float | None:
    """Standardized ridge regression with unpenalized intercept; returns one horizon estimate."""
    pairs = [(list(map(float, x)), float(y)) for x, y in zip(rows_x, rows_y)
             if len(x) == len(query_x) and all(math.isfinite(float(v)) for v in x) and math.isfinite(float(y))]
    if len(pairs) < max(12, len(query_x) + 3):
        return None
    width = len(query_x)
    means = [mean(pair[0][j] for pair in pairs) for j in range(width)]
    scales = [pstdev(pair[0][j] for pair in pairs) for j in range(width)]
    scales = [scale if scale > 1e-8 else 1.0 for scale in scales]
    design = [[1.0] + [(x[j] - means[j]) / scales[j] for j in range(width)] for x, _ in pairs]
    ys = [y for _, y in pairs]
    p = width + 1
    gram = [[0.0 for _ in range(p)] for _ in range(p)]
    rhs = [0.0 for _ in range(p)]
    for row, y in zip(design, ys):
        for i in range(p):
            rhs[i] += row[i] * y
            for j in range(p):
                gram[i][j] += row[i] * row[j]
    # Penalize slopes, not intercept. Strong regularization is deliberate for this small sample.
    for i in range(1, p):
        gram[i][i] += max(float(alpha), 0.0)
    beta = _solve_linear_system(gram, rhs)
    if beta is None:
        return None
    query = [1.0] + [(float(query_x[j]) - means[j]) / scales[j] for j in range(width)]
    prediction = sum(beta[i] * query[i] for i in range(p))
    if not math.isfinite(prediction):
        return None
    # Guard against wildly extrapolated predictions in a short history.
    y_mean = mean(ys)
    y_sd = pstdev(ys) if len(ys) > 1 else 0.0
    if y_sd > 1e-9:
        prediction = max(y_mean - 3.0 * y_sd, min(y_mean + 3.0 * y_sd, prediction))
    return float(prediction)


def _build_ridge_training_examples(
    history: list[dict[str, Any]], min_training_sessions: int
) -> list[dict[str, Any]]:
    """Create strict chronological examples: features from prior sessions, targets from next session."""
    examples: list[dict[str, Any]] = []
    for idx, target in enumerate(history):
        if not target.get("path_eligible"):
            continue
        prior = history[:idx]
        prior_eligible_count = sum(1 for item in prior if item.get("path_eligible"))
        if prior_eligible_count < min_training_sessions or not prior:
            continue
        previous_close = float(prior[-1].get("close_price") or 0.0)
        target_open = float(target.get("open_price") or 0.0)
        if previous_close <= 0 or target_open <= 0:
            continue
        try:
            features = _state_features(prior)
        except (ValueError, TypeError):
            continue
        examples.append({
            "trade_date": str(target.get("trade_date", "")),
            "features": features,
            "gap_pct": (target_open / previous_close - 1.0) * 100.0,
            "path_returns_pct": dict(target.get("path_returns_pct") or {}),
        })
    return examples


def _build_ridge_main_path(
    history: list[dict[str, Any]], min_training_sessions: int = 20, alpha: float = 8.0
) -> dict[str, Any] | None:
    """Predict an independent path from expanding-window market features using per-slot Ridge models.

    No P10/P50/P90 line is used as this model's target. Historical news is intentionally not
    a training feature because complete point-in-time historical news labels are not available.
    Current-news overlay is handled separately and explicitly marked experimental.
    """
    if not history:
        return None
    examples = _build_ridge_training_examples(history, min_training_sessions)
    minimum = max(12, len(STATE_FEATURES) + 3)
    if len(examples) < minimum:
        return None
    try:
        query_x = _state_features(history)
    except (ValueError, TypeError):
        return None
    x_gap = [e["features"] for e in examples]
    y_gap = [float(e["gap_pct"]) for e in examples]
    gap_prediction = _ridge_predict(x_gap, y_gap, query_x, alpha=alpha)
    if gap_prediction is None:
        return None
    previous_close = float(history[-1].get("close_price") or 0.0)
    if previous_close <= 0:
        return None
    predicted_open = previous_close * (1.0 + gap_prediction / 100.0)
    path: list[dict[str, Any]] = []
    per_slot_samples: dict[str, int] = {}
    for slot in EXPECTED_SLOTS:
        selected = [(e["features"], e["path_returns_pct"].get(slot)) for e in examples]
        selected = [(x, float(y)) for x, y in selected if y is not None and math.isfinite(float(y))]
        per_slot_samples[slot] = len(selected)
        estimate = _ridge_predict([x for x, _ in selected], [y for _, y in selected], query_x, alpha=alpha)
        if estimate is None:
            continue
        price = predicted_open * (1.0 + estimate / 100.0)
        if not math.isfinite(price) or price <= 0:
            continue
        path.append({
            "time_ny": slot,
            "return_from_open_pct": round(estimate, 6),
            "predicted_price": round(price, 4),
        })
    if len(path) < 20:
        return None
    price_by_time = {point["time_ny"]: point["predicted_price"] for point in path}
    final_point = next((point for point in reversed(path) if point["time_ny"] == "15:45"), path[-1])
    return {
        "model_name": "main-path-ridge-market-v1",
        "model_family": "ridge_regression_per_time_slot",
        "selection_method": "expanding_window_ridge_regression_per_intraday_horizon",
        "independent_of_quantile_curves": True,
        "selected_neighbor_date": None,
        "candidate_count": len(examples),
        "training_rows": len(examples),
        "feature_names": list(STATE_FEATURES),
        "ridge_alpha": float(alpha),
        "predicted_open_price": round(predicted_open, 4),
        "predicted_open_gap_pct": round(gap_prediction, 6),
        "predicted_close_price": round(float(final_point["predicted_price"]), 4),
        "path": path,
        "price_by_time": price_by_time,
        "per_slot_training_rows": per_slot_samples,
        "news_adjustment_applied": False,
        "news_training_note": (
            "本次回归模型使用过往市场特征训练；由于缺少覆盖足够历史交易日的点时新闻训练标签，"
            "新闻不会被伪装成已训练的模型特征。当前新闻仅可能通过单独标记的实验性叠加影响路径。"
        ),
        "explanation": (
            "此主路径由历史市场特征对开盘跳空和各盘中时间点收益分别进行岭回归预测；"
            "不由 P10/P50/P90 拼接产生，也不保证为校准后的最高概率路径。"
        ),
    }


def _apply_experimental_news_overlay(
    base_path: dict[str, Any], signal: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], bool]:
    """Apply a deliberately small transparent overlay; this is not a learned news coefficient."""
    out = json.loads(json.dumps(base_path))
    eligible = bool(signal.get("adjustment_eligible"))
    strength = float(signal.get("signal") or 0.0)
    if not eligible or abs(strength) < 0.15:
        out["news_adjustment_applied"] = False
        out["news_signal"] = signal
        out["news_adjustment_note"] = "新闻方向性证据不足，主路径保持市场特征模型原始输出。"
        return out, False
    closes = [float(c.get("close_return_from_open_pct", 0.0)) for c in candidates]
    gaps = [float(c.get("gap_pct", 0.0)) for c in candidates]
    close_iqr = max(0.0, float((_quantile(closes, 0.75) or 0.0) - (_quantile(closes, 0.25) or 0.0)))
    gap_iqr = max(0.0, float((_quantile(gaps, 0.75) or 0.0) - (_quantile(gaps, 0.25) or 0.0)))
    # Capped, transparent experimental shifts. Values are percentage points, not probabilities.
    close_shift = math.copysign(min(0.25, max(0.03, close_iqr * 0.12)) * abs(strength), strength)
    gap_shift = math.copysign(min(0.10, max(0.01, gap_iqr * 0.06)) * abs(strength), strength)
    previous_close = float(base_path["predicted_open_price"]) / (1.0 + float(base_path.get("predicted_open_gap_pct", 0.0)) / 100.0)
    new_gap = float(base_path.get("predicted_open_gap_pct", 0.0)) + gap_shift
    new_open = previous_close * (1.0 + new_gap / 100.0)
    points = base_path.get("path", [])
    count = max(len(points), 1)
    adjusted_path: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        progress = (idx + 1) / count
        base_return = float(point["return_from_open_pct"])
        adjusted_return = base_return + close_shift * progress
        adjusted_path.append({
            **point,
            "return_from_open_pct": round(adjusted_return, 6),
            "predicted_price": round(new_open * (1.0 + adjusted_return / 100.0), 4),
            "news_overlay_adjustment_pct_points": round(close_shift * progress, 6),
        })
    by_time = {point["time_ny"]: point["predicted_price"] for point in adjusted_path}
    close_price = next((p["predicted_price"] for p in reversed(adjusted_path) if p["time_ny"] == "15:45"), adjusted_path[-1]["predicted_price"])
    out.update({
        "model_name": "main-path-ridge-market-plus-news-overlay-v1",
        "selection_method": "ridge_market_path_with_experimental_point_in_time_news_overlay",
        "predicted_open_price": round(new_open, 4),
        "predicted_open_gap_pct": round(new_gap, 6),
        "predicted_close_price": round(float(close_price), 4),
        "path": adjusted_path,
        "price_by_time": by_time,
        "news_adjustment_applied": True,
        "news_signal": signal,
        "news_adjustment_method": "capped_heuristic_overlay_not_trained_coefficient",
        "news_adjustment_close_shift_pct_points": round(close_shift, 6),
        "news_adjustment_open_gap_shift_pct_points": round(gap_shift, 6),
        "news_adjustment_note": (
            "新闻调整为透明、封顶的实验性叠加；历史训练阶段未估计新闻系数，必须通过未来逐日样本验证是否有帮助。"
        ),
    })
    return out, True


def _attach_path_selection(
    forecast: dict[str, Any], news_context: dict[str, Any] | None = None
) -> None:
    candidates = forecast.get("candidate_paths", [])
    previous_close = float(forecast.get("previous_session_close") or 0.0)
    if not candidates or previous_close <= 0:
        forecast["historical_cluster_path"] = None
        forecast["market_only_main_path"] = forecast.get("ml_market_path")
        forecast["main_path"] = forecast.get("ml_market_path")
        forecast["path_selection_diagnostics"] = {
            "available": bool(forecast.get("ml_market_path")),
            "selection_version": "ridge-main-path-news-shadow-v2",
            "reason": "没有足够的完整相似历史候选路径；如 Ridge 路径可用则仍展示市场特征模型。",
            "main_path_model": (forecast.get("main_path") or {}).get("model_name"),
            "news_adjustment_applied": False,
        }
        return

    # Preserve the old cluster medoid as a baseline/diagnostic, not as the default main line.
    historical_market_path = _select_cluster_path(candidates, previous_close, "similarity_weight")
    signal = _news_direction_signal(news_context)

    # Save current heuristic cluster weights as an additional shadow comparison only.
    adjusted = [dict(c) for c in candidates]
    values = [float(c.get("close_return_from_open_pct", 0.0)) for c in candidates]
    gaps = [float(c.get("gap_pct", 0.0)) for c in candidates]
    ret_mean = mean(values) if values else 0.0
    gap_mean = mean(gaps) if gaps else 0.0
    ret_scale = pstdev(values) if len(values) > 1 else 0.0
    gap_scale = pstdev(gaps) if len(gaps) > 1 else 0.0
    news_strength = 0.55
    for candidate in adjusted:
        ret_z = (float(candidate["close_return_from_open_pct"]) - ret_mean) / ret_scale if ret_scale > 1e-9 else 0.0
        gap_z = (float(candidate["gap_pct"]) - gap_mean) / gap_scale if gap_scale > 1e-9 else 0.0
        combined_z = max(-2.0, min(2.0, 0.70 * ret_z + 0.30 * gap_z))
        factor = math.exp(news_strength * float(signal["signal"]) * combined_z) if signal["adjustment_eligible"] else 1.0
        candidate["news_adjustment_factor"] = round(factor, 8)
        candidate["news_adjusted_weight"] = float(candidate.get("similarity_weight", 0.0)) * factor
    adjusted_total = sum(max(float(c.get("news_adjusted_weight", 0.0)), 0.0) for c in adjusted)
    adjusted_by_date = {c["trade_date"]: c for c in adjusted}
    for candidate in candidates:
        match = adjusted_by_date.get(candidate.get("trade_date"), {})
        candidate["news_adjustment_factor"] = match.get("news_adjustment_factor", 1.0)
        candidate["news_adjusted_weight"] = match.get("news_adjusted_weight", candidate.get("similarity_weight", 0.0))
        candidate["news_adjusted_normalized_weight"] = round(
            float(candidate["news_adjusted_weight"]) / adjusted_total if adjusted_total > 0 else 0.0, 8
        )
    news_cluster_path = _select_cluster_path(adjusted, previous_close, "news_adjusted_weight")
    if news_cluster_path:
        news_cluster_path["news_signal"] = signal
        news_cluster_path["news_adjustment_applied"] = bool(signal["adjustment_eligible"])

    market_model_path = forecast.get("ml_market_path")
    if not isinstance(market_model_path, dict) or not market_model_path.get("price_by_time"):
        market_model_path = historical_market_path
        fallback_reason = "Ridge 主路径训练样本不足，回退到历史代表路径；这是兼容回退，不是独立 ML 路径。"
    else:
        fallback_reason = None
    if market_model_path is None:
        forecast["historical_cluster_path"] = historical_market_path
        forecast["market_only_main_path"] = None
        forecast["main_path"] = None
        forecast["path_selection_diagnostics"] = {
            "available": False,
            "selection_version": "ridge-main-path-news-shadow-v2",
            "reason": "没有可用的独立模型路径或历史路径回退。",
        }
        return

    news_path, news_applied = _apply_experimental_news_overlay(market_model_path, signal, candidates)
    # The UI's main path is model-generated, independent of quantile lines. News overlay is explicit and gated.
    forecast["historical_cluster_path"] = historical_market_path
    forecast["news_adjusted_cluster_shadow_path"] = news_cluster_path
    forecast["market_only_main_path"] = market_model_path
    forecast["main_path"] = news_path
    forecast["path_selection_diagnostics"] = {
        "available": True,
        "selection_version": "ridge-main-path-news-shadow-v2",
        "main_path_model": (news_path or {}).get("model_name"),
        "main_path_source": "ridge_market_features" if not news_applied else "ridge_market_features_plus_experimental_news_overlay",
        "independent_of_quantile_curves": bool((news_path or {}).get("independent_of_quantile_curves", False)),
        "news_adjustment_applied": news_applied,
        "news_signal": signal,
        "market_model_training_rows": (market_model_path or {}).get("training_rows"),
        "historical_cluster_neighbor_date": (historical_market_path or {}).get("selected_neighbor_date"),
        "historical_cluster_size": (historical_market_path or {}).get("cluster_member_count"),
        "news_cluster_shadow_neighbor_date": (news_cluster_path or {}).get("selected_neighbor_date"),
        "fallback_reason": fallback_reason,
        "warning": (
            "主预测路径由市场特征岭回归独立生成；当前历史样本较少，新闻仅在高相关、高置信方向性证据足够时进行封顶实验叠加。"
            "该新闻系数尚未通过历史点时样本训练，必须与无新闻路径做前瞻比较；主路径也不是经过校准的最高概率路径。"
        ),
    }

def _build_point_in_time_news_context(db_path: Path, cutoff_utc: str) -> dict[str, Any]:
    try:
        from app.news_event_context import build_point_in_time_news_context
        return build_point_in_time_news_context(
            db_path=db_path,
            forecast_cutoff=cutoff_utc,
            window_days=30,
            max_items=100,
        )
    except Exception as exc:
        return {
            "available": False,
            "included_article_count": 0,
            "items": [],
            "eligible_for_historical_training": False,
            "reason": f"结构化新闻上下文不可用，未影响候选路径：{type(exc).__name__}: {exc}",
        }


def _write_immutable_forecast_archive(forecast: dict[str, Any], report_dir: Path) -> Path | None:
    raw_target_date = forecast.get("forecast_date_weekday_estimate")
    if not raw_target_date or not forecast.get("forecast_snapshot_version"):
        return None
    target_date = _parse_date_value(raw_target_date).isoformat()
    archive_dir = report_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"QQQM_intraday_forecast_{target_date}.json"
    if archive_path.exists():
        # Keep the first prospective snapshot for an exact target/as-of/feed/model version.
        # If an archive with this target belongs to another version or data vintage, retain
        # it and write a separate timestamped snapshot rather than silently mixing them.
        try:
            existing = json.loads(archive_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        exact_same_snapshot = bool(
            existing.get("forecast_snapshot_version") == forecast.get("forecast_snapshot_version")
            and str(existing.get("as_of_date") or "") == str(forecast.get("as_of_date") or "")
            and str(existing.get("forecast_date_weekday_estimate") or "") == target_date
            and str(existing.get("feed") or "").lower() == str(forecast.get("feed") or "").lower()
            and bool(existing.get("pre_session_snapshot"))
        )
        if exact_same_snapshot:
            return archive_path
        generated = str(forecast.get("generated_at_utc", "")).replace(":", "").replace("-", "")
        generated = generated.replace("T", "_").replace("Z", "")[:15] or "new"
        archive_path = archive_dir / f"QQQM_intraday_forecast_{target_date}_live_{generated}.json"
        if not archive_path.exists():
            archive_path.write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        archive_path.write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
    return archive_path


def _build_forecast_from_profile(
    history: list[dict[str, Any]], profile: dict[str, dict[str, float | int]], gaps: list[float],
    method: str, min_training_sessions: int,
) -> dict[str, Any]:
    if not history:
        raise ValueError("没有可用的已完成盘中交易日。")
    eligible = [s for s in history if s["path_eligible"]]
    if len(eligible) < min_training_sessions:
        raise ValueError(f"有效交易日只有 {len(eligible)} 天，至少需要 {min_training_sessions} 天。")
    latest = max(history, key=lambda s: s["trade_date"])
    gap_vals = gaps[-min(60, len(gaps)):]
    if not gap_vals:
        gap_vals = [0.0]
    open_low_pct = float(_quantile(gap_vals, 0.10) or 0.0)
    open_mid_pct = float(_quantile(gap_vals, 0.50) or 0.0)
    open_high_pct = float(_quantile(gap_vals, 0.90) or 0.0)
    if method == "conditional_neighbors" and gaps:
        # Nearest-neighbour weighted mean is the center estimate; quantiles remain empirical scenarios.
        gap_mid = float(_weighted_mean([(g, 1.0 / (abs(g - open_mid_pct) + 0.15)) for g in gaps]) or open_mid_pct)
        open_mid_pct = gap_mid
    previous_close = float(latest["close_price"])
    open_low = previous_close * (1.0 + open_low_pct / 100.0)
    open_mid = previous_close * (1.0 + open_mid_pct / 100.0)
    open_high = previous_close * (1.0 + open_high_pct / 100.0)

    curve: list[dict[str, Any]] = []
    for slot in EXPECTED_SLOTS:
        point = profile.get(slot)
        if not point:
            continue
        low_ret = float(point["p10_return_pct"])
        mid_ret = float(point["p50_return_pct"])
        high_ret = float(point["p90_return_pct"])
        curve.append({
            "time_ny": slot,
            "time_hkt": None,
            "p10_return_from_open_pct": round(low_ret, 4),
            "p50_return_from_open_pct": round(mid_ret, 4),
            "p90_return_from_open_pct": round(high_ret, 4),
            "low_price": round(open_low * (1 + low_ret / 100), 4),
            "median_price": round(open_mid * (1 + mid_ret / 100), 4),
            "high_price": round(open_high * (1 + high_ret / 100), 4),
            "historical_slot_samples": int(point["sample_count"]),
        })
    if not curve:
        raise RuntimeError("无法构建盘中路径：可用时间槽样本不足。")
    latest_date = _parse_date_value(latest["trade_date"])
    # Normalize any pandas Timestamp returned by an exchange calendar to date.
    forecast_date = _parse_date_value(_next_weekday(latest_date))
    return {
        "ticker": "QQQM", "feed": latest["feed"], "timeframe": latest["timeframe"],
        "as_of_date": latest["trade_date"],
        "forecast_date_weekday_estimate": forecast_date.isoformat(),
        "market_calendar_source": market_calendar_label(),
        "previous_session_close": round(previous_close, 4),
        "estimated_open": {
            "p10_price": round(open_low, 4), "p50_price": round(open_mid, 4), "p90_price": round(open_high, 4),
            "p10_gap_pct": round(open_low_pct, 4), "p50_gap_pct": round(open_mid_pct, 4), "p90_gap_pct": round(open_high_pct, 4),
        },
        "estimated_close_from_path": {"p10_price": curve[-1]["low_price"], "p50_price": curve[-1]["median_price"], "p90_price": curve[-1]["high_price"]},
        "eligible_training_sessions": len(eligible), "path_training_sessions": len(eligible),
        "curve_method": method,
        "profile_method": (
            "按此前市场状态选择相似历史交易日，使用邻近日子的盘中累计收益形成条件路径；实验性候选模型。"
            if method == "conditional_neighbors" else
            "历史同一时段累计收益的经验 P10/P50/P90；开盘采用历史隔夜跳空经验分位。"
        ),
        "curve": curve,
        "caveats": [
            "曲线是历史数据估计，不是对未来每根K线的确定性预测。",
            "Alpaca IEX 只覆盖单一交易所；缺失的 15 分钟 bar 未被补造。",
            "P10/P50/P90 是历史经验情景，不是校准后的未来概率或保证。",
            f"预测日期日历来源：{market_calendar_label()}。若仅按周末估算，遇美股节假日可能不准确。",
            "当前历史数据约 60 个交易日，候选模型可能不稳定；未证明优于简单基线。",
            "开盘和盘中曲线未计交易费用，不构成买卖信号。",
        ],
    }


def predict_intraday_path(
    history: list[dict[str, Any]], min_training_sessions: int = 20, method: str = "historical_profile"
) -> dict[str, Any]:
    """Create either a historical-profile baseline or similar-session conditional candidate."""
    if min_training_sessions < 10:
        raise ValueError("min_training_sessions 至少为 10。")
    if not history:
        raise ValueError("没有可用的已完成盘中交易日。")
    eligible = [s for s in history if s["path_eligible"]]
    if len(eligible) < min_training_sessions:
        raise ValueError(f"有效完整度交易日只有 {len(eligible)} 天，至少需要 {min_training_sessions} 天才能生成预测。")

    if method == "historical_profile":
        profile = _build_path_profile(eligible)
        gaps = _overnight_gaps(history)[-min(60, len(history)):]
        forecast = _build_forecast_from_profile(history, profile, gaps, method, min_training_sessions)
        forecast["ml_market_path"] = _build_ridge_main_path(history, min_training_sessions=min_training_sessions)
        _attach_path_selection(forecast, None)
        return forecast
    if method == "conditional_neighbors":
        profile, ranked, weighted_gaps = _conditional_profile(history, min_training_sessions)
        # Use a distribution of neighbour gaps for P10/P90; model's weighted center is handled below.
        forecast = _build_forecast_from_profile(history, profile, [v for v, _ in weighted_gaps], method, min_training_sessions)
        weighted_gap_mid = _weighted_mean(weighted_gaps)
        # P50 remains the empirical median; weighted neighbour centre is retained as a separate diagnostic.
        if weighted_gap_mid is not None:
            forecast["estimated_open"]["similarity_weighted_mean_gap_pct"] = round(weighted_gap_mid, 4)
            forecast["estimated_open"]["similarity_weighted_mean_price"] = round(
                float(forecast["previous_session_close"]) * (1 + weighted_gap_mid / 100), 4
            )
        forecast["similar_session_count"] = len(ranked)
        forecast["candidate_paths"] = _candidate_records(ranked)
        forecast["ml_market_path"] = _build_ridge_main_path(history, min_training_sessions=min_training_sessions)
        _attach_path_selection(forecast, None)
        return forecast
    raise ValueError(f"未知预测方法：{method}")


def _mae(values: list[float]) -> float | None:
    return mean(abs(v) for v in values) if values else None


def run_walk_forward_backtest(
    sessions: list[dict[str, Any]], min_training_sessions: int = 20
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Walk-forward compare historical profile, conditional-neighbour path and flat baseline."""
    eligible_count = sum(1 for s in sessions if s["path_eligible"])
    opening_conditional_errors: list[float] = []
    opening_profile_errors: list[float] = []
    opening_zero_gap_errors: list[float] = []
    conditional_path_errors: list[float] = []
    profile_path_errors: list[float] = []
    flat_path_errors: list[float] = []
    ridge_open_errors: list[float] = []
    ridge_path_errors: list[float] = []
    ridge_test_dates = 0
    daily_rows: list[dict[str, Any]] = []
    tested_dates = 0

    for index, target in enumerate(sessions):
        if not target["path_eligible"]:
            continue
        history = sessions[:index]
        eligible_history = [s for s in history if s["path_eligible"]]
        if len(eligible_history) < min_training_sessions:
            continue
        try:
            profile_forecast = predict_intraday_path(history, min_training_sessions, "historical_profile")
            conditional_forecast = predict_intraday_path(history, min_training_sessions, "conditional_neighbors")
        except (ValueError, RuntimeError):
            continue

        actual_open = float(target["open_price"])
        prev_close = float(profile_forecast["previous_session_close"])
        predicted_profile_open = float(profile_forecast["estimated_open"]["p50_price"])
        predicted_cond_open = float(conditional_forecast["estimated_open"]["p50_price"])
        opening_profile = abs(predicted_profile_open - actual_open) / actual_open * 10000 if actual_open else float("nan")
        opening_cond = abs(predicted_cond_open - actual_open) / actual_open * 10000 if actual_open else float("nan")
        zero_gap_error = abs(prev_close - actual_open) / actual_open * 10000 if actual_open else float("nan")
        opening_profile_errors.append(opening_profile)
        opening_conditional_errors.append(opening_cond)
        opening_zero_gap_errors.append(zero_gap_error)

        cond_by_time = {p["time_ny"]: p for p in conditional_forecast["curve"]}
        prof_by_time = {p["time_ny"]: p for p in profile_forecast["curve"]}
        cond_errors: list[float] = []
        prof_errors: list[float] = []
        flat_errors: list[float] = []
        for slot, bar in target["bars"].items():
            if slot not in EXPECTED_SLOTS or slot not in cond_by_time or slot not in prof_by_time:
                continue
            actual = float(bar["close"])
            if actual <= 0:
                continue
            cond_errors.append(abs(float(cond_by_time[slot]["median_price"]) - actual) / actual * 10000)
            prof_errors.append(abs(float(prof_by_time[slot]["median_price"]) - actual) / actual * 10000)
            flat_errors.append(abs(predicted_cond_open - actual) / actual * 10000)
        conditional_path_errors.extend(cond_errors)
        profile_path_errors.extend(prof_errors)
        flat_path_errors.extend(flat_errors)

        # Strict walk-forward independent market-feature model; historical news is not backfilled.
        ridge_path_obj = _build_ridge_main_path(history, min_training_sessions=min_training_sessions)
        ridge_daily_errors: list[float] = []
        ridge_open_error = None
        if ridge_path_obj is not None:
            ridge_open_price = float(ridge_path_obj["predicted_open_price"])
            ridge_open_error = abs(ridge_open_price - actual_open) / actual_open * 10000 if actual_open else None
            if ridge_open_error is not None:
                ridge_open_errors.append(ridge_open_error)
            ridge_prices = ridge_path_obj.get("price_by_time", {})
            for slot, bar in target["bars"].items():
                if slot not in EXPECTED_SLOTS or slot not in ridge_prices:
                    continue
                actual = float(bar["close"])
                if actual > 0:
                    ridge_daily_errors.append(abs(float(ridge_prices[slot]) - actual) / actual * 10000)
            ridge_path_errors.extend(ridge_daily_errors)
            if ridge_daily_errors:
                ridge_test_dates += 1

        daily_rows.append({
            "trade_date_ny": target["trade_date"], "bars_observed": target["slot_count"], "coverage_pct": target["coverage_pct"],
            "actual_open": round(actual_open, 4), "predicted_open_conditional": round(predicted_cond_open, 4),
            "predicted_open_historical_profile": round(predicted_profile_open, 4), "zero_gap_open_baseline": round(prev_close, 4),
            "opening_error_conditional_bps": round(opening_cond, 2), "opening_error_historical_profile_bps": round(opening_profile, 2),
            "zero_gap_baseline_error_bps": round(zero_gap_error, 2),
            "conditional_path_points": len(cond_errors), "conditional_path_mae_bps": round(mean(cond_errors), 2) if cond_errors else None,
            "historical_profile_path_mae_bps": round(mean(prof_errors), 2) if prof_errors else None,
            "flat_path_mae_bps": round(mean(flat_errors), 2) if flat_errors else None,
            "ridge_ml_predicted_open": round(float(ridge_path_obj["predicted_open_price"]), 4) if ridge_path_obj else None,
            "ridge_ml_open_error_bps": round(float(ridge_open_error), 2) if ridge_open_error is not None else None,
            "ridge_ml_path_points": len(ridge_daily_errors),
            "ridge_ml_path_mae_bps": round(mean(ridge_daily_errors), 2) if ridge_daily_errors else None,
        })
        tested_dates += 1

    cond_mae = _mae(conditional_path_errors)
    profile_mae = _mae(profile_path_errors)
    flat_mae = _mae(flat_path_errors)
    def improvement(candidate: float | None, baseline: float | None) -> float | None:
        if candidate is None or baseline is None or baseline <= 0:
            return None
        return round((baseline - candidate) / baseline * 100.0, 2)

    summary = {
        "symbol": "QQQM", "feed": "iex", "timeframe": "15Min",
        "completed_sessions_available": len(sessions), "eligible_path_sessions_available": eligible_count,
        "walk_forward_test_sessions": tested_dates,
        "first_test_date": daily_rows[0]["trade_date_ny"] if daily_rows else None,
        "last_test_date": daily_rows[-1]["trade_date_ny"] if daily_rows else None,
        "opening_mae_bps_conditional_neighbor": round(_mae(opening_conditional_errors), 2) if opening_conditional_errors else None,
        "opening_mae_bps_historical_profile": round(_mae(opening_profile_errors), 2) if opening_profile_errors else None,
        "opening_mae_bps_zero_gap_baseline": round(_mae(opening_zero_gap_errors), 2) if opening_zero_gap_errors else None,
        "path_mae_bps_conditional_neighbor": round(cond_mae, 2) if cond_mae is not None else None,
        "path_mae_bps_historical_profile": round(profile_mae, 2) if profile_mae is not None else None,
        "path_mae_bps_flat_after_open_baseline": round(flat_mae, 2) if flat_mae is not None else None,
        "conditional_improvement_pct_vs_flat_baseline": improvement(cond_mae, flat_mae),
        "conditional_improvement_pct_vs_historical_profile": improvement(cond_mae, profile_mae),
        "ridge_ml_path_test_sessions": ridge_test_dates,
        "ridge_ml_opening_mae_bps": round(_mae(ridge_open_errors), 2) if ridge_open_errors else None,
        "ridge_ml_path_mae_bps": round(_mae(ridge_path_errors), 2) if ridge_path_errors else None,
        "ridge_ml_improvement_pct_vs_flat_baseline": improvement(_mae(ridge_path_errors), flat_mae),
        "ridge_ml_improvement_pct_vs_historical_profile": improvement(_mae(ridge_path_errors), profile_mae),
        "evaluated_intraday_points": len(conditional_path_errors),
        "ridge_ml_evaluated_intraday_points": len(ridge_path_errors),
        "interpretation": (
            "P10/P50/P90 是逐时间点的历史分位情景；主预测路径另由扩展窗口岭回归独立生成，"
            "不由情景线拼接。历史新闻样本未通过点时训练资格审计，因此新闻只作为当前预测的封顶实验叠加。"
            "所有模型必须经样本外误差与简单基线比较；不代表已验证的赚钱策略。"
        ),
    }
    return summary, daily_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_intraday_prototype(
    db_path: Path = DEFAULT_DB_PATH,
    report_dir: Path = REPORT_DIR,
    symbol: str = "QQQM",
    feed: str = "iex",
    timeframe: str = "15Min",
    min_training_sessions: int = 20,
    forecast_target_date: str | None = None,
) -> dict[str, Any]:
    sessions = load_intraday_sessions(db_path, symbol, feed, timeframe, min_coverage=0.70)
    eligible = [s for s in sessions if s["path_eligible"]]
    if len(eligible) < min_training_sessions + 5:
        raise RuntimeError(f"有效完整度交易日只有 {len(eligible)} 天。至少需要 {min_training_sessions + 5} 天才能做初步滚动回测。")

    backtest_summary, daily_rows = run_walk_forward_backtest(sessions, min_training_sessions)
    baseline_forecast = predict_intraday_path(sessions, min_training_sessions, "historical_profile")
    try:
        conditional_forecast = predict_intraday_path(sessions, min_training_sessions, "conditional_neighbors")
        forecast = conditional_forecast
        forecast["conditional_model_available"] = True
    except (ValueError, RuntimeError) as exc:
        forecast = baseline_forecast
        forecast["conditional_model_available"] = False
        forecast["conditional_model_fallback_reason"] = str(exc)

    if forecast_target_date:
        requested_target = _parse_date_value(forecast_target_date)
        as_of_day = _parse_date_value(forecast.get("as_of_date"))
        if requested_target <= as_of_day:
            raise ValueError(
                f"指定预测目标日 {requested_target.isoformat()} 必须晚于数据基准日 {as_of_day.isoformat()}。"
            )
        try:
            from app.services.market_calendar import is_nasdaq_session
            if not is_nasdaq_session(requested_target):
                raise ValueError(f"指定预测目标日 {requested_target.isoformat()} 不是纳斯达克交易日。")
        except ImportError:
            if requested_target.weekday() >= 5:
                raise ValueError(f"指定预测目标日 {requested_target.isoformat()} 落在周末。")
        forecast["forecast_date_weekday_estimate"] = requested_target.isoformat()
        forecast["forecast_target_override_reason"] = (
            "根据当前美东时间调整到下一次尚未开始的完整交易日；若当前交易日已开盘，"
            "当日不再作为全天事前预测目标。"
        )

    forecast["baseline_curve"] = baseline_forecast["curve"]
    forecast["backtest_summary"] = backtest_summary
    generated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    forecast["generated_at_utc"] = generated_at_utc
    forecast["forecast_snapshot_version"] = "intraday-learning-snapshot-v2-main-path-ridge"
    forecast["scenario_line_semantics"] = {
        "p10": "各 15 分钟时间点候选价格分布的第 10 百分位，偏低情景参考，不是 10% 的下跌概率。",
        "p50": "各 15 分钟时间点候选价格分布的中位数，不是主预测路径。",
        "p90": "各 15 分钟时间点候选价格分布的第 90 百分位，偏高情景参考，不是 90% 的上涨概率。",
        "important": "分位数曲线是逐时间点边际统计，连接后不必对应一条历史上真实走过的完整路径。主预测路径由独立模型生成。",
    }
    now_ny = datetime.now(timezone.utc).astimezone(NEW_YORK)
    target_day = _parse_date_value(forecast.get("forecast_date_weekday_estimate"))
    pre_session_snapshot = target_day > now_ny.date() or (
        target_day == now_ny.date() and now_ny.time().replace(tzinfo=None) < time(9, 30)
    )
    forecast["capture_mode"] = "live_prediction" if pre_session_snapshot else "late_or_stale_forecast"
    forecast["pre_session_snapshot"] = bool(pre_session_snapshot)
    forecast["market_state_features"] = {
        name: round(float(value), 8) for name, value in zip(STATE_FEATURES, _state_features(sessions))
    }
    forecast["news_context"] = _build_point_in_time_news_context(db_path, generated_at_utc)
    _attach_path_selection(forecast, forecast.get("news_context"))
    forecast["data_quality"] = {
        "eligible_sessions": len(eligible), "all_sessions_with_open_close_anchors": len(sessions),
        "median_coverage_pct": round(median(s["coverage_pct"] for s in eligible), 2),
        "feed": feed, "timeframe": timeframe,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    latest_json = report_dir / f"{symbol}_intraday_forecast_latest.json"
    archive_path = _write_immutable_forecast_archive(forecast, report_dir)
    # The published prediction and the forecast later scored against the actual day must
    # be byte-for-byte consistent in their path data. If this target/as-of/feed already
    # has an immutable prospective archive, surface that saved snapshot as latest instead
    # of showing a newly regenerated line that will not be used in the review.
    if archive_path and archive_path.exists():
        try:
            archived_forecast = json.loads(archive_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            archived_forecast = None
        if (
            isinstance(archived_forecast, dict)
            and archived_forecast.get("forecast_snapshot_version") == forecast.get("forecast_snapshot_version")
            and str(archived_forecast.get("as_of_date") or "") == str(forecast.get("as_of_date") or "")
            and str(archived_forecast.get("forecast_date_weekday_estimate") or "") == str(forecast.get("forecast_date_weekday_estimate") or "")
            and str(archived_forecast.get("feed") or "").lower() == str(forecast.get("feed") or "").lower()
            and bool(archived_forecast.get("pre_session_snapshot"))
        ):
            forecast = archived_forecast
    latest_json.write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
    stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
    (report_dir / f"{symbol}_intraday_path_backtest_{stamped}.json").write_text(json.dumps(backtest_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(report_dir / f"{symbol}_intraday_path_backtest_daily_latest.csv", daily_rows)
    curve_rows = []
    baseline_by_time = {r["time_ny"]: r for r in baseline_forecast["curve"]}
    main_price_by_time = (forecast.get("main_path") or {}).get("price_by_time", {})
    market_only_price_by_time = (forecast.get("market_only_main_path") or {}).get("price_by_time", {})
    historical_cluster_price_by_time = (forecast.get("historical_cluster_path") or {}).get("price_by_time", {})
    for row in forecast["curve"]:
        baseline = baseline_by_time.get(row["time_ny"], {})
        slot = row["time_ny"]
        curve_rows.append({
            "time_ny": slot, "low_price": row["low_price"], "median_price": row["median_price"],
            "high_price": row["high_price"], "historical_baseline_median_price": baseline.get("median_price"),
            "market_only_main_path_price": market_only_price_by_time.get(slot, ""),
            "news_aware_main_path_price": main_price_by_time.get(slot, ""),
            "historical_cluster_medoid_price": historical_cluster_price_by_time.get(slot, ""),
            "historical_slot_samples": row["historical_slot_samples"],
        })
    _write_csv(report_dir / f"{symbol}_intraday_forecast_curve_latest.csv", curve_rows)
    return {
        "forecast_json": str(latest_json.resolve()), "backtest_summary": backtest_summary, "forecast": forecast,
        "daily_backtest_csv": str((report_dir / f"{symbol}_intraday_path_backtest_daily_latest.csv").resolve()),
        "curve_csv": str((report_dir / f"{symbol}_intraday_forecast_curve_latest.csv").resolve()),
        "archive_json": str(archive_path.resolve()) if archive_path else None,
    }
