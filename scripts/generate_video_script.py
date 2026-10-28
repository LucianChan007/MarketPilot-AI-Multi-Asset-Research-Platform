from __future__ import annotations
"""Generate a Chinese-first daily video package using real intraday bars and explicit freshness checks."""

import argparse
import json
import re
import sqlite3
import sys
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.services.market_calendar import next_nasdaq_session, is_nasdaq_session
from scripts.video_market_data import DB_PATH, load_intraday_from_db, refresh_market_data

LATEST_LEGACY = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
ARCHIVE = ROOT / "reports" / "intraday" / "archive"
VIDEO_FORECAST_LATEST = ROOT / "reports" / "intraday" / "QQQM_video_forecast_latest.json"
OUTROOT = ROOT / "reports" / "video_scripts"
NY = ZoneInfo("America/New_York")
SLOTS = [(datetime.combine(date(2000, 1, 1), time(9, 30)) + timedelta(minutes=15 * i)).strftime("%H:%M") for i in range(26)]

# Prefer common local CJK fonts. On Windows, Microsoft YaHei is normally available.
_available_fonts = {font.name for font in fm.fontManager.ttflist}
for _font in ("Microsoft YaHei", "Microsoft JhengHei", "SimHei", "Noto Sans CJK SC", "PingFang SC", "WenQuanYi Zen Hei", "Arial Unicode MS"):
    if _font in _available_fonts:
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
else:
    # Register common system CJK fonts by path too. This avoids unreadable chart labels
    # on Windows installations where Matplotlib's font cache has not refreshed yet.
    _font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for _font_path in _font_paths:
        if _font_path.exists():
            try:
                fm.fontManager.addfont(str(_font_path))
                _font_name = fm.FontProperties(fname=str(_font_path)).get_name()
                plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
                break
            except (OSError, RuntimeError):
                continue
plt.rcParams["axes.unicode_minus"] = False


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def latest_daily_date(db_path: Path = DB_PATH) -> str | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
        return str(row[0])[:10] if row and row[0] else None
    except sqlite3.Error:
        return None


def actual_daily_ohlc(day: str, db_path: Path = DB_PATH) -> dict[str, float] | None:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT open,high,low,close FROM market_daily WHERE UPPER(ticker)='QQQM' AND trade_date=?", (day,)).fetchone()
        if not row:
            return None
        return dict(zip(("open", "high", "low", "close"), map(float, row)))
    except (sqlite3.Error, TypeError, ValueError):
        return None


def load_news(db_path: Path = DB_PATH, limit: int = 10) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """SELECT title,summary,source_name,category,published_at,url
                   FROM news_items WHERE COALESCE(title,'')<>''
                   ORDER BY published_at DESC LIMIT ?""", (limit,)
            ).fetchall()
        return [dict(zip(("title", "summary", "source", "category", "published_at", "url"), row)) for row in rows]
    except sqlite3.Error:
        return []


def _published_day(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo:
            parsed = parsed.astimezone(NY)
        return parsed.date().isoformat()
    except ValueError:
        return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}", text) else ""


def _source_zh(source: Any) -> str:
    text = str(source or "").lower()
    if "federal reserve" in text or "monetary" in text or "fomc" in text:
        return "美联储"
    if "bls" in text or "labor statistics" in text or "labour statistics" in text:
        return "美国劳工统计局"
    if "fred" in text or "st. louis" in text:
        return "圣路易斯联邦储备银行经济数据库"
    if "sec" in text:
        return "美国证券交易委员会"
    return "公开财经信息来源"


def translate_news(row: dict[str, Any]) -> dict[str, str]:
    """Map official English headlines to truthful Chinese descriptions without leaking English copy."""
    raw = f"{row.get('title') or ''} {row.get('summary') or ''}".lower()
    category = str(row.get("category") or "").lower()
    summary = "这类信息可能影响市场对经济、利率或科技股估值的判断，但不能单独决定价格涨跌。"
    if any(key in raw for key in ("discount rate", "federal open market committee", "fomc", "monetary policy", "policy statement")):
        title = "美联储公布货币政策相关纪要或声明"
        summary = "市场会继续关注利率路径、金融条件和政策制定者对经济的判断。纪要反映讨论背景，不等于下一交易日必然涨跌。"
    elif any(key in raw for key in ("payroll", "unemployment rate", "employment situation", "jobless claims", "nonfarm")):
        title = "美国就业数据成为市场观察重点"
        numbers = []
        jobs = re.search(r"(?:payroll|employment)[^0-9]{0,40}\+?\s*([0-9][0-9,]*)", raw)
        unemployment = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:percent|%)", raw)
        if jobs:
            numbers.append(f"新增就业约{int(jobs.group(1).replace(',', '')):,}人")
        if unemployment:
            numbers.append(f"失业率约为百分之{unemployment.group(1)}")
        details = "，".join(numbers)
        summary = (details + "。" if details else "就业增长和失业率的变化") + "可能改变市场对经济韧性和未来利率的预期。"
    elif any(key in raw for key in ("consumer price index", "cpi", "inflation", "price index")) or category == "inflation":
        title = "美国通胀数据影响利率预期"
        summary = "通胀是市场判断未来利率的重要线索；需要同时观察数据变化和市场原有预期，不能只看单个数字。"
    elif any(key in raw for key in ("economic projections", "economic outlook", "gross domestic product", "gdp", "growth forecast")):
        title = "美国经济前景预测受到关注"
        summary = "经济增长、通胀与利率判断会影响科技股估值，实际市场反应还取决于数据相对预期的差异。"
    elif any(key in raw for key in ("earnings", "revenue", "quarterly results", "profit")):
        title = "企业财报影响科技股预期"
        summary = "投资者会关注收入、盈利和未来展望；个别公司的表现不代表整个纳斯达克指数一定同向变化。"
    elif any(key in raw for key in ("tariff", "trade restriction", "export control", "trade policy")):
        title = "贸易政策变化带来企业经营不确定性"
        summary = "政策可能影响企业成本、供应链和市场风险偏好，具体影响仍要看措施内容与后续执行。"
    elif any(key in raw for key in ("rate cut", "interest rate", "discount window", "rate hike")) or category == "monetary_policy":
        title = "市场继续关注利率政策走向"
        summary = "利率预期会影响成长股估值，但价格反应通常取决于政策相对市场预期的变化。"
    else:
        category_zh = {
            "employment": "美国就业市场相关资讯",
            "monetary_policy": "美国货币政策相关资讯",
            "inflation": "美国通胀相关资讯",
            "economy": "美国经济数据相关资讯",
            "market": "金融市场相关资讯",
        }.get(category, "宏观经济与市场相关资讯")
        title = category_zh
    day = _published_day(row.get("published_at"))
    if day:
        year, month, day_number = day.split("-")
        when = f"{year}年{int(month)}月{int(day_number)}日发布"
    else:
        when = "发布时间未能确认"
    return {"title_zh": title, "summary_zh": summary, "source_zh": _source_zh(row.get("source")), "published_day": day, "when_zh": when}


def _valid_intraday_sessions(frame: pd.DataFrame, now_ny: datetime | None = None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    now_ny = now_ny or datetime.now(timezone.utc).astimezone(NY)
    work = frame.copy()
    if "timestamp_ny" not in work:
        if "timestamp_utc" in work:
            work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
            work["timestamp_ny"] = work["timestamp_utc"].dt.tz_convert(NY)
        else:
            return []
    work = work.dropna(subset=["timestamp_ny"]).copy()
    work["trade_date_ny"] = work["timestamp_ny"].dt.strftime("%Y-%m-%d")
    work["session_time_ny"] = work["timestamp_ny"].dt.strftime("%H:%M")
    work = work[(work["session_time_ny"] >= "09:30") & (work["session_time_ny"] < "16:00")]
    sessions: list[dict[str, Any]] = []
    for session_day, group in work.groupby("trade_date_ny", sort=True):
        group = group.sort_values("timestamp_ny").drop_duplicates("session_time_ny", keep="last").copy()
        parsed_day = date.fromisoformat(str(session_day))
        if parsed_day > now_ny.date():
            continue
        if parsed_day == now_ny.date() and now_ny.time().replace(tzinfo=None) < time(16, 15):
            continue
        if "09:30" not in set(group["session_time_ny"]) or group["session_time_ny"].max() < "15:45":
            continue
        if len(group) < 24:
            continue
        first = group[group["session_time_ny"] == "09:30"].iloc[0]
        last = group.iloc[-1]
        try:
            open_price, close_price = float(first["open"]), float(last["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if open_price <= 0 or close_price <= 0:
            continue
        sessions.append({"date": str(session_day), "bars": group.reset_index(drop=True), "open": open_price, "close": close_price})
    return sessions


def _quantile(values: list[float], q: float) -> float | None:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.quantile(clean, q)) if clean else None


def build_profile_forecast(sessions: list[dict[str, Any]], target_day: str, lookback: int = 20) -> dict[str, Any] | None:
    """Make a transparent historical-profile path using only sessions before target_day."""
    history = [s for s in sessions if s["date"] < target_day][-lookback:]
    if len(history) < 5:
        return None
    history.sort(key=lambda item: item["date"])
    latest = history[-1]
    close = float(latest["close"])
    gap_returns: list[float] = []
    previous_close: float | None = None
    for sess in history:
        if previous_close and previous_close > 0:
            gap_returns.append((float(sess["open"]) / previous_close - 1) * 100)
        previous_close = float(sess["close"])
    if not gap_returns:
        gap_returns = [0.0]
    gap_p10, gap_p50, gap_p90 = (_quantile(gap_returns, q) for q in (0.1, 0.5, 0.9))
    gap_p10, gap_p50, gap_p90 = float(gap_p10 or 0), float(gap_p50 or 0), float(gap_p90 or 0)
    open_prices = {
        "p10_price": close * (1 + gap_p10 / 100),
        "p50_price": close * (1 + gap_p50 / 100),
        "p90_price": close * (1 + gap_p90 / 100),
    }
    points: list[dict[str, Any]] = []
    for slot in SLOTS:
        path_returns: list[float] = []
        for sess in history:
            bars = sess["bars"]
            match = bars[bars["session_time_ny"] == slot]
            if not match.empty:
                price = float(match.iloc[-1]["close"])
                if sess["open"] > 0 and price > 0:
                    path_returns.append((price / sess["open"] - 1) * 100)
        if len(path_returns) < 4:
            continue
        ret10, ret50, ret90 = (_quantile(path_returns, q) for q in (0.1, 0.5, 0.9))
        ret10, ret50, ret90 = float(ret10 or 0), float(ret50 or 0), float(ret90 or 0)
        points.append({
            "time_ny": slot,
            "p10_return_from_open_pct": round(ret10, 5),
            "p50_return_from_open_pct": round(ret50, 5),
            "p90_return_from_open_pct": round(ret90, 5),
            "low_price": round(open_prices["p10_price"] * (1 + ret10 / 100), 4),
            "median_price": round(open_prices["p50_price"] * (1 + ret50 / 100), 4),
            "high_price": round(open_prices["p90_price"] * (1 + ret90 / 100), 4),
            "historical_slot_samples": len(path_returns),
        })
    if len(points) < 18:
        return None
    return {
        "ticker": "QQQM", "feed": "yahoo", "timeframe": "15Min",
        "as_of_date": history[-1]["date"], "forecast_date_weekday_estimate": target_day,
        "previous_session_close": close, "estimated_open": open_prices,
        "estimated_close_from_path": {"p10_price": points[-1]["low_price"], "p50_price": points[-1]["median_price"], "p90_price": points[-1]["high_price"]},
        "curve_method": "近20个完整交易日的分时历史中位路径与隔夜跳空分布",
        "curve": points, "eligible_training_sessions": len(history),
        "news_used_as_direct_price_signal": False,
        "news_note": "新闻用于解释宏观背景；未把无法验证方向的标题强行转换成价格涨跌。",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "caveats": ["这是历史分时统计情景，不是确定价格。", "分位区间不是涨跌概率保证。", "新闻用于背景解释，不单独决定曲线方向。"],
    }


def _archive_forecast(target_day: str) -> dict[str, Any] | None:
    """Retrieve the forecast originally generated for this actual trading date.

    V2 adds the video package's immutable forecast archive and the prior daily
    output folder. Matching is done by forecast target date, not by assuming
    that yesterday's calendar folder is always the previous exchange session.
    """
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sources: list[tuple[Path, str]] = []
    if ARCHIVE.exists():
        sources.extend((path, "intraday_archive") for path in ARCHIVE.glob(f"QQQM_intraday_forecast_{target_day}*.json"))
    video_archive = OUTROOT / "forecast_archive"
    if video_archive.exists():
        sources.extend((path, "video_forecast_archive") for path in video_archive.glob(f"QQQM_forecast_asof_*_target_{target_day}.json"))
    if OUTROOT.exists():
        for day_dir in OUTROOT.iterdir():
            if day_dir.is_dir() and day_dir.name != "forecast_archive":
                candidate = day_dir / "下一交易日走势情景数据.json"
                if candidate.is_file():
                    sources.append((candidate, "daily_video_folder"))
    seen: set[str] = set()
    for path, source_type in sources:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        payload = read_json(path)
        if not payload:
            continue
        payload_target = str(payload.get("forecast_date_weekday_estimate") or payload.get("target_date") or "")[:10]
        if payload_target != target_day or not isinstance(payload.get("curve"), list) or len(payload.get("curve") or []) < 12:
            continue
        as_of = str(payload.get("as_of_date") or payload.get("feature_as_of_date") or "")[:10]
        # Images digitized from a previously published source are a special historical
        # case and retain their original audit metadata. Other archive files must carry
        # a valid generated-at timestamp before target day's market open.
        is_image_restored = payload.get("source_kind") == "published_chart_digitized"
        raw_generated = str(payload.get("generated_at_utc") or payload.get("created_at_utc") or payload.get("generated_at") or "").strip()
        prospective = False
        if is_image_restored:
            prospective = bool(as_of and as_of < target_day)
        elif as_of and as_of < target_day and raw_generated:
            try:
                generated = datetime.fromisoformat(raw_generated.replace("Z", "+00:00"))
                if generated.tzinfo is not None:
                    target_open = datetime.combine(date.fromisoformat(target_day), time(9, 30), tzinfo=NY).astimezone(timezone.utc)
                    prospective = generated.astimezone(timezone.utc) < target_open
            except (TypeError, ValueError, OverflowError):
                prospective = False
        # Preserve compatibility with the original research archive, whose timestamp
        # scheme predates V2. Video-package files are held to the stronger rule above.
        if source_type != "intraday_archive" and not prospective:
            continue
        if source_type == "intraday_archive" and not is_image_restored:
            if not as_of or as_of >= target_day:
                continue
            if raw_generated:
                try:
                    generated = datetime.fromisoformat(raw_generated.replace("Z", "+00:00"))
                    target_open = datetime.combine(date.fromisoformat(target_day), time(9, 30), tzinfo=NY).astimezone(timezone.utc)
                    if generated.tzinfo is None or generated.astimezone(timezone.utc) >= target_open:
                        continue
                except (TypeError, ValueError, OverflowError):
                    continue
        payload = dict(payload)
        payload["_archive_source_path"] = str(path.resolve())
        payload["_archive_source_type"] = source_type
        if source_type == "daily_video_folder":
            source_image = path.parent / "02_下一交易日走势情景参考.png"
            payload["_archive_source_image_path"] = str(source_image.resolve()) if source_image.is_file() else None
        elif source_type == "video_forecast_archive":
            # Locate the exact chart 02 folder that generated this immutable numeric forecast.
            # The overlaid curve is read from JSON, not reverse-engineered from screenshot pixels.
            source_image = None
            for day_dir in OUTROOT.iterdir() if OUTROOT.exists() else []:
                if not day_dir.is_dir() or day_dir.name == "forecast_archive":
                    continue
                candidate_json = day_dir / "下一交易日走势情景数据.json"
                candidate_payload = read_json(candidate_json) if candidate_json.is_file() else None
                if candidate_payload and str(candidate_payload.get("forecast_date_weekday_estimate") or candidate_payload.get("target_date") or "")[:10] == target_day:
                    candidate_png = day_dir / "02_下一交易日走势情景参考.png"
                    if candidate_png.is_file():
                        source_image = candidate_png
                        break
            payload["_archive_source_image_path"] = str(source_image.resolve()) if source_image else None
        immutable = bool(payload.get("forecast_snapshot_version") or payload.get("immutable_snapshot") or payload.get("pre_session_snapshot"))
        rank = (int(source_type == "video_forecast_archive"), int(source_type == "daily_video_folder"), int(is_image_restored), int(immutable), raw_generated, path.name)
        candidates.append((rank, payload))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _forecast_points_for_plot(forecast: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    if not forecast:
        return result
    allowed_slots = set(SLOTS + ["16:00"])
    for point in forecast.get("curve") or []:
        slot = str(point.get("time_ny") or "")[:5]
        if slot not in allowed_slots:
            continue
        try:
            median = float(point.get("median_price"))
            low = float(point.get("low_price", median))
            high = float(point.get("high_price", median))
        except (TypeError, ValueError):
            continue
        if median > 0 and low > 0 and high > 0:
            result[slot] = {"median": median, "low": min(low, high, median), "high": max(low, high, median)}
    # Chart 02 may draw an explicit Ridge main path on top of the historical median.
    # For the later review, compare actual data with that exact main path, not a
    # visually different fallback curve hidden underneath the orange line.
    main_path = forecast.get("main_path") if isinstance(forecast.get("main_path"), dict) else {}
    main_values = main_path.get("price_by_time") if isinstance(main_path.get("price_by_time"), dict) else {}
    if main_values:
        for slot_raw, value_raw in main_values.items():
            slot = str(slot_raw)[:5]
            if slot not in allowed_slots:
                continue
            try:
                value = float(value_raw)
            except (TypeError, ValueError):
                continue
            if value <= 0 or not np.isfinite(value):
                continue
            point = result.setdefault(slot, {"median": value, "low": value, "high": value})
            point["main_path"] = value
            point["median"] = value
    else:
        main_points = main_path.get("path") if isinstance(main_path.get("path"), list) else []
        for item in main_points:
            try:
                slot = str(item.get("time_ny") or "")[:5]
                value = float(item.get("predicted_price", item.get("price")))
            except (TypeError, ValueError):
                continue
            if slot not in allowed_slots or value <= 0 or not np.isfinite(value):
                continue
            point = result.setdefault(slot, {"median": value, "low": value, "high": value})
            point["main_path"] = value
            point["median"] = value
    return result


def _end_time_from_start(slot: str) -> str:
    try:
        parsed = datetime.strptime(slot, "%H:%M") + timedelta(minutes=15)
        return parsed.strftime("%H:%M")
    except ValueError:
        return slot


def plot_review(session: dict[str, Any] | None, forecast: dict[str, Any] | None, out: Path, source_label: str) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    if not session:
        ax.axis("off")
        ax.text(.5, .61, "暂时没有可核验的完整分时行情", ha="center", va="center", fontsize=18, weight="bold", transform=ax.transAxes)
        ax.text(.5, .48, "请先刷新公开行情。没有真实十五分钟数据，就不会拿日线四个价格点拼成一整天。", ha="center", va="center", fontsize=11, wrap=True, transform=ax.transAxes)
        fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
        return {"review_date": None, "review_method": "unavailable", "mae_usd": None, "actual_close": None, "reference_close": None}

    actual_day = session["date"]
    bars = session["bars"].copy().sort_values("timestamp_ny")
    start_times = bars["session_time_ny"].astype(str).tolist()
    opens = pd.to_numeric(bars["open"], errors="coerce").astype(float).tolist()
    closes = pd.to_numeric(bars["close"], errors="coerce").astype(float).tolist()
    highs = pd.to_numeric(bars["high"], errors="coerce").astype(float).tolist()
    lows = pd.to_numeric(bars["low"], errors="coerce").astype(float).tolist()

    # Forecast image labels denote interval boundaries (09:30 ... 16:00).
    # Plot actual opening print at 09:30, then each actual 15-minute bar close at its end.
    actual_times = [start_times[0]] + [_end_time_from_start(t) for t in start_times]
    actual_prices = [opens[0]] + closes
    actual_x = list(range(len(actual_prices)))
    ax.plot(actual_x, actual_prices, color="#1565C0", linewidth=2.7, label="真实十五分钟走势（开盘价＋各时段实际收盘）", zorder=5)

    point_map = _forecast_points_for_plot(forecast)
    reference_slots = [slot for slot in actual_times if slot in point_map]
    method = str((forecast or {}).get("source_kind") or (forecast or {}).get("video_source_kind") or "")
    uses_main_path = any("main_path" in point_map.get(slot, {}) for slot in reference_slots)
    if point_map and len(reference_slots) >= 12:
        x = [actual_times.index(slot) for slot in reference_slots]
        mids = [point_map[slot]["median"] for slot in reference_slots]
        band_lows = [point_map[slot]["low"] for slot in reference_slots]
        band_highs = [point_map[slot]["high"] for slot in reference_slots]
        is_image_restored = method == "published_chart_digitized"
        has_original_snapshot = bool(forecast and (
            forecast.get("forecast_snapshot_version") or forecast.get("immutable_snapshot")
            or forecast.get("pre_session_snapshot")
            or forecast.get("_archive_source_type") in {"video_forecast_archive", "daily_video_folder"}
        ) and not is_image_restored)
        band_label = "原图所示±0.75%情景参考带（非统计置信区间）" if is_image_restored else "预测参考区间"
        line_label = (
            "昨天已发布预测图还原线（中间点为近似值）" if is_image_restored
            else "当时存档 Ridge 机器学习主路径" if has_original_snapshot and uses_main_path
            else "当时存档预测中位线" if has_original_snapshot
            else "此前历史数据回算参考线"
        )
        ax.fill_between(x, band_lows, band_highs, alpha=.20, color="#EF6C00", label=band_label, zorder=1)
        ax.plot(x, mids, color="#EF6C00", linestyle="--", linewidth=2.4, label=line_label, zorder=4)
        actual_by_time = dict(zip(actual_times, actual_prices))
        errors = [abs(actual_by_time[t] - point_map[t]["median"]) for t in reference_slots if t in actual_by_time]
        mae = float(np.mean(errors)) if errors else None
        ref_close = point_map[reference_slots[-1]]["median"]
        if is_image_restored:
            ax.text(.012, .02, "说明：橙线依据已发布图片还原；中间点为近似值。阴影为原图±0.75%参考带，不是统计置信区间。",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6,
                    bbox={"facecolor":"white", "alpha":0.85, "edgecolor":"none", "pad":4})
            review_method = "published_chart_digitized"
        elif has_original_snapshot:
            review_method = "saved_forecast"
        else:
            review_method = "historical_backcast"
    else:
        review_method = "actual_only"
        mae = None
        ref_close = None
        ax.text(.02, .97, "未找到同日可核验的预测记录；本图只展示真实走势，不编造预测对比线。", transform=ax.transAxes, va="top", ha="left", fontsize=10)

    tick_labels = ["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]
    tick_positions = []
    tick_actual_labels = []
    for label in tick_labels:
        if label in actual_times:
            tick_positions.append(actual_times.index(label))
            tick_actual_labels.append(label)
        else:
            # If some bars are missing, anchor each requested tick to the nearest observed end time.
            def minute_value(value: str) -> int:
                hh, mm = map(int, value.split(":")); return hh * 60 + mm
            pos = min(range(len(actual_times)), key=lambda i: abs(minute_value(actual_times[i]) - minute_value(label)))
            if pos not in tick_positions:
                tick_positions.append(pos); tick_actual_labels.append(label)
    ax.set_xticks(tick_positions, tick_actual_labels)
    ax.set_xlabel("美东时间（纽约股市正常交易时段）")
    ax.set_ylabel("QQQM价格（美元）")
    if method == "published_chart_digitized":
        title = f"QQQM {actual_day} 复盘：昨天已发布的预测图，对上当天真实走势"
    else:
        title = f"QQQM {actual_day}：真实分时走势与同日预测参考对照"
    ax.set_title(title, fontsize=15, pad=14)
    source_note = f"真实分时来源：{source_label}"
    if forecast and forecast.get("_archive_source_image_path"):
        source_note += f"\n预测原图：{Path(str(forecast['_archive_source_image_path'])).name}"
    if forecast and forecast.get("_archive_source_path"):
        source_note += f"\n原始预测数据：{Path(str(forecast['_archive_source_path'])).name}"
    ax.text(.012, .985, source_note, transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
            bbox={"facecolor":"white", "alpha":0.84, "edgecolor":"none", "pad":3})
    ax.grid(alpha=.23); ax.legend(loc="best", fontsize=8.8, framealpha=.94)
    actual_open = float(opens[0]) if opens else None
    actual_close = float(closes[-1]) if closes else None
    actual_high = float(max(highs)) if highs else None
    actual_low = float(min(lows)) if lows else None
    day_change_pct = ((actual_close / actual_open) - 1) * 100 if actual_open and actual_close else None
    footer = (f"真实开盘 {actual_open:.2f}｜最高 {actual_high:.2f}｜最低 {actual_low:.2f}｜收盘 {actual_close:.2f} 美元"
              if None not in (actual_open, actual_high, actual_low, actual_close) else "实际分时价格")
    if day_change_pct is not None:
        footer += f"｜开收盘变化 {day_change_pct:+.2f}%"
    ax.text(.99, .02, footer, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"facecolor":"white", "alpha":0.86, "edgecolor":"none", "pad":3})
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    return {"review_date": actual_day, "review_method": review_method,
            "mae_usd": round(mae, 4) if mae is not None else None,
            "actual_open": round(actual_open, 4) if actual_open is not None else None,
            "actual_high": round(actual_high, 4) if actual_high is not None else None,
            "actual_low": round(actual_low, 4) if actual_low is not None else None,
            "actual_close": round(actual_close, 4) if actual_close is not None else None,
            "day_change_pct": round(day_change_pct, 4) if day_change_pct is not None else None,
            "reference_close": round(float(ref_close), 4) if ref_close is not None else None,
            "actual_bar_count": len(start_times), "actual_source": source_label,
            "forecast_source_kind": method or None}

def plot_next(forecast: dict[str, Any] | None, out: Path, news_note: str) -> bool:
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    curve = forecast.get("curve") if forecast else None
    target = str((forecast or {}).get("forecast_date_weekday_estimate") or "")
    if curve and len(curve) >= 18:
        frame = pd.DataFrame(curve)
        xs = list(range(len(frame)))
        lows = pd.to_numeric(frame["low_price"], errors="coerce").astype(float).tolist()
        mids = pd.to_numeric(frame["median_price"], errors="coerce").astype(float).tolist()
        highs = pd.to_numeric(frame["high_price"], errors="coerce").astype(float).tolist()
        times = frame["time_ny"].astype(str).tolist()
        ax.fill_between(xs, lows, highs, color="#90CAF9", alpha=.32, label="历史分位波动参考区间")
        ax.plot(xs, mids, color="#0D47A1", linewidth=2.0, linestyle="--", label="历史中位参考路径")
        main_path = forecast.get("main_path") if isinstance(forecast, dict) else None
        main_prices = main_path.get("price_by_time", {}) if isinstance(main_path, dict) else {}
        main_slots, main_values = [], []
        if isinstance(main_prices, dict):
            for slot in times:
                try:
                    value = float(main_prices[slot])
                    if np.isfinite(value) and value > 0:
                        main_slots.append(slot)
                        main_values.append(value)
                except (KeyError, TypeError, ValueError):
                    continue
        model_family = str((main_path or {}).get("model_family") or "") if isinstance(main_path, dict) else ""
        if len(main_slots) >= 18 and "ridge" in model_family.lower():
            slot_to_x = {slot: idx for idx, slot in enumerate(times)}
            ax.plot([slot_to_x[t] for t in main_slots], main_values, color="#E65100", linewidth=2.8,
                    label="机器学习候选主路径（尚未自动晋级）")
        tick_idxs = sorted(set([0, 4, 8, 12, 16, 20, len(times) - 1]))
        ax.set_xticks(tick_idxs, [times[i] for i in tick_idxs])
        ax.set_xlabel("美东时间")
        ax.set_ylabel("价格（美元）")
        ax.set_title(f"下一交易日 {target}：走势情景参考\n根据此前完整交易日分时历史生成，非确定价格", fontsize=14)
        ax.text(.01, .02, news_note[:140], transform=ax.transAxes, ha="left", va="bottom", fontsize=9, wrap=True)
        ax.grid(alpha=.25); ax.legend(loc="best")
        available = True
    else:
        ax.axis("off")
        ax.text(.5, .63, "下一交易日走势曲线暂不可用", ha="center", va="center", fontsize=18, weight="bold", transform=ax.transAxes)
        reason = "公开行情源暂时无法刷新，且本地没有足够新的完整分时样本。" if forecast is None else "可用分时交易日不足，无法构建可靠的历史路径参考。"
        ax.text(.5, .49, reason, ha="center", va="center", fontsize=12, wrap=True, transform=ax.transAxes)
        ax.text(.5, .37, "不会用过期曲线冒充下一交易日预测。", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        available = False
    fig.tight_layout(); fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    return available


def _next_target_day(latest_actual_day: str, now_ny: datetime | None = None) -> str:
    """Choose the next session correctly around the pre-market boundary.

    Before 09:30 New York, if the latest completed data is the prior session,
    today's session is still forecastable. Once the session has opened, today is
    no longer a full-session forecast target; use the next exchange session.
    """
    now_ny = now_ny or datetime.now(timezone.utc).astimezone(NY)
    if now_ny.tzinfo is None:
        now_ny = now_ny.replace(tzinfo=NY)
    else:
        now_ny = now_ny.astimezone(NY)
    today_ny = now_ny.date()
    latest_day = date.fromisoformat(str(latest_actual_day)[:10])
    try:
        today_is_session = is_nasdaq_session(today_ny)
    except Exception:
        today_is_session = today_ny.weekday() < 5
    current_time = now_ny.time().replace(tzinfo=None)
    if today_is_session and current_time < time(9, 30) and latest_day < today_ny:
        return today_ny.isoformat()
    base = max(latest_day, today_ny) if today_is_session else latest_day
    return next_nasdaq_session(base).isoformat()


def _chinese_news_context(news: list[dict[str, Any]], forecast_date: str) -> tuple[list[dict[str, str]], str, bool]:
    translated = [translate_news(row) for row in news]
    translated = [item for item in translated if item["published_day"]]
    translated.sort(key=lambda item: item["published_day"], reverse=True)
    # Different source feeds often repeat the same policy announcement. Keep one
    # item per Chinese headline so the voiceover does not read the same news twice.
    unique_news: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    for item in translated:
        key = item["title_zh"]
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique_news.append(item)
    translated = unique_news
    latest_day = translated[0]["published_day"] if translated else ""
    cutoff = date.fromisoformat(forecast_date)
    stale = bool(latest_day and (cutoff - date.fromisoformat(latest_day)).days > 4)
    if not translated:
        note = "目前没有可确认发布时间的新闻记录，因此本期不编造新闻依据；走势曲线只根据完整分时行情统计生成。"
    elif stale:
        note = f"本地新闻库可确认的最新发布时间为{latest_day}，距离预测交易日已有数天，只作为宏观背景，不当作最新盘前消息。"
    else:
        note = f"新闻库可确认的较新消息更新至{latest_day}，以下仅用于解释宏观背景，不把标题直接当作涨跌信号。"
    return translated[:4], note, stale


def _price(v: Any) -> str:
    try:
        return f"{float(v):.2f}美元"
    except (TypeError, ValueError):
        return "暂时无法确认"


def _make_script(today: str, review: dict[str, Any], forecast: dict[str, Any] | None,
                 forecast_available: bool, news_items: list[dict[str, str]], news_note: str,
                 actual_latest: str | None, source_label: str, refresh_errors: list[str]) -> tuple[str, str, dict[str, Any]]:
    target = str((forecast or {}).get("forecast_date_weekday_estimate") or "")
    asof = str((forecast or {}).get("as_of_date") or actual_latest or "")
    current_price = review.get("actual_close")
    reference_price = review.get("reference_close")
    mae = review.get("mae_usd")
    review_method = review.get("review_method")
    if review_method == "published_chart_digitized":
        day_move = review.get("day_change_pct")
        day_move_text = f"按分时开盘和收盘计算，当天开收盘变化约为{day_move:+.2f}%。" if day_move is not None else "当天开收盘变化暂时无法确认。"
        mae_text = f"把当天实际曲线与图片还原后的参考线粗略对齐，平均绝对偏差约为{_price(mae)}。" if mae is not None else "实际曲线与图片参考线暂时无法计算完整的对照误差。"
        review_narration = (
            f"先把证据摆上桌：橙色虚线来自昨天已经发布的那张预测图，不是今天收盘后重新补画的。图上标出的开盘情景参考是295.16美元，收盘情景参考是294.55美元；"
            f"蓝线才是{review.get('review_date')}当天抓取到的真实十五分钟走势。{day_move_text}"
            f"{mae_text}不过先划重点：原始逐点预测文件当时没有随图片保存，中间点是从图片坐标近似还原的，所以这次误差只能算图像还原后的粗略对照，不能包装成原始模型的精确成绩。"
            f"原图上的上下阴影是上下各百分之零点七五的情景参考带，不是统计置信区间。说白了，橙线是昨天公开交卷的证据，但尺子有一点像素误差，我们不装作拿着原始电子答题卡。"
        )
    elif review_method == "saved_forecast":
        review_narration = f"第一张图把真实分时走势和交易日前保存的原始预测曲线放在一起。当天真实收盘参考是{_price(current_price)}，参考线末端为{_price(reference_price)}，逐点平均绝对偏差约为{_price(mae)}。别光看最后一笔有没有猜中，盘中拐点和波动范围也得对账，不能只截自己猜对的那一秒发朋友圈。"
    elif review_method == "historical_backcast":
        review_narration = f"这一天没有找到提前保存的原始预测，所以图中的橙色参考线是事后用此前历史数据回算的，不是昨天真正发出去的预测，绝不能拿它冒充提前猜中。蓝线才是当天实际走势，收盘参考为{_price(current_price)}；这张图只用于观察形状，不计入正式预测成绩。"
    elif review_method == "actual_only":
        review_narration = f"第一张图展示了最近一个可确认完整交易日的真实分时走势，收盘参考为{_price(current_price)}。没有找到同日预测原件，就不硬凑对比线。今天这张属于真实行情单人照，不会强行给它P一个AI同框。"
    else:
        review_narration = "目前没有可核验的完整分时数据，第一张图不会用日线开高低收四个点拼出一条假的全天走势。"

    if forecast_available and forecast:
        curve = forecast.get("curve") or []
        main_path = forecast.get("main_path") if isinstance(forecast.get("main_path"), dict) else {}
        is_ml_main = "ridge" in str(main_path.get("model_family") or "").lower() and bool(main_path.get("price_by_time"))
        open_est = main_path.get("predicted_open_price") if is_ml_main else (forecast.get("estimated_open") or {}).get("p50_price")
        close_est = main_path.get("predicted_close_price") if is_ml_main else (forecast.get("estimated_close_from_path") or {}).get("p50_price")
        low = min((float(p.get("low_price")) for p in curve if p.get("low_price") is not None), default=None)
        high = max((float(p.get("high_price")) for p in curve if p.get("high_price") is not None), default=None)
        method_text = ("橙线是根据此前完整交易日训练得到的盘中回归候选主路径，蓝色虚线和阴影是历史中位与分位参考范围；候选模型仍需接受真实行情检验。" if is_ml_main else "这条线来自此前完整交易日的历史分时统计，并不是机器学习主路径，只作情景参考。")
        forecast_narration = f"第二张图是{target}的走势情景参考。{method_text}开盘参考约为{_price(open_est)}，收盘参考约为{_price(close_est)}，历史波动区间约为{_price(low)}到{_price(high)}。利率、就业或通胀新闻只用于解释背景，不把新闻标题硬换算成涨跌幅。区间不保证价格，也不是买卖指令。"
    else:
        forecast_narration = "第二张图目前没有新的、可确认有效的下一交易日走势曲线。原因是联网行情没有刷新成功，或者完整分时样本还不够。宁可把缺口写清楚，也不拿旧数据给未来编剧情。"

    if news_items:
        news_narration = []
        for item in news_items[:2]:
            news_narration.append(f"{item['when_zh']}，{item['source_zh']}的公开消息提到，{item['title_zh']}。{item['summary_zh']}")
        news_text = " ".join(news_narration)
        titles_for_context = " ".join(str(item.get("title_zh") or "") for item in news_items[:2])
        if "美联储" in titles_for_context or "利率" in titles_for_context:
            news_text += "这类消息和QQQM的联系在于：如果市场因此上调未来利率预期，科技股估值可能承压；如果利率预期下降，估值压力也可能缓和。关键是消息相对市场原本预期有多大意外，不能只凭标题判定方向。"
        if "就业" in titles_for_context:
            news_text += "就业数据也不是越好股市就一定越涨：就业偏强可能说明经济有韧性，也可能让市场担心利率维持高位；就业偏弱可能带来降息期待，也可能引发经济放缓担忧。市场不是只看一个数字做判断。"
        if "通胀" in titles_for_context:
            news_text += "通胀数据会影响市场对利率的判断，但真正值得比较的是实际数据与市场预期之间的差异。"
    else:
        news_text = "这次没有足够可靠、带发布时间的新闻记录可以引用，所以不凭空编造所谓的利好或利空。"
    warnings = ""
    if refresh_errors:
        # Never copy raw exception text (which may contain English provider errors) into a voiceover.
        warnings = "本次刷新并非所有环节都成功。图表日期以可核验的数据为准；若第二张图标注暂不可用，请先检查网络与行情源。本片不会把缓存数据伪装成最新行情。"
    is_published_chart_replay = review_method == "published_chart_digitized"
    main_title = (
        f"人工智能昨天画线，QQQM今天交卷！{review.get('review_date') or '每日行情复盘'}"
        if is_published_chart_replay else
        f"QQQM真实走势复盘：人工智能预测到底准不准？{review.get('review_date') or ''}"
    )
    backup_title = (
        "昨天预测收盘参考294.55美元，今天把真实曲线摊开对账！"
        if is_published_chart_replay else "把事前预测和真实走势放在一起，看看模型到底几分"
    )
    xhs_title = (
        "人工智能昨天画的线，今天被QQQM真实走势‘批改’了！"
        if is_published_chart_replay else "人工智能预测QQQM，今天拿真实走势来对账！"
    )
    douyin_title = (
        "昨天预测收盘参考294.55美元，今天真实走势来对账！"
        if is_published_chart_replay else "QQQM真实走势出炉，人工智能这次猜得怎么样？"
    )
    channels_title = (
        "人工智能昨天给QQQM画路线，今天市场按真实走势打分"
        if is_published_chart_replay else "QQQM每日复盘：事前预测与真实走势对照"
    )
    bilibili_title = (
        "QQQM复盘实验：昨天公开预测图与今天十五分钟真实行情逐点对账"
        if is_published_chart_replay else "QQQM每日预测实验：真实分时曲线、新闻背景与模型复盘"
    )
    if is_published_chart_replay:
        opening_narration = "昨天人工智能交了一张QQQM走势路线图，今天市场来交批改意见了！今天咱们不靠记忆滤镜，也不许模型偷偷擦掉错题，把昨天已经发布的原图和真实行情放在一起，看看它到底是有两把刷子，还是画图很积极、考试看运气。"
    else:
        opening_narration = "今天继续检查QQQM行情作业。能找到事前保存的预测，我们就跟真实走势对账；找不到原始预测，就只展示真实行情，不临时补画一条线假装昨天猜中过。市场已经够会反转了，我们的文案就别再反转事实。"
    # Short-form voiceover: one hook, two charts, one brief context sentence, one disclaimer.
    # Keep the directly readable script around 30-60 seconds and never narrate the whole data pipeline.
    if review_method == "published_chart_digitized":
        chart1_short = (
            f"先看第一张图：橙线是昨天已经发布的预测，开盘参考{_price(review.get('reference_open')) if review.get('reference_open') else '见图中标注'}，收盘参考{_price(reference_price)}；蓝线是当天真实走势。"
            f"当天开收盘变化约{review.get('day_change_pct'):+.2f}%。中间预测点是从原图估读的，所以只做近似复盘，不把像素当精确模型记录。"
            if review.get("day_change_pct") is not None else
            f"先看第一张图：橙线是昨天已经发布的预测，蓝线是当天真实走势。原始逐点预测没有留档，中间点由图片估读，只能做近似复盘。"
        )
    elif review_method == "saved_forecast":
        chart1_short = f"先看第一张图：蓝线是实际走势，橙线是事前保存的预测。实际收盘{_price(current_price)}，预测参考收盘{_price(reference_price)}，平均偏差约{_price(mae)}。这次不只看收盘猜没猜中，也看看盘中拐点。"
    elif review_method == "historical_backcast":
        chart1_short = f"先看第一张图：蓝线是实际走势。当天没有找到事前保存的预测，所以不会临时补一条橙线冒充猜中；这张只看真实行情，不计入预测成绩。收盘{_price(current_price)}。"
    elif review_method == "actual_only":
        chart1_short = f"先看第一张图：这是最近一个可核验交易日的真实分时走势，收盘{_price(current_price)}。没有原始预测，就不硬凑对比线，复盘先把事实摆出来。"
    else:
        chart1_short = "先看第一张图：目前没有可核验的完整分时数据，所以这次不伪造真实走势，也不把旧数据拿来充数。"

    if forecast_available and forecast:
        curve = forecast.get("curve") or []
        main_path = forecast.get("main_path") if isinstance(forecast.get("main_path"), dict) else {}
        is_ml_main = "ridge" in str(main_path.get("model_family") or "").lower() and bool(main_path.get("price_by_time"))
        open_est = main_path.get("predicted_open_price") if is_ml_main else (forecast.get("estimated_open") or {}).get("p50_price")
        close_est = main_path.get("predicted_close_price") if is_ml_main else (forecast.get("estimated_close_from_path") or {}).get("p50_price")
        low = min((float(p.get("low_price")) for p in curve if p.get("low_price") is not None), default=None)
        high = max((float(p.get("high_price")) for p in curve if p.get("high_price") is not None), default=None)
        if is_ml_main:
            chart2_short = f"再看第二张图，这是{target}的机器学习候选路径：开盘参考{_price(open_est)}，收盘参考{_price(close_est)}；历史波动参考范围约{_price(low)}到{_price(high)}。候选模型还要继续接受真实行情检验。"
        else:
            chart2_short = f"再看第二张图，这是{target}的历史统计情景参考：开盘中位参考{_price(open_est)}，收盘中位参考{_price(close_est)}，历史波动参考范围约{_price(low)}到{_price(high)}。这不是机器学习主路径。"
    else:
        chart2_short = "第二张图暂时没有可靠的新曲线，行情数据还没跟上。宁可留白，也不让旧预测穿越到明天上班。"

    if news_items:
        item = news_items[0]
        news_short = f"背景只补一句：{item['title_zh']}，可能影响市场对利率和科技股估值的判断，但不能单凭新闻断言涨跌。"
    else:
        news_short = "这次没有可核验的最新新闻，就不硬编利好利空。"

    if is_published_chart_replay:
        opening_short = "昨天的预测今天交卷！别急着夸，也别急着骂，先把两张图摆出来对账。"
    elif review_method == "saved_forecast":
        opening_short = "预测不是画得像就算赢，今天直接拿真实走势来对答案。"
    else:
        opening_short = "今天只看两张图：昨天实际怎么走，下一交易日可能怎么走。"

    spoken = f"""{opening_short}

{chart1_short}

{chart2_short}

{news_short} 这只是历史数据情景参考，不保证准确，也不是买卖建议。"""
    # Keep operational warnings in the status/metadata, not in the spoken voiceover.
    # The viewer-facing narration should stay concise and focus on the two charts.

    md = f"""# {main_title}

## 视频标题（建议首发）
**AI昨天画线，QQQM今天交卷！真实走势到底给几分？**

备用标题：{backup_title}｜人工智能预测不能只靠一张漂亮图

## 本期数据
- 最近一个完整交易日：{review.get('review_date') or '暂不可确认'}
- 日线数据库最近日期：{actual_latest or '暂不可确认'}
- 分时数据来源：{source_label}
- 情景参考目标日：{target or '暂不可生成'}
- 数据说明：{news_note}

## 口播稿，可直接照念

{spoken}

## 两张图片说明
第一张图：蓝线为刷新后取得的真实十五分钟走势；若复盘日期为2026年10月20日，橙线使用本项目随包保存的用户已发布预测图进行近似数字化还原。图上能读到的开盘情景参考为295.16美元、收盘情景参考为294.55美元；原图中间点会标为近似值，±0.75%阴影带不是统计置信区间。若真实分时没有取到，程序不输出虚假的真实走势。

第二张图：下一交易日的分时情景参考。使用此前完整交易日的分时路径和隔夜开盘变化分布。如果样本或联网数据不足，图片会明确显示暂不可用。

## 平台发布文案
- 总标题：{main_title}真实走势到底给几分？
- 小红书标题：{xhs_title}
- 抖音标题：{douyin_title}
- 微信视频号标题：{channels_title}
- 哔哩哔哩标题：{bilibili_title}
- 视频简介：我们持续记录 QQQM 下一交易日走势预测，用真实行情复盘昨天的预测，再检查新情景参考。预测不是算命，准不准交给真实行情检验。本视频为行情研究与模型实验记录，不保证准确，不构成投资建议。
- 话题：#美股 #纳斯达克 #投资复盘 #人工智能 #行情分析

## 剪辑顺序（约30–60秒）
1. 先说开场钩子；2. 插入第一张图，讲真实走势与昨天预测的差别；3. 插入第二张图，讲下一交易日情景参考；4. 用一句新闻背景和风险提示收尾。不需要逐条朗读新闻，也不需要解释模型原理。

## 透明度说明
第一张图的真实蓝线必须来自行情源返回的完整十五分钟数据。2026年10月20日橙色预测线来自本项目保存的用户已发布图片，开盘和收盘数字取自原图标签，中间点按像素位置估读并插值，因此误差是近似复盘，不是原始逐点模型输出的精确成绩。原图±0.75%阴影只表示图中写明的情景参考范围，不是统计置信区间。新闻只作为有来源的背景解释，不直接等同于价格方向；真实数据没取到时，程序必须明确写不可用。
"""
    txt = spoken.replace("\n\n", "\n\n").strip() + "\n"
    publish_description = (
        f"我们持续记录 QQQM 下一交易日走势预测，用真实行情复盘昨天的预测，再检查新情景参考。"
        f"这次目标日：{target or '暂不可确认'}。预测不是算命，准不准交给真实行情检验。\n\n"
        "本视频为行情研究与模型实验记录，不保证准确，不构成投资建议。\n"
        "#QQQM #纳斯达克 #美股预测 #AI分析 #投资复盘 #美股"
    )
    publish_txt = (
        f"推荐标题：{main_title}\n"
        f"备用标题：{backup_title}\n\n"
        f"视频简介：\n{publish_description}\n\n"
        "平台标题：\n"
        f"小红书：{xhs_title}\n"
        f"抖音：{douyin_title}\n"
        f"微信视频号：{channels_title}\n"
        f"哔哩哔哩：{bilibili_title}\n"
    )
    meta = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "main_video_title": main_title,
        "latest_daily_date": actual_latest, "review_date": review.get("review_date"),
        "review_method": review.get("review_method"), "review_metrics": review,
        "review_forecast_source_kind": review.get("forecast_source_kind"),
        "review_forecast_source_file": None,
        "forecast_as_of_date": asof or None, "forecast_target_date": target or None,
        "next_forecast_available": bool(forecast_available), "source_label": source_label,
        "news_items_used": news_items, "news_note_zh": news_note, "refresh_errors": refresh_errors,
        "forecast_method_zh": (forecast or {}).get("curve_method"),
        "forecast_source_kind": (forecast or {}).get("video_source_kind"),
        "forecast_feed": (forecast or {}).get("feed"),
        "forecast_model_family": ((forecast or {}).get("main_path") or {}).get("model_family"),
        "model_candidate_used": bool((forecast or {}).get("video_source_kind") == "trained_intraday_candidate"),
    }
    return md, txt, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="生成中文行情口播稿、真实分时复盘图和下一交易日情景图。")
    parser.add_argument("--refresh", action="store_true", help="先刷新公开行情和官方新闻，再生成视频内容。")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite 行情数据库路径")
    parser.add_argument("--output-dir", default="", help="可选的输出目录")
    args = parser.parse_args()
    db_path = Path(args.db)
    refresh_errors: list[str] = []
    source_label = "本地历史缓存"
    if args.refresh:
        status = refresh_market_data(db_path)
        refresh_errors.extend(status.get("errors") or [])
        source_label = str(status.get("data_source_used") or "本地历史缓存")
        try:
            from app.news_data import sync_news
            news_status = sync_news()
            if news_status.get("failed"):
                refresh_errors.append("官方新闻同步未全部成功；已使用本地可核验新闻")
        except Exception as exc:
            refresh_errors.append(f"官方新闻同步失败：{type(exc).__name__}: {exc}")
    else:
        source_label = "本地历史缓存；需要联网更新时请使用页面上的刷新按钮"

    model_forecast_candidate = read_json(LATEST_LEGACY)
    requested_feed = None
    if args.refresh:
        provider = str(status.get("intraday_provider") or "").lower()
        requested_feed = "eodhd" if provider == "eodhd" else "yahoo" if "雅虎" in provider else None
    if requested_feed is None and model_forecast_candidate and str(model_forecast_candidate.get("feed") or "").lower() in {"iex", "yahoo", "eodhd"}:
        requested_feed = str(model_forecast_candidate.get("feed")).lower()

    # Select one provider's time series; never concatenate vendor feeds into a fake path.
    actual_latest = latest_daily_date(db_path)
    provider_frames: list[tuple[tuple[int, str, int, int], str, pd.DataFrame, list[dict[str, Any]]]] = []
    for rank, feed_name in enumerate(([requested_feed] if requested_feed else []) + [f for f in ("eodhd", "yahoo", "iex") if f != requested_feed]):
        if not feed_name:
            continue
        try:
            candidate_frame = load_intraday_from_db(db_path, feed=feed_name)
            candidate_sessions = _valid_intraday_sessions(candidate_frame)
        except Exception:
            continue
        if not candidate_sessions:
            continue
        candidate_latest_day = str(candidate_sessions[-1]["date"])
        aligned = int(not actual_latest or candidate_latest_day >= actual_latest)
        # Prefer data aligned with daily close, then the most recent complete session,
        # then the caller's explicitly requested provider, then fuller local history.
        provider_frames.append(((aligned, candidate_latest_day, int(feed_name == requested_feed), len(candidate_sessions)), feed_name, candidate_frame, candidate_sessions))
    if provider_frames:
        provider_frames.sort(key=lambda item: item[0], reverse=True)
        _, selected_feed, frame, sessions = provider_frames[0]
        source_label = f"{source_label}；分时曲线统一来源：{selected_feed}"
    else:
        selected_feed, frame, sessions = requested_feed or "unknown", pd.DataFrame(), []
    intraday_candidate = sessions[-1] if sessions else None
    news = load_news(db_path)
    # Do not present cached 15-minute bars as the latest full session if daily data has moved on.
    data_aligned = bool(intraday_candidate and (not actual_latest or intraday_candidate["date"] >= actual_latest))
    latest_session = intraday_candidate if data_aligned else None
    if intraday_candidate and not data_aligned:
        refresh_errors.append(
            f"分时数据只到{intraday_candidate['date']}，但日线已到{actual_latest}；为避免旧图冒充最近交易日，停用过期分时曲线。"
        )

    forecast: dict[str, Any] | None = None
    if latest_session:
        try:
            target = _next_target_day(latest_session["date"])
            # Use only a candidate whose as-of date and target match the current
            # complete session. A stale model JSON can never be passed off as today.
            candidate = model_forecast_candidate or read_json(LATEST_LEGACY)
            candidate_is_current = bool(
                candidate
                and str(candidate.get("feed") or "").lower() == str(latest_session.get("feed") or "").lower()
                and str(candidate.get("as_of_date") or "") == latest_session["date"]
                and str(candidate.get("forecast_date_weekday_estimate") or "") == target
                and candidate.get("capture_mode") == "live_prediction"
                and bool(candidate.get("pre_session_snapshot"))
                and isinstance(candidate.get("curve"), list) and len(candidate.get("curve") or []) >= 18
            )
            if candidate_is_current:
                forecast = candidate
                is_ridge = "ridge" in str(((forecast.get("main_path") or {}).get("model_family")) or "").lower()
                forecast["video_source_kind"] = "trained_intraday_candidate" if is_ridge else "intraday_model_fallback"
            else:
                forecast = build_profile_forecast(sessions, target)
                if forecast:
                    forecast["video_source_kind"] = "historical_profile_only"
                    refresh_errors.append("没有找到与本次数据日期完全匹配的盘中模型预测；第二张图使用历史统计情景，不会标成机器学习结果。")
            if forecast:
                forecast["news_background_latest_date"] = max((_published_day(n.get("published_at")) for n in news), default="")
                forecast["available"] = True
                VIDEO_FORECAST_LATEST.parent.mkdir(parents=True, exist_ok=True)
                VIDEO_FORECAST_LATEST.write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
                (VIDEO_FORECAST_LATEST.parent / f"QQQM_video_forecast_{target}.json").write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                refresh_errors.append("完整分时历史样本不足，暂时不能生成下一交易日曲线。")
        except Exception as exc:
            refresh_errors.append(f"下一交易日情景生成失败：{type(exc).__name__}: {exc}")
    else:
        refresh_errors.append("没有与最新日线日期对齐的完整分时交易日；下一交易日曲线已停用。")

    if forecast is None:
        VIDEO_FORECAST_LATEST.parent.mkdir(parents=True, exist_ok=True)
        VIDEO_FORECAST_LATEST.write_text(json.dumps({
            "available": False, "as_of_date": actual_latest,
            "forecast_date_weekday_estimate": None, "curve": [],
            "reason_zh": refresh_errors[-1] if refresh_errors else "当前没有可用的新预测。",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    if latest_session:
        # A review overlay must be a forecast actually saved or published for this
        # exact session. Never backcast a curve and then visually imply it was
        # published before the session happened.
        review_forecast = _archive_forecast(latest_session["date"], expected_feed=str(selected_feed or ""))
    else:
        review_forecast = None
    translated_news, news_note, news_stale = _chinese_news_context(news, forecast.get("forecast_date_weekday_estimate") if forecast else (latest_session["date"] if latest_session else date.today().isoformat()))

    outdir = Path(args.output_dir) if args.output_dir else OUTROOT / datetime.now().strftime("%Y%m%d")
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    review_img = outdir / "01_上一完整交易日真实走势与预测对比.png"
    next_img = outdir / "02_下一交易日走势情景参考.png"
    review = plot_review(latest_session, review_forecast, review_img, source_label)
    next_available = plot_next(forecast, next_img, news_note)
    md, txt, meta = _make_script(
        datetime.now().strftime("%Y-%m-%d"), review, forecast, next_available,
        translated_news, news_note, actual_latest, source_label, refresh_errors,
    )
    md_path = outdir / "每日行情视频口播稿与发布文案.md"
    txt_path = outdir / "每日视频口播稿_可直接照念.txt"
    publish_path = outdir / "视频标题与简介_直接复制发布.txt"
    meta_path = outdir / "视频内容生成记录.json"
    meta["review_forecast_source_file"] = (review_forecast or {}).get("source_image_file")
    meta["review_forecast_archive_path"] = (review_forecast or {}).get("_archive_source_path")
    meta["review_forecast_source_image_path"] = (review_forecast or {}).get("_archive_source_image_path")
    md_path.write_text(md, encoding="utf-8")
    txt_path.write_text(txt, encoding="utf-8")
    publish_path.write_text(publish_txt, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    zpath = outdir / "每日行情视频脚本与两张配图.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in (md_path, txt_path, publish_path, meta_path, review_img, next_img):
            zf.write(path, arcname=path.name)
        if review_forecast and review_forecast.get("source_image_file"):
            source_image = ARCHIVE / str(review_forecast["source_image_file"])
            if source_image.is_file():
                zf.write(source_image, arcname="复盘依据_此前已发布的预测图.png")
        elif review_forecast and review_forecast.get("_archive_source_image_path"):
            source_image = Path(str(review_forecast["_archive_source_image_path"]))
            if source_image.is_file():
                zf.write(source_image, arcname="复盘依据_此前第二张预测图.png")
        if forecast:
            local_forecast = outdir / "下一交易日走势情景数据.json"
            local_forecast.write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
            zf.write(local_forecast, arcname=local_forecast.name)
            # Keep the exact prediction JSON under reports for next-session scoring;
            # a downloaded PNG alone is not machine-learning feedback.
            forecast_archive_dir = ROOT / "reports" / "video_scripts" / "forecast_archive"
            forecast_archive_dir.mkdir(parents=True, exist_ok=True)
            archive_target = str(forecast.get("forecast_date_weekday_estimate") or "unknown")
            archive_asof = str(forecast.get("as_of_date") or "unknown")
            archived_forecast = forecast_archive_dir / f"QQQM_forecast_asof_{archive_asof}_target_{archive_target}.json"
            if not archived_forecast.exists():
                archived_forecast.write_text(json.dumps(forecast, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "content_dir": str(outdir), "zip": str(zpath), "script_markdown": str(md_path), "voiceover_txt": str(txt_path), "publish_copy_txt": str(publish_path),
        "review_image": str(review_img), "next_image": str(next_img), "daily_latest_date": actual_latest,
        "last_completed_intraday_date": latest_session["date"] if latest_session else None,
        "forecast_target": (forecast or {}).get("forecast_date_weekday_estimate"),
        "forecast_as_of_date": (forecast or {}).get("as_of_date"),
        "forecast_source_kind": (forecast or {}).get("video_source_kind"),
        "forecast_feed": (forecast or {}).get("feed"),
        "forecast_model_family": ((forecast or {}).get("main_path") or {}).get("model_family"),
        "next_forecast_available": next_available, "intraday_source": source_label,
        "news_used": len(translated_news), "news_stale": news_stale, "refresh_errors": refresh_errors,
    }, ensure_ascii=False, indent=2))
    # Content package generation should still finish even if a remote source fails;
    # the output itself explicitly tells the user which chart/data were unavailable.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
