from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit(
        r"缺少 matplotlib。请在项目虚拟环境中执行："
        r" .\.venv\Scripts\python.exe -m pip install matplotlib"
    ) from exc

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
TICKER = "QQQM"
EXPECTED_TIMES = [
    f"{hour:02d}:{minute:02d}"
    for hour, minute in (
        [(9, 30), (9, 45), (10, 0), (10, 15), (10, 30), (10, 45)]
        + [(hour, minute) for hour in range(11, 16) for minute in (0, 15, 30, 45)]
        + [(16, 0)]
    )
    if (hour, minute) != (16, 0)
]

TIMESTAMP_ALIASES = (
    "timestamp_utc", "bar_timestamp", "timestamp", "datetime_utc", "datetime",
    "date_time", "bar_time", "time_utc", "ts", "time", "t", "date",
)
SYMBOL_ALIASES = ("symbol", "ticker", "asset", "instrument", "code")
TIMEFRAME_ALIASES = ("timeframe", "interval", "bar_size", "resolution", "timespan")
FEED_ALIASES = ("feed", "data_feed", "source_feed", "provider_feed")
OPEN_ALIASES = ("open", "open_price", "o")
CLOSE_ALIASES = ("close", "close_price", "c")


@dataclass
class Bar:
    session_date: date
    time_ny: str
    timestamp_ny: datetime
    open_price: float
    close_price: float


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias in by_lower:
            return by_lower[alias]
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        try:
            if number > 1e17:  # nanoseconds
                number /= 1e9
            elif number > 1e14:  # microseconds
                number /= 1e6
            elif number > 1e11:  # milliseconds
                number /= 1e3
            dt = datetime.fromtimestamp(number, tz=UTC)
            return dt.astimezone(NY)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return _parse_timestamp(int(text))
        except ValueError:
            return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # A date-only value cannot define an intraday slot.
        return None
    # Alpaca bars are normally stored in UTC. Naive timestamps are interpreted
    # as UTC deliberately, then converted to New York wall time.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(NY)


def _normalize_timeframe(value: Any) -> bool:
    if value is None:
        return True
    normalized = str(value).lower().replace(" ", "").replace("_", "")
    return normalized in {"15min", "15m", "15minute", "15minutes", "15mins", "15"}


def _find_intraday_table(connection: sqlite3.Connection) -> tuple[str, list[str]]:
    table_rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    preferred = [row[0] for row in table_rows if row[0].lower() == "intraday_bars"]
    others = [row[0] for row in table_rows if row[0] not in preferred]

    for table in preferred + others:
        quoted = '"' + table.replace('"', '""') + '"'
        try:
            columns = [row[1] for row in connection.execute(f"PRAGMA table_info({quoted})")]
        except sqlite3.Error:
            continue
        if (
            _pick_column(columns, TIMESTAMP_ALIASES)
            and _pick_column(columns, OPEN_ALIASES)
            and _pick_column(columns, CLOSE_ALIASES)
            and _pick_column(columns, SYMBOL_ALIASES)
        ):
            return table, columns
    raise RuntimeError(
        "数据库中没有找到可识别的分钟行情表。需要一张包含 symbol/ticker、"
        "timestamp、open、close 字段的表；预期表名通常是 intraday_bars。"
    )


def load_intraday_bars(db_path: Path, ticker: str, timeframe: str = "15Min") -> list[Bar]:
    if not db_path.exists():
        raise FileNotFoundError(f"找不到行情数据库：{db_path}")
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        table, columns = _find_intraday_table(connection)
        symbol_col = _pick_column(columns, SYMBOL_ALIASES)
        timestamp_col = _pick_column(columns, TIMESTAMP_ALIASES)
        timeframe_col = _pick_column(columns, TIMEFRAME_ALIASES)
        feed_col = _pick_column(columns, FEED_ALIASES)
        open_col = _pick_column(columns, OPEN_ALIASES)
        close_col = _pick_column(columns, CLOSE_ALIASES)
        assert symbol_col and timestamp_col and open_col and close_col

        # Quote all names because these are schema-derived SQLite identifiers.
        qtable = '"' + table.replace('"', '""') + '"'
        select_cols = list(dict.fromkeys(
            [symbol_col, timestamp_col, open_col, close_col]
            + ([timeframe_col] if timeframe_col else [])
            + ([feed_col] if feed_col else [])
        ))
        select_sql = ", ".join('"' + c.replace('"', '""') + '"' for c in select_cols)
        rows = connection.execute(f"SELECT {select_sql} FROM {qtable}").fetchall()

        bars: list[Bar] = []
        for row in rows:
            symbol = str(row[symbol_col]).upper() if row[symbol_col] is not None else ""
            if symbol != ticker.upper():
                continue
            if timeframe_col and not _normalize_timeframe(row[timeframe_col]):
                continue
            if feed_col and row[feed_col] is not None and str(row[feed_col]).lower() != "iex":
                continue
            stamp = _parse_timestamp(row[timestamp_col])
            if stamp is None:
                continue
            # Keep regular-session bars only. The model's curve is indexed by NY time.
            slot_time = stamp.strftime("%H:%M")
            if slot_time not in EXPECTED_TIMES:
                continue
            try:
                open_price = float(row[open_col])
                close_price = float(row[close_col])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(open_price) and math.isfinite(close_price)):
                continue
            if open_price <= 0 or close_price <= 0:
                continue
            bars.append(Bar(stamp.date(), slot_time, stamp, open_price, close_price))
        return sorted(bars, key=lambda bar: (bar.session_date, bar.timestamp_ny))
    finally:
        connection.close()


def _select_representative_path(
    bars: list[Bar], forecast: dict[str, Any]
) -> tuple[date | None, dict[str, float], float | None, int]:
    """Choose the completed historical session whose observed path is closest to P50.

    This is an empirical medoid-like representative path, NOT a calibrated
    maximum-probability path. Only observations available by as_of_date are used.
    """
    as_of = date.fromisoformat(str(forecast["as_of_date"]))
    curve = forecast.get("curve", [])
    p50_by_time: dict[str, float] = {}
    for point in curve:
        value = point.get("p50_return_from_open_pct")
        if value is not None:
            p50_by_time[str(point["time_ny"])] = float(value)

    by_day: dict[date, dict[str, Bar]] = defaultdict(dict)
    for bar in bars:
        if bar.session_date <= as_of:
            by_day[bar.session_date][bar.time_ny] = bar

    candidates: list[tuple[float, date, dict[str, float]]] = []
    for session_date, day_bars in by_day.items():
        # Use the 09:30 bar's open as the session opening anchor.
        first = day_bars.get("09:30")
        if first is None or len(day_bars) < 20:
            continue
        anchor = first.open_price
        path_returns: dict[str, float] = {}
        for time_label, bar in day_bars.items():
            path_returns[time_label] = (bar.close_price / anchor - 1.0) * 100.0
        common = [t for t in p50_by_time if t in path_returns]
        if len(common) < 20:
            continue
        mse = sum((path_returns[t] - p50_by_time[t]) ** 2 for t in common) / len(common)
        rmse_pct = math.sqrt(mse)
        candidates.append((rmse_pct, session_date, path_returns))

    if not candidates:
        return None, {}, None, 0
    rmse_pct, best_date, best_returns = min(candidates, key=lambda item: item[0])
    return best_date, best_returns, rmse_pct, len(candidates)


def _choose_chinese_font() -> None:
    try:
        from matplotlib import font_manager
        available = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in (
            "Microsoft YaHei", "Microsoft JhengHei", "SimHei", "Noto Sans CJK SC",
            "Source Han Sans SC", "PingFang SC", "Arial Unicode MS",
        ):
            if candidate in available:
                plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
                break
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def _safe_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"预测 JSON 顶层应为对象：{path}")
    return data


def _save_forecast_archive(forecast_path: Path, output_dir: Path, target_date: str) -> Path:
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"QQQM_intraday_forecast_{target_date}.json"
    if forecast_path.resolve().parent == archive_dir.resolve():
        # An explicitly selected dated archive is the source of truth for after-the-fact review.
        return forecast_path
    source = _safe_json(forecast_path)
    if not archive_path.exists():
        shutil.copy2(forecast_path, archive_path)
        return archive_path
    try:
        existing = _safe_json(archive_path)
    except Exception:
        existing = {}
    # Never overwrite an existing forecast. If a legacy archive predates the snapshot schema,
    # keep it and preserve the new prospective snapshot under a versioned filename.
    if source.get("forecast_snapshot_version") and not existing.get("forecast_snapshot_version"):
        stamp = str(source.get("generated_at_utc", "")).replace(":", "").replace("-", "")
        stamp = stamp.replace("T", "_").replace("Z", "")[:15] or "live"
        versioned = archive_dir / f"QQQM_intraday_forecast_{target_date}_live_{stamp}.json"
        if not versioned.exists():
            shutil.copy2(forecast_path, versioned)
        return versioned
    return archive_path


def _extract_path_prices(path_obj: Any) -> dict[str, float]:
    if not isinstance(path_obj, dict):
        return {}
    values = path_obj.get("price_by_time")
    if isinstance(values, dict):
        result: dict[str, float] = {}
        for slot, value in values.items():
            try:
                price = float(value)
                if math.isfinite(price) and price > 0:
                    result[str(slot)] = price
            except (TypeError, ValueError):
                continue
        if result:
            return result
    result = {}
    for point in path_obj.get("path", []) if isinstance(path_obj.get("path"), list) else []:
        if not isinstance(point, dict):
            continue
        try:
            result[str(point["time_ny"])] = float(point.get("predicted_price", point.get("price")))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _interpolate_short_visual_gaps(
    values: list[float | None], max_gap_points: int = 2
) -> tuple[list[float | None], list[int]]:
    """Fill short gaps for display only; never mutate the raw prediction path.

    A gap is bridged only when it is bounded by valid prices on both sides and has at most
    ``max_gap_points`` missing 15-minute slots. Returned indices are the interpolated points.
    These values are for chart readability only and must not be treated as model observations.
    """
    result = list(values)
    interpolated_indices: list[int] = []
    n = len(result)
    i = 0
    while i < n:
        if result[i] is not None and math.isfinite(float(result[i])):
            i += 1
            continue
        start = i
        while i < n and (result[i] is None or not math.isfinite(float(result[i]))):
            i += 1
        end_exclusive = i
        gap_size = end_exclusive - start
        left_idx = start - 1
        right_idx = end_exclusive
        if (
            gap_size > 0
            and gap_size <= max_gap_points
            and left_idx >= 0
            and right_idx < n
            and result[left_idx] is not None
            and result[right_idx] is not None
            and math.isfinite(float(result[left_idx]))
            and math.isfinite(float(result[right_idx]))
        ):
            left = float(result[left_idx])
            right = float(result[right_idx])
            for offset, idx in enumerate(range(start, end_exclusive), start=1):
                fraction = offset / (gap_size + 1)
                result[idx] = left + (right - left) * fraction
                interpolated_indices.append(idx)
    return result, interpolated_indices


def _write_comparison_csv(
    output_path: Path,
    forecast: dict[str, Any],
    main_path_prices: dict[str, float],
    market_only_prices: dict[str, float],
    actual_by_time: dict[str, float],
    main_path_visual_prices: dict[str, float] | None = None,
    interpolated_slots: set[str] | None = None,
) -> None:
    curve = forecast.get("curve", [])
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        visual_prices = main_path_visual_prices or {}
        interpolated = interpolated_slots or set()
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_ny", "p10_price", "p50_price", "p90_price", "main_path_price",
                        "main_path_visual_price", "main_path_visual_interpolated",
                        "market_only_path_price", "actual_close_price", "historical_slot_samples"],
        )
        writer.writeheader()
        for point in curve:
            slot = str(point["time_ny"])
            writer.writerow({
                "time_ny": slot,
                "p10_price": point.get("low_price", ""),
                "p50_price": point.get("median_price", ""),
                "p90_price": point.get("high_price", ""),
                # Raw candidate path stays missing when the source session had no bar.
                # Interpolated values are separately labelled as display-only and must not
                # be used as model training targets or source observations.
                "main_path_price": main_path_prices.get(slot, ""),
                "main_path_visual_price": visual_prices.get(slot, ""),
                "main_path_visual_interpolated": "yes" if slot in interpolated else "no",
                "market_only_path_price": market_only_prices.get(slot, ""),
                "actual_close_price": actual_by_time.get(slot, ""),
                "historical_slot_samples": point.get("historical_slot_samples", ""),
            })


def _path_mae_bps(predicted: dict[str, float], actual_by_time: dict[str, float]) -> float | None:
    errors = [abs(predicted[t] - actual_by_time[t]) / actual_by_time[t] * 10000.0
              for t in predicted if t in actual_by_time and actual_by_time[t] > 0]
    return round(sum(errors) / len(errors), 3) if errors else None

def create_chart(
    forecast_path: Path,
    db_path: Path,
    output_dir: Path,
    explicit_output: Path | None = None,
) -> dict[str, Any]:
    forecast = _safe_json(forecast_path)
    target_date = str(forecast.get("forecast_date_weekday_estimate", ""))
    as_of_date = str(forecast.get("as_of_date", ""))
    if not target_date or not as_of_date:
        raise ValueError("预测 JSON 缺少 as_of_date 或 forecast_date_weekday_estimate")
    target = date.fromisoformat(target_date)
    estimated_open = forecast.get("estimated_open", {})
    open_p10 = float(estimated_open["p10_price"])
    open_p50 = float(estimated_open["p50_price"])
    open_p90 = float(estimated_open["p90_price"])
    prev_close = float(forecast.get("previous_session_close", open_p50))
    close_summary = forecast.get("estimated_close_from_path", {})

    bars = load_intraday_bars(db_path, str(forecast.get("ticker", TICKER)), "15Min")
    main_path_obj = forecast.get("main_path")
    market_only_path_obj = forecast.get("market_only_main_path")
    main_path_prices = _extract_path_prices(main_path_obj)
    market_only_prices = _extract_path_prices(market_only_path_obj)
    if isinstance(main_path_obj, dict) and main_path_prices:
        representative_date = None
        date_value = main_path_obj.get("selected_neighbor_date")
        try:
            representative_date = date.fromisoformat(str(date_value)) if date_value else None
        except ValueError:
            representative_date = None
        representative_rmse_pct = None
        candidate_count = int(main_path_obj.get("candidate_count") or len(forecast.get("candidate_paths", [])))
    else:
        representative_date, representative_returns, representative_rmse_pct, candidate_count = \
            _select_representative_path(bars, forecast)
        open_anchor = float(estimated_open.get("p50_price") or open_p50)
        main_path_prices = {
            slot: open_anchor * (1.0 + ret / 100.0)
            for slot, ret in representative_returns.items()
        }

    target_bars = [bar for bar in bars if bar.session_date == target]
    actual_by_time = {bar.time_ny: bar.close_price for bar in target_bars}
    actual_open = next((bar.open_price for bar in target_bars if bar.time_ny == "09:30"), None)
    actual_close = actual_by_time.get("15:45")
    # A real-close marker can be found at the final slot even if some intermediate bars are missing.
    if actual_close is None and target_bars:
        actual_close = target_bars[-1].close_price

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = _save_forecast_archive(forecast_path, output_dir, target_date)
    chart_archive_dir = output_dir / "archive" / "charts"
    chart_archive_dir.mkdir(parents=True, exist_ok=True)
    dated_png = output_dir / f"QQQM_intraday_forecast_vs_actual_{target_date}.png"
    latest_png = explicit_output or (output_dir / "QQQM_intraday_forecast_vs_actual_latest.png")
    dated_csv = output_dir / f"QQQM_intraday_forecast_vs_actual_{target_date}.csv"
    latest_csv = output_dir / "QQQM_intraday_forecast_vs_actual_latest.csv"

    curve = forecast.get("curve", [])
    if not curve:
        raise ValueError("预测 JSON 的 curve 数组为空")
    times = [str(point["time_ny"]) for point in curve]
    p10 = [float(point["low_price"]) for point in curve]
    p50 = [float(point["median_price"]) for point in curve]
    p90 = [float(point["high_price"]) for point in curve]
    x_positions = list(range(len(times)))
    x_by_time = {label: index for index, label in enumerate(times)}
    main_path_prices_aligned: list[float | None] = [main_path_prices.get(slot) for slot in times]
    # The selected historical IEX path may lack individual 15-minute bars. Matplotlib
    # breaks a line at None, so bridge only short gaps for visual continuity. Raw prices,
    # error calculations and training data remain untouched.
    main_path_display_aligned, interpolated_indices = _interpolate_short_visual_gaps(
        main_path_prices_aligned, max_gap_points=2
    )
    interpolated_slots = {times[idx] for idx in interpolated_indices}
    main_path_visual_prices = {
        slot: float(value) for slot, value in zip(times, main_path_display_aligned)
        if value is not None and math.isfinite(float(value))
    }
    path_selection = forecast.get("path_selection_diagnostics", {}) or {}
    main_path_news_applied = bool(path_selection.get("news_adjustment_applied", False))
    if isinstance(main_path_obj, dict) and main_path_prices:
        family = str(main_path_obj.get("model_family", ""))
        if family == "ridge_regression_per_time_slot":
            main_path_label = (
                "主预测路径（市场特征岭回归 + 新闻实验调整）" if main_path_news_applied
                else "主预测路径（独立市场特征岭回归）"
            )
        elif main_path_obj.get("selected_neighbor_date"):
            main_path_label = "历史代表路径（对照，不是主模型预测）"
        else:
            main_path_label = "主预测路径（实验模型）"
    else:
        main_path_label = "历史代表路径（旧版兼容；不是独立模型预测）"

    _choose_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=150)
    ax.fill_between(x_positions, p10, p90, alpha=0.12, label="P10–P90 历史情景范围")
    ax.plot(x_positions, p10, linewidth=1.8, linestyle="--", label="P10 偏弱情景")
    ax.plot(x_positions, p50, linewidth=2.3, label="P50 中位路径")
    if any(value is not None for value in main_path_prices_aligned):
        neighbor_label = (
            f"｜历史参考日 {representative_date.isoformat()}"
            if representative_date and (main_path_obj or {}).get("selected_neighbor_date") else ""
        )
        display_label = main_path_label + neighbor_label
        if interpolated_indices:
            display_label += "（短缺口仅图示插值）"
        main_line, = ax.plot(
            x_positions, main_path_display_aligned, linewidth=3.0, linestyle="-.",
            label=display_label,
        )
        if interpolated_indices:
            # Hollow markers make chart-only interpolated points visible to a human reviewer.
            ax.scatter(
                [x_positions[idx] for idx in interpolated_indices],
                [float(main_path_display_aligned[idx]) for idx in interpolated_indices],
                marker="o", s=34, facecolors="white", edgecolors=main_line.get_color(),
                linewidths=1.5, zorder=6,
            )
    else:
        ax.text(
            0.02, 0.96,
            "暂时没有可用的完整候选路径；请检查预测 JSON 的 candidate_paths/main_path。",
            transform=ax.transAxes, va="top", fontsize=10,
        )
    ax.plot(x_positions, p90, linewidth=1.8, linestyle="--", label="P90 偏强情景")

    if actual_by_time:
        actual_x = [x_by_time[t] for t in times if t in actual_by_time]
        actual_y = [actual_by_time[t] for t in times if t in actual_by_time]
        ax.plot(actual_x, actual_y, color="black", linewidth=2.6, marker="o", markersize=2.5,
                label=f"实际盘中价格（{len(actual_y)} 个点）")
        if actual_open is not None:
            ax.scatter([0], [actual_open], marker="D", s=58, zorder=5, label=f"实际开盘 ${actual_open:.2f}")
        if actual_close is not None:
            close_x = x_by_time.get("15:45", actual_x[-1] if actual_x else len(times) - 1)
            ax.scatter([close_x], [actual_close], marker="X", s=75, zorder=5,
                       label=f"实际收盘附近 ${actual_close:.2f}")
    else:
        ax.text(
            0.99, 0.02,
            f"目标交易日 {target_date} 的实际 15 分钟行情尚未在数据库中找到",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        )

    ax.axhline(prev_close, color="gray", linewidth=1.0, linestyle=":",
               label=f"前一交易日收盘 ${prev_close:.2f}")
    ax.scatter([0], [open_p50], marker="D", s=48, zorder=5,
               label=f"P50 预计开盘 ${open_p50:.2f}")
    main_open = (main_path_obj or {}).get("predicted_open_price") if isinstance(main_path_obj, dict) else None
    main_close = (main_path_obj or {}).get("predicted_close_price") if isinstance(main_path_obj, dict) else None
    if main_open is not None:
        ax.scatter([0], [float(main_open)], marker="*", s=115, zorder=6,
                   label=f"主路径预计开盘 ${float(main_open):.2f}")
    ax.scatter([len(times) - 1], [float(close_summary.get("p50_price", p50[-1]))],
               marker="s", s=45, zorder=5,
               label=f"P50 预计收盘 ${float(close_summary.get('p50_price', p50[-1])):.2f}")
    if main_close is not None:
        ax.scatter([len(times) - 1], [float(main_close)], marker="*", s=100, zorder=6,
                   label=f"主路径预计收盘 ${float(main_close):.2f}")

    ax.set_title(
        f"{forecast.get('ticker', TICKER)} 下一交易日盘中路径｜预测日期 {target_date}\n"
        f"预测基于 {as_of_date} 收盘；历史来源 Alpaca IEX / 15Min",
        fontsize=14, pad=14,
    )
    ax.set_xlabel("纽约时间（ET）")
    ax.set_ylabel("价格（美元）")
    tick_indices = [i for i, t in enumerate(times) if t.endswith(":30") or t in {"09:30", "12:00", "15:00", "15:45"}]
    if len(tick_indices) < 5:
        tick_indices = list(range(0, len(times), 4))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([times[i] for i in tick_indices])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.2f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncols=2, fontsize=8.5, framealpha=0.94)
    news_context = forecast.get("news_context", {}) or {}
    news_signal_obj = path_selection.get("news_signal", {}) or {}
    fig.text(
        0.01, 0.01,
        "说明：主预测路径独立于 P10/P50/P90；当前若为岭回归版本，则来自市场特征对开盘跳空和各盘中时点收益的独立回归预测。"
        "新闻方向信号仅在至少两条高相关、高置信方向性新闻时做封顶实验叠加，尚未证明提高准确度。"
        "P10/P50/P90 是各时间点的历史分位情景，不是整条路径的概率，也不是未来概率保证。"
        f"当前纳入新闻 {news_context.get('included_article_count', 0)} 条；新闻加权={'是' if main_path_news_applied else '否'}；"
        f"方向信号={news_signal_obj.get('signal', 0)}。"
        + (f" 主路径图示对 {len(interpolated_indices)} 个短缺失点做了线性连接（空心点标记）；原始预测 CSV、误差计算和训练数据不使用插值值。" if interpolated_indices else " 主路径未做图示插值。"),
        fontsize=8.2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(dated_png, bbox_inches="tight")
    if latest_png != dated_png:
        shutil.copy2(dated_png, latest_png)
    plt.close(fig)

    # Keep the first forecast-only picture; after the target session closes, keep a separate
    # completed actual-vs-forecast image as human-review material.
    if not actual_by_time:
        forecast_only = chart_archive_dir / f"QQQM_intraday_forecast_only_{target_date}.png"
        if not forecast_only.exists():
            shutil.copy2(dated_png, forecast_only)
    if "15:45" in actual_by_time:
        final_review = chart_archive_dir / f"QQQM_intraday_forecast_vs_actual_{target_date}.png"
        shutil.copy2(dated_png, final_review)

    _write_comparison_csv(
        dated_csv, forecast, main_path_prices, market_only_prices, actual_by_time,
        main_path_visual_prices=main_path_visual_prices, interpolated_slots=interpolated_slots,
    )
    if latest_csv != dated_csv:
        shutil.copy2(dated_csv, latest_csv)

    summary = {
        "ticker": forecast.get("ticker", TICKER),
        "as_of_date": as_of_date,
        "target_date": target_date,
        "forecast_archive": str(archive_path),
        "chart_latest": str(latest_png),
        "chart_dated": str(dated_png),
        "comparison_csv": str(latest_csv),
        "previous_close": prev_close,
        "predicted_open_p10": open_p10,
        "predicted_open_p50": open_p50,
        "predicted_open_p90": open_p90,
        "predicted_close_p10": close_summary.get("p10_price"),
        "predicted_close_p50": close_summary.get("p50_price"),
        "predicted_close_p90": close_summary.get("p90_price"),
        "main_path_selected_neighbor_date": (main_path_obj or {}).get("selected_neighbor_date") if isinstance(main_path_obj, dict) else (representative_date.isoformat() if representative_date else None),
        "representative_historical_session": (main_path_obj or {}).get("selected_neighbor_date") if isinstance(main_path_obj, dict) else (representative_date.isoformat() if representative_date else None),
        "representative_path_rmse_vs_p50_pct": round(representative_rmse_pct, 5) if representative_rmse_pct is not None else None,
        "main_path_selection_method": (main_path_obj or {}).get("selection_method") if isinstance(main_path_obj, dict) else "legacy_p50_nearest_historical_path",
        "main_path_cluster_member_count": (main_path_obj or {}).get("cluster_member_count") if isinstance(main_path_obj, dict) else None,
        "main_path_cluster_weight_share_not_calibrated_probability": (main_path_obj or {}).get("cluster_weight_share_not_calibrated_probability") if isinstance(main_path_obj, dict) else None,
        "main_path_open_price": main_open,
        "main_path_close_price": main_close,
        "news_adjustment_applied": main_path_news_applied,
        "news_signal": news_signal_obj,
        "news_article_count_at_forecast_cutoff": news_context.get("included_article_count", 0),
        "historical_sessions_considered": candidate_count,
        "main_path_raw_points": sum(value is not None for value in main_path_prices_aligned),
        "main_path_expected_points": len(times),
        "main_path_visual_interpolated_count": len(interpolated_indices),
        "main_path_visual_interpolated_slots": sorted(interpolated_slots),
        "main_path_visual_interpolation_is_training_data": False,
        "actual_points_found": len(actual_by_time),
        "actual_open": actual_open,
        "actual_close_near_last_bar": actual_close,
        "main_path_curve_mae_bps_vs_actual": _path_mae_bps(main_path_prices, actual_by_time) if actual_by_time else None,
        "market_only_path_curve_mae_bps_vs_actual": _path_mae_bps(market_only_prices, actual_by_time) if actual_by_time else None,
        "p50_curve_mae_bps_vs_actual": _path_mae_bps({str(point['time_ny']): float(point['median_price']) for point in curve}, actual_by_time) if actual_by_time else None,
        "main_path_model": (main_path_obj or {}).get("model_name") if isinstance(main_path_obj, dict) else None,
        "main_path_model_family": (main_path_obj or {}).get("model_family") if isinstance(main_path_obj, dict) else None,
        "main_path_independent_of_quantile_curves": bool((main_path_obj or {}).get("independent_of_quantile_curves", False)) if isinstance(main_path_obj, dict) else False,
        "path_definition": (
            "主路径由市场特征的扩展窗口岭回归对开盘跳空和各盘中时间点收益独立预测；"
            "新闻只在高相关、高置信方向性证据足够时做封顶实验叠加。它不是校准后的最高概率路径。"
            if isinstance(main_path_obj, dict) and main_path_obj.get("model_family") == "ridge_regression_per_time_slot"
            else "当前主路径来自历史候选代表路径，是旧版兼容/对照方法，不是校准后的最高概率路径。"
        ),
    }
    summary_path = output_dir / "QQQM_intraday_forecast_vs_actual_summary_latest.json"
    summary_dated_path = output_dir / f"QQQM_intraday_forecast_vs_actual_summary_{target_date}.json"
    for path in {summary_path, summary_dated_path}:
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="绘制 QQQM 下一交易日盘中预测四线图，并在实际行情存在时叠加真实走势。"
    )
    parser.add_argument("--forecast", default="reports/intraday/QQQM_intraday_forecast_latest.json",
                        help="预测 JSON 路径；默认读取 latest 文件")
    parser.add_argument("--db", default="data/market_data.db", help="SQLite 行情数据库路径")
    parser.add_argument("--output-dir", default="reports/intraday", help="图表与对比文件输出目录")
    parser.add_argument("--output", default=None, help="可选：指定最新图 PNG 的完整输出路径")
    parser.add_argument("--all-archives", action="store_true",
                        help="重画全部预测归档；有完整实际行情的日期会叠加真实走势")
    args = parser.parse_args()

    try:
        db_path = Path(args.db)
        output_dir = Path(args.output_dir)
        if args.all_archives:
            archive_dir = output_dir / "archive"
            paths = sorted(archive_dir.glob("QQQM_intraday_forecast_*.json")) if archive_dir.exists() else []
            # Prefer prospective snapshots over older legacy/chart-only files for the same target date.
            chosen: dict[str, Path] = {}
            for path in paths:
                try:
                    data = _safe_json(path)
                    target = str(data.get("forecast_date_weekday_estimate", ""))
                except Exception:
                    continue
                if not target:
                    continue
                current = chosen.get(target)
                if current is None:
                    chosen[target] = path
                    continue
                try:
                    current_data = _safe_json(current)
                except Exception:
                    current_data = {}
                new_is_versioned = bool(data.get("forecast_snapshot_version"))
                current_is_versioned = bool(current_data.get("forecast_snapshot_version"))
                if new_is_versioned and not current_is_versioned:
                    chosen[target] = path
                elif new_is_versioned == current_is_versioned:
                    # Keep first immutable version for that target date.
                    continue
            if not chosen:
                raise FileNotFoundError(f"归档目录没有可绘制预测 JSON：{archive_dir}")
            summaries = []
            for target, path in sorted(chosen.items()):
                summaries.append(create_chart(path, db_path, output_dir))
            summary = {"archive_mode": True, "charts_generated": len(summaries), "summaries": summaries,
                       "latest_chart": summaries[-1].get("chart_latest") if summaries else None}
        else:
            summary = create_chart(
                Path(args.forecast), db_path, output_dir,
                Path(args.output) if args.output else None,
            )
    except Exception as exc:
        print(f"生成盘中预测图失败：{exc}", file=sys.stderr)
        return 1

    print("========== QQQM 盘中预测与实际走势图 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n图表已保存。目标交易日未结束时显示预测；完成交易日的归档会叠加实际走势，并保留人工复盘图片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
