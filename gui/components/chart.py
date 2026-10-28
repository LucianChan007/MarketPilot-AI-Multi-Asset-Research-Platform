"""GUI chart: show the independent main prediction path by default; scenario quantiles are optional."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _resolve_archive(root: Path, summary: dict[str, Any]) -> Path | None:
    raw = summary.get("forecast_archive")
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw.replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate if candidate.exists() else None


def _path_points(payload: dict[str, Any] | None) -> list[tuple[str, float]]:
    if not payload:
        return []
    for key in ("main_path", "market_only_main_path", "historical_cluster_path"):
        candidate = payload.get(key)
        if not isinstance(candidate, dict):
            continue
        price_by_time = candidate.get("price_by_time")
        if isinstance(price_by_time, dict):
            pairs = []
            for slot, raw_price in price_by_time.items():
                price = _to_float(raw_price)
                if price is not None and price > 0:
                    pairs.append((str(slot), price))
            if pairs:
                return sorted(pairs, key=lambda item: item[0])
        path = candidate.get("path")
        if isinstance(path, list):
            pairs = []
            for point in path:
                if not isinstance(point, dict):
                    continue
                slot = point.get("time_ny") or point.get("time")
                price = _to_float(point.get("predicted_price", point.get("price")))
                if slot is not None and price is not None and price > 0:
                    pairs.append((str(slot), price))
            if pairs:
                return sorted(pairs, key=lambda item: item[0])
    # Legacy archives sometimes store a flat list of points.
    for key in ("main_path_curve", "main_path_points", "selected_path_curve"):
        points = payload.get(key)
        if isinstance(points, list):
            result = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                slot = point.get("time_ny") or point.get("time")
                price = _to_float(point.get("predicted_price", point.get("price", point.get("value"))))
                if slot is not None and price is not None and price > 0:
                    result.append((str(slot), price))
            if result:
                return sorted(result, key=lambda item: item[0])
    return []


def _actual_points(root: Path, target_date: str) -> list[tuple[str, float]]:
    summary = _read_json(root / "reports" / "intraday" / "QQQM_intraday_forecast_vs_actual_summary_latest.json") or {}
    if summary.get("target_date") and str(summary.get("target_date")) != str(target_date):
        return []  # Never overlay actual bars from another forecast date.
    path = root / "reports" / "intraday" / "QQQM_intraday_forecast_vs_actual_latest.csv"
    if not path.exists():
        return []
    result: list[tuple[str, float]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                slot = row.get("time_ny") or row.get("time")
                price = _to_float(row.get("actual_close_price", row.get("actual_price")))
                if slot and price is not None and price > 0:
                    result.append((str(slot), price))
    except OSError:
        return []
    return sorted(result, key=lambda item: item[0])


def _interpolate_short_gaps(values: list[float | None], max_gap: int = 2) -> tuple[list[float | None], list[int]]:
    """Visual-only interpolation for short internal gaps. Does not mutate source/metrics."""
    out = list(values)
    interpolated: list[int] = []
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        start = i
        while i < len(out) and out[i] is None:
            i += 1
        size = i - start
        left, right = start - 1, i
        if size and size <= max_gap and left >= 0 and right < len(out) and out[left] is not None and out[right] is not None:
            for j in range(size):
                out[start + j] = float(out[left]) + (float(out[right]) - float(out[left])) * ((j + 1) / (size + 1))
                interpolated.append(start + j)
    return out, interpolated


def build_forecast_figure(root: Path, show_scenarios: bool = False, forecast_override: dict | None = None, summary_override: dict | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    forecast_path = root / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
    summary_path = root / "reports" / "intraday" / "QQQM_intraday_forecast_vs_actual_summary_latest.json"
    forecast = forecast_override if forecast_override is not None else _read_json(forecast_path)
    summary = summary_override if summary_override is not None else (_read_json(summary_path) or {})
    if not forecast or not isinstance(forecast.get("curve"), list) or not forecast["curve"]:
        return None, "没有有效的盘中预测 JSON。请先运行预测生成流程。"

    curve = forecast["curve"]
    times = [str(point.get("time_ny", "")) for point in curve]
    if not times:
        return None, "预测曲线没有时间点。"
    p10 = [_to_float(point.get("low_price")) for point in curve]
    p50 = [_to_float(point.get("median_price")) for point in curve]
    p90 = [_to_float(point.get("high_price")) for point in curve]
    main_path = _path_points(forecast)
    archive_path = _resolve_archive(root, summary)
    if not main_path and archive_path:
        main_path = _path_points(_read_json(archive_path))

    main_by_time = dict(main_path)
    raw_main = [main_by_time.get(slot) for slot in times]
    visual_main, interpolated_indices = _interpolate_short_gaps(raw_main, max_gap=2)
    actual = dict(_actual_points(root, str(forecast.get("forecast_date_weekday_estimate", ""))))
    actual_open = _to_float(summary.get("actual_open")) if actual else None
    actual_close = _to_float(summary.get("actual_close_near_last_bar")) if actual else None

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=125)
    if show_scenarios:
        clean_p10 = [float(v) if v is not None else float("nan") for v in p10]
        clean_p50 = [float(v) if v is not None else float("nan") for v in p50]
        clean_p90 = [float(v) if v is not None else float("nan") for v in p90]
        ax.fill_between(times, clean_p10, clean_p90, alpha=0.10, label="P10–P90 情景范围")
        ax.plot(times, clean_p10, linewidth=1.2, linestyle="--", alpha=0.75, label="P10 · 较低情景")
        ax.plot(times, clean_p50, linewidth=1.3, linestyle=":", alpha=0.85, label="P50 · 中位参考")
        ax.plot(times, clean_p90, linewidth=1.2, linestyle="--", alpha=0.75, label="P90 · 较高情景")

    main_obj = forecast.get("main_path") if isinstance(forecast.get("main_path"), dict) else {}
    family = str(main_obj.get("model_family", ""))
    news_applied = bool(main_obj.get("news_adjustment_applied", False))
    if family == "ridge_regression_per_time_slot":
        label = "主预测路径 · 独立模型"
        if news_applied:
            label += " + 新闻实验调整"
    elif main_path:
        label = "历史参考路径（主模型尚未生成）"
    else:
        label = "主预测路径（暂不可用）"

    if any(v is not None for v in visual_main):
        plot_values = [float(v) if v is not None else float("nan") for v in visual_main]
        main_line, = ax.plot(times, plot_values, linewidth=3.0, linestyle="-", label=label)
        if interpolated_indices:
            ax.scatter(
                [times[i] for i in interpolated_indices],
                [float(visual_main[i]) for i in interpolated_indices],
                marker="o", s=38, facecolors="white", edgecolors=main_line.get_color(),
                linewidths=1.4, zorder=6, label="少量缺失点的图示连接（不参与评估）",
            )

    if actual:
        actual_times = [slot for slot in times if slot in actual]
        if actual_times:
            ax.plot(actual_times, [actual[slot] for slot in actual_times], color="black", linewidth=2.0,
                    marker="o", markersize=2.5, label="实际盘中走势")
    previous_close = _to_float(forecast.get("previous_session_close"))
    if previous_close is not None:
        ax.axhline(previous_close, color="gray", linewidth=1.0, linestyle="--", alpha=0.7,
                   label=f"前一收盘参考 ${previous_close:.2f}")
    main_open = _to_float(main_obj.get("predicted_open_price"))
    main_close = _to_float(main_obj.get("predicted_close_price"))
    if main_open is not None:
        ax.scatter([times[0]], [main_open], marker="D", s=48, zorder=6, label=f"主路径预计开盘 ${main_open:.2f}")
    if main_close is not None:
        ax.scatter([times[-1]], [main_close], marker="X", s=56, zorder=6, label=f"主路径预计收盘 ${main_close:.2f}")
    if actual_open is not None:
        ax.scatter([times[0]], [actual_open], marker="^", s=46, zorder=7, label=f"实际开盘 ${actual_open:.2f}")
    if actual_close is not None:
        ax.scatter([times[-1]], [actual_close], marker="P", s=48, zorder=7, label=f"实际收盘附近 ${actual_close:.2f}")

    target_date = forecast.get("forecast_date_weekday_estimate", "下一交易日")
    ax.set_title(f"QQQM 主预测路径 | 目标日期：{target_date}")
    ax.set_xlabel("美东时间")
    ax.set_ylabel("价格（美元）")
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:.2f}"))
    ax.legend(loc="best", fontsize=8.5, ncols=2)
    fig.tight_layout()

    notes = []
    if family != "ridge_regression_per_time_slot":
        notes.append("当前没有可用的独立主模型结果，显示的是历史参考路径；请先重新生成预测再检查。")
    else:
        notes.append("粗线是系统单独计算的主预测，不是把 P50 当作答案；它仍需用真实行情检验。")
    if news_applied:
        notes.append("这次新闻线索触发了小幅实验调整；目前还没有证据证明它能让预测更准。")
    else:
        notes.append("本次没有足够可靠、方向明确的新闻，因此没有因为新闻改变主预测线。")
    if show_scenarios:
        notes.append("P10/P50/P90 是每个时间点的偏低/中间/偏高参考，不是整条曲线的概率。")
    if not actual:
        notes.append("目标交易日实际 15 分钟数据尚未匹配；收盘同步后才能复盘。")
    if interpolated_indices:
        notes.append(f"为了方便看图，{len(interpolated_indices)} 个短缺时间点做了图示连接；这些补点不参加误差计算或训练。")
    return fig, " ".join(notes)


def render_intraday_chart(st, root: Path, show_scenarios: bool = False, forecast_override: dict | None = None, summary_override: dict | None = None) -> None:
    fig, note = build_forecast_figure(
        root, show_scenarios=show_scenarios,
        forecast_override=forecast_override, summary_override=summary_override,
    )
    if fig is None:
        st.warning(note)
        return
    st.pyplot(fig, use_container_width=True)
    st.caption(note)
    import matplotlib.pyplot as plt
    plt.close(fig)
