from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
FORECAST_PATH = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
VIDEO_FORECAST_PATH = ROOT / "reports" / "intraday" / "QQQM_video_forecast_latest.json"
MARKET_DB_PATH = ROOT / "data" / "market_data.db"
NEW_YORK = ZoneInfo("America/New_York")

st.title("📈 QQQM 下一交易日走势预测")
st.caption("先看预计开盘、盘中路径与预计收盘。曲线是模型估计，不是未来真实行情，也不保证预测准确。")
with st.expander("新手读图：看哪几条线？", expanded=False):
    st.markdown("- **主路径**：当前主模型给出的盘中价格参考。\n- **偏弱 / 偏强范围**：从历史样本整理出的情景，不是保证最低价或最高价。\n- **历史中位基线**：简单对照组。若模型曲线长期没有比它更好，就不应自称预测能力更强。\n- **K 线**：每根蜡烛汇总一个时间段的开盘、最高、最低、收盘；5 分钟 K 线比日 K 线更细，但需要真实、可靠的盘中数据。")

legacy_data: dict[str, Any] | None = None
video_data: dict[str, Any] | None = None
try:
    if FORECAST_PATH.exists():
        legacy_data = json.loads(FORECAST_PATH.read_text(encoding="utf-8-sig"))
    if VIDEO_FORECAST_PATH.exists():
        video_data = json.loads(VIDEO_FORECAST_PATH.read_text(encoding="utf-8-sig"))
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"读取预测文件失败：{exc}")
    st.stop()

# If the dedicated public-source workflow generated a newer valid curve, show it here
# too. Otherwise retain the legacy model snapshot, but never display it as a future path
# once its target day has passed or the real daily database has moved beyond it.
data: dict[str, Any] | None = legacy_data
if video_data and video_data.get("available") is True and isinstance(video_data.get("curve"), list) and video_data.get("curve"):
    video_target = str(video_data.get("forecast_date_weekday_estimate") or "")
    video_asof = str(video_data.get("as_of_date") or "")
    legacy_target = str((legacy_data or {}).get("forecast_date_weekday_estimate") or "")
    try:
        with sqlite3.connect(MARKET_DB_PATH) as _conn:
            _row = _conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
        _daily_latest = str(_row[0])[:10] if _row and _row[0] else None
    except sqlite3.Error:
        _daily_latest = None
    video_curve_is_fresh = bool(video_target and video_asof and (_daily_latest is None or (video_asof >= _daily_latest and video_target > _daily_latest)))
    if video_curve_is_fresh and (not legacy_data or video_target >= legacy_target):
        data = video_data
if data is None:
    st.warning("还没有可用的盘中预测结果。请到“多平台视频脚本与配图”刷新公开行情并生成新情景。")
    st.stop()

summary = data.get("backtest_summary", {})
quality = data.get("data_quality", {})
open_est = data.get("estimated_open", {})
close_est = data.get("estimated_close_from_path", {})
prev_close = float(data.get("previous_session_close") or 0)
forecast_date = str(data.get("forecast_date_weekday_estimate", "未知"))
as_of_date = str(data.get("as_of_date", "未知"))
latest_daily_date = None
try:
    with sqlite3.connect(MARKET_DB_PATH) as conn:
        row = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
        latest_daily_date = str(row[0])[:10] if row and row[0] else None
except sqlite3.Error:
    pass
now_ny = datetime.now(timezone.utc).astimezone(NEW_YORK)
market_session_closed = now_ny.time().replace(tzinfo=None) >= time(16, 15)
stale_reasons = []
if latest_daily_date and forecast_date not in ("", "未知") and forecast_date <= latest_daily_date:
    stale_reasons.append(f"预测目标日 {forecast_date} 已不晚于真实日线最新日期 {latest_daily_date}")
if forecast_date not in ("", "未知"):
    try:
        target_day = date.fromisoformat(forecast_date[:10])
        if target_day < now_ny.date() or (target_day == now_ny.date() and market_session_closed):
            stale_reasons.append(f"预测目标日 {forecast_date} 的美股交易时段已经结束或已过去")
    except ValueError:
        pass
if stale_reasons:
    st.error("当前保存的曲线已过期：" + "；".join(dict.fromkeys(stale_reasons)) + "。为避免把旧数据冒充新预测，本页已停止显示该曲线。请到“多平台视频脚本与配图”刷新行情并生成新情景；原有学习闭环需要密钥时，再到高级功能单独运行。")
    st.stop()

if data.get("curve_method"):
    source_name = {"yahoo": "公开免密行情", "eodhd": "已配置的备用行情源"}.get(str(data.get("feed") or "").lower(), "公开行情历史统计")
    st.success(f"当前显示新生成的情景曲线。依据：{data.get('curve_method')}；行情来源：{source_name}。")

if data.get("curve_method"):
    st.info(f"预测基准：美股 {as_of_date} 收盘；下一交易日情景目标为 {forecast_date}。目标日按项目交易日历估算，曲线来自历史分时统计。")
else:
    st.info(
        f"预测基准：美股 {as_of_date} 收盘；下一交易日暂估为 {forecast_date}。"
        "原有模型的目标日期算法可能没有覆盖全部美股假日，请核对交易日。"
    )

if data.get("curve_method"):
    st.caption("这条新曲线是按完整分时历史统计出的走势情景，不是保证价格；新闻会在视频内容页面作中文背景解释，不会被强行当成涨跌信号。")
elif data.get("conditional_model_available"):
    st.success("当前显示：相似历史行情候选路径。模型会寻找过去市场状态较接近的交易日，再汇总这些交易日的盘中走势。")
else:
    st.warning("相似历史行情候选模型当前不可用，页面使用历史中位路径基线。")

if not data.get("curve_method"):
    st.warning(
        "当前曲线来自原有盘中模型，使用的历史样本与数据源有限。"
        "这版是实验性统计候选模型，未证明能稳定优于简单基线；情景区间也不是经过校准的真实概率。"
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("上一交易日收盘", f"${prev_close:,.2f}" if prev_close else "暂无")
col2.metric("预计开盘中位情景", f"${float(open_est.get('p50_price') or 0):,.2f}" if open_est else "暂无")
col3.metric("预计收盘中位情景", f"${float(close_est.get('p50_price') or 0):,.2f}" if close_est else "暂无")
col4.metric("可用历史交易日", str(data.get("eligible_training_sessions", 0)))

curve = data.get("curve", [])
baseline_curve = data.get("baseline_curve", [])
if curve:
    df = pd.DataFrame(curve)
    base_df = pd.DataFrame(baseline_curve) if baseline_curve else pd.DataFrame()
    chart = pd.DataFrame({
        "时间（美东）": df["time_ny"],
        "偏弱情景": pd.to_numeric(df["low_price"], errors="coerce"),
        "相似行情中位预测": pd.to_numeric(df["median_price"], errors="coerce"),
        "偏强情景": pd.to_numeric(df["high_price"], errors="coerce"),
    })
    if not base_df.empty:
        baseline_by_time = base_df.set_index("time_ny")["median_price"].to_dict()
        chart["历史中位基线"] = chart["时间（美东）"].map(baseline_by_time)

    st.subheader("下一交易日：预计价格路径")
    st.line_chart(chart.set_index("时间（美东）"), width="stretch", height=420)
    st.caption(
        "蓝/中位线是相似历史交易日给出的候选路径；上下边界是所选历史样本的经验情景。"
        "灰/参考线（如存在）是旧的历史中位基线。我们不人为加波动：如果线条仍较平，代表现有数据没有提供稳定的盘中特征。"
    )

    # A change-from-open chart makes small intraday movements easier to see without exaggerating price levels.
    pct_chart = pd.DataFrame({
        "时间（美东）": df["time_ny"],
        "偏弱情景（%）": pd.to_numeric(df["p10_return_from_open_pct"], errors="coerce"),
        "中位路径（%）": pd.to_numeric(df["p50_return_from_open_pct"], errors="coerce"),
        "偏强情景（%）": pd.to_numeric(df["p90_return_from_open_pct"], errors="coerce"),
    })
    st.subheader("盘中累计涨跌幅（相对预计开盘）")
    st.line_chart(pct_chart.set_index("时间（美东）"), width="stretch", height=300)
    st.caption("这张图用百分比显示盘中变化，更容易看出小幅波动；它和上面的美元价格图来自同一条预测路径。")

    with st.expander("查看每个时间点的预测价格和样本数"):
        table = df.rename(columns={
            "time_ny": "美东时间",
            "low_price": "偏弱情景价格",
            "median_price": "中位预测价格",
            "high_price": "偏强情景价格",
            "historical_slot_samples": "该时间点可用历史样本数",
            "p10_return_from_open_pct": "偏弱累计涨跌幅（%）",
            "p50_return_from_open_pct": "中位累计涨跌幅（%）",
            "p90_return_from_open_pct": "偏强累计涨跌幅（%）",
        })
        st.dataframe(table, hide_index=True, width="stretch")

st.subheader("这条曲线过去测试得怎么样？")
if summary:
    c1, c2, c3 = st.columns(3)
    cond_mae = summary.get("path_mae_bps_conditional_neighbor")
    base_mae = summary.get("path_mae_bps_historical_profile")
    flat_mae = summary.get("path_mae_bps_flat_after_open_baseline")
    c1.metric("相似行情候选路径误差", f"{cond_mae:.1f} bps" if cond_mae is not None else "样本不足", help="数值越低越好，仅计实际存在的盘中 K 线点。")
    c2.metric("历史中位基线误差", f"{base_mae:.1f} bps" if base_mae is not None else "样本不足", help="旧版按各时段历史累计收益分位构建的曲线。")
    c3.metric("开盘后保持不变基线", f"{flat_mae:.1f} bps" if flat_mae is not None else "样本不足", help="简单对照：开盘后每个时点都预测为开盘价。")

    improvement = summary.get("conditional_improvement_pct_vs_flat_baseline")
    if improvement is not None:
        st.write(f"相对‘开盘后保持不变’基线，候选路径误差改善：**{improvement:+.2f}%**。")
    profile_improvement = summary.get("conditional_improvement_pct_vs_historical_profile")
    if profile_improvement is not None:
        st.write(f"相对旧版历史中位路径，候选路径误差改善：**{profile_improvement:+.2f}%**。")
    st.caption(
        f"滚动测试区间：{summary.get('first_test_date') or '暂无'} 至 {summary.get('last_test_date') or '暂无'}；"
        f"测试交易日 {summary.get('walk_forward_test_sessions', 0)} 天，"
        f"实际对比点 {summary.get('evaluated_intraday_points', 0)} 个。"
    )
    st.caption(summary.get("interpretation", "这是探索性回测，不是收益保证。"))

with st.expander("为什么之前看起来像直线？模型到底怎么预测？"):
    st.write(
        "旧版曲线是把所有历史交易日按同一时间点求中位数。每个时间点的涨跌中位数通常靠近 0%，"
        "互相平均后自然可能接近一条直线。这不一定是绘图 bug，而是简单基线没有找到强烈的可预测盘中模式。"
    )
    st.write(
        "新版增加了相似历史行情候选：先根据最近收益、20 日波动、上一交易日盘中表现和数据覆盖率寻找相似历史日，"
        "再汇总这些日期的盘中路径。它有可能更有曲线变化，但是否更准确必须看下方滚动回测；不会为了视觉效果人工制造波动。"
    )
    for note in data.get("caveats", []):
        st.write(f"- {note}")

st.caption(f"数据源：Alpaca IEX {data.get('timeframe', '15Min')} 数据。页面时间使用美东时间；项目其他模块可使用香港时间。覆盖率中位数：{quality.get('median_coverage_pct', '未知')}%。")
