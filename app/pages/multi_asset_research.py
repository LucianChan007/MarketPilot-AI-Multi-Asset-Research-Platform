from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.multi_asset_engine import (
    DB_PATH, NEWS_ROOT, REPORT_ROOT, add_instrument, analyze_news_with_deepseek,
    get_evaluations, latest_dates, list_news, load_instruments, run_cycle, sync_market_data,
)

st.title("🧭 MarketPilot AI · 美股多资产研究台")
st.caption("四个默认标的平行运行，支持继续添加美股。行情同步、预测生成和历史复盘分开操作，打开页面不会自动联网。")
st.info("行情抓取会依次尝试 Yahoo 公开图表接口与 yfinance 备用路径；若仍失败且日线过旧，会尝试 Stooq 日线备用源。Stooq 日线不会与其他来源的分时混用来生成预测。所有价格预测均为研究情景，不是买卖指令。")

with st.expander("新增研究标的", expanded=False):
    with st.form("multi_asset_add_instrument"):
        c1, c2, c3 = st.columns([1, 2, 1])
        new_symbol = c1.text_input("股票代码", placeholder="例如 AMD", max_chars=15).strip().upper()
        new_name = c2.text_input("公司/ETF 名称（可选）", placeholder="Advanced Micro Devices")
        new_type = c3.selectbox("类型", ["stock", "etf"], format_func=lambda x: "股票" if x == "stock" else "ETF")
        submitted = st.form_submit_button("添加到配置", type="primary")
    if submitted:
        try:
            added = add_instrument(new_symbol, new_name or new_symbol, new_type)
            st.success(f"已添加 {added['symbol']}。下次同步时将尝试获取其行情与新闻。")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

instruments = load_instruments(enabled_only=False)
enabled = [item for item in instruments if item["enabled"]]
if not enabled:
    st.warning("当前没有启用的研究标的，请先在 config/instruments.json 添加标的或启用已有标的。")
    st.stop()

symbols = [item["symbol"] for item in enabled]
selected = st.multiselect("本轮要处理的标的", symbols, default=symbols, format_func=lambda s: f"{s} · {next((x['name'] for x in enabled if x['symbol']==s), s)}")

col_sync, col_run, col_local, col_ai = st.columns([1.0, 1.65, 1.0, 1.15])
sync_clicked = col_sync.button("🔄 仅更新行情", type="primary", use_container_width=True, disabled=not selected, help="只拉取行情并检查日期，不训练、不生成预测，也不请求新闻。")
run_clicked = col_run.button("运行完整研究流程", use_container_width=True, disabled=not selected, help="同步行情与新闻、复盘到期预测并生成下一交易日候选。")
local_clicked = col_local.button("只处理本地缓存", use_container_width=True, disabled=not selected)
selected_symbols_for_ai = selected or symbols
ai_symbol = col_ai.selectbox("AI 新闻标的", selected_symbols_for_ai, key="multi_asset_ai_symbol")
ai_clicked = st.button("📰 调用 DeepSeek 分析所选标的新闻（可选）", use_container_width=True)

if sync_clicked:
    try:
        with st.spinner("正在同步行情。每个标的会单独记录数据源、最新日期和接口错误……"):
            result = sync_market_data(selected)
        st.session_state["multi_asset_last_status"] = result
        if result.get("overall_success"):
            st.success("所选标的的最新日线已通过日期检查。下方会分别标明分时是否足够新、能否用于预测。")
        else:
            st.warning("至少一个标的没有取得可验证的最新日线。请展开完整状态查看各接口结果；旧缓存不会被标记成新行情。")
        if not result.get("forecast_ready_all"):
            st.info("行情同步与预测可用性是两回事。部分标的的完整分时可能仍旧或缺失，因此本次只更新行情，没有生成新预测。")
    except Exception as exc:
        st.error(f"行情同步失败：{type(exc).__name__}: {exc}")

if run_clicked or local_clicked:
    try:
        with st.spinner("正在逐个标的同步/读取行情、复盘到期预测并计算候选模型……"):
            result = run_cycle(selected, refresh=run_clicked, sync_news_flag=run_clicked)
        st.session_state["multi_asset_last_status"] = result
        if result.get("overall_success"):
            st.success("本轮选中的标的均已生成或复用有效预测快照。请检查各资产的日期、模型和复盘状态。")
        else:
            st.warning("本轮部分标的未完成。已成功的资产结果会保留，失败原因列在下方；旧图不会冒充新预测。")
    except Exception as exc:
        st.error(f"研究流程运行失败：{type(exc).__name__}: {exc}")

if ai_clicked:
    try:
        with st.spinner(f"正在整理 {ai_symbol} 的新闻背景并调用 DeepSeek……"):
            ai_result = analyze_news_with_deepseek(ai_symbol)
        st.session_state["multi_asset_ai_result"] = ai_result
        st.success("AI 新闻解读已保存。它只用于解释背景，不直接决定价格曲线。")
    except Exception as exc:
        st.error(str(exc))

status = st.session_state.get("multi_asset_last_status")
status_file = REPORT_ROOT / "latest_status.json"
if status is None and status_file.exists():
    try:
        status = json.loads(status_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        status = None

st.subheader("标的行情与模型状态")
rows = []
for instrument in enabled:
    symbol = instrument["symbol"]
    try:
        dates = latest_dates(symbol)
        evals = get_evaluations(symbol, 1)
        last_eval = evals[0] if evals else {}
        latest_forecast = REPORT_ROOT / symbol / "forecast_latest.json"
        forecast = json.loads(latest_forecast.read_text(encoding="utf-8-sig")) if latest_forecast.exists() else {}
        rows.append({
            "代码": symbol, "名称": instrument["name"],
            "最新日线日期": dates.get("latest_daily_any_feed_date") or dates.get("daily_latest_date") or "缺失",
            "日线来源": dates.get("latest_daily_any_feed_source") or dates.get("feed") or "未知",
            "预测同源日线": dates.get("daily_latest_date") or "缺失",
            "完整分时最新日": dates.get("intraday_latest_complete_date") or "缺失",
            "分时样本交易日": dates.get("complete_intraday_sessions", 0),
            "最近分时抓取时间": dates.get("intraday_fetched_at") or "尚无抓取记录",
            "预测目标日": forecast.get("target_date") or "暂无",
            "模型": forecast.get("model_version") or "暂无",
            "最近复盘MAE($)": last_eval.get("mae_usd", "暂无"),
        })
    except Exception as exc:
        rows.append({"代码": symbol, "名称": instrument["name"], "状态": f"读取失败：{type(exc).__name__}: {exc}"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if status:
    with st.expander("最近一次工作流完整状态", expanded=not bool(status.get("overall_success"))):
        st.json(status)

st.divider()
show_symbol = st.selectbox("查看标的详情", symbols, key="multi_asset_detail_symbol")
asset_dir = REPORT_ROOT / show_symbol
forecast_path = asset_dir / "forecast_latest.json"
forecast: dict[str, Any] = {}
if forecast_path.exists():
    try:
        forecast = json.loads(forecast_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        forecast = {}

m1, m2, m3, m4 = st.columns(4)
dates = latest_dates(show_symbol)
m1.metric("最新可用日线日期", dates.get("latest_daily_any_feed_date") or dates.get("daily_latest_date") or "缺失")
m2.metric("预测同源日线日期", dates.get("daily_latest_date") or "缺失")
m3.metric("完整分时最新日期", dates.get("intraday_latest_complete_date") or "缺失")
m4.metric("候选预测目标日", forecast.get("target_date") or "暂无")
st.caption(f"最新日线显示来源：{dates.get('latest_daily_any_feed_source') or dates.get('feed') or '未知'} · 当前预测数据源：{dates.get('feed') or '未知'} · 最近分时抓取：{dates.get('intraday_fetched_at') or '暂无记录'}")

if forecast:
    forecast_asof = str(forecast.get("as_of_date") or "")
    forecast_feed = str(forecast.get("feed") or "").lower()
    latest_any_date = str(dates.get("latest_daily_any_feed_date") or "")
    forecast_is_current = bool(
        dates.get("dates_aligned")
        and forecast_asof
        and forecast_asof == str(dates.get("daily_latest_date") or "")
        and forecast_feed == str(dates.get("feed") or "").lower()
        and (not latest_any_date or forecast_asof == latest_any_date)
    )
    if forecast_is_current:
        st.success("该候选快照与当前可用行情日期、数据源一致。它仍是研究情景，不是买卖指令。")
    else:
        st.warning(f"当前文件是已保存的历史候选快照，基准日期为 {forecast_asof or '未知'}；最新可用日线为 {latest_any_date or '未知'}。它尚未与最新数据对齐，请勿当作本轮新预测。")
    st.caption(f"模型：{forecast.get('model_version', '未知')} · 基准：{forecast.get('as_of_date', '未知')} · 数据源：{forecast.get('feed', '未知')} · 训练交易日：{forecast.get('training_sessions') or 0} · 不自动晋级。")
    st.write(f"快照开盘参考：**${float(forecast.get('predicted_open') or 0):.2f}**；收盘参考：**${float(forecast.get('predicted_close') or 0):.2f}**。这些是模型情景点，不是保证价格。")
    forecast_image = asset_dir / f"forecast_{forecast.get('target_date')}.png"
    if forecast_image.exists():
        caption = f"{show_symbol} 当前候选曲线 · {forecast.get('target_date')}" if forecast_is_current else f"{show_symbol} 历史候选快照（未与最新行情对齐）· {forecast.get('target_date')}"
        st.image(str(forecast_image), caption=caption, use_container_width=True)
    backtest = forecast.get("backtest") or {}
    with st.expander("查看该标的滚动样本外回测", expanded=True):
        if backtest.get("available"):
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("测试交易日", str(backtest.get("test_sessions", 0)))
            b2.metric("Ridge 路径 MAE", f"{backtest.get('ridge_mae_bps')} bps")
            b3.metric("历史统计基准 MAE", f"{backtest.get('historical_profile_mae_bps')} bps")
            b4.metric("相对基准改善", f"{backtest.get('improvement_pct_vs_profile')}%" if backtest.get("improvement_pct_vs_profile") is not None else "暂无")
            st.caption(str(backtest.get("note") or ""))
        else:
            st.info(str(backtest.get("reason") or "目前没有可用样本外成绩。"))
else:
    st.info("当前标的还没有新预测。先在页面上方点击同步流程；如果历史分时不足或数据日期不一致，会在状态中说明原因。")

reviews = sorted(asset_dir.glob("review_*.png"), key=lambda p: p.stat().st_mtime, reverse=True) if asset_dir.exists() else []
if reviews:
    st.subheader("历史预测与真实走势复盘")
    st.image(str(reviews[0]), caption=f"最近一次复盘：{reviews[0].stem.replace('review_', '')}", use_container_width=True)
    st.caption("复盘图使用目标日期开盘前保存的原始预测 JSON，与同源实际十五分钟行情对照；缺少事前快照时不会事后补画预测线。")

evaluations = get_evaluations(show_symbol, 50)
if evaluations:
    st.subheader("历史预测成绩表")
    st.dataframe(pd.DataFrame(evaluations), use_container_width=True, hide_index=True)
else:
    st.info("目前还没有到期且成功结算的多股票预测记录。至少需要先生成一个预测，再等目标交易日结束并成功取得完整实际分时数据。")

news_tab, ai_tab = st.tabs(["新闻背景", "AI 新闻解读"])
with news_tab:
    news = list_news(show_symbol, 20)
    if news:
        for item in news:
            title = str(item.get("title") or "（无标题）")
            url = str(item.get("url") or "")
            with st.expander(f"{item.get('published_at') or '发布时间未确认'} · {title[:140]}"):
                st.caption(f"来源：{item.get('source') or '未知'} · 类型：{item.get('category') or '未知'}")
                if item.get("summary"):
                    st.write(item["summary"][:900])
                if url.startswith("https://") or url.startswith("http://"):
                    st.markdown(f"[打开来源文章]({url})")
    else:
        st.info("尚无新闻记录；点击上方同步工作流后会尝试读取美联储官方 RSS 与股票关键词新闻 RSS。")
with ai_tab:
    ai_path = NEWS_ROOT / f"{show_symbol}_latest.json"
    ai = st.session_state.get("multi_asset_ai_result")
    if ai and ai.get("symbol") != show_symbol:
        ai = None
    if ai is None and ai_path.exists():
        try:
            ai = json.loads(ai_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            ai = None
    if ai:
        st.write(ai.get("overall_summary") or "暂无摘要")
        st.caption(f"分析模型：{ai.get('model', '未知')} · 生成时间：{ai.get('generated_at', '未知')}")
        for event in ai.get("events") or []:
            st.markdown(f"**{event.get('title', '事件')}** · 方向参考：{event.get('possible_direction', 'unclear')} · 置信度：{event.get('confidence', '未知')}")
            st.write(str(event.get("rationale") or event.get("risk") or ""))
        st.warning(str(ai.get("notice") or "AI 新闻解读用于背景参考，不构成投资建议。"))
    else:
        st.info("尚无保存的 AI 新闻解读。请先配置 DeepSeek Key，再点击页面上方的可选分析按钮。")

st.caption("MarketPilot AI 是可审计的研究流程，不会自动交易。公开行情是否可用取决于当前网络和供应商；DeepSeek 新闻分析需要自行配置密钥。请区分‘最新日线已更新’与‘完整分时足够新、可以生成预测’。")
