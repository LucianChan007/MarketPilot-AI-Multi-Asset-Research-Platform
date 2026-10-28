from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import load_history
from app.multi_asset_engine import (
    DB_PATH as MULTI_ASSET_DB,
    REPORT_ROOT as MULTI_ASSET_REPORTS,
    _daily_rows,
    get_evaluations,
    latest_dates,
    list_news,
    load_instruments,
    run_cycle,
    sync_market_data,
)
from app.services.local_data_bootstrap import (
    ensure_bundled_qqqm_history,
    mirror_qqqm_daily_from_multi_asset,
)
from app.services.system_health import expected_last_completed_session
from app.services.ui_workflows import run_project_command

LEGACY_DB = ROOT / "data" / "market_data.db"
USERS_DB = ROOT / "data" / "users.db"
AUTH_KEY = "qqqm_authenticated_user"

st.markdown(
    """
    <style>
    .block-container {padding-top:1.5rem; padding-bottom:2.2rem; max-width:1500px;}
    [data-testid="stSidebar"] {border-right:1px solid rgba(100,116,139,.18);}
    .mp-hero {padding:27px 29px;border-radius:20px;color:white;background:linear-gradient(118deg,#101d35 0%,#1d3e78 58%,#2765e8 100%);position:relative;overflow:hidden;margin-bottom:18px;}
    .mp-kicker {font-size:11px;font-weight:750;letter-spacing:.16em;text-transform:uppercase;color:#b9d1ff;margin-bottom:10px;}
    .mp-hero h1 {font-size:31px;line-height:1.2;margin:0 0 10px;color:white;font-weight:760;letter-spacing:-.035em;}
    .mp-hero p {font-size:14px;line-height:1.65;color:#e0eaff;max-width:900px;margin:0;}
    .mp-pill {display:inline-block;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.09);border-radius:999px;padding:6px 10px;font-size:11px;color:#f0f5ff;margin-top:15px;}
    .mp-card {border:1px solid rgba(100,116,139,.22);border-radius:15px;padding:15px 16px 13px;background:var(--secondary-background-color);min-height:137px;}
    .mp-card-symbol {font-size:12px;color:#6b87b8;font-weight:800;letter-spacing:.06em;}
    .mp-card-price {font-size:27px;font-weight:780;letter-spacing:-.04em;color:var(--text-color);margin-top:8px;}
    .mp-card-note {font-size:11px;line-height:1.5;color:var(--text-color);opacity:.72;margin-top:7px;}
    .mp-section {font-size:19px;font-weight:760;letter-spacing:-.025em;margin:19px 0 4px;color:var(--text-color);}
    .mp-sub {font-size:12px;line-height:1.55;color:var(--text-color);opacity:.7;margin-bottom:12px;}
    div[data-testid="stButton"] button, div[data-testid="stPageLink"] a {border-radius:10px;font-weight:650;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_portfolio(user_id: Any) -> dict[str, float | int]:
    empty: dict[str, float | int] = {
        "shares": 0.0, "avg_cost": 0.0, "realized": 0.0,
        "transactions": 0, "dividends": 0.0,
    }
    if not user_id or not USERS_DB.exists():
        return empty
    try:
        with sqlite3.connect(USERS_DB) as conn:
            tx = conn.execute(
                """SELECT symbol, side, quantity, price, fees, realized_pnl, shares_after, average_cost_after
                   FROM portfolio_transactions t WHERE user_id=? AND NOT EXISTS (
                     SELECT 1 FROM portfolio_transaction_voids v WHERE v.transaction_id=t.transaction_id
                   ) ORDER BY executed_at_utc DESC, transaction_id DESC""", (user_id,),
            ).fetchall()
            dividends = conn.execute(
                """SELECT COALESCE(SUM(net_amount),0) FROM portfolio_dividends d
                   WHERE user_id=? AND NOT EXISTS (
                     SELECT 1 FROM portfolio_dividend_voids v WHERE v.dividend_id=d.dividend_id
                   ) AND UPPER(symbol)='QQQM'""", (user_id,),
            ).fetchone()[0]
        qqqm = [row for row in tx if str(row[0]).upper() == "QQQM"]
        if not qqqm:
            return empty
        return {
            "shares": float(qqqm[0][6] or 0),
            "avg_cost": float(qqqm[0][7] or 0),
            "realized": sum(float(row[5] or 0) for row in qqqm),
            "transactions": len(qqqm),
            "dividends": float(dividends or 0),
        }
    except sqlite3.Error:
        return empty


def _legacy_frame(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if symbol != "QQQM":
        return pd.DataFrame(), {}
    try:
        frame = load_history(symbol)
    except Exception:
        return pd.DataFrame(), {}
    if frame.empty:
        return pd.DataFrame(), {}
    frame = frame.reset_index().rename(columns={"trade_date": "date", "index": "date"})
    date_col = "date" if "date" in frame.columns else frame.columns[0]
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["price"] = pd.to_numeric(frame.get("adj_close"), errors="coerce").fillna(
        pd.to_numeric(frame.get("close"), errors="coerce")
    )
    frame = frame.dropna(subset=["date", "price"]).sort_values("date")
    last = frame.iloc[-1]
    return frame, {
        "date": pd.Timestamp(last["date"]).date().isoformat(),
        "price": float(last["price"]),
        "source": str(last.get("source") or "legacy_market_database"),
        "fetched_at": str(last.get("fetched_at") or ""),
        "record_count": len(frame),
        "feed": "legacy",
    }


def read_asset_history(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read stored rows only; no provider is contacted while rendering the page."""
    options: list[tuple[pd.DataFrame, dict[str, Any], int]] = []
    try:
        dates = latest_dates(symbol)
        feed = dates.get("latest_daily_any_feed_source")
        if feed:
            frame = _daily_rows(symbol, MULTI_ASSET_DB, str(feed))
            if not frame.empty:
                frame = frame.copy()
                frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
                frame["price"] = pd.to_numeric(frame["adj_close"], errors="coerce").fillna(
                    pd.to_numeric(frame["close"], errors="coerce")
                )
                frame = frame.dropna(subset=["date", "price"]).sort_values("date")
                last = frame.iloc[-1]
                meta = {
                    "date": pd.Timestamp(last["date"]).date().isoformat(),
                    "price": float(last["price"]),
                    "source": str(feed),
                    "fetched_at": str(dates.get("daily_fetched_at") or ""),
                    "record_count": len(frame),
                    "feed": str(feed),
                    "daily_latest_date": dates.get("daily_latest_date"),
                    "intraday_latest_complete_date": dates.get("intraday_latest_complete_date"),
                    "complete_intraday_sessions": dates.get("complete_intraday_sessions", 0),
                }
                # Prefer new multi-asset feed rows over the packaged legacy history on ties.
                options.append((frame, meta, 2))
    except Exception as exc:
        # Keep a visible diagnostic for the page instead of failing the whole dashboard.
        multi_error = f"{type(exc).__name__}: {exc}"
    else:
        multi_error = ""

    legacy, legacy_meta = _legacy_frame(symbol)
    if not legacy.empty:
        options.append((legacy, legacy_meta, 1))

    if not options:
        return pd.DataFrame(), {"error": multi_error or "本地还没有该标的的日线记录。"}
    options.sort(key=lambda item: (item[1].get("date") or "", item[2]), reverse=True)
    frame, meta, _rank = options[0]
    meta = dict(meta)
    if multi_error:
        meta["read_warning"] = f"多资产数据库读取提示：{multi_error}"
    return frame, meta


def money(value: float | int) -> str:
    return f"${float(value):,.2f}"


def bridge_qqqm_cache_if_selected(selected_symbols: list[str]) -> dict[str, Any]:
    if "QQQM" not in selected_symbols:
        return {"applied": False, "reason": "QQQM_not_selected"}
    try:
        return mirror_qqqm_daily_from_multi_asset()
    except Exception as exc:
        # A compatibility copy must never hide the primary multi-asset sync result.
        return {"applied": False, "reason": "bridge_failed", "error": f"{type(exc).__name__}: {exc}"}


# Local CSV bootstrap only. No network request, forecast or model run happens at page load.
bootstrap_status = ensure_bundled_qqqm_history()
st.session_state["_local_market_bootstrap_status"] = bootstrap_status

instruments = load_instruments(enabled_only=True)
if not instruments:
    instruments = [
        {"symbol": "QQQM", "name": "QQQM", "enabled": True},
        {"symbol": "AAPL", "name": "Apple Inc.", "enabled": True},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "enabled": True},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "enabled": True},
    ]
symbols = [str(item["symbol"]).upper() for item in instruments]
try:
    expected_date = expected_last_completed_session()
except Exception:
    expected_date = None

st.markdown(
    """
    <div class="mp-hero">
      <div class="mp-kicker">AI-POWERED INVESTMENT RESEARCH</div>
      <h1>MarketPilot AI · 美股多资产研究台</h1>
      <p>把市场行情、宏观信息与个人交易记录放在同一条研究流程里。先看数据，再读解释，最后用真实结果检验预测。</p>
      <span class="mp-pill">MarketPilot AI · Multi-Asset Research&nbsp;&nbsp;·&nbsp;&nbsp;Research, not trading</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("本页展示四只默认标的的平行行情状态。打开页面只读本地文件与数据库，不会自动发起行情、新闻或宏观 API 请求。")

if bootstrap_status.get("applied"):
    st.info(
        f"首次启动已从项目自带的历史 CSV 导入 {bootstrap_status.get('rows_added', 0):,} 条 QQQM 日线，"
        f"数据截至 {bootstrap_status.get('latest_date') or '未知'}。这是离线历史缓存，不是实时行情。"
    )
elif bootstrap_status.get("error"):
    st.warning(f"本地 QQQM 历史数据初始化未完成：{bootstrap_status['error']}")
if st.session_state.get("_local_market_bootstrap_error"):
    st.warning(f"启动时的本地历史数据初始化提示：{st.session_state['_local_market_bootstrap_error']}")

# Four assets are treated equally. The packaged CSV only provides historical QQQM data;
# the other assets remain honestly marked as pending until a real provider returns data.
asset_views: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
for item in instruments:
    symbol = str(item["symbol"]).upper()
    asset_views[symbol] = read_asset_history(symbol)

st.markdown('<div class="mp-section">四资产行情总览</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mp-sub">价格卡显示本地保存的最近日线，不代表实时成交价。标记「历史缓存」的记录不会被当作本次同步成功。</div>',
    unsafe_allow_html=True,
)
for row_start in range(0, len(instruments), 4):
    row_items = instruments[row_start:row_start + 4]
    card_cols = st.columns(len(row_items))
    for index, item in enumerate(row_items):
        symbol = str(item["symbol"]).upper()
        frame, meta = asset_views.get(symbol, (pd.DataFrame(), {}))
        with card_cols[index]:
            if not frame.empty and meta.get("price") is not None:
                price_text = money(meta["price"])
                latest = str(meta.get("date") or "日期未知")
                source = str(meta.get("source") or "来源未知")
                if source.startswith("bundled_"):
                    status_text = "本地历史缓存 · 非实时"
                elif expected_date and latest < expected_date:
                    status_text = f"行情可能滞后 · 应至少检查至 {expected_date}"
                else:
                    status_text = "已保存日线 · 仍需核对来源与日期"
            else:
                price_text = "暂无数据"
                latest = "尚未取得日线"
                source = "等待首次同步"
                status_text = "尚未同步该标的"
            st.markdown(
                f'<div class="mp-card"><div class="mp-card-symbol">{symbol}</div>'
                f'<div class="mp-card-price">{price_text}</div>'
                f'<div class="mp-card-note">{item.get("name", symbol)}<br>数据日期：{latest}<br>来源：{source}<br>{status_text}</div></div>',
                unsafe_allow_html=True,
            )

st.markdown('<div class="mp-section">数据同步与研究流程</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mp-sub">先单独同步行情并检查失败原因，再运行完整研究流程。完整研究会额外请求新闻数据、结算到期预测，并在数据对齐时生成候选预测。</div>',
    unsafe_allow_html=True,
)
selected = st.multiselect(
    "本轮要处理的标的", symbols, default=symbols,
    format_func=lambda symbol: f"{symbol} · {next((x.get('name', symbol) for x in instruments if x['symbol'] == symbol), symbol)}",
    key="home_multi_asset_selected_symbols",
)
button_sync, button_full, button_cache = st.columns([1, 1.5, 1])
sync_clicked = button_sync.button("🔄 仅更新行情", type="primary", use_container_width=True, disabled=not selected)
full_clicked = button_full.button("运行完整多资产研究", use_container_width=True, disabled=not selected)
cache_clicked = button_cache.button("只处理本地缓存", use_container_width=True, disabled=not selected)

if sync_clicked:
    try:
        with st.spinner("正在逐个标的请求行情并校验数据日期……"):
            result = sync_market_data(selected)
            bridge = bridge_qqqm_cache_if_selected(selected)
        result["legacy_qqqm_cache_bridge"] = bridge
        st.session_state["home_multi_asset_result"] = result
        st.rerun()
    except Exception as exc:
        st.error(f"行情同步流程异常：{type(exc).__name__}: {exc}")

if full_clicked:
    try:
        with st.spinner("正在同步行情与新闻、复盘到期预测，并检查每个标的能否生成新候选……"):
            result = run_cycle(selected, refresh=True, sync_news_flag=True)
            bridge = bridge_qqqm_cache_if_selected(selected)
        result["legacy_qqqm_cache_bridge"] = bridge
        st.session_state["home_multi_asset_result"] = result
        st.rerun()
    except Exception as exc:
        st.error(f"多资产研究流程异常：{type(exc).__name__}: {exc}")

if cache_clicked:
    try:
        with st.spinner("正在检查本地行情与预测文件，不主动同步外部数据……"):
            result = run_cycle(selected, refresh=False, sync_news_flag=False)
            bridge = bridge_qqqm_cache_if_selected(selected)
        result["legacy_qqqm_cache_bridge"] = bridge
        st.session_state["home_multi_asset_result"] = result
        st.rerun()
    except Exception as exc:
        st.error(f"本地缓存流程异常：{type(exc).__name__}: {exc}")

last_result = st.session_state.get("home_multi_asset_result")
if last_result:
    succeeded = bool(last_result.get("overall_success"))
    if succeeded:
        st.success("所选标的的日线已通过当前流程的日期检查。请继续核对各资产的分时日期和预测可用性。")
    else:
        st.warning("本轮没有全部成功。旧数据已保留；请查看下面每个标的的供应商尝试、日期和错误，不要把旧缓存当作同步成功。")
    with st.expander("查看最近一次多资产同步/预测完整日志", expanded=not succeeded):
        status_rows = []
        for symbol, item in (last_result.get("symbols") or {}).items():
            attempts = item.get("provider_attempts") or []
            errors = [str(error) for error in (item.get("errors") or [])]
            for attempt in attempts:
                errors.extend(str(error) for error in (attempt.get("errors") or []))
            status_rows.append({
                "标的": symbol,
                "日线最新日": item.get("latest_daily_any_feed_date") or item.get("daily_latest_date") or "缺失",
                "日线来源": item.get("latest_daily_any_feed_source") or item.get("feed") or "未知",
                "完整分时最新日": item.get("intraday_latest_complete_date") or "缺失",
                "日线已更新": item.get("daily_fresh", item.get("success", False)),
                "可生成预测": item.get("forecast_ready", False),
                "错误详情": "；".join(dict.fromkeys(errors)) or "没有记录到错误",
            })
        if status_rows:
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
        st.json(last_result)
        st.caption(f"状态文件：{last_result.get('status_file') or MULTI_ASSET_REPORTS / 'latest_status.json'}")

st.markdown('<div class="mp-section">各资产行情状态与预测准备度</div>', unsafe_allow_html=True)
status_rows = []
for item in instruments:
    symbol = str(item["symbol"]).upper()
    frame, meta = asset_views.get(symbol, (pd.DataFrame(), {}))
    try:
        dates = latest_dates(symbol)
    except Exception as exc:
        dates = {"latest_daily_any_feed_date": None, "latest_daily_any_feed_source": None, "intraday_latest_complete_date": None, "complete_intraday_sessions": 0}
        meta = {**meta, "error": f"{type(exc).__name__}: {exc}"}
    daily_date = str(meta.get("date") or dates.get("latest_daily_any_feed_date") or "")
    if not daily_date:
        status_text = "缺少本地日线，等待同步"
    elif str(meta.get("source") or "").startswith("bundled_"):
        status_text = "离线历史缓存，非实时"
    elif expected_date and daily_date < expected_date:
        status_text = "日线落后于预期交易日"
    else:
        status_text = "已保存日线，需检查完整分时"
    status_rows.append({
        "代码": symbol,
        "最新可读日线": daily_date or "缺失",
        "日线来源": meta.get("source") or dates.get("latest_daily_any_feed_source") or "未知",
        "记录条数": meta.get("record_count", 0),
        "最新完整分时日": dates.get("intraday_latest_complete_date") or "缺失",
        "完整分时样本交易日": dates.get("complete_intraday_sessions", 0),
        "预期最近已收盘交易日": expected_date or "无法确认",
        "状态": status_text,
    })
st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

if expected_date:
    st.caption(f"按美股交易日历估算，最近已收盘交易日为 {expected_date}。行情更新时间和行情交易日期是两回事；本地文件刚导入，也不代表数据就是最新的。")

st.divider()
chart_symbol = st.selectbox("查看某只资产的日线走势", symbols, key="home_chart_symbol")
chart_frame, chart_meta = read_asset_history(chart_symbol)
if not chart_frame.empty:
    chart = chart_frame.copy()
    chart["date"] = pd.to_datetime(chart["date"], errors="coerce")
    chart["price"] = pd.to_numeric(chart["price"], errors="coerce")
    chart = chart.dropna(subset=["date", "price"]).sort_values("date").tail(90)
    if not chart.empty:
        st.line_chart(chart.set_index("date")[["price"]].rename(columns={"price": f"{chart_symbol} 收盘参考价"}), use_container_width=True, height=280)
        first, last = chart.iloc[0], chart.iloc[-1]
        change = (float(last["price"]) / float(first["price"]) - 1) * 100 if float(first["price"]) else 0
        st.caption(f"图表区间：{first['date']:%Y-%m-%d} 至 {last['date']:%Y-%m-%d} · 区间变化 {change:+.2f}% · 来源 {chart_meta.get('source') or '未知'}。这不是盘中实时曲线。")
else:
    st.info(f"{chart_symbol} 暂无本地日线。先点「仅更新行情」，系统会记录接口尝试和失败详情。")
if chart_meta.get("read_warning"):
    st.caption(chart_meta["read_warning"])

# Retain the original personal ledger feature without making QQQM the market dashboard's only asset.
st.markdown('<div class="mp-section">个人账本概况（QQQM）</div>', unsafe_allow_html=True)
principal = st.session_state.get(AUTH_KEY, {})
user_id = principal.get("user_id") if isinstance(principal, dict) else None
portfolio = read_portfolio(user_id)
qqqm_frame, qqqm_meta = read_asset_history("QQQM")
latest_qqqm_price = float(qqqm_meta.get("price")) if qqqm_meta.get("price") is not None else None
market_value = portfolio["shares"] * latest_qqqm_price if latest_qqqm_price is not None else 0.0
cost_value = portfolio["shares"] * portfolio["avg_cost"]
unrealized = market_value - cost_value
p1, p2, p3, p4 = st.columns(4)
p1.metric("当前持有股数", f"{portfolio['shares']:g}")
p2.metric("QQQM 持仓市值估算", money(market_value) if latest_qqqm_price is not None else "暂无行情")
p3.metric("已实现盈亏累计", money(portfolio["realized"]))
p4.metric("股息净额累计", money(portfolio["dividends"]))
if latest_qqqm_price is not None:
    st.caption(f"估值参考价 ${latest_qqqm_price:.2f}，对应行情日期 {qqqm_meta.get('date', '未知')}，数据源 {qqqm_meta.get('source', '未知')}。持仓估值不等于实时市值。")
else:
    st.caption("没有可用的 QQQM 参考价格，当前仅展示账本数量与历史盈亏。")
st.page_link("app/pages/portfolio_planner.py", label="打开个人投资组合与交易账本 →", icon="🧾", use_container_width=True)

st.markdown('<div class="mp-section">新闻背景</div>', unsafe_allow_html=True)
news_symbol = st.selectbox("新闻标的", symbols, key="home_news_symbol")
news_items = list_news(news_symbol, 8)
if news_items:
    for entry in news_items[:8]:
        title = str(entry.get("title") or "（无标题）")
        with st.expander(f"{entry.get('published_at') or '发布时间未确认'} · {title[:130]}"):
            st.caption(f"来源：{entry.get('source') or '未知'} · 类型：{entry.get('category') or '未知'}")
            if entry.get("summary"):
                st.write(str(entry["summary"])[:900])
            url = str(entry.get("url") or "")
            if url.startswith(("https://", "http://")):
                st.markdown(f"[打开来源文章]({url})")
else:
    st.info("该标的暂无已保存新闻。行情同步与新闻同步分开，使用「运行完整多资产研究」才会尝试收录新闻。")

st.markdown('<div class="mp-section">保留的旧版 QQQM 研究流程</div>', unsafe_allow_html=True)
with st.expander("旧版行情、宏观、新闻与 QQQM 预测反馈（兼容工作区）", expanded=False):
    st.write("保留旧版完整工作流，供原有 QQQM 预测、宏观数据、新闻报告等页面继续使用。它与新版四资产研究数据库分开记录，建议先在新版总览确认行情情况。")
    if st.button("运行旧版 QQQM 综合更新", key="legacy_qqqm_full_update", use_container_width=True):
        with st.spinner("正在运行旧版 QQQM 日常工作流……"):
            legacy_result = run_project_command(["-m", "app.daily_automation"], timeout=2400)
        st.session_state["home_legacy_update_result"] = legacy_result
    legacy_result = st.session_state.get("home_legacy_update_result")
    if legacy_result:
        if legacy_result.get("returncode") == 0:
            st.success("旧版综合任务返回成功。仍需检查日志中的各阶段结果。")
        else:
            st.warning(f"旧版综合任务未完全成功，退出码 {legacy_result.get('returncode')}。新版多资产行情数据库与旧版数据库分开维护。")
        with st.expander("旧版同步日志", expanded=legacy_result.get("returncode") != 0):
            if legacy_result.get("stdout"):
                st.code(str(legacy_result["stdout"])[-12000:])
            if legacy_result.get("stderr"):
                st.code(str(legacy_result["stderr"])[-6000:])

st.markdown('<div class="mp-section">其他研究工具</div>', unsafe_allow_html=True)
l1, l2, l3, l4 = st.columns(4)
with l1:
    st.page_link("app/dashboard.py", label="市场与持仓", icon="📊", use_container_width=True)
with l2:
    st.page_link("app/pages/next_day_forecast.py", label="预测情景", icon="🔭", use_container_width=True)
with l3:
    st.page_link("app/pages/model_learning_review.py", label="预测复盘", icon="🧠", use_container_width=True)
with l4:
    st.page_link("app/pages/multi_asset_research.py", label="多资产研究详情", icon="🧭", use_container_width=True)
st.page_link("app/pages/system_status.py", label="打开数据更新与系统诊断 →", icon="🩺", use_container_width=True)
st.caption("研究与学习用途，不构成投资建议。所有行情均应核对数据日期和来源；没有完整、同源分时数据时，不生成或发布伪装成实际行情的预测曲线。")
