from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from app.services.market_calendar import next_nasdaq_session

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market_data.db"
FORECAST_DIR = ROOT / "reports" / "forecast"
CONTEXT_DIR = ROOT / "reports" / "context_features"
CONTEXT_AGENT_DIR = ROOT / "reports" / "context_agent"
MODEL_LAB_DIR = ROOT / "reports" / "model_lab"
APP_TZ = ZoneInfo("Asia/Hong_Kong")


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = list(directory.glob(pattern)) if directory.exists() else []
    return max(files, key=lambda p: (p.stat().st_mtime, p.name)) if files else None


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def query_db(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None


def money(value: Any) -> str:
    number = as_float(value)
    return f"${number:,.2f}" if number is not None else "暂无数据"


def pct(value: Any, signed: bool = False, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "暂无数据"
    return f"{number:+.{digits}f}%" if signed else f"{number:.{digits}f}%"


def format_hkt(value: Any) -> str:
    if not value:
        return "暂无时间记录"
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            # Stored generated_at_local values in this project use UTC+8.
            dt = dt.replace(tzinfo=APP_TZ)
        return dt.astimezone(APP_TZ).strftime("%Y-%m-%d %H:%M:%S HKT")
    except (TypeError, ValueError):
        return str(value)


def pct_price(price: Any, return_pct: Any) -> float | None:
    p, r = as_float(price), as_float(return_pct)
    if p is None or r is None:
        return None
    return p * (1 + r / 100)


st.title("📈 QQQM AI 投资助手")
st.caption("先看结论，再看依据。这个页面帮助你理解行情与持仓，不会自动下单，也不会保证预测准确。")
now_hkt = datetime.now(APP_TZ).strftime("%Y-%m-%d %H:%M HKT")
st.caption(f"页面时间：{now_hkt} · 应用统一显示香港时间（UTC+8）；美股交易日期直接采用行情数据的交易日，不做时区换算。")

forecast_path = latest_file(FORECAST_DIR, "QQQM_return_forecast_latest_*.json")
forecast = read_json(forecast_path)
context = read_json(CONTEXT_DIR / "QQQM_context_snapshot_latest.json")
readiness = read_json(MODEL_LAB_DIR / "QQQM_context_training_readiness_latest.json")
champion = read_json(MODEL_LAB_DIR / "QQQM_champion_challenger_latest.json")
agent_report_path = CONTEXT_AGENT_DIR / "QQQM_context_analysis_latest.md"

# Small, beginner-friendly status strip.
feedback = query_db(
    "SELECT id, as_of_date, status, actual_target_date, actual_adjusted_close, actual_return_pct, "
    "error_zero_return_bps, error_historical_mean_bps, error_ridge_full_bps, error_ridge_returns_only_bps "
    "FROM forecast_feedback WHERE UPPER(ticker)='QQQM' ORDER BY id DESC LIMIT 100"
)
evaluated = feedback[feedback["status"] == "evaluated"].copy() if not feedback.empty else pd.DataFrame()
pending_n = int((feedback["status"] == "pending").sum()) if not feedback.empty else 0
live_snapshots = query_db(
    "SELECT COUNT(*) AS n FROM forecast_context_snapshots WHERE UPPER(ticker)='QQQM' AND capture_mode='live_prediction'"
)
live_n = int(live_snapshots.iloc[0]["n"]) if not live_snapshots.empty else 0

c1, c2, c3 = st.columns(3)
c1.metric("已验证的真实预测", f"{len(evaluated)} 条")
c2.metric("等待收盘验证", f"{pending_n} 条")
c3.metric("实时特征快照", f"{live_n} 条")
if len(evaluated) < 30:
    st.warning("学习状态：正在积累真实预测记录。现阶段样本太少，不能判断哪个模型更准，也不应把模型点估计当作买卖信号。")

st.divider()
st.header("1. 明天可能怎么走？")
if forecast:
    as_of_date = str(forecast.get("as_of_date", "未知"))
    try:
        daily_latest_frame = query_db("SELECT MAX(trade_date) AS latest_date FROM market_daily WHERE UPPER(ticker)='QQQM'")
        daily_latest_date = str(daily_latest_frame.iloc[0]["latest_date"])[:10] if not daily_latest_frame.empty and pd.notna(daily_latest_frame.iloc[0]["latest_date"]) else None
    except Exception:
        daily_latest_date = None
    now_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if daily_latest_date and as_of_date not in ("", "未知") and daily_latest_date > as_of_date:
        st.warning(f"预测报告截至 {as_of_date}，但行情数据库已更新到 {daily_latest_date}。这份结果只能用于历史学习，请刷新数据后重新生成预测。")
    elif as_of_date not in ("", "未知"):
        try:
            target_day = next_nasdaq_session(date.fromisoformat(as_of_date[:10]))
            if target_day < now_ny.date() or (target_day == now_ny.date() and now_ny.time().replace(tzinfo=None) >= time(16, 15)):
                st.warning(f"这份预测原本对应 {target_day.isoformat()}，其美股交易时段已结束。页面中的数字仅作历史研究，不能当成下一交易日预测。")
        except ValueError:
            pass
    current_price = as_float(forecast.get("latest_adjusted_close"))
    point = forecast.get("point_forecasts") or {}
    bands = forecast.get("historical_return_bands") or {}
    band = bands.get("last_60_sessions") or bands.get("last_252_sessions") or {}
    generated_at = format_hkt(forecast.get("generated_at_local"))

    st.caption(f"最新已知美股交易日：{as_of_date} · 预测目标：下一个交易日 · 生成时间：{generated_at}")
    if current_price is not None:
        st.metric("最近收盘价", money(current_price))

    ridge_ret = point.get("ridge_returns_only_return_pct")
    ridge_full = point.get("ridge_full_return_pct")
    hist_mean = point.get("historical_mean_return_pct")
    estimated_price = pct_price(current_price, ridge_ret)

    p1, p2, p3 = st.columns(3)
    p1.metric("模型点估计（收益率特征）", pct(ridge_ret, signed=True), help="这是实验模型给出的点估计，不是已校准概率，也不是保证。")
    p2.metric("对应的估计价格", money(estimated_price))
    p3.metric("历史均值参考", pct(hist_mean, signed=True))

    low = as_float(band.get("p10_price_scenario"))
    mid = as_float(band.get("p50_price_scenario"))
    high = as_float(band.get("p90_price_scenario"))
    if current_price is not None and any(v is not None for v in (low, mid, high)):
        st.subheader("历史价格情景参考")
        a, b, c = st.columns(3)
        a.metric("偏弱情景", money(low), pct(band.get("p10_return_pct"), signed=True))
        b.metric("中间情景", money(mid), pct(band.get("p50_return_pct"), signed=True))
        c.metric("偏强情景", money(high), pct(band.get("p90_return_pct"), signed=True))
        scenario_df = pd.DataFrame(
            {"历史价格情景": [v for v in (low, mid, high) if v is not None]},
            index=[label for label, v in (("偏弱 P10", low), ("中间 P50", mid), ("偏强 P90", high)) if v is not None],
        )
        st.bar_chart(scenario_df, width="stretch")
        st.caption("解释：P10/P50/P90 是过去所选窗口的收益分位映射到当前价格后的情景，不是未来概率保证。")
    with st.expander("这些数字是什么意思？"):
        st.write("模型点估计是算法对下一个交易日收益的单一估算。历史价格情景是过去真实波动的参考范围。两者都可能出错，且当前回测尚未证明机器学习模型稳定优于简单基线。")
        st.write(f"完整 Ridge 估计：{pct(ridge_full, signed=True)}。当前页面不把模型分数转换成未经验证的上涨概率。")
else:
    st.warning("还没有找到最新预测文件。请先运行每日预测流程。")

st.divider()
st.header("2. 最近行情走势")
history = query_db(
    "SELECT trade_date, adj_close FROM market_daily WHERE ticker=? ORDER BY trade_date DESC LIMIT 60",
    ("QQQM",),
)
if not history.empty:
    history["adj_close"] = pd.to_numeric(history["adj_close"], errors="coerce")
    history = history.dropna(subset=["adj_close"]).sort_values("trade_date")
    if not history.empty:
        st.line_chart(history.set_index("trade_date")[["adj_close"]], width="stretch")
        st.caption(f"图表显示 {history['trade_date'].iloc[0]} 至 {history['trade_date'].iloc[-1]} 的最近 {len(history)} 个可用交易日。")
else:
    st.info("目前无法读取历史行情，请检查本地数据库。")

st.divider()
st.header("3. 系统最近预测得怎么样？")
if not evaluated.empty:
    model_names = {
        "error_zero_return_bps": "不预测涨跌（零收益基线）",
        "error_historical_mean_bps": "历史平均涨跌",
        "error_ridge_full_bps": "机器学习模型（全部特征）",
        "error_ridge_returns_only_bps": "机器学习模型（历史收益特征）",
    }
    metric_rows = []
    for col, label in model_names.items():
        vals = pd.to_numeric(evaluated[col], errors="coerce").abs().dropna()
        metric_rows.append({"模型": label, "有效记录": len(vals), "平均误差（bps，越低越好）": round(float(vals.mean()), 2) if len(vals) else None})
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")
else:
    st.info("还没有可评估的真实预测。下一交易日收盘数据到达后，系统会结算待验证预测。")

if not feedback.empty:
    display = feedback[["as_of_date", "status", "actual_target_date", "actual_adjusted_close", "actual_return_pct"]].copy()
    display.columns = ["预测日期", "状态", "验证交易日", "实际收盘价", "实际收益率 %"]
    display["状态"] = display["状态"].map({"pending": "等待验证", "evaluated": "已验证"}).fillna(display["状态"])
    st.dataframe(display.head(10), hide_index=True, width="stretch")

if champion:
    with st.expander("查看历史模型对比实验"):
        st.write("上一轮历史留出实验没有达到模型晋级门槛，因此保留参考模型。历史回测成绩不等于未来表现。")
        st.json(champion, expanded=False)

if readiness:
    eligible = readiness.get("eligible_training_rows", 0)
    st.caption(f"可用于上下文机器学习的实时样本：{eligible} 条。当前采用保守规则，事后补建快照不会进入训练。")

st.divider()
st.header("4. 近期市场消息与 AI 解读")
if context:
    market = context.get("market") or {}
    news = context.get("news") or {}
    macro = context.get("macro") or {}
    m1, m2, m3 = st.columns(3)
    m1.metric("近 30 天已收集新闻", news.get("news_count_30_calendar_days", 0))
    m2.metric("最近交易日", market.get("latest_trade_date", context.get("as_of_date", "未知")))
    m3.metric("宏观记录数", macro.get("rows_in_database", 0))
    items = news.get("recent_items") or []
    if items:
        st.subheader("最近公开新闻")
        for item in items[:5]:
            title = str(item.get("title", "无标题"))
            source = str(item.get("source", "未知来源"))
            pub = str(item.get("published_at", "时间未知"))
            url = str(item.get("url", ""))
            st.markdown(f"- **{title}**  \n  来源：{source} · 发布时间：{pub} · [查看原文]({url})")
    if news.get("limitations"):
        with st.expander("新闻数据的限制"):
            for item in news["limitations"]:
                st.write(item)
    if macro.get("reason"):
        st.warning(str(macro["reason"]))
else:
    st.info("还没有最新市场上下文。可以运行 scripts/preview_market_context.py。")

if agent_report_path.exists():
    with st.expander("查看 DeepSeek 市场分析全文"):
        try:
            st.markdown(agent_report_path.read_text(encoding="utf-8"))
        except OSError as exc:
            st.error(f"无法读取 AI 分析报告：{exc}")
else:
    st.info("尚未找到 DeepSeek 分析报告。生成后会在这里显示。")

st.divider()
st.header("5. 系统状态与下一步")
st.write("**当前系统在做什么？** 每天读取最新行情，保存预测，等下一交易日真实收盘后评估结果。模型是否更准，需要累积更多前瞻样本验证。")
st.write("**新闻是否已经训练模型？** 还没有。新闻与宏观信息目前用于解释市场，等历史发布时间和版本核验充分后才做候选模型实验。")
st.write("**现在能自动交易吗？** 不能。本系统不提交订单；任何方案都需要考虑手续费、价差、税费与自身风险承受能力。")
st.caption("时间规则：应用页面统一显示香港时间（HKT，UTC+8）；美股交易日取行情数据中的 trade_date。不要为了更改页面显示时区而更改 Windows 系统时区。")
