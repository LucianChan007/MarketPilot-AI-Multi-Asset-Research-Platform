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
from app.services.ui_workflows import run_project_command


ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = ROOT / "reports" / "forecast"
DB_PATH = ROOT / "data" / "market_data.db"



def _latest_forecast_path() -> Path | None:
    """Return the newest timestamped return-forecast JSON file."""
    if not FORECAST_DIR.exists():
        return None
    candidates = list(FORECAST_DIR.glob("QQQM_return_forecast_latest_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _load_forecast(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"预测 JSON 顶层不是对象：{path.name}")
    return value


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if pd.notna(result) else default
    except (TypeError, ValueError):
        return default


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float, signed: bool = False) -> str:
    return f"{value:+.2f}%" if signed else f"{value:.2f}%"


def _get_recent_prices(limit: int = 40) -> tuple[pd.DataFrame, str | None]:
    """Read recent QQQM prices from market_daily without assuming exact column names."""
    if not DB_PATH.exists():
        return pd.DataFrame(), f"没有找到本地行情数据库：{DB_PATH}"

    try:
        with sqlite3.connect(DB_PATH) as connection:
            table_row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='market_daily'"
            ).fetchone()
            if not table_row:
                return pd.DataFrame(), "数据库中没有 market_daily 表，暂时只展示预测情景。"

            columns = [row[1] for row in connection.execute('PRAGMA table_info("market_daily")')]
            by_lower = {name.lower(): name for name in columns}

            def choose(*names: str) -> str | None:
                for name in names:
                    if name.lower() in by_lower:
                        return by_lower[name.lower()]
                return None

            ticker_col = choose("ticker", "symbol", "instrument", "asset")
            date_col = choose("date", "trade_date", "session_date", "timestamp")
            price_col = choose("adj_close", "adjusted_close", "adjclose", "adjustedclose", "close")
            if not ticker_col or not date_col or not price_col:
                return pd.DataFrame(), (
                    "market_daily 的字段名与页面预期不同，未能自动识别证券代码、日期或收盘价字段。"
                )

            def q(identifier: str) -> str:
                return '"' + identifier.replace('"', '""') + '"'

            sql = (
                f"SELECT {q(date_col)} AS trade_date, {q(price_col)} AS price "
                f"FROM {q('market_daily')} WHERE UPPER({q(ticker_col)}) = ? "
                f"ORDER BY {q(date_col)} DESC LIMIT ?"
            )
            rows = connection.execute(sql, ("QQQM", limit)).fetchall()

        frame = pd.DataFrame(rows, columns=["date", "price"])
        if frame.empty:
            return frame, "market_daily 中没有找到 QQQM 记录。"
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame = frame.dropna(subset=["date", "price"]).sort_values("date")
        frame = frame.drop_duplicates(subset=["date"], keep="last").tail(limit)
        return frame, None
    except (sqlite3.Error, OSError, ValueError) as exc:
        return pd.DataFrame(), f"读取历史行情失败：{exc}"


def _build_chart(
    history: pd.DataFrame,
    current_price: float,
    scenario_prices: dict[str, float],
    model_price: float,
) -> pd.DataFrame:
    """Build a categorical chart with the final point labelled as next-session scenarios."""
    rows: list[dict[str, Any]] = []
    if not history.empty:
        for row in history.itertuples(index=False):
            label = pd.Timestamp(row.date).strftime("%m-%d")
            rows.append(
                {
                    "日期": label,
                    "历史收盘价": float(row.price),
                    "偏弱情景": None,
                    "历史中位情景": None,
                    "偏强情景": None,
                    "Ridge 点估计": None,
                }
            )

    # Use the forecast's as-of price as the anchor so scenario calculations match the JSON report.
    if not rows or not history.empty:
        rows.append(
            {
                "日期": "最新收盘",
                "历史收盘价": current_price,
                "偏弱情景": current_price,
                "历史中位情景": current_price,
                "偏强情景": current_price,
                "Ridge 点估计": current_price,
            }
        )

    rows.append(
        {
            "日期": "下一交易日情景",
            "历史收盘价": None,
            "偏弱情景": scenario_prices["low"],
            "历史中位情景": scenario_prices["mid"],
            "偏强情景": scenario_prices["high"],
            "Ridge 点估计": model_price,
        }
    )
    return pd.DataFrame(rows).set_index("日期")


st.title("📈 QQQM 下一交易日走势预测")
st.caption("查看最新可用收盘价、实验性收益率点估计，以及历史收益分布形成的价格情景。页面不在打开时自动请求 API；需要时请主动运行下方工作流。")
update_col, local_col = st.columns([1.45, 1])
with update_col:
    if st.button("🔄 同步行情、新闻并重建预测", type="primary", use_container_width=True, key="forecast_full_update"):
        with st.spinner("正在同步各数据源、结算历史预测并生成最新预测……"):
            result = run_project_command(["-m", "app.daily_automation"], timeout=2400)
        st.session_state["forecast_full_update_result"] = result
        st.rerun()
with local_col:
    if st.button("🧪 仅按本地行情重建预测", use_container_width=True, key="forecast_local_build"):
        with st.spinner("正在用本地数据库训练并生成预测……"):
            result = run_project_command(["scripts/run_forecast_daily_cycle.py"], timeout=1800)
        st.session_state["forecast_local_build_result"] = result
        st.rerun()

for result_key, label in [
    ("forecast_full_update_result", "最近一次完整同步工作流"),
    ("forecast_local_build_result", "最近一次本地预测生成"),
]:
    action_result = st.session_state.get(result_key)
    if action_result:
        if action_result.get("returncode") == 0:
            st.success(f"{label}执行成功。请核对实际行情日期、预测目标交易日和下方来源说明。")
        else:
            st.warning(f"{label}未完整成功。旧预测不会被伪装成新预测，具体错误见日志。")
        with st.expander(f"查看：{label}日志", expanded=action_result.get("returncode") != 0):
            if action_result.get("stdout"):
                st.code(action_result["stdout"][-10000:])
            if action_result.get("stderr"):
                st.code(action_result["stderr"][-6000:])

with st.expander("这些数字是什么意思？", expanded=False):
    st.markdown("- **点估计**：模型给出的一个参考数值，不是目标价承诺。\n- **偏弱 / 中位 / 偏强情景**：根据过去行情分布形成的参考范围，不是未来概率保证。\n- **Ridge**：一种较简单的机器学习回归方法；它是否有用，要看历史上对没见过的数据预测得怎么样。")

forecast_path = _latest_forecast_path()
if forecast_path is None:
    st.error("没有找到最新收益幅度预测文件。请运行每日预测闭环生成预测，再检查运行日志。")
    st.code(r"& .\.venv\Scripts\python.exe .\scripts\run_forecast_daily_cycle.py", language="powershell")
    st.stop()

try:
    forecast = _load_forecast(forecast_path)
except (OSError, json.JSONDecodeError, ValueError) as exc:
    st.error(f"读取预测文件失败：{exc}")
    st.stop()

current_price = _num(forecast.get("latest_adjusted_close"))
point_forecasts = forecast.get("point_forecasts", {})
bands = forecast.get("historical_return_bands", {})
as_of_date = str(forecast.get("as_of_date", "未知"))

# A cached return forecast can look valid even after its target session has finished.
try:
    with sqlite3.connect(DB_PATH) as conn:
        latest_row = conn.execute(
            "SELECT trade_date, source, fetched_at FROM market_daily WHERE UPPER(ticker)='QQQM' ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
    db_latest_date = str(latest_row[0])[:10] if latest_row and latest_row[0] else None
    db_latest_source = str(latest_row[1]) if latest_row and latest_row[1] else "未知"
    db_latest_fetched_at = str(latest_row[2]) if latest_row and latest_row[2] else "未知"
except sqlite3.Error:
    db_latest_date = None
    db_latest_source = "未知"
    db_latest_fetched_at = "未知"
now_ny = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
closed_for_day = now_ny.time().replace(tzinfo=None) >= time(16, 15)
stale_reasons = []
if db_latest_date and as_of_date not in ("", "未知") and db_latest_date > as_of_date:
    stale_reasons.append(f"行情数据库已更新到 {db_latest_date}，预测报告仍截至 {as_of_date}")
try:
    target_day = next_nasdaq_session(date.fromisoformat(as_of_date[:10]))
    if target_day < now_ny.date() or (target_day == now_ny.date() and closed_for_day):
        stale_reasons.append(f"该报告对应的目标交易日 {target_day.isoformat()} 已经结束或已过去")
except (ValueError, TypeError):
    pass
if stale_reasons:
    st.error("这份收益预测已经过期：" + "；".join(stale_reasons) + "。页面已停止把它当作下一交易日预测展示。请先刷新行情，再重新运行收益幅度回测。")
    st.stop()

if current_price <= 0 or not isinstance(point_forecasts, dict) or not isinstance(bands, dict):
    st.error("预测 JSON 缺少必要字段或字段值无效。请重新运行收益幅度回测脚本生成最新文件。")
    st.stop()

window_options = {
    "近 20 个交易日": "last_20_sessions",
    "近 60 个交易日": "last_60_sessions",
    "近 252 个交易日": "last_252_sessions",
}
selected_label = st.selectbox("历史参考窗口", list(window_options.keys()), index=1)
selected_key = window_options[selected_label]
selected_band = bands.get(selected_key, {})
if not isinstance(selected_band, dict) or not selected_band:
    st.error(f"预测文件中缺少 {selected_label} 的情景数据。")
    st.stop()

low_price = _num(selected_band.get("p10_price_scenario"))
median_price = _num(selected_band.get("p50_price_scenario"))
high_price = _num(selected_band.get("p90_price_scenario"))
ridge_return_pct = _num(point_forecasts.get("ridge_full_return_pct"))
mean_return_pct = _num(point_forecasts.get("historical_mean_return_pct"))
ridge_price = current_price * (1 + ridge_return_pct / 100)
mean_price = current_price * (1 + mean_return_pct / 100)

if not (0 < low_price <= high_price):
    st.error("价格情景区间无效，请重新生成预测文件。")
    st.stop()

try:
    forecast_target_date = next_nasdaq_session(date.fromisoformat(as_of_date[:10]))
except (ValueError, TypeError, RuntimeError):
    forecast_target_date = None

st.info(
    f"预测基准数据截至 **{as_of_date}** 收盘；本地行情库最新日期 **{db_latest_date or '未知'}**，来源 **{db_latest_source}**；"
    f"最新行情记录更新时间：{db_latest_fetched_at}。读取预测文件 `{forecast_path.name}`。"
    " P10/P50/P90 是历史收益分布的情景分位数，不是经过校准的未来概率。"
)
if db_latest_source.startswith("public_snapshot_"):
    st.warning("当前基准行情来自已标注的公开网页历史快照，不代表本次 API 请求已成功。可用上方完整同步按钮尝试切换为供应商数据源。")
    st.markdown("[打开原始公开历史行情来源](https://stockanalysis.com/etf/qqqm/history/)")
if forecast_target_date:
    st.subheader(f"预测目标交易日：{forecast_target_date.isoformat()}（NASDAQ / 美东时间）")

metric_cols = st.columns(4)
metric_cols[0].metric("最新收盘价", _money(current_price))
metric_cols[1].metric("历史均值点估计", _money(mean_price), _pct(mean_return_pct, signed=True))
metric_cols[2].metric("Ridge 实验性点估计", _money(ridge_price), _pct(ridge_return_pct, signed=True))
metric_cols[3].metric("所选历史窗口", selected_label, f"{int(_num(selected_band.get('sample_size')))} 个样本")

st.subheader("下一交易日价格情景")
scenario_cols = st.columns(3)
scenario_data = [
    (scenario_cols[0], "偏弱情景 · P10", low_price),
    (scenario_cols[1], "历史中位情景 · P50", median_price),
    (scenario_cols[2], "偏强情景 · P90", high_price),
]
for column, label, price in scenario_data:
    move = price - current_price
    move_pct = (price / current_price - 1) * 100
    column.metric(label, _money(price), f"{move:+.2f} USD / {move_pct:+.2f}%")

st.subheader("历史价格 + 下一交易日情景图")
history, history_warning = _get_recent_prices(limit=40)
if history_warning:
    st.warning(history_warning)
if not history.empty:
    db_last_date = pd.Timestamp(history["date"].max()).strftime("%Y-%m-%d")
    if db_last_date != as_of_date:
        st.caption(
            f"注意：行情数据库最近日期为 {db_last_date}，预测报告价格日期为 {as_of_date}。"
            "图表使用预测报告的收盘价作为情景计算锚点。"
        )

chart_df = _build_chart(
    history=history,
    current_price=current_price,
    scenario_prices={"low": low_price, "mid": median_price, "high": high_price},
    model_price=ridge_price,
)
st.line_chart(chart_df, width="stretch", height=420)
st.caption(
    "图中连接线仅用于可视化各情景相对当前收盘价的位置，不代表价格会沿直线运行。"
)

st.subheader("如果你持有 QQQM，价格变化对应多少美元？")
share_count = st.number_input(
    "持有股数（仅用于情景金额换算，不会保存到代码）",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="输入当前想评估的股数；默认 0，不在页面代码中写入个人持仓信息。",
)
scenario_rows = []
for label, price in [("偏弱情景", low_price), ("历史中位情景", median_price), ("偏强情景", high_price), ("Ridge 点估计", ridge_price)]:
    per_share_change = price - current_price
    scenario_rows.append(
        {
            "情景": label,
            "情景价格": round(price, 4),
            "每股价格变化": round(per_share_change, 4),
            "相对当前价格变化": round((price / current_price - 1) * 100, 4),
            "持仓毛变化（未计费用）": round(per_share_change * share_count, 4) if share_count > 0 else None,
        }
    )
st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)

st.subheader("怎样解读这些数字")
with st.expander("模型可靠性与风险说明", expanded=True):
    st.markdown(
        """
        - 历史样本外测试中，Ridge 收益率模型没有优于简单的历史均值基线，因此它的点估计只作为实验参考。
        - 历史 P10/P50/P90 表示所选历史窗口中收益分布的分位数；不意味着下一交易日有 10%、50% 或 90% 的对应概率。
        - 表格中的持仓毛变化只反映价格变化，不包括买入成本、卖出手续费、重新买入费用、滑点或汇兑成本。
        - 不应仅凭这张图自动买卖。对少量持仓，先核对真实手续费，再评估交易预期收益是否足以覆盖成本。
        """
    )

st.caption("研究工具，不构成投资建议。历史表现和历史分位数不保证未来结果。")
