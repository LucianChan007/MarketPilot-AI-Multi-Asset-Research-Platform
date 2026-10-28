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
FORECAST_DIR = ROOT / "reports" / "forecast"
MARKET_DB_PATH = ROOT / "data" / "market_data.db"


def _latest_file(pattern: str) -> Path | None:
    files = list(FORECAST_DIR.glob(pattern)) if FORECAST_DIR.exists() else []
    if not files:
        return None
    # Timestamp is part of these report filenames; mtime is a safe fallback.
    return max(files, key=lambda p: (p.stat().st_mtime, p.name))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层应为对象：{path.name}")
    return value


def _money(value: float, digits: int = 2) -> str:
    return f"${value:,.{digits}f}"


def _pct(value: float, digits: int = 2, signed: bool = False) -> str:
    if signed:
        return f"{value:+.{digits}f}%"
    return f"{value:.{digits}f}%"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if pd.notna(result) else default
    except (TypeError, ValueError):
        return default


def _explain_scenario(
    preference: str,
    target_price: float,
    selected_p90: float,
    target_profit: float,
    qty: float,
    sell_fee: float,
    rebuy_fee: float,
    acceptable_drawdown: float,
) -> str:
    lines: list[str] = []
    if qty <= 0:
        return "请先填写大于 0 的持仓股数，再查看持仓情景。"

    if target_price > selected_p90:
        lines.append(
            f"你设定的净赚目标对应卖出价约 {_money(target_price)}，高于所选历史窗口的 P90 情景价 "
            f"{_money(selected_p90)}。这不代表目标无法达到，只代表它没有落在这段历史经验分布的中间 80% 范围内。"
        )
    else:
        lines.append(
            f"你设定的净赚目标对应卖出价约 {_money(target_price)}，没有高于所选历史窗口的 P90 情景价 "
            f"{_money(selected_p90)}；这仍不代表该目标会在明天或近期实现。"
        )

    if sell_fee + rebuy_fee > 0:
        round_trip_per_share = (sell_fee + rebuy_fee) / qty
        lines.append(
            f"如果卖出后还打算买回相同股数，按你填写的费用估算，卖出加重新买入至少要产生约 "
            f"{_money(round_trip_per_share)} / 股的回落，才足以抵消这两笔费用；这还没有覆盖价格反弹、滑点或汇兑变化风险。"
        )
    else:
        lines.append("当前卖出费和重新买入费都填为 $0。请按众安银行实际收费填写，否则净收益与波段门槛会偏乐观。")

    if preference == "更偏长期持有":
        lines.append(
            "长期持有视角：本次样本外回测没有证明次日模型优于简单历史均值基线。若原本就是长期配置，"
            "不要仅凭这份弱预测频繁进出；更值得复核的是投资期限、仓位和能承受的回撤。"
        )
    elif preference == "希望做波段":
        lines.append(
            "波段视角：先设定净收益目标、可接受回撤和重新买回规则；只有潜在价差明显超过交易成本时，才值得进一步评估。"
            "目前的次日预测不够可靠，不能独立支持买入或卖出结论。"
        )
    else:
        lines.append(
            "尚未决定长期还是波段：可以先把这页当作情景计算器，不急着交易。以当前模型验证结果，暂时没有足够证据支持主动短线进出。"
        )

    lines.append(
        f"按你设定的 {acceptable_drawdown:.1f}% 回撤提醒幅度，价格复核线约为当前价格的 "
        f"{100 - acceptable_drawdown:.1f}%（这里只是提醒线，不是止损指令）。"
    )
    return "\n\n".join(lines)


st.markdown(
    """
    <style>
      .hero { padding: 1.1rem 1.25rem; border-radius: 16px; background: linear-gradient(115deg, #10243a, #244b62); color: white; margin-bottom: 1rem; }
      .hero h1 { font-size: 1.75rem; margin: 0 0 .25rem 0; color: white; }
      .hero p { font-size: .95rem; margin: 0; color: #dce9f3; }
      .section-note { color: #64748b; font-size: .9rem; }
      .small-muted { color: #64748b; font-size: .82rem; }
    </style>
    <div class="hero">
      <h1>QQQM 预测与持仓规划</h1>
      <p>先看可能的价格区间，再换算成自己的盈亏和交易成本。不把模型猜测包装成确定答案。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("先看懂这页的几个词", expanded=False):
    st.markdown("- **持仓成本**：买入时投入的金额，通常还要考虑费用。\n- **浮动盈亏**：按最近已同步价格估算，如果现在卖出可能得到的盈亏，未卖出前并未锁定。\n- **回撤**：资产从之前高点跌下来的幅度。\n- **价格情景**：用来做假设计算，不是模型承诺未来会到达的价格。")

forecast_path = _latest_file("QQQM_return_forecast_latest_*.json")
summary_path = _latest_file("QQQM_return_forecast_summary_*.csv")
interval_path = _latest_file("QQQM_return_forecast_intervals_*.csv")

if forecast_path is None:
    st.error("没有找到次日收益预测结果。请先在项目根目录运行：")
    st.code(r"& .\.venv\Scripts\python.exe .\scripts\run_next_day_return_backtest.py", language="powershell")
    st.stop()

try:
    forecast = _read_json(forecast_path)
except Exception as exc:
    st.error(f"读取预测 JSON 失败：{type(exc).__name__}: {exc}")
    st.stop()

current_price = _safe_float(forecast.get("latest_adjusted_close"), 0.0)
as_of_date = str(forecast.get("as_of_date", "未知"))
try:
    with sqlite3.connect(MARKET_DB_PATH) as conn:
        _daily_latest = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
    _db_latest_date = str(_daily_latest[0])[:10] if _daily_latest and _daily_latest[0] else None
except sqlite3.Error:
    _db_latest_date = None
_ny_now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
_ny_closed = _ny_now.time().replace(tzinfo=None) >= time(16, 15)
_planner_stale = bool(_db_latest_date and as_of_date not in ("", "未知") and _db_latest_date > as_of_date)
try:
    _planner_target = next_nasdaq_session(date.fromisoformat(as_of_date[:10]))
    if _planner_target < _ny_now.date() or (_planner_target == _ny_now.date() and _ny_closed):
        _planner_stale = True
except (ValueError, TypeError):
    _planner_target = None
if _planner_stale:
    st.warning(
        f"当前持仓情景报告以 {as_of_date} 为基准"
        + (f"，本地日线已更新到 {_db_latest_date}" if _db_latest_date and _db_latest_date > as_of_date else "")
        + "。对应的下一交易日可能已经结束；持仓盈亏换算仍可查看，但这些情景不能当作当前的明日预测。请先刷新行情并重新运行回测。"
    )
point = forecast.get("point_forecasts", {}) or {}
bands = forecast.get("historical_return_bands", {}) or {}
if current_price <= 0 or not bands:
    st.error("预测 JSON 缺少有效价格或历史情景数据，请重新运行收益幅度回测脚本。")
    st.stop()

# Start with blank-equivalent values so personal holdings are not exposed in a public repository.
with st.sidebar:
    st.header("我的持仓参数")
    st.caption("初始股数和持仓成本为 0，以保护个人财务信息。请在本地按实际持仓填写；这些输入不会发送给外部服务。")
    quantity = st.number_input("当前持有股数", min_value=0.0, value=0.0, step=0.5, format="%.4f")
    remaining_cost = st.number_input("剩余持仓总成本（USD）", min_value=0.0, value=0.0, step=1.0, format="%.2f")
    sell_fee = st.number_input("预计卖出总费用（USD）", min_value=0.0, value=0.0, step=0.25, format="%.2f")
    rebuy_fee = st.number_input("未来重新买入总费用（USD）", min_value=0.0, value=0.0, step=0.25, format="%.2f")
    target_profit = st.number_input("希望实现的净利润（USD）", min_value=0.0, value=15.0, step=1.0, format="%.2f")
    scenario_window_label = st.selectbox("情景参考窗口", ["最近 20 个交易日", "最近 60 个交易日", "最近 252 个交易日"], index=1)
    preference = st.selectbox("目前更倾向", ["还没决定", "更偏长期持有", "希望做波段"], index=0)
    acceptable_drawdown = st.slider("回撤提醒线（不是止损指令）", min_value=1, max_value=20, value=5, step=1)

window_lookup = {
    "最近 20 个交易日": ("20", "last_20_sessions"),
    "最近 60 个交易日": ("60", "last_60_sessions"),
    "最近 252 个交易日": ("252", "last_252_sessions"),
}
window_n, band_key = window_lookup[scenario_window_label]
selected_band = bands.get(band_key, {}) or {}
if not selected_band:
    st.error(f"预测结果中缺少 {scenario_window_label} 的历史情景。")
    st.stop()

p10 = _safe_float(selected_band.get("p10_price_scenario"))
p50 = _safe_float(selected_band.get("p50_price_scenario"))
p90 = _safe_float(selected_band.get("p90_price_scenario"))
r10 = _safe_float(selected_band.get("p10_return_pct"))
r50 = _safe_float(selected_band.get("p50_return_pct"))
r90 = _safe_float(selected_band.get("p90_return_pct"))
hist_mean = _safe_float(point.get("historical_mean_return_pct"))
ridge_full = _safe_float(point.get("ridge_full_return_pct"))
ridge_returns = _safe_float(point.get("ridge_returns_only_return_pct"))

# Best historical backtest MAE was the expanding mean baseline. Do not elevate the weaker Ridge models.
main_reference_pct = hist_mean
reference_price = current_price * (1 + main_reference_pct / 100)

st.caption(f"行情截至：{as_of_date} 收盘 · 最新收盘价 {_money(current_price)} · 预测报告：{forecast_path.name}")

# Optional freshness warning when the report is no longer based on the latest close the user expects.
try:
    price_day = pd.Timestamp(as_of_date)
    today = pd.Timestamp.now().normalize()
    if (today - price_day).days > 5:
        st.warning("当前预测所用行情日期较旧，请先同步行情并重新运行回测，再参考下面的情景。")
except Exception:
    pass

c1, c2, c3, c4 = st.columns(4)
c1.metric("最近收盘价", _money(current_price), as_of_date)
c2.metric("历史均值点估计", _pct(main_reference_pct, signed=True), f"参考价 {_money(reference_price)}")
c3.metric("所选窗口 P10–P90", f"{_pct(r10)} 至 {_pct(r90)}", f"经验区间 · {window_n} 日")

summary_df: pd.DataFrame | None = None
if summary_path is not None:
    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:
        summary_df = None
mae_value: float | None = None
ridge_mae_value: float | None = None
if summary_df is not None and {"experiment", "mae_bps"}.issubset(summary_df.columns):
    mean_rows = summary_df[summary_df["experiment"] == "expanding_mean_baseline"]
    ridge_rows = summary_df[summary_df["experiment"] == "ridge_full"]
    if not mean_rows.empty:
        mae_value = _safe_float(mean_rows.iloc[0]["mae_bps"])
    if not ridge_rows.empty:
        ridge_mae_value = _safe_float(ridge_rows.iloc[0]["mae_bps"])
if mae_value is not None:
    c4.metric("历史均值基线 MAE", _pct(mae_value / 100), "样本外平均绝对误差")
else:
    c4.metric("历史均值点估计", _pct(main_reference_pct, signed=True), "非真实上涨概率")

st.info(
    "先看结论：样本外回测中，历史均值基线的 MAE 优于完整 Ridge 模型。"
    "当前页面因此把历史均值显示为参考点估计，而不是挑选更乐观的模型数字。"
    "它仍不是经过校准的未来预测，不能单独作为买卖信号。"
)

# Forecast scenarios.
st.subheader("1. 明天可能是什么价格情景？")
st.caption("P10 / P50 / P90 是历史收益分布的分位数情景，不是保证，也不是经过校准的未来概率区间。")
scenarios = [
    {"情景": "偏弱情景（P10）", "收益率": r10, "价格": p10},
    {"情景": "中位情景（P50）", "收益率": r50, "价格": p50},
    {"情景": "偏强情景（P90）", "收益率": r90, "价格": p90},
]
sc_cols = st.columns(3)
for col, scenario in zip(sc_cols, scenarios):
    price_delta = scenario["价格"] - current_price
    position_delta = quantity * price_delta
    with col:
        col.metric(scenario["情景"], _money(scenario["价格"]), f"{_pct(scenario['收益率'], signed=True)} · 每股 {price_delta:+.2f}")
        col.caption(f"{quantity:g} 股市值变化：{position_delta:+,.2f} USD")

scenario_rows = []
for scenario in scenarios:
    hypothetical_value = quantity * scenario["价格"]
    hypothetical_net = hypothetical_value - remaining_cost - sell_fee
    scenario_rows.append(
        {
            "历史情景": scenario["情景"],
            "情景收益率": scenario["收益率"],
            "情景价格（USD）": scenario["价格"],
            "持仓市值（USD）": hypothetical_value,
            "相对当前市值变化（USD）": quantity * (scenario["价格"] - current_price),
            "若该价卖出，扣预估卖费后净盈亏（USD）": hypothetical_net,
        }
    )
st.dataframe(
    pd.DataFrame(scenario_rows).style.format(
        {
            "情景收益率": "{:+.2f}%",
            "情景价格（USD）": "${:,.2f}",
            "持仓市值（USD）": "${:,.2f}",
            "相对当前市值变化（USD）": "{:+,.2f}",
            "若该价卖出，扣预估卖费后净盈亏（USD）": "{:+,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("情景净盈亏 = 情景卖出金额 − 剩余持仓成本 − 预估卖出费用。尚未卖出时，浮盈亏并未实现。")

# Current holding and cost-aware planner.
st.subheader("2. 我的持仓：涨跌后实际差多少钱？")
current_value = quantity * current_price
current_unrealized = current_value - remaining_cost
break_even_price = (remaining_cost + sell_fee) / quantity if quantity > 0 else None
target_price = (remaining_cost + target_profit + sell_fee) / quantity if quantity > 0 else None
price_to_target_pct = (target_price / current_price - 1) * 100 if quantity > 0 and current_price > 0 else None
h1, h2, h3, h4 = st.columns(4)
h1.metric("当前持仓市值", _money(current_value))
h2.metric("当前浮盈 / 浮亏", _money(current_unrealized), _pct(current_unrealized / remaining_cost * 100, signed=True) if remaining_cost > 0 else "成本为 0")
h3.metric("含预估卖费的保本价", _money(break_even_price) if break_even_price is not None else "需输入股数")
h4.metric("实现目标净利润的价格", _money(target_price) if target_price is not None else "需输入股数", _pct(price_to_target_pct, signed=True) if price_to_target_pct is not None else "")

if quantity > 0 and target_price is not None:
    required_gain_per_share = target_price - current_price
    st.write(
        f"按当前输入，若要在卖出后净赚 **{_money(target_profit)}**，估算平均卖出价需要达到 **{_money(target_price)}**，"
        f"比最近收盘价高 **{_money(required_gain_per_share)} / 股（{_pct(price_to_target_pct, signed=True)}）**。"
    )
    if target_price > p90:
        st.warning(
            f"这个目标价高于最近 {window_n} 个交易日收益分布的 P90 参考价 {_money(p90)}。"
            "这不是说目标一定无法达到，而是不要把它当成次日大概率就能完成的目标。"
        )

round_trip_cost = sell_fee + rebuy_fee
if quantity > 0:
    friction_per_share = round_trip_cost / quantity
    friction_pct = friction_per_share / current_price * 100 if current_price else 0.0
    st.write(
        f"**波段费用门槛：** 卖出再按相同股数买回，预估费用合计 {_money(round_trip_cost)}；"
        f"价格至少需要产生约 {_money(friction_per_share)} / 股（当前价的 {friction_pct:.3f}%）的有利回落，"
        "才足以抵消这两笔费用。实际结果还会受成交价、滑点、汇兑和费用规则影响。"
    )

# Compare the three empirical windows and their walk-forward interval coverage.
st.subheader("3. 用不同历史窗口比较风险范围")
window_rows: list[dict[str, Any]] = []
coverage_lookup: dict[str, float] = {}
if interval_path is not None:
    try:
        intervals_df = pd.read_csv(interval_path)
        for _, row in intervals_df.iterrows():
            label = str(row.get("interval", ""))
            for n in (20, 60, 252):
                if f"{n}d" in label:
                    coverage_lookup[str(n)] = _safe_float(row.get("observed_coverage"), float("nan"))
    except Exception:
        pass

for n, key in [(20, "last_20_sessions"), (60, "last_60_sessions"), (252, "last_252_sessions")]:
    band = bands.get(key, {}) or {}
    if not band:
        continue
    row = {
        "历史窗口": f"最近 {n} 日",
        "P10 价格（USD）": _safe_float(band.get("p10_price_scenario")),
        "P50 价格（USD）": _safe_float(band.get("p50_price_scenario")),
        "P90 价格（USD）": _safe_float(band.get("p90_price_scenario")),
        "P10–P90 收益范围": f"{_pct(_safe_float(band.get('p10_return_pct')))} 至 {_pct(_safe_float(band.get('p90_return_pct')))}",
        "样本外实际覆盖率": _pct(coverage_lookup[str(n)] * 100) if str(n) in coverage_lookup else "尚无数据",
    }
    window_rows.append(row)
st.dataframe(
    pd.DataFrame(window_rows).style.format(
        {"P10 价格（USD）": "${:,.2f}", "P50 价格（USD）": "${:,.2f}", "P90 价格（USD）": "${:,.2f}"}
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption("历史样本外覆盖率来自过去的滚动测试。即使接近 80%，也不能保证下一次区间具有同样覆盖率。")

st.subheader("4. 长期持有还是波段？")
if quantity <= 0:
    st.warning("填写持仓股数后，才能计算持仓情景。")
else:
    analysis = _explain_scenario(
        preference=preference,
        target_price=target_price if target_price is not None else current_price,
        selected_p90=p90,
        target_profit=target_profit,
        qty=quantity,
        sell_fee=sell_fee,
        rebuy_fee=rebuy_fee,
        acceptable_drawdown=float(acceptable_drawdown),
    )
    st.markdown(analysis)

    alert_price = current_price * (1 - acceptable_drawdown / 100)
    alert_value_drop = quantity * (current_price - alert_price)
    st.info(
        f"你的回撤复核线是 **{_money(alert_price)}**。如果到达该价格，按 {quantity:g} 股估算，"
        f"相对当前市值减少约 **{_money(alert_value_drop)}**。这是提醒你重新评估，不是自动止损或买卖指令。"
    )

# Show the measurable evidence behind the model choice.
st.subheader("5. 模型到底准不准？")
if summary_df is not None:
    show_rows = summary_df[summary_df.get("experiment", pd.Series(dtype=str)).isin(
        ["expanding_mean_baseline", "zero_return_baseline", "ridge_returns_only", "ridge_full"]
    )].copy()
    if not show_rows.empty:
        names = {
            "expanding_mean_baseline": "历史均值基线",
            "zero_return_baseline": "预测 0% 基线",
            "ridge_returns_only": "收益率特征 Ridge",
            "ridge_full": "完整特征 Ridge",
        }
        show_rows["方法"] = show_rows["experiment"].map(names).fillna(show_rows["experiment"])
        if "mae_bps" in show_rows:
            show_rows["MAE（百分比）"] = pd.to_numeric(show_rows["mae_bps"], errors="coerce") / 100
        if "rmse_bps" in show_rows:
            show_rows["RMSE（百分比）"] = pd.to_numeric(show_rows["rmse_bps"], errors="coerce") / 100
        if "sign_accuracy" in show_rows:
            show_rows["方向命中率"] = pd.to_numeric(show_rows["sign_accuracy"], errors="coerce") * 100
        cols = [c for c in ["方法", "MAE（百分比）", "RMSE（百分比）", "方向命中率"] if c in show_rows.columns]
        st.dataframe(
            show_rows[cols].style.format(
                {"MAE（百分比）": "{:.3f}%", "RMSE（百分比）": "{:.3f}%", "方向命中率": "{:.1f}%"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("没有找到可展示的回测汇总行。")
else:
    st.caption("没有找到回测汇总 CSV，因此暂时无法展示模型与基线的误差比较。")

if mae_value is not None and ridge_mae_value is not None:
    st.warning(
        f"当前历史回测里，完整 Ridge 的 MAE 为 {_pct(ridge_mae_value / 100, 3)}，"
        f"高于历史均值基线 {_pct(mae_value / 100, 3)}。因此页面只把历史均值作为参考，不把 Ridge 预测包装成交易优势。"
    )

with st.expander("数据来源与限制"):
    st.write(f"预测文件：`{forecast_path.relative_to(ROOT)}`")
    st.write(f"行情特征截至：{as_of_date}；收盘价：{_money(current_price)}。")
    st.write(forecast.get("caveat", "历史区间不是校准的未来概率区间。"))
    st.write("本页面的仓位、持仓成本及费用为用户可编辑输入；它们不会自动同步逐笔交易。")
    st.write("此页面尚未调用 DeepSeek 生成个性化报告；当前操作说明是基于回测结果与透明规则生成，不是投资指令。")

st.caption("下一步可以将这里的持仓输入接到主仪表盘已计算的持仓数据，再通过显式同意按钮调用现有 DeepSeek 持仓分析模块。")
