from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import subprocess
import sys
from datetime import date
from uuid import uuid4
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Paths and page configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.portfolio import (
    PortfolioValidationError,
    build_scenario_table,
    calculate_portfolio,
    validate_transactions,
)
from app.decision_support import build_decision_assessment
from app.portfolio_ai import generate_portfolio_analysis

DB_PATH = ROOT / "data" / "market_data.db"
HISTORY_PATH = ROOT / "data" / "QQQM_tiingo_history.csv"
REPORT_DIR = ROOT / "reports"
PORTFOLIO_TRANSACTIONS_PATH = ROOT / "data" / "portfolio_transactions.csv"


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def latest_file(pattern: str) -> Path | None:
    """Return the most recently modified file matching a report pattern."""
    if not REPORT_DIR.exists():
        return None
    files = list(REPORT_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def run_command(args: list[str], timeout: int = 600) -> dict[str, Any]:
    """Run an existing project CLI using the current Python interpreter."""
    try:
        result = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "command": args,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"命令运行超过 {timeout} 秒，已停止等待。",
            "command": args,
        }
    except Exception as exc:  # Keep UI alive and surface the problem.
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "command": args,
        }


def render_command_result(result: dict[str, Any] | None) -> None:
    """Display the result of a command previously run by the dashboard."""
    if not result:
        return

    if result.get("returncode") == 0:
        st.success("命令执行完成。")
    else:
        st.error("命令执行失败，请查看输出和错误信息。")

    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    command = result.get("command", [])

    if command:
        with st.expander("查看运行命令"):
            st.code(subprocess.list2cmdline(command), language="powershell")

    if stdout:
        with st.expander("查看命令输出", expanded=result.get("returncode") != 0):
            st.code(stdout)

    if stderr:
        with st.expander("查看错误信息", expanded=True):
            st.code(stderr)


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file without allowing a rendering error to crash UI."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"读取文件失败：{exc}"


@st.cache_data(ttl=60, show_spinner=False)
def load_market_data() -> pd.DataFrame:
    """Read QQQM daily OHLCV data from the existing SQLite database."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT trade_date, open, high, low, close, volume
            FROM market_daily
            WHERE ticker = ?
            ORDER BY trade_date
            """,
            conn,
            params=("QQQM",),
        )


@st.cache_data(ttl=300, show_spinner=False)
def load_latest_market_prices(tickers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Return the latest locally stored close per ticker and its trade date."""
    if not DB_PATH.exists() or not tickers:
        return {}

    cleaned = tuple(sorted({str(t).strip().upper() for t in tickers if str(t).strip()}))
    if not cleaned:
        return {}

    placeholders = ",".join("?" for _ in cleaned)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            frame = pd.read_sql_query(
                f"SELECT ticker, trade_date, close FROM market_daily "
                f"WHERE UPPER(ticker) IN ({placeholders}) ORDER BY trade_date",
                conn,
                params=cleaned,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return {}

    if frame.empty:
        return {}
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    latest: dict[str, dict[str, Any]] = {}
    for ticker, group in frame.groupby("ticker", sort=False):
        row = group.iloc[-1]
        latest[ticker] = {
            "price": float(row["close"]),
            "date": row["trade_date"].strftime("%Y-%m-%d"),
        }
    return latest


@st.cache_data(ttl=60, show_spinner=False)
def load_fred_macro_data() -> pd.DataFrame:
    """Read existing FRED observations from the local database."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                "SELECT series_id, observation_date, value, retrieved_at "
                "FROM macro_observations ORDER BY observation_date",
                conn,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_bls_macro_data() -> pd.DataFrame:
    """Read existing BLS economic observations from the local database."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                "SELECT series_id, series_name, category, year, period, "
                "period_name, value, unit, footnotes, source_name, source_url, fetched_at "
                "FROM bls_macro_observations",
                conn,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_news_data() -> pd.DataFrame:
    """Read official news items stored locally."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(
                "SELECT news_id, source_id, source_name, category, title, summary, "
                "url, published_at, fetched_at FROM news_items "
                "ORDER BY published_at DESC",
                conn,
            )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def load_backtest_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the most recent rolling-grid trade/equity CSVs and matching JSON."""
    equity_file = latest_file("QQQM_rolling_grid_backtest_*_equity.csv")
    trades_file = latest_file("QQQM_rolling_grid_backtest_*_trades.csv")
    equity_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    summary: dict[str, Any] = {}

    if equity_file:
        try:
            equity_df = pd.read_csv(equity_file)
            if "date" in equity_df.columns:
                equity_df["date"] = pd.to_datetime(
                    equity_df["date"], errors="coerce"
                )
            equity_df = equity_df.dropna(subset=["date"]).sort_values("date")
            equity_df = equity_df.reset_index(drop=True)
        except Exception as exc:
            st.error(f"读取资金曲线失败：{exc}")
            equity_df = pd.DataFrame()

        json_file = equity_file.with_name(
            equity_file.name.removesuffix("_equity.csv") + ".json"
        )
        if json_file.exists():
            try:
                content = json.loads(json_file.read_text(encoding="utf-8"))
                summary = content.get("summary", {})
            except (OSError, json.JSONDecodeError, AttributeError):
                summary = {}

    if trades_file:
        try:
            trades_df = pd.read_csv(trades_file)
            if "date" in trades_df.columns:
                trades_df["date"] = pd.to_datetime(
                    trades_df["date"], errors="coerce"
                )
            if "side" in trades_df.columns:
                trades_df["side"] = (
                    trades_df["side"].astype(str).str.upper()
                )
            if "date" in trades_df.columns:
                trades_df = trades_df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            st.error(f"读取交易记录失败：{exc}")
            trades_df = pd.DataFrame()

    return equity_df, trades_df, summary


def max_drawdown_pct(values: pd.Series | list[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive percentage."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return 0.0
    peak = series.cummax()
    drawdown = series / peak - 1.0
    return abs(float(drawdown.min()) * 100.0)


def annualized_volatility_pct(values: pd.Series | list[float]) -> float:
    """Annualized volatility based on daily account-equity returns."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    returns = series.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * (252.0**0.5) * 100.0)


def render_latest_report(pattern: str, *, title: str) -> None:
    report = latest_file(pattern)
    if report is None:
        st.info("暂时没有找到报告。先运行相应分析任务即可生成。")
        return

    st.caption(f"{title}：{report.name}")
    st.markdown(read_text_file(report))


# -----------------------------------------------------------------------------
# Header and navigation
# -----------------------------------------------------------------------------
st.title("📊 市场、持仓与网格回测")
st.caption("这里保留原来的详细市场分析、个人持仓、网格策略回测和宏观 / 新闻资料。首次使用建议先从‘每日总览与复盘’开始。")

# >>> QQQM INVESTMENT OVERVIEW V0.4.0 >>>
st.divider()
st.header("投资总览")
st.caption(
    "先看整体赚亏和当前持仓；下方各标签页保留详细行情、AI 研究、持仓明细、回测与宏观数据。"
)
st.info("怎么理解这页：‘当前浮动盈亏’是按最新已同步收盘价估算的未实现盈亏；‘已实现盈亏’来自已经卖出的交易；‘网格回测’是在历史数据上模拟规则，不代表未来收益保证。")

if not PORTFOLIO_TRANSACTIONS_PATH.exists():
    st.info(
        "还没有本地交易记录。请进入“我的投资组合”录入或导入交易记录，"
        "投资总览就会根据记录和本地行情自动计算。"
    )
else:
    try:
        overview_raw = pd.read_csv(
            PORTFOLIO_TRANSACTIONS_PATH,
            encoding="utf-8-sig",
        )
        overview_tx, overview_warnings = validate_transactions(overview_raw)

        overview_tickers = tuple(sorted(overview_tx["ticker"].unique().tolist()))
        overview_market_prices = load_latest_market_prices(overview_tickers)
        overview_current_prices: dict[tuple[str, str], float] = {}
        overview_price_dates: dict[str, str] = {}

        for _, overview_pair in overview_tx[["ticker", "currency"]].drop_duplicates().iterrows():
            overview_ticker = str(overview_pair["ticker"]).upper()
            overview_currency = str(overview_pair["currency"]).upper()
            overview_record = (
                overview_market_prices.get(overview_ticker)
                if overview_currency == "USD"
                else None
            )
            if overview_record and overview_record.get("price") is not None:
                overview_current_prices[(overview_ticker, overview_currency)] = float(
                    overview_record["price"]
                )
                overview_price_dates[overview_ticker] = str(
                    overview_record.get("date", "未知日期")
                )

        overview_calculation = calculate_portfolio(
            overview_raw,
            current_prices=overview_current_prices,
        )
        overview_positions = overview_calculation.positions.copy()

        for overview_warning in overview_warnings:
            st.warning(overview_warning)

        if overview_positions.empty:
            st.info("交易记录已读取，但没有可展示的持仓汇总。")
        else:
            overview_currencies = sorted(
                overview_positions["currency"].dropna().unique().tolist()
            )

            for overview_currency in overview_currencies:
                overview_currency_positions = overview_positions[
                    overview_positions["currency"] == overview_currency
                ]
                overview_open = overview_currency_positions[
                    overview_currency_positions["open_position"].astype(bool)
                ]
                overview_missing_price = (
                    not overview_open.empty
                    and overview_open["current_price"].isna().any()
                )

                overview_cost = (
                    float(overview_open["remaining_cost_basis"].sum())
                    if not overview_open.empty
                    else 0.0
                )
                overview_realized = float(
                    overview_currency_positions["realized_pnl"].sum()
                )
                overview_dividends = float(
                    overview_currency_positions["dividend_income"].sum()
                )
                overview_fees = float(
                    overview_currency_positions["total_fees"].sum()
                )
                overview_value = (
                    float(overview_open["market_value"].sum())
                    if not overview_missing_price
                    else None
                )
                overview_unrealized = (
                    float(overview_open["unrealized_pnl"].sum())
                    if not overview_missing_price
                    else None
                )
                overview_cumulative = (
                    overview_realized + overview_unrealized + overview_dividends
                    if overview_unrealized is not None
                    else None
                )

                st.subheader(f"{overview_currency} 投资结果")
                overview_cols = st.columns(4)
                if overview_cumulative is None:
                    overview_cols[0].metric("累计净盈亏", "暂不可计算")
                else:
                    overview_cols[0].metric(
                        "累计净盈亏",
                        f"{overview_cumulative:+,.2f} {overview_currency}",
                        help="已实现盈亏 + 当前未实现盈亏 + 净分红。手续费已按成本或净收入口径计入，不重复扣除。",
                    )
                overview_cols[1].metric(
                    "当前持仓市值",
                    f"{overview_value:,.2f} {overview_currency}"
                    if overview_value is not None
                    else "缺少估值价格",
                )
                overview_cols[2].metric(
                    "当前浮动盈亏",
                    f"{overview_unrealized:+,.2f} {overview_currency}"
                    if overview_unrealized is not None
                    else "暂不可计算",
                )
                overview_cols[3].metric(
                    "剩余持仓成本",
                    f"{overview_cost:,.2f} {overview_currency}",
                )

                if overview_cumulative is None:
                    st.warning(
                        "部分未平仓标的缺少可用估值价格，因此暂时不能得出可靠的累计净盈亏。"
                    )
                elif overview_cumulative > 0:
                    st.success(
                        f"简单结论：按当前已记录交易和最新可用行情，你的已记录投资净盈利 "
                        f"{overview_cumulative:,.2f} {overview_currency}。"
                    )
                elif overview_cumulative < 0:
                    st.error(
                        f"简单结论：按当前已记录交易和最新可用行情，你的已记录投资净亏损 "
                        f"{abs(overview_cumulative):,.2f} {overview_currency}。"
                    )
                else:
                    st.info("简单结论：按当前记录计算，累计净盈亏约为零。")

                st.caption(
                    f"已实现盈亏 {overview_realized:+,.2f} {overview_currency}；"
                    f"净分红 {overview_dividends:+,.2f} {overview_currency}；"
                    f"累计记录手续费 {overview_fees:,.2f} {overview_currency}。"
                    "手续费已包含在相应成本或净收入中，不应再扣一次。"
                )

                overview_table = overview_currency_positions.rename(columns={
                    "ticker": "证券",
                    "quantity": "持有数量",
                    "current_price": "最新可用价格",
                    "market_value": "当前持仓市值",
                    "average_cost": "每股加权平均成本",
                    "remaining_cost_basis": "剩余持仓成本",
                    "unrealized_pnl": "当前浮动盈亏",
                    "realized_pnl": "历史已实现盈亏",
                    "dividend_income": "净分红",
                    "cumulative_net_pnl": "累计净盈亏",
                    "equivalent_cost_per_share": "净收益调整后等效成本",
                    "open_position": "仍有持仓",
                })
                overview_cols_to_show = [
                    "证券", "quantity", "持有数量", "最新可用价格", "当前持仓市值",
                    "每股加权平均成本", "剩余持仓成本", "当前浮动盈亏",
                    "历史已实现盈亏", "净分红", "累计净盈亏",
                    "净收益调整后等效成本", "仍有持仓",
                ]
                overview_cols_to_show = [
                    col for col in overview_cols_to_show if col in overview_table.columns
                ]
                st.dataframe(
                    overview_table[overview_cols_to_show],
                    use_container_width=True,
                    hide_index=True,
                )

                overview_used_dates = [
                    f"{ticker}: {market_date}"
                    for ticker, market_date in sorted(overview_price_dates.items())
                    if ticker in overview_tickers
                ]
                st.caption(
                    "价格是本地数据库保存的最新可用日线收盘价，不是实时行情。"
                    + (" 价格日期：" + "；".join(overview_used_dates) if overview_used_dates else " 当前没有可用的本地美元日线价格。")
                )

            st.caption(
                "口径提示：累计净盈亏仅反映本地录入的交易、净分红与当前可用估值；"
                "因未记录完整的账户入金、出金及现金余额，它不等同于银行账户的真实总回报率。"
                "等效成本是解释性参考，不是券商税务成本、保本价或买卖目标价。"
            )

    except PortfolioValidationError as exc:
        st.error("无法生成投资总览：本地交易记录未通过校验。")
        for overview_error in exc.errors:
            st.write(f"- {overview_error}")
    except Exception as exc:
        st.error(f"投资总览加载失败：{type(exc).__name__}: {exc}")

# <<< QQQM INVESTMENT OVERVIEW V0.4.0 <<<

st.caption("个人投资研究平台 · 行情分析 · AI 报告 · 策略回测")
st.warning("本工具用于学习、研究和历史回测，不构成投资建议。历史表现不代表未来结果。")

market_tab, report_tab, portfolio_tab, backtest_tab, macro_tab, status_tab = st.tabs(
    [
        "📊 市场概览",
        "🧠 AI 分析",
        "👛 我的投资组合",
        "⚙️ 策略回测",
        "🌎 宏观与新闻",
        "🗄️ 数据状态",
    ]
)


# -----------------------------------------------------------------------------
# 1. Market overview
# -----------------------------------------------------------------------------
with market_tab:
    heading_col, refresh_col = st.columns([5, 1])
    with heading_col:
        st.subheader("QQQM 市场行情")
    with refresh_col:
        if st.button("刷新行情", key="refresh_market_data"):
            load_market_data.clear()
            st.rerun()

    if not DB_PATH.exists():
        st.warning(f"没有找到数据库文件：{DB_PATH}")
    else:
        try:
            market_df = load_market_data()
        except Exception as exc:
            market_df = pd.DataFrame()
            st.error(f"读取市场数据失败：{type(exc).__name__}: {exc}")

        if market_df.empty:
            st.warning("数据库中没有读到 QQQM 行情，请先同步行情数据。")
        else:
            market_df["trade_date"] = pd.to_datetime(
                market_df["trade_date"], errors="coerce"
            )
            market_df = (
                market_df.dropna(subset=["trade_date", "close"])
                .sort_values("trade_date")
                .reset_index(drop=True)
            )

            if market_df.empty:
                st.warning("行情记录没有有效的日期或收盘价。")
            else:
                latest = market_df.iloc[-1]
                latest_close = float(latest["close"])

                def period_return(trading_days_back: int) -> float | None:
                    index = len(market_df) - 1 - trading_days_back
                    if index < 0:
                        return None
                    previous_close = float(market_df.iloc[index]["close"])
                    if previous_close == 0:
                        return None
                    return (latest_close / previous_close - 1.0) * 100.0

                daily_return = period_return(1)
                return_5d = period_return(5)
                return_20d = period_return(20)

                metric_cols = st.columns(4)
                metric_cols[0].metric(
                    "最新收盘价",
                    f"${latest_close:,.2f}",
                    f"{daily_return:+.2f}%" if daily_return is not None else None,
                )
                metric_cols[1].metric(
                    "近 5 个交易日",
                    f"{return_5d:+.2f}%" if return_5d is not None else "数据不足",
                )
                metric_cols[2].metric(
                    "近 20 个交易日",
                    f"{return_20d:+.2f}%" if return_20d is not None else "数据不足",
                )
                metric_cols[3].metric("日线记录数", f"{len(market_df):,}")

                st.caption(
                    f"最新交易日：{latest['trade_date'].date()}。"
                    "收益率按收盘价计算，未计入分红和交易成本。"
                )

                st.subheader("历史收盘价")
                price_series = market_df.set_index("trade_date")["close"]
                st.line_chart(price_series, y_label="价格（美元）")

                st.subheader("最近 15 个交易日")
                recent = market_df.tail(15).sort_values(
                    "trade_date", ascending=False
                ).copy()
                recent["trade_date"] = recent["trade_date"].dt.strftime("%Y-%m-%d")
                st.dataframe(
                    recent,
                    use_container_width=True,
                    hide_index=True,
                )


# -----------------------------------------------------------------------------
# 2. AI analysis report
# -----------------------------------------------------------------------------
with report_tab:
    st.subheader("AI 市场分析")
    st.write(
        "调用项目现有分析流程，结合行情、宏观数据与新闻生成研究报告。"
    )

    if st.button("生成 / 刷新 AI 分析报告", type="primary", key="run_ai_analysis"):
        with st.spinner("正在运行分析程序，这可能需要一点时间……"):
            result = run_command(
                [
                    sys.executable,
                    "main.py",
                    "analyze",
                    "--ticker",
                    "QQQM",
                    "--with-macro",
                ],
                timeout=600,
            )
        st.session_state["ai_result"] = result

    render_command_result(st.session_state.get("ai_result"))

    st.divider()
    st.subheader("最近生成的投资报告")
    render_latest_report(
        "QQQM_investment_report_*.md",
        title="投资报告",
    )


# -----------------------------------------------------------------------------
# 3. Personal portfolio import, accounting and scenario analysis
# -----------------------------------------------------------------------------
with portfolio_tab:
    st.subheader("我的投资组合")
    st.caption(
        "支持手动录入单笔交易，也支持上传 CSV。手动记录只保存在当前电脑的本地文件中；"
        "不会自动上传到 AI 服务或远程数据库。"
    )
    st.info(
        "第一版支持买入、卖出和现金分红；成本采用加权平均成本法。"
        "估值价格优先读取本地数据库最近收盘价（仅对 USD 标的自动填入），并允许手动修改。"
        "不同币种分开计算，不会未经汇率换算就直接相加。"
    )

    template_columns = [
        "trade_date", "ticker", "side", "quantity", "price", "fee",
        "currency", "amount", "trade_id", "notes",
    ]
    blank_template = pd.DataFrame(columns=template_columns).to_csv(
        index=False
    ).encode("utf-8-sig")
    st.download_button(
        "下载空白交易记录模板 CSV",
        data=blank_template,
        file_name="portfolio_template.csv",
        mime="text/csv",
        key="download_portfolio_template",
    )

    st.caption(
        f"本地交易记录文件：{PORTFOLIO_TRANSACTIONS_PATH.relative_to(ROOT)}。"
        "请勿把个人交易记录提交到公开 GitHub 仓库。"
    )
    source_mode = st.radio(
        "选择录入方式",
        ["手动录入（本地保存）", "上传 CSV（临时分析）"],
        horizontal=True,
        key="portfolio_source_mode",
    )

    raw_transactions = pd.DataFrame(columns=template_columns)

    if source_mode == "手动录入（本地保存）":
        st.markdown("#### 添加一笔交易")
        st.write(
            "例如：买入 2 股 QQQM，成交价 290.53 美元。"
            "请填写实际成交日期；历史交易不能一律填今天，否则已实现盈亏和持仓顺序会算错。"
        )

        side_label = st.selectbox(
            "交易类型",
            ["买入", "卖出", "现金分红"],
            key="portfolio_manual_side",
        )
        side_value = {"买入": "BUY", "卖出": "SELL", "现金分红": "DIVIDEND"}[side_label]

        with st.form("portfolio_manual_transaction_form", clear_on_submit=True):
            date_col, ticker_col, currency_col = st.columns(3)
            with date_col:
                manual_trade_date = st.date_input(
                    "实际成交日期",
                    value=date.today(),
                    help="建议填写券商交易记录里的成交日期。",
                )
            with ticker_col:
                manual_ticker = st.text_input("证券代码", value="QQQM").strip().upper()
            with currency_col:
                manual_currency = st.text_input("交易币种", value="USD").strip().upper()

            manual_quantity = 0.0
            manual_price = 0.0
            manual_amount = 0.0
            if side_value in {"BUY", "SELL"}:
                quantity_col, price_col, fee_col = st.columns(3)
                with quantity_col:
                    manual_quantity = st.number_input(
                        "股数 / 数量",
                        min_value=0.0,
                        value=2.0,
                        step=1.0,
                        format="%.6f",
                        help="例如买入或卖出 2 股，就填 2。",
                    )
                with price_col:
                    manual_price = st.number_input(
                        "每股成交价",
                        min_value=0.0,
                        value=290.53,
                        step=0.01,
                        format="%.4f",
                    )
                with fee_col:
                    manual_fee = st.number_input(
                        "手续费（总额）",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                    )
            else:
                amount_col, fee_col = st.columns(2)
                with amount_col:
                    manual_amount = st.number_input(
                        "本次现金分红总额",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                    )
                with fee_col:
                    manual_fee = st.number_input(
                        "分红相关费用（如有）",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                    )

            manual_notes = st.text_input(
                "备注（选填）",
                placeholder="例如：众安银行成交记录 / 分批买入",
            )
            submitted = st.form_submit_button(
                "保存这笔交易",
                type="primary",
                use_container_width=True,
            )

        # Read the saved local ledger before processing an added transaction.
        if PORTFOLIO_TRANSACTIONS_PATH.exists():
            try:
                saved_transactions = pd.read_csv(
                    PORTFOLIO_TRANSACTIONS_PATH,
                    encoding="utf-8-sig",
                )
            except Exception as exc:
                saved_transactions = pd.DataFrame(columns=template_columns)
                st.error(f"读取本地交易记录失败：{type(exc).__name__}: {exc}")
        else:
            saved_transactions = pd.DataFrame(columns=template_columns)

        if submitted:
            new_row = {
                "trade_date": manual_trade_date.isoformat(),
                "ticker": manual_ticker,
                "side": side_value,
                "quantity": manual_quantity if side_value in {"BUY", "SELL"} else 0.0,
                "price": manual_price if side_value in {"BUY", "SELL"} else 0.0,
                "fee": float(manual_fee),
                "currency": manual_currency,
                "amount": manual_amount if side_value == "DIVIDEND" else 0.0,
                "trade_id": uuid4().hex[:16],
                "notes": manual_notes.strip(),
            }
            candidate = pd.concat(
                [saved_transactions.reindex(columns=template_columns), pd.DataFrame([new_row])],
                ignore_index=True,
            )
            try:
                # Validate both row fields and chronological position consistency before saving.
                normalized_candidate, duplicate_warnings = validate_transactions(candidate)
                calculate_portfolio(candidate)
                PORTFOLIO_TRANSACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
                candidate.to_csv(
                    PORTFOLIO_TRANSACTIONS_PATH,
                    index=False,
                    encoding="utf-8-sig",
                )
                saved_transactions = candidate
                st.success(
                    f"已保存：{manual_trade_date.isoformat()} · {manual_ticker} · "
                    f"{side_label}。当前本地共有 {len(candidate)} 笔记录。"
                )
                for warning in duplicate_warnings:
                    st.warning(warning)
            except PortfolioValidationError as exc:
                st.error("这笔交易没有保存，因为记录校验未通过：")
                for error in exc.errors:
                    st.write(f"- {error}")
            except Exception as exc:
                st.error(f"保存交易失败：{type(exc).__name__}: {exc}")

        # Refresh the stored ledger after a successful save.
        if PORTFOLIO_TRANSACTIONS_PATH.exists():
            try:
                saved_transactions = pd.read_csv(
                    PORTFOLIO_TRANSACTIONS_PATH,
                    encoding="utf-8-sig",
                )
            except Exception:
                pass

        if not saved_transactions.empty:
            st.markdown("#### 本地交易记录")
            try:
                normalized_saved, saved_warnings = validate_transactions(saved_transactions)
                for warning in saved_warnings:
                    st.warning(warning)
            except PortfolioValidationError as exc:
                st.error("本地记录存在校验问题，暂时不能进行持仓分析。")
                for error in exc.errors:
                    st.write(f"- {error}")
                normalized_saved = pd.DataFrame()

            st.dataframe(
                saved_transactions.sort_values("trade_date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            if not saved_transactions.empty:
                id_to_label = {}
                for _, row in saved_transactions.iterrows():
                    trade_id = str(row.get("trade_id", ""))
                    label = (
                        f"{row.get('trade_date', '')} | {row.get('ticker', '')} | "
                        f"{row.get('side', '')} | qty={row.get('quantity', '')} | "
                        f"price={row.get('price', '')} | ID={trade_id}"
                    )
                    if trade_id:
                        id_to_label[trade_id] = label

                if id_to_label:
                    with st.expander("删除录错的交易记录"):
                        st.warning("删除后会立即更新本地记录。若只是价格或日期填错，也可以先删除，再重新录入。")
                        with st.form("portfolio_delete_transaction_form"):
                            selected_trade_id = st.selectbox(
                                "选择要删除的交易",
                                options=list(id_to_label),
                                format_func=lambda value: id_to_label[value],
                            )
                            confirm_delete = st.checkbox("我确认删除所选记录")
                            delete_submitted = st.form_submit_button("删除所选记录")
                        if delete_submitted:
                            if not confirm_delete:
                                st.warning("请先勾选确认框。")
                            else:
                                remaining = saved_transactions[
                                    saved_transactions["trade_id"].astype(str) != selected_trade_id
                                ].copy()
                                remaining.to_csv(
                                    PORTFOLIO_TRANSACTIONS_PATH,
                                    index=False,
                                    encoding="utf-8-sig",
                                )
                                st.success("已删除所选记录。")
                                st.rerun()

            raw_transactions = saved_transactions.copy()
        else:
            st.info("还没有本地交易记录。用上面的表单添加一笔交易即可，不需要制作 CSV。")

    else:
        st.markdown("#### 上传交易记录 CSV")
        st.caption("CSV 只用于本次页面分析，不会自动保存到本地交易记录。")
        uploaded_file = st.file_uploader(
            "上传交易记录 CSV",
            type=["csv"],
            key="portfolio_transaction_upload",
            help="适合已有交易历史的用户；也可以切回手动录入。",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            try:
                raw_transactions = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding="utf-8-sig",
                )
            except UnicodeDecodeError:
                try:
                    raw_transactions = pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding="gb18030",
                    )
                except Exception as exc:
                    st.error(f"无法解析 CSV 编码：{exc}")
            except Exception as exc:
                st.error(f"读取 CSV 失败：{type(exc).__name__}: {exc}")

    if raw_transactions.empty:
        if source_mode == "上传 CSV（临时分析）":
            st.info("上传 CSV 后，可以先预览并验证记录，再计算持仓分析。")
        else:
            st.info("请先保存至少一笔本地交易记录，然后即可计算持仓与盈亏。")
    else:
        # A deterministic fingerprint prevents calculations from mixing stale results
        # when the transaction input changes, regardless of whether it came from a file or a form.
        fingerprint_source = raw_transactions.to_csv(index=False).encode("utf-8")
        file_fingerprint = hashlib.sha256(fingerprint_source).hexdigest()[:12]
        result_key = f"portfolio_result_{file_fingerprint}"
        price_editor_key = f"portfolio_price_editor_{file_fingerprint}"
        st.subheader("交易记录预览")
        st.dataframe(
            raw_transactions.head(20),
            use_container_width=True,
            hide_index=True,
        )

        try:
            normalized_preview, preview_warnings = validate_transactions(
                raw_transactions
            )
            st.success(
                f"基础校验通过：{len(normalized_preview)} 条交易记录，"
                f"涉及 {normalized_preview['ticker'].nunique()} 个标的。"
            )
            for warning in preview_warnings:
                st.warning(warning)
            valid_upload = True
        except PortfolioValidationError as exc:
            valid_upload = False
            st.error("交易记录未通过校验，请修正 CSV 后重新上传。")
            for error in exc.errors:
                st.write(f"- {error}")
            normalized_preview = pd.DataFrame()
            preview_warnings = []

        if valid_upload:
            ticker_currency_pairs = (
                normalized_preview[["ticker", "currency"]]
                .drop_duplicates()
                .sort_values(["ticker", "currency"])
                .reset_index(drop=True)
            )
            tickers = tuple(sorted(normalized_preview["ticker"].unique().tolist()))
            market_prices = load_latest_market_prices(tickers)

            price_rows = []
            for _, pair in ticker_currency_pairs.iterrows():
                ticker = str(pair["ticker"])
                currency = str(pair["currency"])
                market_record = market_prices.get(ticker) if currency == "USD" else None
                db_price = market_record["price"] if market_record else None
                db_date = market_record["date"] if market_record else "无本地数据"
                price_rows.append({
                    "ticker": ticker,
                    "currency": currency,
                    "数据库最近收盘价": db_price,
                    "数据库交易日期": db_date,
                    "估值使用价格": db_price,
                })

            price_input = pd.DataFrame(price_rows)
            st.subheader("估值价格")
            st.caption(
                "数据库价格只是本地保存的最近收盘价，并非实时价格。"
                "非 USD 币种不会自动套用数据库价格；缺少价格时请手动填写。"
            )
            edited_prices = st.data_editor(
                price_input,
                key=price_editor_key,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                disabled=["ticker", "currency", "数据库最近收盘价", "数据库交易日期"],
                column_config={
                    "ticker": st.column_config.TextColumn("证券代码", disabled=True),
                    "currency": st.column_config.TextColumn("币种", disabled=True),
                    "数据库最近收盘价": st.column_config.NumberColumn(
                        "数据库最近收盘价", format="%.4f", disabled=True
                    ),
                    "数据库交易日期": st.column_config.TextColumn(
                        "数据库交易日期", disabled=True
                    ),
                    "估值使用价格": st.column_config.NumberColumn(
                        "估值使用价格（可编辑）", min_value=0.000001, format="%.4f", step=0.01
                    ),
                },
            )

            current_prices: dict[tuple[str, str], float] = {}
            for _, price_row in edited_prices.iterrows():
                value = pd.to_numeric(
                    pd.Series([price_row.get("估值使用价格")]), errors="coerce"
                ).iloc[0]
                if pd.notna(value) and float(value) > 0:
                    current_prices[(
                        str(price_row["ticker"]).upper(),
                        str(price_row["currency"]).upper(),
                    )] = float(value)

            price_signature = tuple(
                sorted((ticker, currency, round(price, 8))
                       for (ticker, currency), price in current_prices.items())
            )

            calculate_clicked = st.button(
                "计算持仓与盈亏",
                type="primary",
                key=f"calculate_portfolio_{file_fingerprint}",
            )
            calculated_before = st.session_state.get(result_key)
            auto_recalculate = (
                calculated_before is not None
                and calculated_before.get("price_signature") != price_signature
            )

            if calculate_clicked or auto_recalculate:
                try:
                    status_message = (
                        "估值价格已更新，正在自动重算持仓……"
                        if auto_recalculate and not calculate_clicked
                        else "正在校验交易顺序并计算持仓……"
                    )
                    with st.spinner(status_message):
                        calculation = calculate_portfolio(
                            raw_transactions, current_prices=current_prices
                        )
                    st.session_state[result_key] = {
                        "transactions": calculation.transactions,
                        "positions": calculation.positions,
                        "warnings": calculation.warnings,
                        "price_signature": price_signature,
                    }
                    if auto_recalculate and not calculate_clicked:
                        st.success("估值价格已变化，持仓市值与盈亏已自动更新。")
                    else:
                        st.success("持仓计算完成。")
                except PortfolioValidationError as exc:
                    st.error("无法完成持仓计算，请检查交易记录。")
                    for error in exc.errors:
                        st.write(f"- {error}")
                except Exception as exc:
                    st.error(f"持仓计算发生意外错误：{type(exc).__name__}: {exc}")

            calculated = st.session_state.get(result_key)
            if calculated:
                st.divider()
                st.subheader("持仓分析结果")
                if calculated.get("price_signature") != price_signature:
                    st.warning("估值价格与上次计算时不同。请再次点击“计算持仓与盈亏”以更新结果。")
                for warning in calculated.get("warnings", []):
                    st.warning(warning)

                positions = calculated["positions"].copy()
                transaction_results = calculated["transactions"].copy()

                # --- Personal portfolio AI analysis (explicit consent required) ---
                if not positions.empty:
                    ai_report_state_key = f"portfolio_ai_report_{file_fingerprint}"
                    ai_consent_key = f"portfolio_ai_consent_{file_fingerprint}"
                    with st.expander("个人持仓 DeepSeek 分析", expanded=False):
                        st.markdown(
                            "生成一份结合实际持仓、交易汇总、价格情景及本地已保存市场/宏观/新闻证据的分析报告。"
                        )
                        st.warning(
                            "隐私说明：勾选同意后，程序会将聚合持仓、按标的/币种/交易类型汇总的交易统计、"
                            "价格情景，以及本地保存的市场/宏观/新闻上下文发送给 DeepSeek。"
                            "不会发送交易 ID、备注或逐笔交易日期。请求由外部模型服务处理。"
                        )
                        result_prices_current = calculated.get("price_signature") == price_signature
                        if not result_prices_current:
                            st.warning("当前估值价格与已计算结果不同。请先点击“计算持仓与盈亏”，再生成 AI 报告。")
                        consent = st.checkbox(
                            "我已了解以上数据范围，并同意将这些汇总信息发送给 DeepSeek。",
                            key=ai_consent_key,
                            value=False,
                        )
                        generate_ai_clicked = st.button(
                            "生成我的持仓分析",
                            type="primary",
                            key=f"generate_portfolio_ai_{file_fingerprint}",
                            disabled=(not consent) or (not result_prices_current),
                        )
                        if generate_ai_clicked:
                            try:
                                with st.spinner("正在整理持仓摘要并请求 DeepSeek……"):
                                    ai_report_path = generate_portfolio_analysis(
                                        positions,
                                        transaction_results,
                                    )
                                st.session_state[ai_report_state_key] = str(ai_report_path)
                                st.success(f"分析报告已保存：{ai_report_path.name}")
                            except Exception as exc:
                                st.error(f"持仓 AI 分析失败：{type(exc).__name__}: {exc}")

                        saved_ai_report = st.session_state.get(ai_report_state_key)
                        if saved_ai_report:
                            ai_path = Path(saved_ai_report)
                            if ai_path.exists():
                                st.download_button(
                                    "下载个人持仓分析报告",
                                    data=ai_path.read_bytes(),
                                    file_name=ai_path.name,
                                    mime="text/markdown",
                                    key=f"download_portfolio_ai_{file_fingerprint}",
                                )
                                # >>> QQQM PORTFOLIO AI PLAIN LANGUAGE V0.4.0 >>>
                                report_content = ai_path.read_text(encoding="utf-8")
                                summary_start = report_content.find("## 一眼看懂")
                                ai_section_start = report_content.find("## AI 解读：")
                                if summary_start >= 0:
                                    summary_end = (
                                        ai_section_start
                                        if ai_section_start > summary_start
                                        else len(report_content)
                                    )
                                    summary_content = report_content[summary_start:summary_end].strip()
                                    st.markdown(summary_content)
                                    with st.expander("查看完整个人持仓分析报告"):
                                        st.markdown(report_content)
                                else:
                                    with st.expander("查看最近生成的个人持仓分析"):
                                        st.markdown(report_content)


                if positions.empty:
                    st.info("没有可汇总的持仓记录。")
                else:
                    currencies = sorted(positions["currency"].dropna().unique().tolist())
                    for currency in currencies:
                        currency_positions = positions[positions["currency"] == currency]
                        open_positions = currency_positions[
                            currency_positions["open_position"].astype(bool)
                        ]
                        missing_market_prices = (
                            not open_positions.empty
                            and open_positions["current_price"].isna().any()
                        )
                        st.markdown(f"#### {currency} 账户概览")
                        if open_positions.empty:
                            market_value_total = 0.0
                            unrealized_total = 0.0
                        elif missing_market_prices:
                            market_value_total = None
                            unrealized_total = None
                        else:
                            market_value_total = float(open_positions["market_value"].sum())
                            unrealized_total = float(open_positions["unrealized_pnl"].sum())

                        remaining_cost = (
                            float(open_positions["remaining_cost_basis"].sum())
                            if not open_positions.empty else 0.0
                        )
                        realized_total = float(currency_positions["realized_pnl"].sum())
                        dividends_total = float(currency_positions["dividend_income"].sum())
                        cumulative_net_total = (
                            realized_total + unrealized_total + dividends_total
                            if unrealized_total is not None else None
                        )
                        total_fees = float(currency_positions["total_fees"].sum())

                        # 简明视图：先呈现整体结果，再解释盈亏来源。
                        metric_cols = st.columns(3)
                        metric_cols[0].metric(
                            "累计净盈亏",
                            f"{cumulative_net_total:+,.2f} {currency}"
                            if cumulative_net_total is not None else "缺少估值价格",
                        )
                        metric_cols[1].metric(
                            "当前持仓市值",
                            f"{market_value_total:,.2f} {currency}"
                            if market_value_total is not None else "缺少估值价格",
                        )
                        metric_cols[2].metric("剩余持仓成本", f"{remaining_cost:,.2f} {currency}")

                        detail_cols = st.columns(3)
                        detail_cols[0].metric(
                            "未实现盈亏",
                            f"{unrealized_total:+,.2f} {currency}"
                            if unrealized_total is not None else "暂不可计算",
                        )
                        detail_cols[1].metric("已实现盈亏", f"{realized_total:+,.2f} {currency}")
                        detail_cols[2].metric("累计手续费", f"{total_fees:,.2f} {currency}")
                        st.caption(
                            f"当前持仓标的数：{len(open_positions)}。"
                            f"累计分红净收入：{dividends_total:+,.2f} {currency}。"
                            "累计净盈亏 = 已实现盈亏 + 未实现盈亏 + 分红净收入。"
                            "买入手续费已计入持仓成本，卖出手续费已扣除卖出收入，"
                            "分红手续费已从分红中扣除，因此不再重复扣减累计手续费。"
                            "不同币种分开统计，没有合并换算。"
                        )

                    display_positions = positions.rename(columns={
                        "ticker": "标的",
                        "currency": "币种",
                        "quantity": "持有数量",
                        "average_cost": "加权平均成本",
                        "remaining_cost_basis": "剩余持仓成本",
                        "current_price": "估值使用价格",
                        "market_value": "持仓市值",
                        "unrealized_pnl": "未实现盈亏",
                        "unrealized_pnl_pct": "未实现收益率(%)",
                        "realized_pnl": "已实现盈亏",
                        "dividend_income": "分红净收入",
                        "realized_plus_dividends": "已实现盈亏与分红合计",
                        "cumulative_net_pnl": "累计净盈亏",
                        "equivalent_cost_per_share": "净收益调整后等效成本",
                        "total_fees": "累计手续费",
                        "open_position": "仍有持仓",
                    })
                    st.subheader("持仓明细")
                    st.dataframe(
                        display_positions,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "持有数量": st.column_config.NumberColumn(format="%.6f"),
                            "加权平均成本": st.column_config.NumberColumn(format="%.4f"),
                            "剩余持仓成本": st.column_config.NumberColumn(format="%.2f"),
                            "估值使用价格": st.column_config.NumberColumn(format="%.4f"),
                            "持仓市值": st.column_config.NumberColumn(format="%.2f"),
                            "未实现盈亏": st.column_config.NumberColumn(format="%.2f"),
                            "未实现收益率(%)": st.column_config.NumberColumn(format="%.2f"),
                            "已实现盈亏": st.column_config.NumberColumn(format="%.2f"),
                            "累计净盈亏": st.column_config.NumberColumn(format="%+.2f"),
                            "净收益调整后等效成本": st.column_config.NumberColumn(format="%.4f"),
                            "分红净收入": st.column_config.NumberColumn(format="%.2f"),
                            "累计手续费": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )

                    scenario_options = [-30, -20, -10, -5, 0, 5, 10, 20, 30]
                    selected_scenarios = st.multiselect(
                        "价格情景变化幅度（相对当前估值价格）",
                        options=scenario_options,
                        default=[-20, -10, -5, 0, 5, 10, 20],
                        format_func=lambda value: f"{value:+d}%" if value else "0%",
                        key=f"portfolio_scenarios_{file_fingerprint}",
                    )
                    scenario_df = build_scenario_table(positions, selected_scenarios)
                    st.subheader("价格情景模拟")
                    if scenario_df.empty:
                        st.info("没有可模拟的未平仓标的。请补充有效估值价格，或确认是否仍有持仓。")
                    else:
                        scenario_display = scenario_df.rename(columns={
                            "ticker": "标的",
                            "currency": "币种",
                            "price_change_pct": "价格变化(%)",
                            "scenario_price": "情景价格",
                            "scenario_market_value": "情景持仓市值",
                            "scenario_unrealized_pnl": "情景未实现盈亏",
                            "scenario_unrealized_pnl_pct": "情景未实现收益率(%)",
                            "realized_pnl": "历史已实现盈亏",
                            "dividend_income": "历史净分红",
                            "current_cumulative_net_pnl": "当前累计净盈亏",
                            "scenario_cumulative_net_pnl": "情景累计净盈亏",
                            "cumulative_pnl_change": "累计盈亏变化",
                        })
                        st.dataframe(
                            scenario_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "价格变化(%)": st.column_config.NumberColumn(format="%+.1f%%"),
                                "情景价格": st.column_config.NumberColumn(format="%.4f"),
                                "情景持仓市值": st.column_config.NumberColumn(format="%.2f"),
                                "情景未实现盈亏": st.column_config.NumberColumn(format="%+.2f"),
                                "情景未实现收益率(%)": st.column_config.NumberColumn(format="%+.2f"),
                                "历史已实现盈亏": st.column_config.NumberColumn(format="%+.2f"),
                                "历史净分红": st.column_config.NumberColumn(format="%+.2f"),
                                "当前累计净盈亏": st.column_config.NumberColumn(format="%+.2f"),
                                "情景累计净盈亏": st.column_config.NumberColumn(format="%+.2f"),
                                "累计盈亏变化": st.column_config.NumberColumn(format="%+.2f"),
                            },
                        )

                    st.caption(
                        "情景累计净盈亏 = 历史已实现盈亏 + 情景下未实现盈亏 + 历史净分红。"
                        "累计盈亏变化展示相对于当前估值的变化。"
                        "这是基于假设价格的情景分析，不是未来收益预测；不同币种分开计算。"
                    )

                    # V0.4.0-C decision support section
                    st.subheader("下一步怎么做？")
                    st.caption(
                        "根据你自己设定的持有期限、回撤提醒线和预算，生成透明的风险复核计划。"
                        "这是规则化情景分析，不是价格预测、个人化买卖指令或自动交易。"
                    )
                    with st.form("portfolio_decision_support_form"):
                        horizon = st.selectbox(
                            "你计划持有这类投资多久？",
                            options=["少于 1 年", "1–5 年", "5 年以上"],
                            index=1,
                            help="按你自己的计划填写；期限较长并不保证盈利。",
                        )
                        review_drawdown = st.slider(
                            "如果价格从当前估值下跌多少，提醒你重新评估？",
                            min_value=5,
                            max_value=30,
                            value=10,
                            step=5,
                            format="%d%%",
                        )
                        planned_budget = st.number_input(
                            "未来考虑投入的新增预算（参考金额；不代表建议买入）",
                            min_value=0.0,
                            value=0.0,
                            step=50.0,
                            help="填写 0 表示目前不计划追加。多币种持仓时，此预算不自动分配或换汇。",
                        )
                        decision_submitted = st.form_submit_button(
                            "生成我的条件式观察计划",
                            type="primary",
                        )

                    if decision_submitted:
                        try:
                            st.session_state["portfolio_decision_assessment"] = build_decision_assessment(
                                positions=positions,
                                holding_horizon=horizon,
                                review_drawdown_pct=review_drawdown,
                                planned_additional_budget=planned_budget,
                            )
                            st.session_state["portfolio_decision_preferences"] = {
                                "holding_horizon": horizon,
                                "review_drawdown_pct": review_drawdown,
                                "planned_additional_budget": planned_budget,
                            }
                        except Exception as exc:
                            st.error(f"无法生成观察计划：{type(exc).__name__}: {exc}")

                    decision_result = st.session_state.get("portfolio_decision_assessment")
                    if decision_result:
                        st.markdown("#### 你的观察计划")
                        st.write(decision_result.get("conclusion", ""))
                        if decision_result.get("horizon_note"):
                            st.info(decision_result["horizon_note"])
                        if decision_result.get("budget_note"):
                            st.caption(decision_result["budget_note"])
                        for decision_warning in decision_result.get("warnings", []):
                            st.warning(decision_warning)
                        for item in decision_result.get("items", []):
                            with st.container(border=True):
                                st.markdown(f"**{item['ticker']} · {item['currency']}**")
                                metric_cols = st.columns(3)
                                metric_cols[0].metric(
                                    "当前累计净盈亏",
                                    f"{item['current_cumulative_net_pnl']:+,.2f} {item['currency']}",
                                )
                                metric_cols[1].metric(
                                    f"价格下跌 {item['review_scenario_drawdown_pct']:.0f}% 的市值变化",
                                    f"−{item['review_scenario_value_decrease']:,.2f} {item['currency']}",
                                )
                                metric_cols[2].metric(
                                    "该情景下累计净盈亏",
                                    f"{item['review_scenario_cumulative_net_pnl']:+,.2f} {item['currency']}",
                                )
                                st.write(item["review_trigger"])
                                st.caption(item["interpretation"])
                        st.caption(
                            "设置目前保存在当前 Streamlit 会话中。刷新浏览器或重启应用后可能需要重新填写。"
                            "系统不会因浮亏自动建议加仓，也不会发送或执行交易。"
                        )

                    st.subheader("交易历史与已实现盈亏")
                    transaction_columns = [
                        "trade_date", "ticker", "side", "quantity", "price", "fee",
                        "currency", "amount", "realized_pnl", "notes",
                    ]
                    available_columns = [c for c in transaction_columns if c in transaction_results.columns]
                    tx_display = transaction_results[available_columns].rename(columns={
                        "trade_date": "交易日期",
                        "ticker": "标的",
                        "side": "类型",
                        "quantity": "数量",
                        "price": "成交价",
                        "fee": "手续费",
                        "currency": "币种",
                        "amount": "分红金额",
                        "realized_pnl": "本笔已实现盈亏",
                        "notes": "备注",
                    })
                    st.dataframe(
                        tx_display.sort_values("交易日期", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

                    tx_csv = transaction_results.to_csv(index=False).encode("utf-8-sig")
                    pos_csv = positions.to_csv(index=False).encode("utf-8-sig")
                    dl_col1, dl_col2 = st.columns(2)
                    dl_col1.download_button(
                        "下载已处理交易记录 CSV",
                        data=tx_csv,
                        file_name="portfolio_transactions_calculated.csv",
                        mime="text/csv",
                        key=f"download_portfolio_tx_{file_fingerprint}",
                    )
                    dl_col2.download_button(
                        "下载持仓分析结果 CSV",
                        data=pos_csv,
                        file_name="portfolio_positions.csv",
                        mime="text/csv",
                        key=f"download_portfolio_positions_{file_fingerprint}",
                    )



# -----------------------------------------------------------------------------
# 4. Rolling-grid backtest and visualization
# -----------------------------------------------------------------------------
with backtest_tab:
    st.subheader("滚动网格策略回测")
    st.caption(
        "使用本地 Tiingo 历史日线；策略按收盘信号、下一交易日开盘执行。"
        "参数用于历史研究，不代表建议采用的真实交易配置。"
    )

    if not HISTORY_PATH.exists():
        st.warning(
            f"没有找到历史行情 CSV：{HISTORY_PATH}\n\n"
            "请确认 data/QQQM_tiingo_history.csv 存在。"
        )
    else:
        with st.form("rolling_backtest_form"):
            left_col, right_col = st.columns(2)

            with left_col:
                lookback = st.number_input(
                    "移动平均线周期（交易日）",
                    min_value=20,
                    max_value=250,
                    value=60,
                    step=5,
                )
                step_pct_ui = st.select_slider(
                    "网格步长",
                    options=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                    value=2.5,
                    format_func=lambda value: f"{value:.1f}%",
                )

            with right_col:
                levels = st.number_input(
                    "网格层数",
                    min_value=1,
                    max_value=10,
                    value=6,
                    step=1,
                )
                cash_per_grid = st.number_input(
                    "每层买入资金（美元）",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                )

            initial_cash = st.number_input(
                "初始账户资金（美元）",
                min_value=1000,
                max_value=1_000_000,
                value=10000,
                step=1000,
            )

            run_backtest = st.form_submit_button(
                "运行回测",
                type="primary",
            )

        if run_backtest:
            command = [
                sys.executable,
                "-m",
                "app.rolling_backtest",
                "--ticker",
                "QQQM",
                "--data-file",
                str(HISTORY_PATH),
                "--lookback",
                str(int(lookback)),
                "--step-pct",
                str(float(step_pct_ui) / 100.0),
                "--levels",
                str(int(levels)),
                "--cash-per-grid",
                str(float(cash_per_grid)),
                "--initial-cash",
                str(float(initial_cash)),
            ]

            with st.spinner("正在计算回测，请稍候……"):
                backtest_result = run_command(command, timeout=600)

            st.session_state["backtest_result"] = backtest_result
            st.session_state["backtest_params"] = {
                "lookback": int(lookback),
                "step_pct": float(step_pct_ui),
                "levels": int(levels),
                "cash_per_grid": float(cash_per_grid),
                "initial_cash": float(initial_cash),
            }

        render_command_result(st.session_state.get("backtest_result"))

        params = st.session_state.get("backtest_params")
        if params:
            st.caption(
                "最近一次回测参数："
                f"均线 {params['lookback']} 日 · "
                f"步长 {params['step_pct']:.1f}% · "
                f"层数 {params['levels']} · "
                f"每层 ${params['cash_per_grid']:,.0f} · "
                f"初始资金 ${params['initial_cash']:,.0f}"
            )

        # Load the latest outputs after a successful execution, or show the
        # latest existing backtest when the page is first opened.
        equity_df, trades_df, summary = load_backtest_outputs()
        equity_file = latest_file("QQQM_rolling_grid_backtest_*_equity.csv")
        trades_file = latest_file("QQQM_rolling_grid_backtest_*_trades.csv")
        report_file = latest_file("QQQM_rolling_grid_backtest_*.md")

        st.divider()
        st.subheader("回测绩效分析")

        if equity_df.empty:
            st.info("目前没有可展示的回测资金曲线，请先成功运行一次回测。")
        else:
            required_equity_cols = {
                "date", "close", "cash", "shares", "position_value", "equity",
                "buy_hold_position_value", "buy_hold_equity",
            }
            if not required_equity_cols.issubset(equity_df.columns):
                missing = sorted(required_equity_cols - set(equity_df.columns))
                st.error(f"资金曲线缺少必要字段：{', '.join(missing)}")
            else:
                grid_summary = summary.get("grid_strategy", {})
                bh_summary = summary.get("buy_and_hold", {})
                parameters = summary.get("parameters", {})

                # Prefer the metrics emitted by the backtest engine. Calculated
                # fallbacks keep old reports readable if their JSON is missing.
                starting_cash = float(
                    parameters.get("initial_cash", equity_df["equity"].iloc[0])
                )
                final_grid_equity = float(
                    grid_summary.get("final_equity", equity_df["equity"].iloc[-1])
                )
                final_bh_equity = float(
                    bh_summary.get("final_equity", equity_df["buy_hold_equity"].iloc[-1])
                )
                grid_return = float(
                    grid_summary.get(
                        "total_return_pct",
                        (final_grid_equity / starting_cash - 1.0) * 100.0,
                    )
                )
                bh_return = float(
                    bh_summary.get(
                        "total_return_pct",
                        (final_bh_equity / starting_cash - 1.0) * 100.0,
                    )
                )
                grid_mdd = float(
                    grid_summary.get("max_drawdown_pct", max_drawdown_pct(equity_df["equity"]))
                )
                bh_mdd_fallback = max_drawdown_pct(
                    pd.concat(
                        [pd.Series([starting_cash]), equity_df["buy_hold_equity"]],
                        ignore_index=True,
                    )
                )
                bh_mdd = float(
                    bh_summary.get("max_drawdown_pct", bh_mdd_fallback)
                )
                grid_vol = float(
                    grid_summary.get(
                        "annualized_volatility_pct",
                        annualized_volatility_pct(equity_df["equity"]),
                    )
                )
                bh_vol = float(
                    bh_summary.get(
                        "annualized_volatility_pct",
                        annualized_volatility_pct(
                            pd.concat(
                                [pd.Series([starting_cash]), equity_df["buy_hold_equity"]],
                                ignore_index=True,
                            )
                        ),
                    )
                )
                avg_exposure = float(
                    grid_summary.get(
                        "average_position_exposure_pct",
                        (
                            equity_df["position_value"]
                            / equity_df["equity"].where(equity_df["equity"] != 0)
                            * 100.0
                        ).mean(),
                    )
                )
                bh_avg_exposure = float(
                    bh_summary.get(
                        "average_position_exposure_pct",
                        (
                            equity_df["buy_hold_position_value"]
                            / equity_df["buy_hold_equity"].where(equity_df["buy_hold_equity"] != 0)
                            * 100.0
                        ).mean(),
                    )
                )
                total_fees = float(grid_summary.get("fees_paid", trades_df.get("fee", pd.Series(dtype=float)).sum()))
                buy_count = int(
                    grid_summary.get(
                        "buy_orders_filled",
                        int((trades_df.get("side", pd.Series(dtype=str)) == "BUY").sum()),
                    )
                )
                sell_count = int(
                    grid_summary.get(
                        "sell_orders_filled",
                        int((trades_df.get("side", pd.Series(dtype=str)) == "SELL").sum()),
                    )
                )
                dividends = float(grid_summary.get("dividends_received", 0.0))

                metric_cols = st.columns(5)
                metric_cols[0].metric("网格总收益率", f"{grid_return:+.2f}%")
                metric_cols[1].metric("买入持有收益率", f"{bh_return:+.2f}%")
                metric_cols[2].metric("网格最大回撤", f"{grid_mdd:.2f}%")
                metric_cols[3].metric("平均股票仓位", f"{avg_exposure:.2f}%")
                metric_cols[4].metric("网格手续费", f"${total_fees:,.2f}")

                metric_table = pd.DataFrame(
                    [
                        {
                            "指标": "最终账户权益",
                            "滚动网格": f"${final_grid_equity:,.2f}",
                            "买入持有": f"${final_bh_equity:,.2f}",
                        },
                        {
                            "指标": "总收益率",
                            "滚动网格": f"{grid_return:+.2f}%",
                            "买入持有": f"{bh_return:+.2f}%",
                        },
                        {
                            "指标": "最大回撤",
                            "滚动网格": f"{grid_mdd:.2f}%",
                            "买入持有": f"{bh_mdd:.2f}%",
                        },
                        {
                            "指标": "年化波动率",
                            "滚动网格": f"{grid_vol:.2f}%",
                            "买入持有": f"{bh_vol:.2f}%",
                        },
                        {
                            "指标": "平均股票仓位",
                            "滚动网格": f"{avg_exposure:.2f}%",
                            "买入持有": f"{bh_avg_exposure:.2f}%",
                        },
                        {
                            "指标": "已成交买入 / 卖出",
                            "滚动网格": f"{buy_count} / {sell_count}",
                            "买入持有": "不适用",
                        },
                        {
                            "指标": "交易手续费",
                            "滚动网格": f"${total_fees:,.2f}",
                            "买入持有": (
                                f"${float(bh_summary.get('entry_fee_paid', 0.0)):,.2f} 建仓费"
                                if bh_summary
                                else "见回测报告"
                            ),
                        },
                        {
                            "指标": "收到的现金分红",
                            "滚动网格": f"${dividends:,.2f}",
                            "买入持有": (
                                f"${float(bh_summary.get('dividends_received_in_cash', 0.0)):,.2f}"
                                if bh_summary
                                else "见回测报告"
                            ),
                        },
                    ]
                )
                st.dataframe(metric_table, use_container_width=True, hide_index=True)

                # Equity curves: grid and buy/hold share the same initial principal.
                st.subheader("账户权益曲线")
                curve_df = equity_df[["date", "equity", "buy_hold_equity"]].copy()
                curve_df = curve_df.rename(
                    columns={
                        "equity": "滚动网格",
                        "buy_hold_equity": "买入持有",
                    }
                ).set_index("date")
                st.line_chart(curve_df, y_label="账户权益（美元）")
                st.caption(
                    "买入持有基准包含初始建仓手续费与滑点估算；"
                    "两条策略的股票仓位不同，不能只按收益率判断优劣。"
                )

                # Drawdown series include the initial account capital in benchmark peak.
                st.subheader("账户回撤曲线")
                grid_equity = pd.to_numeric(equity_df["equity"], errors="coerce")
                bh_equity = pd.to_numeric(equity_df["buy_hold_equity"], errors="coerce")
                grid_peak = pd.concat(
                    [pd.Series([starting_cash]), grid_equity], ignore_index=True
                ).cummax().iloc[1:].reset_index(drop=True)
                bh_peak = pd.concat(
                    [pd.Series([starting_cash]), bh_equity], ignore_index=True
                ).cummax().iloc[1:].reset_index(drop=True)
                drawdown_df = pd.DataFrame(
                    {
                        "date": equity_df["date"],
                        "滚动网格": (grid_equity.reset_index(drop=True) / grid_peak - 1.0) * 100.0,
                        "买入持有": (bh_equity.reset_index(drop=True) / bh_peak - 1.0) * 100.0,
                    }
                ).set_index("date")
                st.line_chart(drawdown_df, y_label="回撤（%）")
                st.caption("回撤为负数；离 0 越远，表示较历史权益高点跌得越深。")

                # Price line with actual fills marked as green buys and red sells.
                st.subheader("QQQM 收盘价与策略买卖点")
                price_data = equity_df[["date", "close"]].dropna().copy()
                price_line = alt.Chart(price_data).mark_line().encode(
                    x=alt.X("date:T", title="交易日期"),
                    y=alt.Y("close:Q", title="QQQM 收盘价（美元）", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("date:T", title="日期"),
                        alt.Tooltip("close:Q", title="收盘价", format=".2f"),
                    ],
                )
                price_chart = price_line

                trade_columns = {"date", "side", "fill_price"}
                if not trades_df.empty and trade_columns.issubset(trades_df.columns):
                    valid_trades = trades_df.dropna(subset=["date", "fill_price"]).copy()
                    points_tooltip = [
                        alt.Tooltip("date:T", title="交易日期"),
                        alt.Tooltip("side:N", title="方向"),
                        alt.Tooltip("fill_price:Q", title="成交价格", format=".2f"),
                    ]
                    if "shares" in valid_trades.columns:
                        points_tooltip.append(
                            alt.Tooltip("shares:Q", title="股数", format=".0f")
                        )
                    if "fee" in valid_trades.columns:
                        points_tooltip.append(
                            alt.Tooltip("fee:Q", title="手续费", format=".2f")
                        )

                    buy_rows = valid_trades[valid_trades["side"] == "BUY"]
                    sell_rows = valid_trades[valid_trades["side"] == "SELL"]

                    if not buy_rows.empty:
                        buy_points = alt.Chart(buy_rows).mark_point(
                            shape="triangle-up", size=85, color="#16a34a"
                        ).encode(
                            x=alt.X("date:T"),
                            y=alt.Y("fill_price:Q"),
                            tooltip=points_tooltip,
                        )
                        price_chart = price_chart + buy_points

                    if not sell_rows.empty:
                        sell_points = alt.Chart(sell_rows).mark_point(
                            shape="triangle-down", size=85, color="#dc2626"
                        ).encode(
                            x=alt.X("date:T"),
                            y=alt.Y("fill_price:Q"),
                            tooltip=points_tooltip,
                        )
                        price_chart = price_chart + sell_points

                st.altair_chart(price_chart.interactive(), use_container_width=True)
                st.caption("绿色上三角是买入成交，红色下三角是卖出成交。悬停可以查看价格和交易信息。")

                # Trade details, sortable and filterable.
                st.subheader("交易明细")
                if trades_df.empty:
                    st.info("这次回测没有生成交易记录。")
                else:
                    trade_view = trades_df.copy()
                    if "side" in trade_view.columns:
                        directions = sorted(trade_view["side"].dropna().unique().tolist())
                        selected_directions = st.multiselect(
                            "筛选交易方向",
                            options=directions,
                            default=directions,
                            key="trade_direction_filter",
                        )
                        trade_view = trade_view[
                            trade_view["side"].isin(selected_directions)
                        ]

                    if "date" in trade_view.columns:
                        trade_view = trade_view.sort_values("date", ascending=False)

                    visible_columns = [
                        column
                        for column in [
                            "date", "side", "tier", "trigger_price", "fill_price",
                            "target_sell_price", "shares", "notional", "fee",
                            "realized_pnl", "cash_after", "held_shares_after", "reason",
                        ]
                        if column in trade_view.columns
                    ]
                    display_df = trade_view[visible_columns].copy()
                    if "date" in display_df.columns:
                        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                # Downloads refer to the exact latest files shown above.
                dl1, dl2 = st.columns(2)
                if trades_file:
                    dl1.download_button(
                        "下载交易记录 CSV",
                        data=trades_file.read_bytes(),
                        file_name=trades_file.name,
                        mime="text/csv",
                        key="download_trades_csv",
                    )
                if equity_file:
                    dl2.download_button(
                        "下载资金曲线 CSV",
                        data=equity_file.read_bytes(),
                        file_name=equity_file.name,
                        mime="text/csv",
                        key="download_equity_csv",
                    )

                if report_file:
                    with st.expander("查看回测 Markdown 报告"):
                        st.caption(report_file.name)
                        st.markdown(read_text_file(report_file))

                if equity_file:
                    st.caption(f"当前展示的资金曲线文件：{equity_file.name}")
                if trades_file:
                    st.caption(f"当前展示的交易记录文件：{trades_file.name}")


# -----------------------------------------------------------------------------
# 5. Macro data, news and sync status
# -----------------------------------------------------------------------------
with macro_tab:
    st.subheader("宏观经济与官方新闻")
    st.caption(
        "读取本地 SQLite 已保存的数据。宏观指标来自 FRED / BLS；新闻来自项目已配置的官方来源。"
    )

    sync_col1, sync_col2, sync_col3 = st.columns([1, 1, 1])
    with sync_col1:
        if st.button("同步 FRED 宏观数据", key="sync_fred_macro", type="primary"):
            with st.spinner("正在同步 FRED 数据……"):
                result = run_command(
                    [sys.executable, "main.py", "macro-sync"], timeout=600
                )
            st.session_state["macro_sync_fred_result"] = result
            load_fred_macro_data.clear()
    with sync_col2:
        if st.button("同步新闻与 BLS 数据", key="sync_news_bls"):
            with st.spinner("正在同步新闻与 BLS 指标……"):
                result = run_command(
                    [sys.executable, "-m", "app.news_data", "sync"], timeout=600
                )
            st.session_state["macro_sync_news_result"] = result
            load_news_data.clear()
            load_bls_macro_data.clear()
    with sync_col3:
        if st.button("刷新本地数据", key="refresh_macro_news"):
            load_fred_macro_data.clear()
            load_bls_macro_data.clear()
            load_news_data.clear()
            st.rerun()

    render_command_result(st.session_state.get("macro_sync_fred_result"))
    render_command_result(st.session_state.get("macro_sync_news_result"))

    fred_tab, bls_tab, news_tab, sync_tab = st.tabs(
        ["FRED 指标", "BLS 通胀与就业", "官方新闻", "同步状态"]
    )

    with fred_tab:
        st.markdown("#### 美国利率与宏观指标")
        fred_df = load_fred_macro_data()
        if fred_df.empty:
            st.info("本地没有读取到 FRED 数据。可点击上方“同步 FRED 宏观数据”尝试更新。")
        else:
            fred_df["observation_date"] = pd.to_datetime(
                fred_df["observation_date"], errors="coerce"
            )
            fred_df["value"] = pd.to_numeric(fred_df["value"], errors="coerce")
            fred_df = fred_df.dropna(subset=["observation_date", "value"])
            labels = {
                "EFFR": "联邦基金有效利率（EFFR）",
                "DGS10": "美国 10 年期国债收益率（DGS10）",
                "DFII10": "美国 10 年期实际利率（DFII10）",
                "CPIAUCSL": "美国 CPI 指数（CPIAUCSL）",
            }

            preferred_order = ["EFFR", "DGS10", "DFII10", "CPIAUCSL"]
            latest_by_series = (
                fred_df.sort_values("observation_date")
                .groupby("series_id", as_index=False)
                .tail(1)
            )
            existing_ids = set(fred_df["series_id"].astype(str))
            metric_ids = [sid for sid in preferred_order if sid in existing_ids]
            if metric_ids:
                metric_cols = st.columns(len(metric_ids))
                indexed_latest = latest_by_series.set_index("series_id")
                for idx, sid in enumerate(metric_ids):
                    row = indexed_latest.loc[sid]
                    label = labels.get(sid, sid)
                    value = float(row["value"])
                    value_text = f"{value:.3f}" if sid == "CPIAUCSL" else f"{value:.2f}%"
                    metric_cols[idx].metric(label, value_text)
                    metric_cols[idx].caption(
                        f"观测日期：{row['observation_date'].date()}"
                    )

            available_ids = sorted(existing_ids, key=lambda x: (preferred_order.index(x) if x in preferred_order else 999, x))
            selected_series = st.selectbox(
                "选择要查看的指标",
                options=available_ids,
                format_func=lambda value: f"{labels.get(value, value)} · {value}",
                key="fred_series_select",
            )
            selected_df = fred_df[fred_df["series_id"] == selected_series].copy()
            selected_df = selected_df.sort_values("observation_date")
            chart_title = labels.get(selected_series, selected_series)
            if selected_series == "CPIAUCSL":
                st.caption("CPIAUCSL 为消费者价格指数，不是同比通胀率；需要用指数序列自行计算同比或环比变化。")
            chart_df = selected_df.set_index("observation_date")["value"]
            st.line_chart(chart_df, y_label="指数值" if selected_series == "CPIAUCSL" else "百分比 / 数值")
            st.caption(f"当前图表：{chart_title}。缺失日期可能反映该指标本身的发布频率。")
            recent_fred = selected_df.sort_values("observation_date", ascending=False).head(30).copy()
            recent_fred["observation_date"] = recent_fred["observation_date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                recent_fred[["series_id", "observation_date", "value", "retrieved_at"]],
                use_container_width=True,
                hide_index=True,
            )

    with bls_tab:
        st.markdown("#### BLS 通胀与就业数据")
        bls_df = load_bls_macro_data()
        if bls_df.empty:
            st.info("本地没有读取到 BLS 宏观记录。可点击上方“同步新闻与 BLS 数据”尝试更新。")
        else:
            bls_df["value_numeric"] = pd.to_numeric(bls_df["value"], errors="coerce")
            bls_df["year"] = pd.to_numeric(bls_df["year"], errors="coerce")
            bls_df["period_number"] = pd.to_numeric(
                bls_df["period"].astype(str).str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )
            # BLS monthly series use M01-M12; non-monthly periods remain visible in the table.
            month_text = bls_df["period_number"].where(
                bls_df["period"].astype(str).str.startswith("M")
                & bls_df["period_number"].between(1, 12)
            )
            bls_df["period_date"] = pd.to_datetime(
                bls_df["year"].astype("Int64").astype(str)
                + "-"
                + month_text.astype("Int64").astype(str).str.zfill(2)
                + "-01",
                errors="coerce",
            )

            categories = ["全部"] + sorted(
                bls_df["category"].dropna().astype(str).unique().tolist()
            )
            selected_category = st.selectbox(
                "按类别筛选",
                categories,
                key="bls_category_filter",
            )
            filtered_bls = bls_df.copy()
            if selected_category != "全部":
                filtered_bls = filtered_bls[
                    filtered_bls["category"] == selected_category
                ]

            if filtered_bls.empty:
                st.info("该类别暂时没有数据。")
            else:
                latest_bls = (
                    filtered_bls.sort_values(
                        ["year", "period_number"], ascending=[False, False]
                    )
                    .drop_duplicates(subset=["series_id"], keep="first")
                )
                metric_items = latest_bls.to_dict("records")
                metric_cols = st.columns(min(3, len(metric_items)))
                for idx, row in enumerate(metric_items[:3]):
                    numeric_value = row.get("value_numeric")
                    value_text = (
                        f"{numeric_value:,.3f} {row.get('unit') or ''}".strip()
                        if pd.notna(numeric_value)
                        else str(row.get("value", "暂无数据"))
                    )
                    metric_cols[idx].metric(
                        str(row.get("series_name") or row.get("series_id")),
                        value_text,
                    )
                    metric_cols[idx].caption(
                        f"{row.get('period_name') or row.get('period')} {row.get('year')}"
                    )

                series_options = (
                    filtered_bls[["series_id", "series_name"]]
                    .drop_duplicates()
                    .sort_values("series_name")
                )
                selected_bls_series = st.selectbox(
                    "选择趋势图指标",
                    options=series_options["series_id"].tolist(),
                    format_func=lambda sid: str(
                        series_options.loc[series_options["series_id"] == sid, "series_name"].iloc[0]
                    ),
                    key="bls_series_select",
                )
                series_df = filtered_bls[
                    filtered_bls["series_id"] == selected_bls_series
                ].copy()
                series_df = series_df.dropna(subset=["period_date", "value_numeric"])
                if not series_df.empty:
                    series_df = series_df.sort_values("period_date")
                    st.line_chart(
                        series_df.set_index("period_date")["value_numeric"],
                        y_label=str(series_df["unit"].dropna().iloc[0]) if series_df["unit"].notna().any() else "数值",
                    )
                else:
                    st.info("该指标没有可绘制的月度时间序列，可在下方表格查看原始记录。")

                display_bls = filtered_bls.sort_values(
                    ["year", "period_number"], ascending=[False, False]
                ).head(40).copy()
                display_columns = [
                    "series_name", "category", "year", "period", "period_name",
                    "value", "unit", "source_name", "source_url", "fetched_at",
                ]
                display_bls = display_bls[[c for c in display_columns if c in display_bls.columns]]
                st.dataframe(
                    display_bls,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "source_url": st.column_config.LinkColumn("官方数据来源", display_text="打开来源")
                    },
                )

    with news_tab:
        st.markdown("#### 官方经济新闻")
        news_df = load_news_data()
        if news_df.empty:
            st.info("本地新闻表没有记录。可点击上方“同步新闻与 BLS 数据”尝试更新。")
        else:
            news_df["published_at_dt"] = pd.to_datetime(
                news_df["published_at"], errors="coerce", utc=True
            )
            news_df = news_df.sort_values("published_at_dt", ascending=False)
            sources = ["全部"] + sorted(news_df["source_name"].dropna().astype(str).unique().tolist())
            categories = ["全部"] + sorted(news_df["category"].dropna().astype(str).unique().tolist())
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                selected_source = st.selectbox("新闻来源", sources, key="news_source_filter")
            with filter_col2:
                selected_news_category = st.selectbox("新闻类别", categories, key="news_category_filter")
            search = st.text_input("搜索标题或摘要", key="news_search_text", placeholder="例如 CPI、interest rate、employment")
            filtered_news = news_df.copy()
            if selected_source != "全部":
                filtered_news = filtered_news[filtered_news["source_name"] == selected_source]
            if selected_news_category != "全部":
                filtered_news = filtered_news[filtered_news["category"] == selected_news_category]
            if search.strip():
                pattern = search.strip()
                mask = (
                    filtered_news["title"].fillna("").str.contains(pattern, case=False, regex=False)
                    | filtered_news["summary"].fillna("").str.contains(pattern, case=False, regex=False)
                )
                filtered_news = filtered_news[mask]

            st.caption(f"找到 {len(filtered_news)} 条新闻，共 {len(news_df)} 条本地记录。")
            for _, item in filtered_news.head(30).iterrows():
                published = item.get("published_at_dt")
                published_text = published.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(published) else "日期未知"
                source_label = str(item.get("source_name") or item.get("source_id") or "未知来源")
                category_label = str(item.get("category") or "未分类")
                title = str(item.get("title") or "（无标题）")
                summary_text = str(item.get("summary") or "暂无摘要")
                url = str(item.get("url") or "")
                with st.expander(f"{title}  ·  {source_label} / {category_label}  ·  {published_text}"):
                    st.write(summary_text)
                    if url.startswith(("https://", "http://")):
                        st.markdown(f"[阅读官方原文]({url})")
                    st.caption(f"抓取时间：{item.get('fetched_at') or '未知'}")

    with sync_tab:
        st.markdown("#### 同步状态与数据新鲜度")
        st.write("这些信息反映数据目前是否存储在本地，不代表数据源一定已经发布了最新一期数据。")

        if not DB_PATH.exists():
            st.warning("数据库文件不存在。")
        else:
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    bls_state = pd.read_sql_query(
                        "SELECT series_id, series_name, category, source_url, "
                        "last_attempt_at, last_success_at, last_observation_count, last_error "
                        "FROM bls_macro_sync_state ORDER BY category, series_id",
                        conn,
                    )
                    if not bls_state.empty:
                        st.markdown("**BLS 同步记录**")
                        st.dataframe(
                            bls_state,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "source_url": st.column_config.LinkColumn("官方数据来源", display_text="打开来源")
                            },
                        )
                    else:
                        st.info("暂时没有 BLS 同步状态记录。")

                    fred_status = pd.read_sql_query(
                        "SELECT series_id, COUNT(*) AS observation_count, "
                        "MAX(observation_date) AS latest_observation, "
                        "MAX(retrieved_at) AS last_retrieved_at "
                        "FROM macro_observations GROUP BY series_id ORDER BY series_id",
                        conn,
                    )
                    if not fred_status.empty:
                        st.markdown("**FRED 数据覆盖情况**")
                        st.dataframe(
                            fred_status,
                            use_container_width=True,
                            hide_index=True,
                        )

                    news_counts = pd.read_sql_query(
                        "SELECT source_name, category, COUNT(*) AS item_count, "
                        "MAX(published_at) AS latest_published_at, "
                        "MAX(fetched_at) AS last_fetched_at "
                        "FROM news_items GROUP BY source_name, category "
                        "ORDER BY source_name, category",
                        conn,
                    )
                    if not news_counts.empty:
                        st.markdown("**本地新闻数量与更新时间**")
                        st.dataframe(news_counts, use_container_width=True, hide_index=True)

                    state_tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name LIKE '%sync_state%' ORDER BY name",
                        conn,
                    )["name"].tolist()
                    for table_name in state_tables:
                        if table_name in {"bls_macro_sync_state"}:
                            continue
                        escaped_name = table_name.replace('"', '""')
                        extra_state = pd.read_sql_query(
                            f'SELECT * FROM "{escaped_name}" LIMIT 100', conn
                        )
                        if not extra_state.empty:
                            st.markdown(f"**{table_name}**")
                            st.dataframe(extra_state, use_container_width=True, hide_index=True)
            except (sqlite3.Error, pd.errors.DatabaseError) as exc:
                st.error(f"读取同步状态失败：{type(exc).__name__}: {exc}")


# -----------------------------------------------------------------------------
# 5. Data status
# -----------------------------------------------------------------------------
with status_tab:
    st.subheader("本地数据库状态")
    st.write(f"数据库路径：`{DB_PATH}`")

    if not DB_PATH.exists():
        st.warning("数据库文件不存在。")
    else:
        try:
            with sqlite3.connect(DB_PATH) as conn:
                table_names = pd.read_sql_query(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    ORDER BY name
                    """,
                    conn,
                )["name"].tolist()

                table_info = []
                for name in table_names:
                    safe_name = name.replace('"', '""')
                    count = conn.execute(
                        f'SELECT COUNT(*) FROM "{safe_name}"'
                    ).fetchone()[0]
                    table_info.append({"数据表": name, "记录数": int(count)})

            if table_info:
                st.dataframe(
                    pd.DataFrame(table_info),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("数据库存在，但目前没有数据表。")
        except Exception as exc:
            st.error(f"读取数据库结构失败：{type(exc).__name__}: {exc}")

    st.divider()
    st.subheader("本地文件状态")
    status_rows = [
        {
            "项目文件": str(HISTORY_PATH.relative_to(ROOT)),
            "状态": "已找到" if HISTORY_PATH.exists() else "不存在",
        },
        {
            "项目文件": str(DB_PATH.relative_to(ROOT)),
            "状态": "已找到" if DB_PATH.exists() else "不存在",
        },
        {
            "项目文件": "reports/",
            "状态": "已找到" if REPORT_DIR.exists() else "不存在",
        },
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    if REPORT_DIR.exists():
        st.caption(f"报告目录内文件数量：{sum(1 for item in REPORT_DIR.iterdir() if item.is_file())}")
