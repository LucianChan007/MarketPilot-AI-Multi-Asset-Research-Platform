from __future__ import annotations

import ast
import shutil
from datetime import datetime
from pathlib import Path

DASHBOARD = Path("app/dashboard.py")
MARKER = "# >>> QQQM INVESTMENT OVERVIEW V0.4.0 >>>"
END_MARKER = "# <<< QQQM INVESTMENT OVERVIEW V0.4.0 <<<"

BLOCK = r'''# >>> QQQM INVESTMENT OVERVIEW V0.4.0 >>>
st.divider()
st.header("投资总览")
st.caption(
    "先看整体赚亏和当前持仓；下方各标签页保留详细行情、AI 研究、持仓明细、回测与宏观数据。"
)

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
'''


def main() -> int:
    if not DASHBOARD.exists():
        raise SystemExit(f"找不到 {DASHBOARD}。请在项目根目录运行脚本。")

    source = DASHBOARD.read_text(encoding="utf-8")
    if MARKER in source or END_MARKER in source:
        raise SystemExit("投资总览代码似乎已经安装过。未作修改。")

    title_anchor = 'st.title("📈 QQQM Investment Assistant")'
    if source.count(title_anchor) != 1:
        raise SystemExit(
            "找不到唯一的 dashboard 页面标题锚点，未修改文件。\n"
            "请检查 app/dashboard.py 中的页面标题后再调整脚本。"
        )

    # Syntax-check the injected block as a standalone top-level fragment.
    try:
        ast.parse(BLOCK, filename="investment_overview_block.py")
    except SyntaxError as exc:
        raise SystemExit(f"插入代码语法检查失败，未修改文件：{exc}") from exc

    updated = source.replace(title_anchor, title_anchor + "\n\n" + BLOCK, 1)
    try:
        ast.parse(updated, filename=str(DASHBOARD))
    except SyntaxError as exc:
        raise SystemExit(f"修改后的 dashboard.py 语法检查失败，未修改文件：{exc}") from exc

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DASHBOARD.with_name(f"dashboard.py.{stamp}.bak")
    shutil.copy2(DASHBOARD, backup)
    DASHBOARD.write_text(updated, encoding="utf-8", newline="\n")

    print(f"投资总览已添加：{DASHBOARD}")
    print(f"原文件备份：{backup}")
    print("注意：总览显示本地最新可用日线估值，不是实时行情。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
