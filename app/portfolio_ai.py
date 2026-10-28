from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.portfolio import build_scenario_table

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
PORTFOLIO_REPORT_DIR = REPORTS_DIR / "portfolio"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 120


def _number(value: Any) -> float | None:
    """Return a finite float or None, avoiding NaN/Infinity in prompts."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _round_number(value: Any, digits: int = 4) -> float | None:
    number = _number(value)
    return round(number, digits) if number is not None else None


def build_portfolio_payload(
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
) -> dict[str, Any]:
    """Build a privacy-minimized payload; never include IDs, notes, or raw trades."""
    position_fields = {
        "ticker": "ticker",
        "currency": "currency",
        "quantity": "quantity",
        "average_cost": "average_cost",
        "remaining_cost_basis": "remaining_cost_basis",
        "current_price": "current_price",
        "market_value": "market_value",
        "unrealized_pnl": "unrealized_pnl",
        "unrealized_pnl_pct": "unrealized_pnl_pct",
        "realized_pnl": "realized_pnl",
        "dividend_income": "dividend_income",
        "total_fees": "total_fees",
        "cumulative_net_pnl": "cumulative_net_pnl",
        "equivalent_cost_per_share": "equivalent_cost_per_share",
        "open_position": "open_position",
    }

    position_items: list[dict[str, Any]] = []
    for _, row in positions.iterrows():
        item: dict[str, Any] = {}
        for output_key, source_key in position_fields.items():
            if source_key not in row.index:
                continue
            if output_key in {"ticker", "currency"}:
                item[output_key] = str(row[source_key])
            elif output_key == "open_position":
                item[output_key] = bool(row[source_key])
            else:
                item[output_key] = _round_number(row[source_key])
        position_items.append(item)

    # Aggregate transaction history by ticker/currency/action. Do not send trade IDs,
    # free-text notes, exact individual dates, or per-transaction rows to the model.
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not transactions.empty and {"ticker", "currency", "side"}.issubset(transactions.columns):
        for _, row in transactions.iterrows():
            key = (
                str(row.get("ticker", "")),
                str(row.get("currency", "")),
                str(row.get("side", "")).upper(),
            )
            entry = aggregate.setdefault(key, {
                "ticker": key[0],
                "currency": key[1],
                "side": key[2],
                "transaction_count": 0,
                "quantity_total": 0.0,
                "fees_total": 0.0,
                "realized_pnl_total": 0.0,
                "cash_amount_total": 0.0,
            })
            entry["transaction_count"] += 1
            entry["quantity_total"] += _number(row.get("quantity")) or 0.0
            entry["fees_total"] += _number(row.get("fee")) or 0.0
            entry["realized_pnl_total"] += _number(row.get("realized_pnl")) or 0.0
            entry["cash_amount_total"] += _number(row.get("amount")) or 0.0

    transaction_summary = []
    for entry in aggregate.values():
        transaction_summary.append({
            **entry,
            "quantity_total": round(entry["quantity_total"], 6),
            "fees_total": round(entry["fees_total"], 4),
            "realized_pnl_total": round(entry["realized_pnl_total"], 4),
            "cash_amount_total": round(entry["cash_amount_total"], 4),
        })

    scenarios = build_scenario_table(positions, [-10, -5, 0, 5, 10])
    scenario_fields = [
        "ticker", "currency", "price_change_pct", "scenario_price",
        "scenario_market_value", "scenario_unrealized_pnl",
        "current_cumulative_net_pnl", "scenario_cumulative_net_pnl",
        "cumulative_pnl_change",
    ]
    scenario_items: list[dict[str, Any]] = []
    for _, row in scenarios.iterrows():
        item = {}
        for field in scenario_fields:
            if field not in row.index:
                continue
            if field in {"ticker", "currency"}:
                item[field] = str(row[field])
            else:
                item[field] = _round_number(row[field])
        scenario_items.append(item)

    currency_totals: list[dict[str, Any]] = []
    if not positions.empty and "currency" in positions.columns:
        for currency in sorted(positions["currency"].dropna().astype(str).unique()):
            group = positions[positions["currency"].astype(str) == currency]
            def sum_column(column: str) -> float:
                if column not in group.columns:
                    return 0.0
                vals = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
                return round(float(vals.sum()), 4)

            open_rows = group[group.get("open_position", pd.Series(False, index=group.index)).astype(bool)]
            prices_missing = bool(
                not open_rows.empty
                and ("current_price" not in open_rows.columns or open_rows["current_price"].isna().any())
            )
            unrealized_sum = None if prices_missing else sum_column("unrealized_pnl")
            currency_totals.append({
                "currency": currency,
                "open_position_count": int(len(open_rows)),
                "remaining_cost_basis": sum_column("remaining_cost_basis"),
                "market_value": None if prices_missing else sum_column("market_value"),
                "realized_pnl": sum_column("realized_pnl"),
                "unrealized_pnl": unrealized_sum,
                "net_dividend_income": sum_column("dividend_income"),
                "total_recorded_fees": sum_column("total_fees"),
                "cumulative_net_pnl": (
                    None if unrealized_sum is None else round(
                        sum_column("realized_pnl") + unrealized_sum + sum_column("dividend_income"), 4
                    )
                ),
            })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accounting_method": "weighted-average cost; fees included in buy basis/sell proceeds/dividend net income",
        "currency_policy": "currencies are reported separately; no FX conversion is assumed",
        "positions": position_items,
        "currency_totals": currency_totals,
        "transaction_aggregates": transaction_summary,
        "price_scenarios": scenario_items,
        "privacy_note": "Contains aggregate portfolio/transaction figures only; no trade IDs, notes, or individual transaction dates are included.",
    }



def _format_money(value: Any, currency: str, signed: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "暂不可计算"
    if signed:
        sign = "+" if number >= 0 else "−"
        return f"{sign}{abs(number):,.2f} {currency}"
    return f"{number:,.2f} {currency}"


def build_plain_language_summary(
    payload: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> str:
    """Create a deterministic, beginner-friendly summary before AI commentary.

    All displayed accounting amounts come from Python's already-calculated payload;
    the language model is not allowed to calculate or replace them.
    """
    evidence = evidence or {}
    lines: list[str] = []
    totals = payload.get("currency_totals", [])
    positions = payload.get("positions", [])
    scenarios = payload.get("price_scenarios", [])

    market = evidence.get("qqqm_market_summary", {})
    market_date = market.get("latest_date") if isinstance(market, dict) else None
    if market_date:
        lines.append(f"**行情数据日期：** QQQM {market_date}（本地保存的最近可用日线，不是实时行情）")
    else:
        lines.append("**行情数据：** 使用本地最近可用估值；报告未能从市场摘要中确认行情日期。")
    lines.append("")

    if not totals:
        lines.append("目前没有可用的币种汇总，因此无法可靠展示累计盈亏。请先计算持仓并确认交易记录有效。")
        return "\n".join(lines)

    lines.append("### 先看整体是赚还是亏")
    for total in totals:
        currency = str(total.get("currency", "未知币种"))
        cumulative = total.get("cumulative_net_pnl")
        realized = total.get("realized_pnl")
        unrealized = total.get("unrealized_pnl")
        dividends = total.get("net_dividend_income")
        fees = total.get("total_recorded_fees")
        value = total.get("market_value")
        cost = total.get("remaining_cost_basis")

        lines.append(f"#### {currency}")
        if cumulative is None:
            lines.append("- **累计净盈亏：暂不可可靠计算。** 至少一个未平仓标的缺少有效估值价格。")
        elif cumulative > 0:
            lines.append(f"- **整体结果：累计净盈利 {_format_money(cumulative, currency, signed=True)}。**")
        elif cumulative < 0:
            lines.append(f"- **整体结果：累计净亏损 {_format_money(cumulative, currency, signed=True)}。**")
        else:
            lines.append("- **整体结果：累计净盈亏约为 0。**")

        lines.append(
            f"- 已经卖出产生的盈亏：{_format_money(realized, currency, signed=True)}；"
            f"当前持仓浮动盈亏：{_format_money(unrealized, currency, signed=True)}；"
            f"净分红：{_format_money(dividends, currency, signed=True)}。"
        )
        lines.append(
            f"- 当前持仓市值：{_format_money(value, currency)}；"
            f"剩余持仓成本：{_format_money(cost, currency)}；"
            f"累计记录手续费：{_format_money(fees, currency)}。"
        )
        if cumulative is not None and unrealized is not None:
            if cumulative > 0 and unrealized < 0:
                lines.append("- **简单理解：** 整体记录仍然盈利，但手上还持有的股票目前是浮亏；历史已实现盈利和净分红抵消了这部分浮亏。")
            elif cumulative < 0 and unrealized > 0:
                lines.append("- **简单理解：** 当前持仓本身有浮盈，但加上历史已实现结果和分红后，整体记录仍为亏损。")
            elif cumulative < 0:
                lines.append("- **简单理解：** 按目前已记录交易和估值，整体结果为亏损；这不是对未来价格的预测。")
            else:
                lines.append("- **简单理解：** 按目前已记录交易和估值，整体结果为盈利；不代表未来仍会盈利。")
        lines.append("")

    open_positions = [p for p in positions if p.get("open_position")]
    lines.append("### 你现在持有什么")
    if not open_positions:
        lines.append("当前没有未平仓持仓。已实现盈亏和分红仍会保留在累计结果中。")
    else:
        for position in open_positions:
            ticker = str(position.get("ticker", "未知标的"))
            currency = str(position.get("currency", "未知币种"))
            quantity = _number(position.get("quantity"))
            quantity_text = f"{quantity:g} 股" if quantity is not None else "股数未知"
            avg_cost = position.get("average_cost")
            current_price = position.get("current_price")
            market_value = position.get("market_value")
            unrealized = position.get("unrealized_pnl")
            remaining_cost = position.get("remaining_cost_basis")
            equivalent = position.get("equivalent_cost_per_share")
            lines.append(
                f"- **{ticker}：** {quantity_text}；当前估值价 {_format_money(current_price, currency)}；"
                f"市值 {_format_money(market_value, currency)}；"
                f"持仓浮动盈亏 {_format_money(unrealized, currency, signed=True)}；"
                f"剩余成本 {_format_money(remaining_cost, currency)}；"
                f"每股加权平均成本 {_format_money(avg_cost, currency)}。"
            )
            if _number(equivalent) is not None:
                lines.append(
                    f"  - 净收益调整后等效成本：{_format_money(equivalent, currency)}/股。"
                    "这只是扣除历史已实现盈亏和净分红后的解释性参考，不是券商税务成本或保本价。"
                )

    lines.append("")
    lines.append("### 如果价格涨跌，累计盈亏会怎样变化")
    selected_changes = {-10.0, -5.0, 0.0, 5.0, 10.0}
    selected = [s for s in scenarios if _number(s.get("price_change_pct")) in selected_changes]
    if not selected:
        lines.append("暂无可用价格情景。请先确认持仓估值价格有效。")
    else:
        for scenario in selected:
            currency = str(scenario.get("currency", "未知币种"))
            ticker = str(scenario.get("ticker", "未知标的"))
            pct = _number(scenario.get("price_change_pct"))
            if pct is None:
                continue
            lines.append(
                f"- {ticker} 假设价格 {pct:+.0f}%：情景价格 {_format_money(scenario.get('scenario_price'), currency)}；"
                f"情景市值 {_format_money(scenario.get('scenario_market_value'), currency)}；"
                f"累计净盈亏 {_format_money(scenario.get('scenario_cumulative_net_pnl'), currency, signed=True)}；"
                f"相对当前累计盈亏变化 {_format_money(scenario.get('cumulative_pnl_change'), currency, signed=True)}。"
            )
    lines.append("")
    lines.append("**重要口径：** 累计净盈亏 = 已实现盈亏 + 当前持仓未实现盈亏 + 净分红。手续费已按现有记账方法计入成本或净收入，不应重复扣除。不同币种分开计算。价格情景是假设，不是预测。由于没有完整的账户入金、出金和现金余额，这也不等同于银行账户的真实总回报率。")
    return "\n".join(lines)


def _load_market_and_external_context() -> dict[str, Any]:
    """Read available market/macro/news context from existing local project modules."""
    context: dict[str, Any] = {}
    try:
        # Reuse the established analysis pipeline and local data, rather than making
        # a second market-data client here.
        from main import build_summary, load_history
        history = load_history("QQQM")
        context["qqqm_market_summary"] = build_summary("QQQM", history)
    except Exception as exc:
        context["market_context_note"] = f"Could not load local market summary: {type(exc).__name__}: {exc}"

    try:
        from app.macro_data import get_macro_snapshot
        context["fred_macro"] = get_macro_snapshot()
    except Exception as exc:
        context["fred_macro_note"] = f"Could not load local FRED snapshot: {type(exc).__name__}: {exc}"

    try:
        # Existing helper loads local stored news/BLS context; it does not sync sources.
        from app.ai_report import _collect_external_context
        news_context, bls_context = _collect_external_context()
        context["stored_news_context"] = news_context[:12000]
        context["stored_bls_context"] = bls_context[:8000]
    except Exception as exc:
        context["external_context_note"] = f"Could not load stored news/BLS context: {type(exc).__name__}: {exc}"

    return context


def generate_portfolio_analysis(
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
) -> Path:
    """Generate a DeepSeek report from sanitized portfolio aggregates and local evidence."""
    # Streamlit may not have imported the project's CLI/config module first.
    # Load the local .env if python-dotenv is available, without printing secrets.
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key or api_key == "replace_with_your_deepseek_api_key":
        raise RuntimeError(
            "未配置有效的 DEEPSEEK_API_KEY。请检查项目根目录 .env；不要将 API Key 发到聊天中。"
        )

    try:
        from app.ai_report import DEFAULT_BASE_URL, DEFAULT_MODEL, REQUEST_TIMEOUT
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip()
        model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
        timeout = float(REQUEST_TIMEOUT)
    except Exception:
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip()
        model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
        timeout = DEFAULT_TIMEOUT

    if not base_url or not model:
        raise RuntimeError("DEEPSEEK_BASE_URL 或 DEEPSEEK_MODEL 配置为空。")
    if positions.empty:
        raise ValueError("当前没有持仓计算结果，无法生成个人持仓分析。")

    payload = build_portfolio_payload(positions, transactions)
    evidence = _load_market_and_external_context()

    system_prompt = """你是一名谨慎、面向新手的投资研究助手。你的首要任务是让用户先看懂结果，而不是展示专业术语。
必须遵守：
1. 所有持仓、成本、盈亏和情景数值均由程序计算；不得重新计算、修改、四舍五入成不同金额或补齐缺失数字。引用时保持给定数值和币种。
2. 第一屏先给一句话结论（最多 2 句），直说“目前整体是赚还是亏、金额多少、当前持仓是浮盈还是浮亏”。
3. 使用日常中文解释术语。首次提到已实现盈亏时解释为已卖出交易的结果；未实现盈亏解释为仍持有标的按当前估值计算的浮动结果。
4. 先给简明判断，再给“当前状态”“接下来怎么观察”“什么条件下重新评估买入/持有/减仓”“风险与数据局限”，最后才是专业指标细节。重要结论不能埋在长文末尾。
5. 不得发出无条件的“立即买入/卖出”命令，不得承诺收益。把行动辅助写成条件式规则：当前可观察什么、哪些证据变化时应重新评估、有哪些相反证据。若用户未提供持有期限、风险承受能力和仓位预算，必须说明无法据此给出充分个性化的买卖时点。
6. 价格情景是假设，不是价格预测。不得声称访问了实时行情；标明实际提供的行情与宏观数据日期。
7. 只有在所提供数据中有依据时才引用新闻、宏观和技术指标。缺失数据明确说明，不得推断其数值或编造新闻内容。
8. 不同币种绝不相加。手续费已按给定的记账方法进入买入成本、卖出净收入或净分红，不得重复扣除。净收益调整后等效成本仅是解释性参考，不是券商税务成本、保本价或目标价。
9. 报告属于研究与风险情景分析，不替用户执行交易，不把市场指标和 QQQM 涨跌之间的相关性写成已证明的因果关系。
"""

    user_prompt = (
        "请用面向投资新手的方式，生成一份简洁、可行动但不作确定性预测的个人持仓分析。\n\n"
        "请严格使用以下结构：\n"
        "## 一句话结论\n"
        "最多 2 句。直说按当前已记录数据总体赚/亏多少钱，当前仍持有的仓位是浮盈还是浮亏。不得省略币种。\n"
        "## 当前状态：这意味着什么\n"
        "用 3-5 个短要点解释结果构成，不要复制整张数据表。\n"
        "## 接下来怎么做：条件式观察清单\n"
        "给出最多 3 条条件式观察建议。每条包括要观察的信号、何种变化值得重新评估、为什么。不得无条件下达买卖指令。没有用户风险偏好/持有期限时，明确说这类信息缺失。\n"
        "## 哪些情况会改变当前判断\n"
        "列出最多 3 项有数据依据的变化条件，以及对应的不确定性。\n"
        "## 风险与数据局限\n"
        "说明价格、数据日期、币种、费用及缺少完整账户现金流的影响。\n"
        "## 详细依据（供想深入了解的用户阅读）\n"
        "再解释技术指标、宏观和新闻。不要把专业术语放在前面，不要杜撰因果关系。全文应精炼，避免重复。\n\n"
        "【程序核算的持仓与交易摘要】\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + "\n\n【可用的本地市场、宏观与新闻证据】\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2, default=str)[:24000]
    )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境未安装 openai SDK，请使用项目虚拟环境运行。") from exc

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content if response.choices else None
    report_text = str(content or "").strip()
    if not report_text:
        raise RuntimeError("DeepSeek 返回了空报告，请检查模型配置、接口状态或额度。")

    PORTFOLIO_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_path = PORTFOLIO_REPORT_DIR / f"portfolio_ai_report_{stamp}.md"
    report = (
        "# 个人持仓 DeepSeek 分析\n\n"
        f"- 生成时间（UTC）：{payload['generated_at_utc']}\n"
        f"- 模型：{model}\n"
        "- 数据来源：本地交易汇总、最近可用行情、已保存宏观数据与新闻\n\n"
        "> 本报告用于研究与风险情景分析，不构成买卖指令。模型输出可能出错；市场数据可能滞后。\n\n"
        "## 本次提供给模型的数据范围\n\n"
        "仅发送聚合持仓、按标的/币种/交易类型汇总的交易统计、价格情景，以及本地保存的市场/宏观/新闻上下文；"
        "不发送交易 ID、备注或逐笔交易日期。\n\n"
        "## 一眼看懂\n\n"
        + build_plain_language_summary(payload, evidence)
        + "\n\n## AI 解读：当前怎么看、下一步关注什么\n\n"
        + report_text + "\n"
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path
