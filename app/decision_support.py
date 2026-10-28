from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

SUPPORTED_HORIZONS = {
    "少于 1 年",
    "1–5 年",
    "5 年以上",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if pd.notna(result) else default
    except (TypeError, ValueError):
        return default


def build_decision_assessment(
    positions: pd.DataFrame,
    holding_horizon: str,
    review_drawdown_pct: float,
    planned_additional_budget: float = 0.0,
) -> dict[str, Any]:
    """Build a transparent review plan from positions and user preferences.

    ``review_drawdown_pct`` is a user-defined reminder threshold for a price
    decline from the current valuation, not a forecast or guaranteed stop-loss.
    Values are kept separate by ticker and currency.
    """
    if holding_horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"holding_horizon must be one of: {', '.join(SUPPORTED_HORIZONS)}")

    drawdown = _number(review_drawdown_pct, float("nan"))
    budget = _number(planned_additional_budget, float("nan"))
    if pd.isna(drawdown) or not 1 <= drawdown <= 50:
        raise ValueError("review_drawdown_pct must be between 1 and 50.")
    if pd.isna(budget) or budget < 0:
        raise ValueError("planned_additional_budget must be zero or greater.")

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holding_horizon": holding_horizon,
        "review_drawdown_pct": drawdown,
        "planned_additional_budget": budget,
        "items": [],
        "warnings": [],
    }

    required = {"ticker", "currency", "quantity", "remaining_cost_basis", "current_price", "open_position"}
    if not isinstance(positions, pd.DataFrame) or positions.empty or not required.issubset(positions.columns):
        result["warnings"].append("当前没有可用于分析的持仓数据。请先完成持仓计算。")
        result["conclusion"] = "暂时无法生成条件式观察计划，请先完成持仓计算并提供有效估值价格。"
        return result

    open_positions = positions[positions["open_position"].astype(bool)]
    if open_positions.empty:
        result["warnings"].append("当前没有未平仓持仓。")
        result["conclusion"] = "目前没有未平仓持仓，因此不生成当前持仓的价格回撤提醒。历史已实现盈亏仍可在持仓报告中查看。"
        return result

    for _, row in open_positions.iterrows():
        ticker = str(row["ticker"]).upper()
        currency = str(row["currency"]).upper()
        quantity = _number(row["quantity"])
        cost_basis = _number(row["remaining_cost_basis"])
        current_price = _number(row["current_price"], float("nan"))

        if quantity <= 0:
            continue
        if pd.isna(current_price) or current_price <= 0:
            result["warnings"].append(f"{ticker}/{currency} 缺少有效估值价格，已跳过情景计算。")
            continue

        market_value = quantity * current_price
        unrealized_pnl = market_value - cost_basis
        realized_pnl = _number(row.get("realized_pnl", 0.0))
        dividends = _number(row.get("dividend_income", 0.0))
        current_cumulative = realized_pnl + unrealized_pnl + dividends

        scenario_price = current_price * (1 - drawdown / 100.0)
        scenario_value = quantity * scenario_price
        scenario_unrealized = scenario_value - cost_basis
        scenario_cumulative = realized_pnl + scenario_unrealized + dividends
        value_change = market_value - scenario_value

        average_cost = _number(row.get("average_cost", float("nan")), float("nan"))
        below_average_cost = not pd.isna(average_cost) and current_price < average_cost

        item = {
            "ticker": ticker,
            "currency": currency,
            "quantity": quantity,
            "current_price": current_price,
            "average_cost": average_cost,
            "market_value": market_value,
            "remaining_cost_basis": cost_basis,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "dividend_income": dividends,
            "current_cumulative_net_pnl": current_cumulative,
            "review_scenario_drawdown_pct": drawdown,
            "review_scenario_price": scenario_price,
            "review_scenario_market_value": scenario_value,
            "review_scenario_value_decrease": value_change,
            "review_scenario_cumulative_net_pnl": scenario_cumulative,
            "price_below_average_cost": below_average_cost,
        }

        if below_average_cost:
            item["interpretation"] = (
                "当前估值低于剩余持仓的加权平均成本，说明这部分持仓目前浮亏。"
                "浮亏本身不能单独证明应该卖出或加仓，需要结合资金用途和你设定的风险边界复核。"
            )
        else:
            item["interpretation"] = (
                "当前估值不低于剩余持仓的加权平均成本；这不代表未来不会下跌，也不是继续持有或买入的保证。"
            )

        item["review_trigger"] = (
            f"如果 {ticker} 从当前估值下跌 {drawdown:.1f}%，价格情景约为 "
            f"{scenario_price:,.2f} {currency}，持仓市值将比当前减少 "
            f"{value_change:,.2f} {currency}；在其他项目不变时，累计净盈亏约为 "
            f"{scenario_cumulative:+,.2f} {currency}。达到该情景时，按你的规则重新评估，"
            "而不是自动买入或卖出。"
        )
        result["items"].append(item)

    if not result["items"]:
        result["warnings"].append("没有可用的未平仓持仓和有效估值价格组合。")
        result["conclusion"] = "暂时无法生成价格回撤提醒，请检查持仓数量和估值价格。"
        return result

    if holding_horizon == "少于 1 年":
        horizon_note = (
            "你的计划持有期少于 1 年。优先确认这笔钱是否会在期间内用于必要支出；"
            "如果是近期必需资金，不应仅因为想回本而承受计划之外的波动。"
        )
    elif holding_horizon == "1–5 年":
        horizon_note = (
            "你的计划持有期为 1–5 年。建议把这次回撤提醒与预先设定的投入/退出规则一起复核；"
            "中期持有并不保证回本。"
        )
    else:
        horizon_note = (
            "你的计划持有期为 5 年以上。较长持有期不等于没有大幅回撤；"
            "可以定期复核投资理由、仓位集中度和风险承受能力，而不是只盯着买入成本。"
        )

    if budget == 0:
        budget_note = "你填写的新增投入预算为 0；本计划只关注现有持仓，不把浮亏当作加仓理由。"
    else:
        budget_note = (
            f"你填写的未来新增投入预算为 {budget:,.2f}（预算单位需与你计划投入的币种一致）。"
            "这只是用户提供的预算参数，不代表系统建议把这笔钱投入该标的。"
        )

    result["horizon_note"] = horizon_note
    result["budget_note"] = budget_note
    result["conclusion"] = (
        f"本计划按你设定的 {drawdown:.1f}% 价格回撤提醒线检查现有持仓。"
        "它用于明确什么时候重新检查风险，不是价格预测、止损保证或自动买卖信号。"
    )
    return result
