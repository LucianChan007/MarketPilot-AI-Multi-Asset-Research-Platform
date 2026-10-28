"""Position calculations for the GUI.

This module only calculates an estimate from user-entered shares/cost and a
provided reference price. It does not connect to or sync any brokerage account.
"""
from __future__ import annotations

from typing import Any


def calculate_position(
    shares: float,
    avg_cost: float,
    current_price: float,
    fees_paid: float = 0.0,
) -> dict[str, Any]:
    """Calculate position value and unrealized P/L.

    Args:
        shares: Number of shares held. Must be non-negative.
        avg_cost: Average acquisition price per share, excluding fees if fees
            are entered separately.
        current_price: Reference market price per share, such as the latest
            available closing price (not necessarily real-time).
        fees_paid: Optional total fees/costs to include in the cost basis.

    Returns:
        Dictionary of numeric metrics rounded for display.
    """
    values = {
        "shares": float(shares),
        "avg_cost": float(avg_cost),
        "current_price": float(current_price),
        "fees_paid": float(fees_paid),
    }
    if any(v < 0 for v in values.values()):
        raise ValueError("shares、avg_cost、current_price 和 fees_paid 不能为负数。")

    base_cost = values["shares"] * values["avg_cost"]
    total_cost = base_cost + values["fees_paid"]
    market_value = values["shares"] * values["current_price"]
    profit = market_value - total_cost
    profit_pct = profit / total_cost * 100 if total_cost else 0.0

    return {
        "shares": values["shares"],
        "avg_cost": round(values["avg_cost"], 4),
        "current_price": round(values["current_price"], 4),
        "fees_paid": round(values["fees_paid"], 2),
        "base_cost": round(base_cost, 2),
        "total_cost": round(total_cost, 2),
        "market_value": round(market_value, 2),
        "profit": round(profit, 2),
        "profit_pct": round(profit_pct, 2),
    }
