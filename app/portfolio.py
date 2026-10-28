from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

SUPPORTED_SIDES = {"BUY", "SELL", "DIVIDEND"}
REQUIRED_COLUMNS = {"trade_date", "ticker", "side"}
OPTIONAL_COLUMNS = {
    "quantity", "price", "fee", "currency", "amount", "trade_id", "notes"
}

COLUMN_ALIASES = {
    "trade_date": {"trade_date", "date", "transaction_date", "成交日期", "交易日期"},
    "ticker": {"ticker", "symbol", "证券代码", "标的"},
    "side": {"side", "action", "type", "交易方向", "交易类型"},
    "quantity": {"quantity", "qty", "shares", "数量", "股数"},
    "price": {"price", "fill_price", "成交价", "成交价格"},
    "fee": {"fee", "commission", "手续费"},
    "currency": {"currency", "ccy", "币种"},
    "amount": {"amount", "cash_amount", "dividend_amount", "金额", "分红金额"},
    "trade_id": {"trade_id", "transaction_id", "成交编号"},
    "notes": {"notes", "memo", "备注"},
}


class PortfolioValidationError(ValueError):
    """Raised when a transaction file has invalid or inconsistent rows."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass
class PortfolioCalculation:
    transactions: pd.DataFrame
    positions: pd.DataFrame
    warnings: list[str]


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize recognized column names and retain unknown columns."""
    aliases = {
        alias.casefold(): canonical
        for canonical, names in COLUMN_ALIASES.items()
        for alias in (names | {canonical})
    }
    rename: dict[object, str] = {}
    for column in frame.columns:
        cleaned = str(column).strip().casefold().replace(" ", "_")
        rename[column] = aliases.get(cleaned, cleaned)

    result = frame.rename(columns=rename).copy()
    if result.columns.duplicated().any():
        merged: dict[str, pd.Series] = {}
        for column in dict.fromkeys(result.columns):
            same = result.loc[:, result.columns == column]
            if same.shape[1] == 1:
                merged[column] = same.iloc[:, 0]
            else:
                merged[column] = same.bfill(axis=1).iloc[:, 0]
        result = pd.DataFrame(merged, index=result.index)
    return result


def validate_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize transaction data, validate rows and warn about exact duplicates.

    Supported transaction types:
      * BUY: quantity and price required; fee defaults to zero.
      * SELL: quantity and price required; must not sell more than held.
      * DIVIDEND: positive gross cash amount required; fee is optional.

    Input order is retained for transactions on the same date. Across different
    dates, rows are sorted chronologically before position calculation.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise PortfolioValidationError(["交易文件为空，请至少提供一条交易记录。"])

    frame = _canonicalize_columns(raw)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise PortfolioValidationError([
            "缺少必需字段：" + ", ".join(missing),
            "必需字段为 trade_date、ticker、side；建议同时提供 quantity、price、fee、currency 和 amount。",
        ])

    defaults = {
        "quantity": pd.NA,
        "price": pd.NA,
        "fee": 0.0,
        "currency": "USD",
        "amount": pd.NA,
        "trade_id": "",
        "notes": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame = frame.reset_index(drop=True)
    frame["_source_row"] = range(2, len(frame) + 2)  # Header is row 1.
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip().str.upper()
    frame["side"] = frame["side"].fillna("").astype(str).str.strip().str.upper()
    frame["currency"] = (
        frame["currency"].fillna("USD").astype(str).str.strip().str.upper()
    )
    frame["trade_id"] = frame["trade_id"].fillna("").astype(str).str.strip()
    frame["notes"] = frame["notes"].fillna("").astype(str).str.strip()

    errors: list[str] = []
    invalid_numeric: dict[str, pd.Series] = {}
    for column in ("quantity", "price", "fee", "amount"):
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")
        # Blank optional cells are allowed; nonblank values that cannot be
        # parsed as numbers should not silently turn into zero or missing data.
        nonblank = original.notna() & original.astype(str).str.strip().ne("")
        invalid_numeric[column] = nonblank & converted.isna()
        frame[column] = converted

    frame["fee"] = frame["fee"].fillna(0.0)

    for idx, row in frame.iterrows():
        row_no = int(row["_source_row"])
        side = str(row["side"])

        if pd.isna(row["trade_date"]):
            errors.append(f"第 {row_no} 行：trade_date 不是有效日期。")
        if not row["ticker"]:
            errors.append(f"第 {row_no} 行：ticker 不能为空。")
        if side not in SUPPORTED_SIDES:
            errors.append(
                f"第 {row_no} 行：不支持 side={side!r}；只支持 BUY、SELL、DIVIDEND。"
            )
        if not row["currency"]:
            errors.append(f"第 {row_no} 行：currency 不能为空。")

        for column, invalid_rows in invalid_numeric.items():
            if bool(invalid_rows.iloc[idx]):
                errors.append(f"第 {row_no} 行：{column} 必须是数字。")

        if float(row["fee"]) < 0:
            errors.append(f"第 {row_no} 行：fee 必须大于或等于 0。")

        if side in {"BUY", "SELL"}:
            if pd.isna(row["quantity"]) or float(row["quantity"]) <= 0:
                errors.append(f"第 {row_no} 行：{side} 的 quantity 必须大于 0。")
            if pd.isna(row["price"]) or float(row["price"]) <= 0:
                errors.append(f"第 {row_no} 行：{side} 的 price 必须大于 0。")
            if pd.notna(row["amount"]) and float(row["amount"]) != 0:
                errors.append(
                    f"第 {row_no} 行：BUY/SELL 的 amount 应留空或填 0，金额由 quantity × price 计算。"
                )
        elif side == "DIVIDEND":
            if pd.isna(row["amount"]) or float(row["amount"]) <= 0:
                errors.append(f"第 {row_no} 行：DIVIDEND 的 amount 必须是正数。")
            if pd.notna(row["quantity"]) and float(row["quantity"]) != 0:
                errors.append(f"第 {row_no} 行：DIVIDEND 的 quantity 应留空或填 0。")
            if pd.notna(row["price"]) and float(row["price"]) != 0:
                errors.append(f"第 {row_no} 行：DIVIDEND 的 price 应留空或填 0。")

    ids = frame.loc[frame["trade_id"] != "", "trade_id"]
    duplicate_ids = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicate_ids:
        errors.append("trade_id 存在重复值：" + ", ".join(duplicate_ids))

    if errors:
        raise PortfolioValidationError(errors)

    warnings: list[str] = []
    duplicate_subset = [
        "trade_date", "ticker", "side", "quantity", "price", "fee", "currency", "amount"
    ]
    duplicates = frame.duplicated(subset=duplicate_subset, keep=False)
    if duplicates.any():
        count = int(duplicates.sum())
        warnings.append(
            f"检测到 {count} 行内容完全重复的交易记录。系统不会自动删除，请核对是否为重复导入。"
        )

    frame = frame.sort_values(
        ["trade_date", "_source_row"], kind="mergesort"
    ).reset_index(drop=True)
    return frame, warnings


def calculate_portfolio(
    raw: pd.DataFrame,
    current_prices: Mapping[str | tuple[str, str], float] | None = None,
) -> PortfolioCalculation:
    """Calculate positions with weighted-average cost accounting.

    Buy-side fees are added to cost basis. Sell-side fees are deducted from
    proceeds. Dividend fees are deducted from dividend cash income. Therefore,
    the resulting P&L already accounts for those fees and must not subtract the
    total fee amount again.

    Currency totals are kept separate. No foreign-exchange conversion is done.
    """
    tx, warnings = validate_transactions(raw)

    normalized_prices: dict[str | tuple[str, str], float] = {}
    for key, value in (current_prices or {}).items():
        if isinstance(key, tuple) and len(key) == 2:
            normalized_key: str | tuple[str, str] = (
                str(key[0]).strip().upper(), str(key[1]).strip().upper()
            )
        else:
            normalized_key = str(key).strip().upper()
        normalized_prices[normalized_key] = value

    states: dict[tuple[str, str], dict[str, float]] = {}
    processed_rows: list[dict] = []

    for _, row in tx.iterrows():
        ticker = str(row["ticker"])
        currency = str(row["currency"])
        key = (ticker, currency)
        state = states.setdefault(key, {
            "quantity": 0.0,
            "cost_basis": 0.0,
            "realized_pnl": 0.0,
            "dividend_income": 0.0,
            "buy_fees": 0.0,
            "sell_fees": 0.0,
            "dividend_fees": 0.0,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
        })

        side = str(row["side"])
        qty = float(row["quantity"]) if pd.notna(row["quantity"]) else 0.0
        price = float(row["price"]) if pd.notna(row["price"]) else 0.0
        fee = float(row["fee"])
        amount = float(row["amount"]) if pd.notna(row["amount"]) else 0.0
        realized = 0.0
        allocated_cost = 0.0
        cash_flow = 0.0

        row_no = int(row["_source_row"])
        if side == "BUY":
            notional = qty * price
            state["quantity"] += qty
            state["cost_basis"] += notional + fee
            state["buy_fees"] += fee
            state["buy_notional"] += notional
            cash_flow = -(notional + fee)
        elif side == "SELL":
            available = state["quantity"]
            if qty > available + 1e-8:
                raise PortfolioValidationError([
                    f"第 {row_no} 行：尝试卖出 {qty:g} 股 {ticker}，"
                    f"但此前可用持仓只有 {available:g} 股。"
                ])
            average_cost = state["cost_basis"] / available if available > 0 else 0.0
            allocated_cost = average_cost * qty
            notional = qty * price
            realized = notional - fee - allocated_cost
            state["quantity"] = max(0.0, available - qty)
            state["cost_basis"] = max(0.0, state["cost_basis"] - allocated_cost)
            if state["quantity"] <= 1e-8:
                state["quantity"] = 0.0
                state["cost_basis"] = 0.0
            state["realized_pnl"] += realized
            state["sell_fees"] += fee
            state["sell_notional"] += notional
            cash_flow = notional - fee
        else:  # DIVIDEND
            state["dividend_income"] += amount - fee
            state["dividend_fees"] += fee
            cash_flow = amount - fee

        processed = row.to_dict()
        processed.update({
            "notional": qty * price if side in {"BUY", "SELL"} else amount,
            "allocated_cost_basis": allocated_cost,
            "realized_pnl": realized,
            "cash_flow": cash_flow,
            "shares_after_transaction": state["quantity"],
            "holding_cost_basis_after": state["cost_basis"],
        })
        processed_rows.append(processed)

    position_rows: list[dict] = []
    for (ticker, currency), state in states.items():
        qty = state["quantity"]
        cost_basis = state["cost_basis"]
        average_cost = cost_basis / qty if qty > 0 else float("nan")

        current = normalized_prices.get(
            (ticker, currency), normalized_prices.get(ticker)
        )
        try:
            current_price = (
                float(current)
                if current is not None and pd.notna(current)
                else float("nan")
            )
        except (TypeError, ValueError):
            current_price = float("nan")
        if not pd.isna(current_price) and current_price <= 0:
            current_price = float("nan")

        market_value = (
            qty * current_price
            if qty > 0 and not pd.isna(current_price)
            else (0.0 if qty <= 1e-8 else float("nan"))
        )
        unrealized_pnl = (
            market_value - cost_basis
            if not pd.isna(market_value)
            else float("nan")
        )
        unrealized_pct = (
            unrealized_pnl / cost_basis * 100
            if not pd.isna(unrealized_pnl) and cost_basis > 0
            else float("nan")
        )
        total_fees = state["buy_fees"] + state["sell_fees"] + state["dividend_fees"]

        if qty > 1e-8:
            cumulative_net_pnl = (
                state["realized_pnl"] + unrealized_pnl + state["dividend_income"]
                if not pd.isna(unrealized_pnl)
                else float("nan")
            )
            equivalent_cost_per_share = (
                cost_basis - state["realized_pnl"] - state["dividend_income"]
            ) / qty
        else:
            # A closed position has no per-share equivalent cost, but all
            # realized profits/losses and net dividends remain part of history.
            cumulative_net_pnl = state["realized_pnl"] + state["dividend_income"]
            equivalent_cost_per_share = float("nan")

        position_rows.append({
            "ticker": ticker,
            "currency": currency,
            "quantity": qty,
            "average_cost": average_cost,
            "remaining_cost_basis": cost_basis,
            "current_price": current_price,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pct,
            "realized_pnl": state["realized_pnl"],
            "dividend_income": state["dividend_income"],
            "realized_plus_dividends": state["realized_pnl"] + state["dividend_income"],
            "cumulative_net_pnl": cumulative_net_pnl,
            "equivalent_cost_per_share": equivalent_cost_per_share,
            "total_fees": total_fees,
            "buy_fees": state["buy_fees"],
            "sell_fees": state["sell_fees"],
            "dividend_fees": state["dividend_fees"],
            "buy_notional": state["buy_notional"],
            "sell_notional": state["sell_notional"],
            "open_position": qty > 1e-8,
        })

    positions = pd.DataFrame(position_rows)
    if not positions.empty:
        positions = positions.sort_values(["currency", "ticker"]).reset_index(drop=True)
        numeric_columns = positions.select_dtypes(include="number").columns
        positions[numeric_columns] = positions[numeric_columns].round(6)

    processed_df = pd.DataFrame(processed_rows)
    if not processed_df.empty and "trade_date" in processed_df.columns:
        processed_df["trade_date"] = pd.to_datetime(
            processed_df["trade_date"]
        ).dt.strftime("%Y-%m-%d")

    return PortfolioCalculation(
        transactions=processed_df,
        positions=positions,
        warnings=warnings,
    )


def build_scenario_table(
    positions: pd.DataFrame,
    changes_pct: Sequence[float] = (-20, -10, -5, 0, 5, 10, 20),
) -> pd.DataFrame:
    """Calculate price scenarios for open positions, including total recorded P&L.

    Scenario cumulative P&L is defined as historical realized P&L plus the
    scenario unrealized P&L plus historical net dividend income. All values
    remain separated by ticker and currency; no FX conversion is performed.
    This is a hypothetical sensitivity analysis, not a price forecast.
    """
    output_columns = [
        "ticker",
        "currency",
        "price_change_pct",
        "scenario_price",
        "scenario_market_value",
        "scenario_unrealized_pnl",
        "scenario_unrealized_pnl_pct",
        "realized_pnl",
        "dividend_income",
        "current_cumulative_net_pnl",
        "scenario_cumulative_net_pnl",
        "cumulative_pnl_change",
    ]
    required = {
        "ticker",
        "currency",
        "quantity",
        "remaining_cost_basis",
        "current_price",
        "open_position",
    }
    if positions.empty or not required.issubset(positions.columns):
        return pd.DataFrame(columns=output_columns)

    def safe_float(value, default=0.0):
        try:
            number = float(value)
            return number if pd.notna(number) else default
        except (TypeError, ValueError):
            return default

    rows: list[dict] = []
    for _, position in positions.iterrows():
        if not bool(position["open_position"]):
            continue

        current_price = safe_float(position["current_price"], float("nan"))
        quantity = safe_float(position["quantity"])
        cost_basis = safe_float(position["remaining_cost_basis"])
        if pd.isna(current_price) or current_price <= 0 or quantity <= 0:
            continue

        realized = safe_float(position.get("realized_pnl", 0.0))
        dividends = safe_float(position.get("dividend_income", 0.0))
        current_market_value = quantity * current_price
        current_unrealized = current_market_value - cost_basis
        current_cumulative = realized + current_unrealized + dividends

        for change in changes_pct:
            pct = safe_float(change, float("nan"))
            if pd.isna(pct) or pct < -100:
                continue

            scenario_price = current_price * (1.0 + pct / 100.0)
            scenario_market_value = quantity * scenario_price
            scenario_unrealized = scenario_market_value - cost_basis
            scenario_cumulative = realized + scenario_unrealized + dividends

            rows.append({
                "ticker": position["ticker"],
                "currency": position["currency"],
                "price_change_pct": pct,
                "scenario_price": scenario_price,
                "scenario_market_value": scenario_market_value,
                "scenario_unrealized_pnl": scenario_unrealized,
                "scenario_unrealized_pnl_pct": (
                    scenario_unrealized / cost_basis * 100.0
                    if cost_basis > 0
                    else float("nan")
                ),
                "realized_pnl": realized,
                "dividend_income": dividends,
                "current_cumulative_net_pnl": current_cumulative,
                "scenario_cumulative_net_pnl": scenario_cumulative,
                "cumulative_pnl_change": scenario_cumulative - current_cumulative,
            })

    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows, columns=output_columns).round(6)
