from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "market_data.db"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None
    volume: float | None
    dividends: float
    source: str


@dataclass
class Lot:
    buy_level: float
    shares: int
    cost_basis: float
    buy_date: str
    buy_price: float


@dataclass(frozen=True)
class PendingOrder:
    side: str  # "BUY" or "SELL"
    trigger_level: float
    buy_level: float | None = None


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than or equal to 0")
    return number


def _read_price_bars(
    db_path: Path,
    ticker: str,
    *,
    data_file: Path | None = None,
) -> list[PriceBar]:
    """Read validated bars from a CSV file when supplied, otherwise SQLite."""
    if data_file is not None:
        return _read_price_bars_from_csv(data_file, ticker)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        found = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_daily'"
        ).fetchone()
        if not found:
            raise RuntimeError("SQLite table market_daily was not found.")

        rows = conn.execute(
            """
            SELECT ticker, trade_date, open, high, low, close,
                   adj_close, volume, dividends, source
            FROM market_daily
            WHERE UPPER(ticker) = UPPER(?)
            ORDER BY trade_date ASC
            """,
            (ticker,),
        ).fetchall()
    finally:
        conn.close()

    bars: list[PriceBar] = []
    seen_dates: set[str] = set()
    for row in rows:
        try:
            trade_date = str(row["trade_date"])
            # Validate ISO-like date values and reject malformed records.
            date.fromisoformat(trade_date[:10])
            o = float(row["open"])
            h = float(row["high"])
            low = float(row["low"])
            c = float(row["close"])
            if not all(math.isfinite(x) and x > 0 for x in (o, h, low, c)):
                continue
            if h < max(o, c, low) or low > min(o, c, h):
                continue
            if trade_date in seen_dates:
                continue
            seen_dates.add(trade_date)

            adj_raw = row["adj_close"]
            vol_raw = row["volume"]
            div_raw = row["dividends"]
            adj = float(adj_raw) if adj_raw is not None else None
            if adj is not None and (not math.isfinite(adj) or adj <= 0):
                adj = None
            volume = float(vol_raw) if vol_raw is not None else None
            dividends = float(div_raw or 0.0)
            if not math.isfinite(dividends) or dividends < 0:
                dividends = 0.0

            bars.append(
                PriceBar(
                    ticker=str(row["ticker"]),
                    trade_date=trade_date[:10],
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    adj_close=adj,
                    volume=volume,
                    dividends=dividends,
                    source=str(row["source"] or "unknown"),
                )
            )
        except (TypeError, ValueError, OverflowError):
            # Skip invalid database rows, but do not silently pretend they were used.
            continue

    if len(bars) < 2:
        raise RuntimeError(
            f"Only {len(bars)} valid daily bars found for {ticker}; at least 2 are required."
        )
    return bars



def _read_price_bars_from_csv(csv_path: Path, ticker: str) -> list[PriceBar]:
    """Read a validated Tiingo daily-prices CSV without touching SQLite.

    Uses raw OHLC for fills/mark-to-market, adjClose only as metadata, and
    divCash separately for cash dividend credits. The CSV must contain date,
    open, high, low, close; volume, adjClose and divCash are optional.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV data file not found: {csv_path}")
    if not csv_path.is_file():
        raise RuntimeError(f"CSV data path is not a file: {csv_path}")

    bars: list[PriceBar] = []
    seen_dates: set[str] = set()
    invalid_rows: list[str] = []
    source_name = f"tiingo_csv:{csv_path.name}"

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {str(field).strip() for field in (reader.fieldnames or [])}
        required = {"date", "open", "high", "low", "close"}
        missing = required - fields
        if missing:
            raise RuntimeError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_no, row in enumerate(reader, start=2):
            try:
                date_raw = (row.get("date") or "").strip()
                trade_date = date_raw[:10]
                date.fromisoformat(trade_date)
                if trade_date in seen_dates:
                    raise ValueError(f"duplicate date {trade_date}")

                def number(name: str, default: float | None = None) -> float | None:
                    raw = row.get(name)
                    if raw is None or str(raw).strip() == "":
                        return default
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError(f"{name} is not finite")
                    return value

                o = number("open")
                h = number("high")
                low = number("low")
                c = number("close")
                if any(value is None for value in (o, h, low, c)):
                    raise ValueError("missing OHLC price")
                o, h, low, c = float(o), float(h), float(low), float(c)
                if any(value <= 0 for value in (o, h, low, c)):
                    raise ValueError("OHLC prices must be positive")
                if h < max(o, c, low) or low > min(o, c, h):
                    raise ValueError("invalid OHLC relationship")

                adj = number("adjClose", None)
                if adj is not None and adj <= 0:
                    adj = None
                volume = number("volume", None)
                if volume is not None and volume < 0:
                    raise ValueError("volume cannot be negative")
                dividends = number("divCash", 0.0) or 0.0
                if dividends < 0:
                    raise ValueError("divCash cannot be negative")

                bars.append(
                    PriceBar(
                        ticker=ticker.upper(),
                        trade_date=trade_date,
                        open=o,
                        high=h,
                        low=low,
                        close=c,
                        adj_close=adj,
                        volume=volume,
                        dividends=dividends,
                        source=source_name,
                    )
                )
                seen_dates.add(trade_date)
            except (TypeError, ValueError, OverflowError) as exc:
                invalid_rows.append(f"line {line_no}: {exc}")

    if invalid_rows:
        examples = "; ".join(invalid_rows[:5])
        raise RuntimeError(
            f"CSV validation failed for {len(invalid_rows)} row(s). Examples: {examples}"
        )

    bars.sort(key=lambda bar: bar.trade_date)
    if len(bars) < 2:
        raise RuntimeError(
            f"Only {len(bars)} valid daily bars found in {csv_path}; at least 2 are required."
        )
    return bars


def _build_buy_levels(lower: float, upper: float, step: float) -> list[float]:
    intervals = (upper - lower) / step
    rounded = round(intervals)
    if not math.isclose(intervals, rounded, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("upper - lower must be an exact multiple of step.")
    if rounded < 1:
        raise ValueError("The grid must contain at least one interval.")
    # Buy at each level lower ... upper-step; sell that lot one step above.
    return [round(lower + i * step, 8) for i in range(int(rounded))]


def _max_drawdown(equities: list[float]) -> float:
    peak = -math.inf
    largest = 0.0
    for value in equities:
        peak = max(peak, value)
        if peak > 0:
            largest = max(largest, (peak - value) / peak)
    return largest


def _cagr(start_value: float, end_value: float, start_date: str, end_date: str) -> float | None:
    days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
    if days <= 0 or start_value <= 0 or end_value <= 0:
        return None
    return (end_value / start_value) ** (365.25 / days) - 1.0


def _annualized_volatility(equities: list[float]) -> float | None:
    returns: list[float] = []
    for previous, current in zip(equities, equities[1:]):
        if previous > 0:
            returns.append(current / previous - 1.0)
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(252.0)


def run_backtest(
    bars: list[PriceBar],
    *,
    initial_cash: float,
    lower: float,
    upper: float,
    step: float,
    cash_per_grid: float,
    fee_rate: float,
    slippage_bps: float,
    max_position_value: float,
) -> dict[str, Any]:
    """Run daily-close-confirmed grid signals with next-session-open execution."""
    if initial_cash <= 0 or cash_per_grid <= 0 or max_position_value <= 0:
        raise ValueError("Cash and position values must be greater than zero.")
    if lower <= 0 or upper <= lower or step <= 0:
        raise ValueError("Require 0 < lower < upper and step > 0.")
    if fee_rate < 0 or fee_rate >= 0.1:
        raise ValueError("fee_rate must be between 0 and 0.1 (exclusive).")
    if slippage_bps < 0 or slippage_bps >= 1000:
        raise ValueError("slippage_bps must be between 0 and 1000.")

    buy_levels = _build_buy_levels(lower, upper, step)
    tolerance = 1e-8
    max_shares = max(0, math.floor(max_position_value / upper))
    if max_shares < 1:
        raise ValueError("max_position_value is too small to buy one share at the lower grid bound.")

    cash = float(initial_cash)
    inventory: dict[float, list[Lot]] = {level: [] for level in buy_levels}
    pending: list[PendingOrder] = []
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    signal_count = 0
    cancelled_orders = 0
    rejected_orders = 0
    total_fees = 0.0
    total_dividends = 0.0
    realized_pnl = 0.0
    total_buy_shares = 0
    total_sell_shares = 0
    max_held_shares = 0

    # Cost-aware buy-and-hold benchmark: buy whole shares at the first close,
    # applying the same adverse slippage and fee assumptions as grid trades.
    # Remaining cash is retained and cash dividends are credited to the account.
    slip_fraction = slippage_bps / 10_000.0
    first_close = bars[0].close
    bh_entry_fill = first_close * (1.0 + slip_fraction)
    bh_shares = math.floor(initial_cash / (bh_entry_fill * (1.0 + fee_rate)))
    bh_entry_notional = bh_shares * bh_entry_fill
    bh_entry_fee = bh_entry_notional * fee_rate
    bh_entry_slippage_cost = bh_shares * (bh_entry_fill - first_close)
    bh_cash = initial_cash - bh_entry_notional - bh_entry_fee
    bh_initial_value = initial_cash

    for index, bar in enumerate(bars):
        # Dividend entitlement is based on shares held at the prior close, before today's open trades.
        if index > 0 and bar.dividends > 0:
            grid_shares_before_open = sum(
                lot.shares for lots in inventory.values() for lot in lots
            )
            dividend_amount = grid_shares_before_open * bar.dividends
            cash += dividend_amount
            total_dividends += dividend_amount

            bh_dividend = bh_shares * bar.dividends
            bh_cash += bh_dividend

        # Execute signals generated from the previous close, at today's open only.
        if index > 0 and pending:
            todays_orders = list(pending)
            pending = []
            # If multiple levels were crossed in a gap, prioritize the deepest eligible bid;
            # sells are handled in ascending target order. Cash/position constraints still apply.
            todays_orders.sort(
                key=lambda order: (
                    0 if order.side == "BUY" else 1,
                    order.trigger_level if order.side == "BUY" else order.trigger_level,
                )
            )
            for order in todays_orders:
                if order.side == "BUY":
                    # A buy limit signal is stale if the next open has recovered above its trigger.
                    if bar.open > order.trigger_level + tolerance:
                        cancelled_orders += 1
                        continue

                    fill_price = bar.open * (1.0 + slip_fraction)
                    held_shares = sum(
                        lot.shares for lots in inventory.values() for lot in lots
                    )
                    available_position_shares = max_shares - held_shares
                    if available_position_shares <= 0:
                        rejected_orders += 1
                        continue

                    budget_shares = math.floor(cash_per_grid / fill_price)
                    cash_shares = math.floor(cash / (fill_price * (1.0 + fee_rate)))
                    quantity = min(budget_shares, cash_shares, available_position_shares)
                    if quantity <= 0:
                        rejected_orders += 1
                        continue

                    notional = quantity * fill_price
                    fee = notional * fee_rate
                    total_cost = notional + fee
                    if total_cost > cash + 1e-8:
                        rejected_orders += 1
                        continue

                    cash -= total_cost
                    total_fees += fee
                    total_buy_shares += quantity
                    buy_level = float(order.trigger_level)
                    lot = Lot(
                        buy_level=buy_level,
                        shares=quantity,
                        cost_basis=total_cost,
                        buy_date=bar.trade_date,
                        buy_price=fill_price,
                    )
                    inventory.setdefault(buy_level, []).append(lot)
                    held_after = sum(
                        item.shares for lots in inventory.values() for item in lots
                    )
                    max_held_shares = max(max_held_shares, held_after)
                    trades.append(
                        {
                            "date": bar.trade_date,
                            "side": "BUY",
                            "trigger_level": order.trigger_level,
                            "fill_price": fill_price,
                            "shares": quantity,
                            "notional": notional,
                            "fee": fee,
                            "realized_pnl": 0.0,
                            "cash_after": cash,
                            "held_shares_after": held_after,
                            "reason": "close crossed down; next open remained at/below trigger",
                        }
                    )

                elif order.side == "SELL":
                    if bar.open < order.trigger_level - tolerance:
                        cancelled_orders += 1
                        continue

                    if order.buy_level is None:
                        rejected_orders += 1
                        continue
                    matching_key = round(order.buy_level, 8)
                    lots = inventory.get(matching_key, [])
                    if not lots:
                        # No matching lot bought one grid interval below the sell trigger.
                        rejected_orders += 1
                        continue

                    lot = lots.pop(0)  # FIFO within each grid level.
                    if not lots:
                        inventory[matching_key] = []

                    fill_price = bar.open * (1.0 - slip_fraction)
                    notional = lot.shares * fill_price
                    fee = notional * fee_rate
                    net_proceeds = notional - fee
                    lot_realized_pnl = net_proceeds - lot.cost_basis
                    cash += net_proceeds
                    total_fees += fee
                    realized_pnl += lot_realized_pnl
                    total_sell_shares += lot.shares
                    held_after = sum(
                        item.shares for remaining in inventory.values() for item in remaining
                    )
                    trades.append(
                        {
                            "date": bar.trade_date,
                            "side": "SELL",
                            "trigger_level": order.trigger_level,
                            "fill_price": fill_price,
                            "shares": lot.shares,
                            "notional": notional,
                            "fee": fee,
                            "realized_pnl": lot_realized_pnl,
                            "cash_after": cash,
                            "held_shares_after": held_after,
                            "reason": "close crossed up; next open remained at/above trigger",
                        }
                    )

        held_shares = sum(lot.shares for lots in inventory.values() for lot in lots)
        position_value = held_shares * bar.close
        grid_equity = cash + position_value
        bh_equity = bh_cash + bh_shares * bar.close
        equity_curve.append(
            {
                "date": bar.trade_date,
                "ticker": bar.ticker,
                "close": bar.close,
                "cash": cash,
                "shares": held_shares,
                "position_value": position_value,
                "equity": grid_equity,
                "buy_hold_cash": bh_cash,
                "buy_hold_shares": bh_shares,
                "buy_hold_position_value": bh_shares * bar.close,
                "buy_hold_equity": bh_equity,
            }
        )

        # Signals are detected after today's close and can only execute at the next session's open.
        if index < len(bars) - 1:
            previous_close = bars[index - 1].close if index > 0 else bar.close
            if index > 0:
                if bar.close < previous_close:
                    for level in buy_levels:
                        if previous_close > level + tolerance and bar.close <= level + tolerance:
                            pending.append(PendingOrder("BUY", level, None))
                            signal_count += 1
                elif bar.close > previous_close:
                    for buy_level in buy_levels:
                        sell_level = round(buy_level + step, 8)
                        if previous_close < sell_level - tolerance and bar.close >= sell_level - tolerance:
                            pending.append(PendingOrder("SELL", sell_level, buy_level))
                            signal_count += 1

    if not equity_curve:
        raise RuntimeError("No equity curve was generated.")

    strategy_equities = [point["equity"] for point in equity_curve]
    benchmark_equities = [point["buy_hold_equity"] for point in equity_curve]
    final_grid = equity_curve[-1]
    final_bh = equity_curve[-1]
    start_date = bars[0].trade_date
    end_date = bars[-1].trade_date
    duration_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
    final_grid_equity = float(final_grid["equity"])
    final_bh_equity = float(final_bh["buy_hold_equity"])

    buys = sum(1 for trade in trades if trade["side"] == "BUY")
    sells = sum(1 for trade in trades if trade["side"] == "SELL")
    buy_and_sell_transactions = len(trades)
    sources = sorted({bar.source for bar in bars if bar.source})

    grid_exposures = [
        point["position_value"] / point["equity"] * 100.0
        for point in equity_curve if point["equity"] > 0
    ]
    grid_cash_weights = [
        point["cash"] / point["equity"] * 100.0
        for point in equity_curve if point["equity"] > 0
    ]
    bh_exposures = [
        point["buy_hold_position_value"] / point["buy_hold_equity"] * 100.0
        for point in equity_curve if point["buy_hold_equity"] > 0
    ]
    bh_equities = [point["buy_hold_equity"] for point in equity_curve]
    bh_annualized_volatility = _annualized_volatility(bh_equities)

    summary: dict[str, Any] = {
        "ticker": bars[0].ticker.upper(),
        "data_source": ", ".join(sources) if sources else "unknown",
        "start_date": start_date,
        "end_date": end_date,
        "bars": len(bars),
        "calendar_days": duration_days,
        "parameters": {
            "initial_cash": initial_cash,
            "grid_lower": lower,
            "grid_upper": upper,
            "grid_step": step,
            "cash_per_grid_order": cash_per_grid,
            "fee_rate": fee_rate,
            "slippage_bps_each_side": slippage_bps,
            "max_position_value_assumption": max_position_value,
            "max_shares_at_upper_bound": max_shares,
        },
        "grid_strategy": {
            "initial_equity": initial_cash,
            "final_equity": final_grid_equity,
            "total_return_pct": (final_grid_equity / initial_cash - 1.0) * 100.0,
            "cagr_pct": (
                _cagr(initial_cash, final_grid_equity, start_date, end_date) * 100.0
                if _cagr(initial_cash, final_grid_equity, start_date, end_date) is not None
                else None
            ),
            "max_drawdown_pct": _max_drawdown(strategy_equities) * 100.0,
            "annualized_volatility_pct": (
                _annualized_volatility(strategy_equities) * 100.0
                if _annualized_volatility(strategy_equities) is not None
                else None
            ),
            "average_position_exposure_pct": (
                statistics.mean(grid_exposures) if grid_exposures else None
            ),
            "average_cash_weight_pct": (
                statistics.mean(grid_cash_weights) if grid_cash_weights else None
            ),
            "buy_orders_filled": buys,
            "sell_orders_filled": sells,
            "filled_orders_total": buy_and_sell_transactions,
            "signals_generated": signal_count,
            "orders_cancelled_on_next_open": cancelled_orders,
            "orders_rejected_for_cash_inventory_or_lot": rejected_orders,
            "shares_bought": total_buy_shares,
            "shares_sold": total_sell_shares,
            "ending_cash": cash,
            "ending_shares": int(final_grid["shares"]),
            "ending_position_value": float(final_grid["position_value"]),
            "max_shares_held": max_held_shares,
            "fees_paid": total_fees,
            "dividends_received": total_dividends,
            "realized_pnl_before_dividends": realized_pnl,
            "final_equity_includes_unsold_inventory_at_close": True,
        },
        "buy_and_hold": {
            "initial_equity": bh_initial_value,
            "final_equity": final_bh_equity,
            "total_return_pct": (final_bh_equity / bh_initial_value - 1.0) * 100.0,
            "cagr_pct": (
                _cagr(bh_initial_value, final_bh_equity, start_date, end_date) * 100.0
                if _cagr(bh_initial_value, final_bh_equity, start_date, end_date) is not None
                else None
            ),
            "max_drawdown_pct": _max_drawdown([bh_initial_value, *benchmark_equities]) * 100.0,
            "annualized_volatility_pct": (
                _annualized_volatility([bh_initial_value, *benchmark_equities]) * 100.0
                if _annualized_volatility([bh_initial_value, *benchmark_equities]) is not None else None
            ),
            "average_position_exposure_pct": (
                statistics.mean(bh_exposures) if bh_exposures else None
            ),
            "buy_shares_at_first_close": bh_shares,
            "entry_fee_paid": bh_entry_fee,
            "entry_slippage_cost_estimate": bh_entry_slippage_cost,
            "ending_cash_including_dividends": bh_cash,
        },
        "assumptions_and_limitations": [
            "Grid starts entirely in cash; it does not hold an initial inventory.",
            "Buy-and-hold benchmark buys whole shares at the first close and applies the same fee-rate and adverse-slippage assumptions to that initial purchase; remaining cash is retained.",
            "The two strategies still have different time-varying market exposure; report returns alongside exposure and drawdown rather than treating them as exposure-matched.",
            "Signals are generated from close-to-close crossings and execute at the next trading session open, not at the triggering grid line.",
            "A buy signal is cancelled if the next open is above its trigger; a sell signal is cancelled if the next open is below its trigger.",
            "Execution slippage is applied adversely to each filled order, plus the configured proportional fee.",
            "Orders buy whole shares only. A buy fills one lot per crossed grid level, subject to cash and maximum share limits.",
            "A sell can only close a lot originally bought one grid step below that sell level.",
            "Dividend cash is credited from the database dividend field to shares held at the prior close. If the source lacks dividend events, total returns may be understated.",
            "The daily-bar model does not infer intraday high/low order, which avoids optimistic intraday path assumptions but can miss valid intraday grid fills.",
            "This is a prototype backtest on the currently available database window, not evidence of long-term out-of-sample performance.",
            "Taxes, market impact, partial fills, bid-ask spread dynamics, slippage variation, corporate action reconciliation and broker constraints are not fully modeled.",
        ],
    }

    return {
        "summary": summary,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        # Create an empty file with a clear header so downstream tooling can find it.
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.2f}{suffix}"


def _write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    timestamp: str,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    ticker = summary["ticker"]
    stem = f"{ticker}_grid_backtest_{timestamp}"
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    trades_path = output_dir / f"{stem}_trades.csv"
    equity_path = output_dir / f"{stem}_equity.csv"

    grid = summary["grid_strategy"]
    bh = summary["buy_and_hold"]
    params = summary["parameters"]
    lines = [
        f"# {ticker} 网格交易回测报告",
        "",
        f"- 回测期间：{summary['start_date']} 至 {summary['end_date']}",
        f"- 行情来源：{summary['data_source']}",
        f"- 有效日线：{summary['bars']} 条；自然日跨度：{summary['calendar_days']} 天",
        "",
        "> 本报告是基于历史日线数据的策略模拟，不构成投资建议，也不代表未来表现。",
        "",
        "## 回测参数",
        "",
        f"- 初始资金：${params['initial_cash']:,.2f}",
        f"- 网格下限 / 上限：${params['grid_lower']:,.2f} / ${params['grid_upper']:,.2f}",
        f"- 网格间距：${params['grid_step']:,.2f}",
        f"- 每个网格订单目标金额：${params['cash_per_grid_order']:,.2f}",
        f"- 手续费率：{params['fee_rate'] * 100:.3f}% / 每笔成交额",
        f"- 单边滑点：{params['slippage_bps_each_side']:.1f} bps",
        f"- 最大持仓约束：最多 {params['max_shares_at_upper_bound']} 股（按上限价格约束，限制区间内买入敞口）",
        "",
        "## 绩效结果",
        "",
        "| 指标 | 网格策略 | 买入并持有基准 |",
        "|---|---:|---:|",
        f"| 期末权益 | ${grid['final_equity']:,.2f} | ${bh['final_equity']:,.2f} |",
        f"| 总收益率 | {_format_metric(grid['total_return_pct'], '%')} | {_format_metric(bh['total_return_pct'], '%')} |",
        f"| 年化收益率（CAGR） | {_format_metric(grid['cagr_pct'], '%')} | {_format_metric(bh['cagr_pct'], '%')} |",
        f"| 最大回撤 | {_format_metric(grid['max_drawdown_pct'], '%')} | {_format_metric(bh['max_drawdown_pct'], '%')} |",
        f"| 年化波动率 | {_format_metric(grid['annualized_volatility_pct'], '%')} | {_format_metric(bh['annualized_volatility_pct'], '%')} |",
        f"| 平均股票仓位 | {_format_metric(grid['average_position_exposure_pct'], '%')} | {_format_metric(bh['average_position_exposure_pct'], '%')} |",
        f"| 已成交买单 / 卖单 | {grid['buy_orders_filled']} / {grid['sell_orders_filled']} | N/A |",
        f"| 成交订单总数 | {grid['filled_orders_total']} | 1 笔初始买入 |",
        f"| 手续费 | ${grid['fees_paid']:,.2f} | ${bh['entry_fee_paid']:,.2f}（初始买入） |",
        f"| 初始滑点成本估计 | 含在逐笔成交价中 | ${bh['entry_slippage_cost_estimate']:,.2f} |",
        f"| 获得现金分红 | ${grid['dividends_received']:,.2f} | 已计入基准现金 |",
        f"| 期末现金 / 持仓 | ${grid['ending_cash']:,.2f} / {grid['ending_shares']} 股 | ${bh['ending_cash_including_dividends']:,.2f} / {bh['buy_shares_at_first_close']} 股 |",
        "",
        "## 成交与执行规则",
        "",
        f"- 产生信号 {grid['signals_generated']} 次；次日开盘条件不再满足而取消 {grid['orders_cancelled_on_next_open']} 次；因资金、库存或单格限制未成交 {grid['orders_rejected_for_cash_inventory_or_lot']} 次。",
        "- 交易信号只在收盘后根据收盘价穿越网格时生成，最早在下一个交易日开盘成交。",
        "- 买入信号仅在次日开盘价仍不高于触发价时执行；卖出信号仅在次日开盘价仍不低于触发价时执行。",
        "- 买入按触发网格层对应的库存记账；只有持有在低一格买入的库存时，才会执行对应的上穿卖出。",
        "- 按整股交易。手续费和不利滑点均纳入网格策略成交成本。",
        "",
        "## 结果解读注意事项",
        "",
        "- 网格从全现金开始，买入并持有基准则在首日收盘买入尽可能多的整股；基准已计入初始买入的手续费和不利滑点，但两者市场暴露仍不同。",
        "- 平均股票仓位用于解释收益差异，不是事前已知的参数，也不能单独证明策略更优。",
        "- 收盘信号在次日开盘执行，因此回测不会假设知道某根日线内部先触发最高价还是最低价；代价是会遗漏部分日内网格交易。",
        "- 如果价格持续单边上涨，网格可能持有过多现金；持续下跌时则可能不断累积持仓并耗尽资金。",
        "- 当前数据库仅包含上述日期范围，回测结果不能替代跨牛市、熊市与震荡期的长期测试。",
        "- 税费、买卖价差的动态变化、市场冲击、部分成交和数据修订未被完整模拟。",
        "",
        "## 数据文件",
        "",
        f"- 成交记录：`{trades_path.name}`",
        f"- 每日权益曲线：`{equity_path.name}`",
        f"- 完整机器可读结果：`{json_path.name}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(trades_path, result["trades"])
    _write_csv(equity_path, result["equity_curve"])
    return md_path, json_path, trades_path, equity_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily-bar grid strategy backtest using SQLite market_daily or an optional validated CSV."
    )
    parser.add_argument("--ticker", default="QQQM", help="Ticker symbol (default: QQQM)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--data-file", type=Path, default=None, help="Optional validated daily OHLCV CSV (e.g. data/QQQM_tiingo_history.csv); overrides SQLite as the price source.")
    parser.add_argument("--initial-cash", type=_positive_float, default=10_000.0, help="Initial cash, default 10000")
    parser.add_argument("--lower", type=_positive_float, default=280.0, help="Lower grid bound, default 280")
    parser.add_argument("--upper", type=_positive_float, default=330.0, help="Upper grid bound, default 330")
    parser.add_argument("--step", type=_positive_float, default=5.0, help="Dollar step between grid lines, default 5")
    parser.add_argument("--cash-per-grid", type=_positive_float, default=1_000.0, help="Target notional per buy order, default 1000")
    parser.add_argument("--fee-rate", type=_nonnegative_float, default=0.001, help="Fee as fraction of notional, default 0.001 = 0.1%%")
    parser.add_argument("--slippage-bps", type=_nonnegative_float, default=5.0, help="Adverse slippage per side in basis points, default 5")
    parser.add_argument("--max-position-value", type=_positive_float, default=None, help="Maximum intended grid exposure; derives a conservative whole-share cap using the upper grid bound; default equals initial cash")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR, help=f"Output directory (default: {DEFAULT_REPORT_DIR})")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("--ticker cannot be empty")

    max_position_value = args.max_position_value
    if max_position_value is None:
        max_position_value = args.initial_cash

    try:
        bars = _read_price_bars(args.db, ticker, data_file=args.data_file)
        result = run_backtest(
            bars,
            initial_cash=args.initial_cash,
            lower=args.lower,
            upper=args.upper,
            step=args.step,
            cash_per_grid=args.cash_per_grid,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            max_position_value=max_position_value,
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = _write_outputs(result, args.output_dir, stamp)
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"[BACKTEST ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = result["summary"]
    grid = summary["grid_strategy"]
    bh = summary["buy_and_hold"]
    print(f"Ticker: {summary['ticker']}")
    print(f"Data: {summary['start_date']} -> {summary['end_date']} ({summary['bars']} bars; {summary['data_source']})")
    print(
        f"Grid: ${args.lower:.2f} - ${args.upper:.2f}, step ${args.step:.2f}; "
        f"cash/grid ${args.cash_per_grid:,.2f}; fee {args.fee_rate * 100:.3f}%; "
        f"slippage {args.slippage_bps:.1f} bps/side"
    )
    print("\nResults")
    print(f"  Grid final equity:       ${grid['final_equity']:,.2f}")
    print(f"  Grid total return:       {grid['total_return_pct']:.2f}%")
    print(f"  Grid max drawdown:       {grid['max_drawdown_pct']:.2f}%")
    print(f"  Grid annualized vol:     {_format_metric(grid['annualized_volatility_pct'], '%')}")
    print(f"  Grid avg stock exposure: {_format_metric(grid['average_position_exposure_pct'], '%')}")
    print(f"  Grid filled buy/sell:    {grid['buy_orders_filled']} / {grid['sell_orders_filled']}")
    print(f"  Grid fees paid:          ${grid['fees_paid']:,.2f}")
    print(f"  Buy-hold final equity:   ${bh['final_equity']:,.2f}")
    print(f"  Buy-hold total return:   {bh['total_return_pct']:.2f}%")
    print(f"  Buy-hold max drawdown:   {bh['max_drawdown_pct']:.2f}%")
    print(f"  Buy-hold annualized vol: {_format_metric(bh['annualized_volatility_pct'], '%')}")
    print(f"  Buy-hold avg exposure:   {_format_metric(bh['average_position_exposure_pct'], '%')}")
    print(f"  Buy-hold entry fee:      ${bh['entry_fee_paid']:,.2f}")
    print(f"  Buy-hold entry slippage: ${bh['entry_slippage_cost_estimate']:,.2f}")
    print("\nSaved files")
    for path in paths:
        print(f"  {path.resolve()}")
    print("\nNote: daily-close signal -> next-session-open execution; this is a research prototype, not trading advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
