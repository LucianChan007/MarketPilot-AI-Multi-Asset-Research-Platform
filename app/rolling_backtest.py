from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.backtest import DEFAULT_DB, DEFAULT_REPORT_DIR, PriceBar, _read_price_bars


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return value


def _nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return value


def _max_drawdown(values: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def _annualized_volatility(values: list[float]) -> float | None:
    returns = [cur / prev - 1.0 for prev, cur in zip(values, values[1:]) if prev > 0]
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(252.0)


def _cagr(initial: float, final: float, first_date: str, last_date: str) -> float | None:
    days = (date.fromisoformat(last_date) - date.fromisoformat(first_date)).days
    if days <= 0 or initial <= 0 or final <= 0:
        return None
    return (final / initial) ** (365.25 / days) - 1.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_rolling_backtest(
    bars: list[PriceBar],
    *,
    initial_cash: float,
    lookback: int,
    step_pct: float,
    levels: int,
    cash_per_grid: float,
    fee_rate: float,
    slippage_bps: float,
    max_position_value: float,
    trade_start_date: str | None = None,
    trade_end_date: str | None = None,
) -> dict[str, Any]:
    if len(bars) <= lookback + 2:
        raise ValueError(f"Need more than lookback + 2 bars; got {len(bars)} with lookback {lookback}.")
    if lookback < 2:
        raise ValueError("lookback must be at least 2 trading bars")
    if levels < 1 or levels > 20:
        raise ValueError("levels must be between 1 and 20")
    if step_pct <= 0 or step_pct * levels >= 0.8:
        raise ValueError("Require step_pct > 0 and step_pct * levels < 0.8")
    if min(initial_cash, cash_per_grid, max_position_value) <= 0:
        raise ValueError("initial_cash, cash_per_grid and max_position_value must be > 0")
    if fee_rate < 0 or fee_rate >= 0.1:
        raise ValueError("fee_rate must be between 0 and 0.1")
    if slippage_bps < 0 or slippage_bps >= 1000:
        raise ValueError("slippage_bps must be between 0 and 1000")

    for label, raw_date in (("trade_start_date", trade_start_date), ("trade_end_date", trade_end_date)):
        if raw_date is not None:
            try:
                date.fromisoformat(raw_date)
            except ValueError as exc:
                raise ValueError(f"{label} must use YYYY-MM-DD format") from exc

    active_start_idx = 0
    if trade_start_date is not None:
        active_start_idx = next(
            (i for i, bar in enumerate(bars) if bar.trade_date >= trade_start_date),
            -1,
        )
        if active_start_idx < 0:
            raise ValueError(f"No bar on or after start date {trade_start_date}")
        if active_start_idx < lookback:
            raise ValueError(
                f"Start date {bars[active_start_idx].trade_date} has only "
                f"{active_start_idx} prior bars; at least {lookback} are required "
                "to warm up the rolling reference without using test-period data."
            )

    active_end_idx = len(bars) - 1
    if trade_end_date is not None:
        eligible = [i for i, bar in enumerate(bars) if bar.trade_date <= trade_end_date]
        if not eligible:
            raise ValueError(f"No bar on or before end date {trade_end_date}")
        active_end_idx = eligible[-1]

    if active_end_idx <= active_start_idx:
        raise ValueError("End date must be after start date and include at least two bars")

    slip = slippage_bps / 10_000.0
    cash = float(initial_cash)
    lots: dict[int, dict[str, Any]] = {}
    next_lot_id = 1
    pending: list[dict[str, Any]] = []
    pending_sell_ids: set[int] = set()
    armed = {tier: True for tier in range(1, levels + 1)}
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    fees_paid = 0.0
    dividends_received = 0.0
    shares_bought = 0
    shares_sold = 0
    signals_generated = 0
    orders_cancelled = 0
    orders_rejected = 0
    max_shares_held = 0
    realized_pnl = 0.0

    # Cost-aware buy-and-hold baseline: purchase whole shares at first close,
    # then retain the shares and cash dividends. Does not rebalance.
    # The benchmark begins on the first evaluated/trading day, not during the
    # indicator warm-up period. The warm-up bars provide history for the SMA only.
    first = bars[active_start_idx]
    bh_fill = first.close * (1.0 + slip)
    bh_shares = math.floor(initial_cash / (bh_fill * (1.0 + fee_rate)))
    bh_notional = bh_shares * bh_fill
    bh_entry_fee = bh_notional * fee_rate
    bh_entry_slippage = bh_shares * (bh_fill - first.close)
    bh_cash = initial_cash - bh_notional - bh_entry_fee

    # Store the trailing reference used to evaluate each day's signal. Ref[i]
    # is derived strictly from bars before i, excluding bar i itself.
    refs: list[float | None] = [None] * len(bars)
    deviations: list[float | None] = [None] * len(bars)
    for i in range(lookback, len(bars)):
        window = bars[i - lookback:i]
        ref = statistics.mean(item.close for item in window)
        refs[i] = ref
        deviations[i] = bars[i].close / ref - 1.0

    for i, bar in enumerate(bars[: active_end_idx + 1]):
        # The bars before active_start_idx are indicator warm-up only. No portfolio
        # is active and the passive benchmark has not been purchased yet.
        if i < active_start_idx:
            continue

        # Dividends are credited only after the first evaluated session to shares
        # already held before that session's open fills. Shares purchased at the
        # test-start close do not receive that day's ex-dividend distribution.
        if i > active_start_idx and bar.dividends > 0:
            held_before_open = sum(lot["shares"] for lot in lots.values())
            dividend = held_before_open * bar.dividends
            cash += dividend
            dividends_received += dividend
            bh_cash += bh_shares * bar.dividends

        # Execute previous-close signals at today's open, subject to limit checks.
        if i > 0 and pending:
            todays = pending
            pending = []
            todays.sort(key=lambda order: (0 if order["side"] == "BUY" else 1, order["trigger_price"]))
            for order in todays:
                trigger = float(order["trigger_price"])
                if order["side"] == "BUY":
                    if bar.open > trigger + 1e-9:
                        orders_cancelled += 1
                        continue
                    fill = bar.open * (1.0 + slip)
                    held_shares = sum(lot["shares"] for lot in lots.values())
                    current_position_value = held_shares * bar.open
                    available_value = max(0.0, max_position_value - current_position_value)
                    available_position_shares = math.floor(available_value / fill)
                    budget_shares = math.floor(cash_per_grid / fill)
                    cash_shares = math.floor(cash / (fill * (1.0 + fee_rate)))
                    quantity = min(available_position_shares, budget_shares, cash_shares)
                    if quantity <= 0:
                        orders_rejected += 1
                        continue
                    notional = quantity * fill
                    fee = notional * fee_rate
                    total_cost = notional + fee
                    if total_cost > cash + 1e-8:
                        orders_rejected += 1
                        continue
                    cash -= total_cost
                    fees_paid += fee
                    shares_bought += quantity
                    lot_id = next_lot_id
                    next_lot_id += 1
                    target = fill * (1.0 + step_pct)
                    lots[lot_id] = {
                        "lot_id": lot_id,
                        "tier": int(order["tier"]),
                        "shares": quantity,
                        "cost_basis": total_cost,
                        "buy_date": bar.trade_date,
                        "buy_price": fill,
                        "target_sell_price": target,
                    }
                    pending_sell_ids.discard(lot_id)
                    held_after = sum(lot["shares"] for lot in lots.values())
                    max_shares_held = max(max_shares_held, held_after)
                    trades.append({
                        "date": bar.trade_date,
                        "side": "BUY",
                        "tier": order["tier"],
                        "trigger_price": trigger,
                        "fill_price": fill,
                        "target_sell_price": target,
                        "shares": quantity,
                        "notional": notional,
                        "fee": fee,
                        "realized_pnl": 0.0,
                        "cash_after": cash,
                        "held_shares_after": held_after,
                        "reason": "close crossed below trailing-SMA percentage band; next-open buy limit remained eligible",
                    })
                else:  # SELL
                    lot_id = int(order["lot_id"])
                    lot = lots.get(lot_id)
                    pending_sell_ids.discard(lot_id)
                    if lot is None:
                        orders_rejected += 1
                        continue
                    if bar.open < float(lot["target_sell_price"]) - 1e-9:
                        orders_cancelled += 1
                        continue
                    fill = bar.open * (1.0 - slip)
                    notional = int(lot["shares"]) * fill
                    fee = notional * fee_rate
                    proceeds = notional - fee
                    pnl = proceeds - float(lot["cost_basis"])
                    cash += proceeds
                    fees_paid += fee
                    shares_sold += int(lot["shares"])
                    realized_pnl += pnl
                    del lots[lot_id]
                    held_after = sum(item["shares"] for item in lots.values())
                    trades.append({
                        "date": bar.trade_date,
                        "side": "SELL",
                        "lot_id": lot_id,
                        "tier": lot["tier"],
                        "trigger_price": lot["target_sell_price"],
                        "fill_price": fill,
                        "shares": lot["shares"],
                        "notional": notional,
                        "fee": fee,
                        "realized_pnl": pnl,
                        "cash_after": cash,
                        "held_shares_after": held_after,
                        "reason": "lot reached percentage rebound target; next-open sell remained eligible",
                    })

        held_shares = sum(lot["shares"] for lot in lots.values())
        position_value = held_shares * bar.close
        equity = cash + position_value
        bh_equity = bh_cash + bh_shares * bar.close
        equity_curve.append({
            "date": bar.trade_date,
            "close": bar.close,
            "reference_sma_prior": refs[i],
            "close_deviation_from_reference_pct": (deviations[i] * 100.0) if deviations[i] is not None else None,
            "cash": cash,
            "shares": held_shares,
            "position_value": position_value,
            "equity": equity,
            "buy_hold_cash": bh_cash,
            "buy_hold_shares": bh_shares,
            "buy_hold_position_value": bh_shares * bar.close,
            "buy_hold_equity": bh_equity,
        })

        # Signal generation after today's close. The reference only uses prior bars.
        # It is a moving-average-anchored percentage grid, not the same strategy as
        # the fixed-dollar grid in app.backtest.
        if i < active_end_idx and i >= lookback:
            ref = refs[i]
            dev = deviations[i]
            if ref is not None and dev is not None:
                prev_dev = deviations[i - 1] if i > lookback else None
                for tier in range(1, levels + 1):
                    threshold_dev = -(tier * step_pct)
                    if dev > threshold_dev:
                        armed[tier] = True
                    elif armed[tier] and prev_dev is not None and prev_dev > threshold_dev and dev <= threshold_dev:
                        trigger_price = ref * (1.0 + threshold_dev)
                        pending.append({
                            "side": "BUY",
                            "trigger_price": trigger_price,
                            "tier": tier,
                        })
                        armed[tier] = False
                        signals_generated += 1

                # Keep trying to exit lots whose target has been reached at a close.
                # If next-open eligibility fails and price remains above target, retry
                # after the next close instead of leaving the lot stranded.
                for lot_id, lot in list(lots.items()):
                    if bar.close >= float(lot["target_sell_price"]) and lot_id not in pending_sell_ids:
                        pending.append({
                            "side": "SELL",
                            "trigger_price": float(lot["target_sell_price"]),
                            "lot_id": lot_id,
                        })
                        pending_sell_ids.add(lot_id)
                        signals_generated += 1

    grid_equities = [point["equity"] for point in equity_curve]
    bh_equities = [point["buy_hold_equity"] for point in equity_curve]
    grid_exposure = [p["position_value"] / p["equity"] * 100.0 for p in equity_curve if p["equity"] > 0]
    bh_exposure = [p["buy_hold_position_value"] / p["buy_hold_equity"] * 100.0 for p in equity_curve if p["buy_hold_equity"] > 0]
    first_date, last_date = bars[active_start_idx].trade_date, bars[active_end_idx].trade_date
    evaluated_bars = bars[active_start_idx : active_end_idx + 1]
    final = equity_curve[-1]
    final_bh = bh_equities[-1]
    cagr_grid = _cagr(initial_cash, float(final["equity"]), first_date, last_date)
    cagr_bh = _cagr(initial_cash, final_bh, first_date, last_date)
    buy_count = sum(t["side"] == "BUY" for t in trades)
    sell_count = sum(t["side"] == "SELL" for t in trades)
    summary: dict[str, Any] = {
        "ticker": bars[0].ticker.upper(),
        "data_source": ", ".join(sorted({bar.source for bar in evaluated_bars if bar.source})) or "unknown",
        "start_date": first_date,
        "end_date": last_date,
        "bars": len(equity_curve),
        "warmup_bars": min(active_start_idx, lookback),
        "parameters": {
            "strategy": "rolling_sma_percentage_grid",
            "initial_cash": initial_cash,
            "lookback_bars": lookback,
            "step_pct": step_pct * 100.0,
            "downside_levels": levels,
            "cash_per_grid_order": cash_per_grid,
            "fee_rate_pct": fee_rate * 100.0,
            "slippage_bps_each_side": slippage_bps,
            "max_position_value": max_position_value,
        },
        "grid_strategy": {
            "final_equity": final["equity"],
            "total_return_pct": (float(final["equity"]) / initial_cash - 1.0) * 100.0,
            "cagr_pct": cagr_grid * 100.0 if cagr_grid is not None else None,
            "max_drawdown_pct": _max_drawdown(grid_equities) * 100.0,
            "annualized_volatility_pct": (_annualized_volatility(grid_equities) or 0.0) * 100.0,
            "average_position_exposure_pct": statistics.mean(grid_exposure) if grid_exposure else 0.0,
            "average_cash_weight_pct": statistics.mean([p["cash"] / p["equity"] * 100.0 for p in equity_curve if p["equity"] > 0]),
            "buy_orders_filled": buy_count,
            "sell_orders_filled": sell_count,
            "signals_generated": signals_generated,
            "orders_cancelled_on_next_open": orders_cancelled,
            "orders_rejected_for_cash_or_position_cap": orders_rejected,
            "fees_paid": fees_paid,
            "dividends_received": dividends_received,
            "realized_pnl_before_dividends": realized_pnl,
            "ending_cash": cash,
            "ending_shares": sum(lot["shares"] for lot in lots.values()),
            "ending_position_value": final["position_value"],
            "max_shares_held": max_shares_held,
        },
        "buy_and_hold": {
            "final_equity": final_bh,
            "total_return_pct": (final_bh / initial_cash - 1.0) * 100.0,
            "cagr_pct": cagr_bh * 100.0 if cagr_bh is not None else None,
            "max_drawdown_pct": _max_drawdown([initial_cash, *bh_equities]) * 100.0,
            "annualized_volatility_pct": (_annualized_volatility([initial_cash, *bh_equities]) or 0.0) * 100.0,
            "average_position_exposure_pct": statistics.mean(bh_exposure) if bh_exposure else 0.0,
            "shares_bought_at_start": bh_shares,
            "entry_fee_paid": bh_entry_fee,
            "entry_slippage_cost_estimate": bh_entry_slippage,
            "dividends_received_in_cash": bh_cash - (initial_cash - bh_notional - bh_entry_fee),
        },
        "assumptions_and_limitations": [
            "This is a rolling percentage grid, a different strategy from the fixed-dollar grid.",
            "Each day's reference SMA uses only the preceding lookback closes and excludes the current close; no future prices are used to calculate the reference.",
            "Buy signals occur when close/reference crosses below a negative percentage band; each buy lot's sell target is its actual fill price times (1 + step_pct).",
            "Signals are generated after the close and execute no earlier than next-session open. Buys are canceled if next open is above their trigger; sells are retried on later closes if the target remains reached.",
            "Raw OHLC prices are used for fills and marking; cash dividends in the input data are separately credited to shares held before the ex-date open.",
            "Whole shares only; fees and adverse per-side slippage included. Taxes, market impact, partial fills and spread dynamics are not fully modeled.",
            "The buy-and-hold benchmark is close to fully invested, so exposure differs from the rolling grid; returns should not be compared without considering exposure.",
            "This prototype is not investment advice and requires further validation on out-of-sample periods and with different cost assumptions.",
        ],
    }
    return {"summary": summary, "trades": trades, "equity_curve": equity_curve}


def _write_outputs(result: dict[str, Any], output_dir: Path, stamp: str) -> list[Path]:
    summary = result["summary"]
    stem = f"{summary['ticker']}_rolling_grid_backtest_{stamp}"
    md = output_dir / f"{stem}.md"
    js = output_dir / f"{stem}.json"
    trades_path = output_dir / f"{stem}_trades.csv"
    equity_path = output_dir / f"{stem}_equity.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = summary["grid_strategy"]
    bh = summary["buy_and_hold"]
    p = summary["parameters"]
    lines = [
        f"# {summary['ticker']} Rolling Percentage-Grid Backtest",
        "",
        f"- Data: {summary['start_date']} to {summary['end_date']} ({summary['bars']} evaluated bars; {summary['data_source']})",
        f"- Indicator warm-up bars excluded from portfolio performance: {summary.get('warmup_bars', 0)}",
        f"- Reference: prior {p['lookback_bars']}-bar SMA; current close excluded",
        f"- Step: {p['step_pct']:.2f}% ; downside levels: {p['downside_levels']}; cash per buy order: ${p['cash_per_grid_order']:,.2f}",
        f"- Initial cash: ${p['initial_cash']:,.2f}; fee: {p['fee_rate_pct']:.3f}%; slippage: {p['slippage_bps_each_side']:.1f} bps per side",
        "",
        "## Results",
        "",
        "| Metric | Rolling grid | Buy and hold |",
        "|---|---:|---:|",
        f"| Final equity | ${grid['final_equity']:,.2f} | ${bh['final_equity']:,.2f} |",
        f"| Total return | {grid['total_return_pct']:.2f}% | {bh['total_return_pct']:.2f}% |",
        f"| CAGR | {grid['cagr_pct']:.2f}% | {bh['cagr_pct']:.2f}% |" if grid["cagr_pct"] is not None and bh["cagr_pct"] is not None else "| CAGR | N/A | N/A |",
        f"| Maximum drawdown | {grid['max_drawdown_pct']:.2f}% | {bh['max_drawdown_pct']:.2f}% |",
        f"| Annualized volatility | {grid['annualized_volatility_pct']:.2f}% | {bh['annualized_volatility_pct']:.2f}% |",
        f"| Average stock exposure | {grid['average_position_exposure_pct']:.2f}% | {bh['average_position_exposure_pct']:.2f}% |",
        f"| Filled buys / sells | {grid['buy_orders_filled']} / {grid['sell_orders_filled']} | N/A |",
        f"| Fees paid | ${grid['fees_paid']:,.2f} | ${bh['entry_fee_paid']:,.2f} entry fee |",
        f"| Dividends received | ${grid['dividends_received']:,.2f} | ${bh['dividends_received_in_cash']:,.2f} |",
        "",
        "## Important assumptions",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["assumptions_and_limitations"])
    lines.extend(["", "## Files", "", f"- Trades: `{trades_path.name}`", f"- Equity curve: `{equity_path.name}`", ""])
    md.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps({"summary": summary, "trades": result["trades"], "equity_curve": result["equity_curve"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(trades_path, result["trades"])
    _write_csv(equity_path, result["equity_curve"])
    return [md, js, trades_path, equity_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest a rolling SMA-anchored percentage grid strategy using daily bars.")
    parser.add_argument("--ticker", default="QQQM")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--data-file", type=Path, default=None, help="Validated CSV; if specified, overrides SQLite.")
    parser.add_argument("--initial-cash", type=_positive_float, default=10_000.0)
    parser.add_argument("--lookback", type=int, default=60, help="Trailing SMA lookback in trading bars; only prior closes are used.")
    parser.add_argument("--step-pct", type=_positive_float, default=0.025, help="Percentage distance per downside tier and rebound target; 0.025 means 2.5%%.")
    parser.add_argument("--levels", type=int, default=4, help="Number of downside buy tiers.")
    parser.add_argument("--cash-per-grid", type=_positive_float, default=1_000.0)
    parser.add_argument("--fee-rate", type=_nonnegative_float, default=0.001, help="Fraction of notional; default 0.001 = 0.1%%.")
    parser.add_argument("--slippage-bps", type=_nonnegative_float, default=5.0)
    parser.add_argument("--max-position-value", type=_positive_float, default=None, help="Maximum intended marked stock exposure; defaults to initial cash.")
    parser.add_argument("--start-date", default=None, help="First date when the portfolio starts trading (YYYY-MM-DD); earlier rows are SMA warm-up only.")
    parser.add_argument("--end-date", default=None, help="Last date to include in evaluation (YYYY-MM-DD).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("--ticker cannot be empty")
    if args.lookback < 2:
        parser.error("--lookback must be at least 2")
    if args.levels < 1 or args.levels > 20:
        parser.error("--levels must be between 1 and 20")
    if args.step_pct * args.levels >= 0.8:
        parser.error("--step-pct * --levels must be < 0.8")
    max_position_value = args.max_position_value if args.max_position_value is not None else args.initial_cash
    try:
        bars = _read_price_bars(args.db, ticker, data_file=args.data_file)
        result = run_rolling_backtest(
            bars,
            initial_cash=args.initial_cash,
            lookback=args.lookback,
            step_pct=args.step_pct,
            levels=args.levels,
            cash_per_grid=args.cash_per_grid,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            max_position_value=max_position_value,
            trade_start_date=args.start_date,
            trade_end_date=args.end_date,
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = _write_outputs(result, args.output_dir, stamp)
    except (OSError, ValueError, RuntimeError, Exception) as exc:
        print(f"[ROLLING BACKTEST ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    s = result["summary"]
    g = s["grid_strategy"]
    b = s["buy_and_hold"]
    p = s["parameters"]
    print(f"Ticker: {s['ticker']}")
    print(f"Data: {s['start_date']} -> {s['end_date']} ({s['bars']} bars; {s['data_source']})")
    print(f"Rolling grid: prior SMA {p['lookback_bars']} bars; step {p['step_pct']:.2f}%; levels {p['downside_levels']}; cash/grid ${p['cash_per_grid_order']:,.2f}")
    print("\nResults")
    print(f"  Grid final equity:       ${g['final_equity']:,.2f}")
    print(f"  Grid total return:       {g['total_return_pct']:.2f}%")
    print(f"  Grid CAGR:               {g['cagr_pct']:.2f}%" if g['cagr_pct'] is not None else "  Grid CAGR:               N/A")
    print(f"  Grid max drawdown:       {g['max_drawdown_pct']:.2f}%")
    print(f"  Grid annualized vol:     {g['annualized_volatility_pct']:.2f}%")
    print(f"  Grid avg stock exposure: {g['average_position_exposure_pct']:.2f}%")
    print(f"  Grid filled buy/sell:    {g['buy_orders_filled']} / {g['sell_orders_filled']}")
    print(f"  Grid fees paid:          ${g['fees_paid']:,.2f}")
    print(f"  Buy-hold final equity:   ${b['final_equity']:,.2f}")
    print(f"  Buy-hold total return:   {b['total_return_pct']:.2f}%")
    print(f"  Buy-hold max drawdown:   {b['max_drawdown_pct']:.2f}%")
    print(f"  Buy-hold annualized vol: {b['annualized_volatility_pct']:.2f}%")
    print(f"  Buy-hold avg exposure:   {b['average_position_exposure_pct']:.2f}%")
    print(f"  Buy-hold entry fee:      ${b['entry_fee_paid']:,.2f}")
    print("\nSaved files")
    for path in paths:
        print(f"  {path.resolve()}")
    print("\nNote: moving reference uses only prior closes. This is a research prototype, not trading advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
