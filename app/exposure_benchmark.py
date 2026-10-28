from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.backtest import (
    DEFAULT_DB,
    DEFAULT_REPORT_DIR,
    _annualized_volatility,
    _cagr,
    _max_drawdown,
    _positive_float,
    _nonnegative_float,
    _read_price_bars,
    run_backtest,
)


def _allocation_percent(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("allocation must be a number from 1 to 100") from exc
    if not math.isfinite(number) or not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("allocation must be from 1 to 100 percent")
    return number


def _run_passive_allocation(
    bars: list[Any],
    *,
    initial_cash: float,
    allocation_pct: float,
    fee_rate: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """Buy whole shares using the target fraction at the first close, then hold."""
    if not bars:
        raise ValueError("No price bars supplied.")

    initial_allocation = allocation_pct / 100.0
    slip_fraction = slippage_bps / 10_000.0
    reference_price = float(bars[0].close)
    fill_price = reference_price * (1.0 + slip_fraction)
    allocation_budget = initial_cash * initial_allocation
    shares = math.floor(allocation_budget / (fill_price * (1.0 + fee_rate)))
    notional = shares * fill_price
    entry_fee = notional * fee_rate
    entry_slippage_cost = shares * (fill_price - reference_price)
    cash = initial_cash - notional - entry_fee
    dividends_received = 0.0
    curve: list[dict[str, Any]] = []

    for index, bar in enumerate(bars):
        # A purchase assumed at the first close does not receive a dividend recorded
        # on the same bar; dividends are credited from later bars onward.
        if index > 0 and float(bar.dividends or 0.0) > 0:
            dividend = shares * float(bar.dividends)
            cash += dividend
            dividends_received += dividend

        position_value = shares * float(bar.close)
        equity = cash + position_value
        curve.append(
            {
                "date": str(bar.trade_date),
                "close": float(bar.close),
                "cash": cash,
                "shares": shares,
                "position_value": position_value,
                "equity": equity,
                "exposure_pct": position_value / equity * 100.0 if equity > 0 else 0.0,
            }
        )

    equities = [float(point["equity"]) for point in curve]
    # Include initial account value so entry costs are reflected in drawdown.
    dd = _max_drawdown([initial_cash, *equities]) * 100.0
    vol = _annualized_volatility([initial_cash, *equities])
    final = curve[-1]
    final_equity = float(final["equity"])
    cagr = _cagr(initial_cash, final_equity, str(bars[0].trade_date), str(bars[-1].trade_date))

    return {
        "strategy": f"passive_initial_allocation_{allocation_pct:g}pct",
        "strategy_type": "passive buy-and-hold with uninvested cash",
        "target_initial_allocation_pct": allocation_pct,
        "initial_equity": initial_cash,
        "allocation_budget": allocation_budget,
        "shares_bought": shares,
        "entry_reference_close": reference_price,
        "entry_fill_price_with_slippage": fill_price,
        "entry_notional": notional,
        "entry_fee": entry_fee,
        "entry_slippage_cost_estimate": entry_slippage_cost,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial_cash - 1.0) * 100.0,
        "cagr_pct": cagr * 100.0 if cagr is not None else None,
        "max_drawdown_pct": dd,
        "annualized_volatility_pct": vol * 100.0 if vol is not None else None,
        "average_position_exposure_pct": statistics.mean(
            point["exposure_pct"] for point in curve
        ),
        "ending_cash": float(final["cash"]),
        "ending_shares": shares,
        "ending_position_value": float(final["position_value"]),
        "dividends_received": dividends_received,
        "fees_paid": entry_fee,
        "equity_curve": curve,
    }


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    # Grid and passive rows have some strategy-specific fields. Use the union
    # of keys so CSV writing does not reject fields from later rows.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    stamp: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker = result["ticker"]
    stem = f"{ticker}_exposure_benchmark_{stamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"

    csv_rows: list[dict[str, Any]] = []
    for item in result["strategies"]:
        csv_rows.append({key: value for key, value in item.items() if key != "equity_curve"})
    _write_csv(csv_path, csv_rows)

    lines = [
        f"# {ticker} Grid vs Initial-Allocation Passive Benchmarks",
        "",
        f"- Period: {result['start_date']} to {result['end_date']} ({result['bars']} daily bars)",
        f"- Data source: {result['data_source']}",
        f"- Initial cash: ${result['initial_cash']:,.2f}",
        f"- Fee: {result['fee_rate'] * 100:.3f}% per fill; adverse slippage: {result['slippage_bps']:.1f} bps per side",
        "",
        "## Results",
        "",
        "| Strategy | Initial stock allocation | Final equity | Total return | Max drawdown | Annualized volatility | Average stock exposure | Fees |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["strategies"]:
        allocation = (
            "dynamic grid"
            if item["strategy"] == "grid"
            else f"{item['target_initial_allocation_pct']:.0f}%"
        )
        lines.append(
            f"| {item['strategy']} | {allocation} | ${item['final_equity']:,.2f} "
            f"| {item['total_return_pct']:.2f}% | {item['max_drawdown_pct']:.2f}% "
            f"| {_fmt(item.get('annualized_volatility_pct'), '%')} "
            f"| {_fmt(item.get('average_position_exposure_pct'), '%')} "
            f"| ${item['fees_paid']:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## Methodology and caveats",
            "",
            "- Passive controls invest the specified fraction of initial equity at the first close, buy whole shares, and hold them; the remainder stays in cash. There is no daily rebalance back to the target allocation, so actual exposure changes with price.",
            "- The grid uses the daily-close signal / next-session-open execution rules in `app.backtest`.",
            "- Both strategies use the same configured proportional fee and adverse-slippage assumptions for purchases; the grid also pays these costs on each subsequent filled order.",
            "- Dividends in the local database are credited as cash on bars after the assumed entry date. If the data source is missing dividend events, total returns may be understated.",
            "- Exposure-matched initial allocations are informative controls, not mathematically identical to the grid's changing exposure. Compare average exposure alongside return and drawdown.",
            "- This is a prototype on the available sample, not proof of long-term out-of-sample performance or investment advice.",
            "",
            f"- CSV: `{csv_path.name}`",
            f"- JSON: `{json_path.name}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    result_for_json = {key: value for key, value in result.items()}
    json_path.write_text(json.dumps(result_for_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare grid performance to passive allocations using SQLite or an optional validated CSV."
    )
    parser.add_argument("--ticker", default="QQQM", help="Ticker symbol (default QQQM)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument("--data-file", type=Path, default=None, help="Optional validated daily OHLCV CSV; overrides SQLite as the price source.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Report output directory")
    parser.add_argument("--initial-cash", type=_positive_float, default=10_000.0)
    parser.add_argument("--lower", type=_positive_float, default=280.0)
    parser.add_argument("--upper", type=_positive_float, default=330.0)
    parser.add_argument("--step", type=_positive_float, default=5.0)
    parser.add_argument("--cash-per-grid", type=_positive_float, default=1_000.0)
    parser.add_argument("--fee-rate", type=_nonnegative_float, default=0.001)
    parser.add_argument("--slippage-bps", type=_nonnegative_float, default=5.0)
    parser.add_argument("--max-position-value", type=_positive_float, default=None)
    parser.add_argument(
        "--allocations",
        nargs="+",
        type=_allocation_percent,
        default=[10.0, 20.0, 25.0, 50.0, 100.0],
        help="Passive initial stock allocation percentages, e.g. --allocations 10 20 25 50 100",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("--ticker cannot be empty")
    max_position_value = args.max_position_value or args.initial_cash

    try:
        bars = _read_price_bars(args.db, ticker, data_file=args.data_file)
        grid_result = run_backtest(
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
        grid_summary = grid_result["summary"]["grid_strategy"]
        strategies: list[dict[str, Any]] = [
            {
                "strategy": "grid",
                "strategy_type": "dynamic grid",
                "target_initial_allocation_pct": None,
                "initial_equity": grid_summary["initial_equity"],
                "final_equity": grid_summary["final_equity"],
                "total_return_pct": grid_summary["total_return_pct"],
                "cagr_pct": grid_summary["cagr_pct"],
                "max_drawdown_pct": grid_summary["max_drawdown_pct"],
                "annualized_volatility_pct": grid_summary["annualized_volatility_pct"],
                "average_position_exposure_pct": grid_summary["average_position_exposure_pct"],
                "ending_cash": grid_summary["ending_cash"],
                "ending_shares": grid_summary["ending_shares"],
                "ending_position_value": grid_summary["ending_position_value"],
                "dividends_received": grid_summary["dividends_received"],
                "fees_paid": grid_summary["fees_paid"],
                "buy_orders_filled": grid_summary["buy_orders_filled"],
                "sell_orders_filled": grid_summary["sell_orders_filled"],
            }
        ]
        for allocation_pct in args.allocations:
            passive = _run_passive_allocation(
                bars,
                initial_cash=args.initial_cash,
                allocation_pct=allocation_pct,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
            )
            passive.pop("equity_curve", None)
            passive["buy_orders_filled"] = 1 if passive["shares_bought"] > 0 else 0
            passive["sell_orders_filled"] = 0
            strategies.append(passive)

        data_source = ", ".join(sorted({bar.source for bar in bars if bar.source})) or "unknown"
        result = {
            "ticker": ticker,
            "data_source": data_source,
            "data_file": str(args.data_file.resolve()) if args.data_file is not None else None,
            "database": str(args.db.resolve()) if args.data_file is None else None,
            "start_date": bars[0].trade_date,
            "end_date": bars[-1].trade_date,
            "bars": len(bars),
            "initial_cash": args.initial_cash,
            "fee_rate": args.fee_rate,
            "slippage_bps": args.slippage_bps,
            "grid_parameters": {
                "lower": args.lower,
                "upper": args.upper,
                "step": args.step,
                "cash_per_grid": args.cash_per_grid,
                "max_position_value": max_position_value,
            },
            "strategies": strategies,
        }
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path, json_path, csv_path = _write_outputs(result, args.output_dir, stamp)

    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"[EXPOSURE BENCHMARK ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Ticker: {ticker}")
    print(f"Data: {bars[0].trade_date} -> {bars[-1].trade_date} ({len(bars)} bars; {data_source})")
    print(f"Grid: ${args.lower:.2f}-${args.upper:.2f}, step ${args.step:.2f}, cash/grid ${args.cash_per_grid:,.2f}")
    print("\nResults (passive allocation is invested at start and then held, not rebalanced)")
    print(f"{'STRATEGY':<27} {'RETURN%':>9} {'MAX DD%':>9} {'VOL%':>9} {'AVG EXP%':>10} {'FEES':>9}")
    for item in strategies:
        print(
            f"{item['strategy']:<27} "
            f"{item['total_return_pct']:>9.2f} "
            f"{item['max_drawdown_pct']:>9.2f} "
            f"{item['annualized_volatility_pct'] if item['annualized_volatility_pct'] is not None else float('nan'):>9.2f} "
            f"{item['average_position_exposure_pct'] if item['average_position_exposure_pct'] is not None else float('nan'):>10.2f} "
            f"{item['fees_paid']:>9.2f}"
        )
    print("\nSaved files")
    print(f"  {md_path.resolve()}")
    print(f"  {json_path.resolve()}")
    print(f"  {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
