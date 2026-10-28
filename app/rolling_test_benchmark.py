from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.backtest import (
    DEFAULT_DB,
    DEFAULT_REPORT_DIR,
    _annualized_volatility,
    _cagr,
    _max_drawdown,
    _read_price_bars,
)
from app.rolling_backtest import run_rolling_backtest


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _allocation(raw: str) -> float:
    value = _positive_float(raw)
    if value > 100:
        raise argparse.ArgumentTypeError("allocation must be <= 100 percent")
    return value


def _passive_control(
    bars: list[Any],
    *,
    initial_cash: float,
    allocation_pct: float,
    fee_rate: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """Buy whole shares at first test-day close, then hold with cash dividends."""
    if not bars:
        raise ValueError("No bars supplied for passive control")

    slip = slippage_bps / 10_000.0
    first = bars[0]
    ref_price = float(first.close)
    fill_price = ref_price * (1.0 + slip)
    budget = initial_cash * allocation_pct / 100.0
    shares = math.floor(budget / (fill_price * (1.0 + fee_rate)))
    notional = shares * fill_price
    fee = notional * fee_rate
    cash = initial_cash - notional - fee
    dividends = 0.0
    curve: list[dict[str, Any]] = []

    for i, bar in enumerate(bars):
        if i > 0 and float(bar.dividends or 0.0) > 0:
            paid = shares * float(bar.dividends)
            cash += paid
            dividends += paid
        position = shares * float(bar.close)
        equity = cash + position
        curve.append({
            "date": str(bar.trade_date),
            "close": float(bar.close),
            "cash": cash,
            "shares": shares,
            "position_value": position,
            "equity": equity,
            "exposure_pct": position / equity * 100.0 if equity > 0 else 0.0,
        })

    equities = [float(p["equity"]) for p in curve]
    dates_for_metrics = [initial_cash, *equities]
    final_equity = equities[-1]
    return {
        "strategy": f"passive_initial_allocation_{allocation_pct:g}pct",
        "initial_allocation_pct": allocation_pct,
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial_cash - 1.0) * 100.0,
        "cagr_pct": _cagr(initial_cash, final_equity, bars[0].trade_date, bars[-1].trade_date) * 100.0,
        "max_drawdown_pct": _max_drawdown(dates_for_metrics) * 100.0,
        "annualized_volatility_pct": (_annualized_volatility(dates_for_metrics) or 0.0) * 100.0,
        "average_position_exposure_pct": statistics.mean(p["exposure_pct"] for p in curve),
        "average_cash_weight_pct": statistics.mean(
            p["cash"] / p["equity"] * 100.0 for p in curve if p["equity"] > 0
        ),
        "shares_bought": shares,
        "entry_fee": fee,
        "entry_slippage_cost_estimate": shares * (fill_price - ref_price),
        "fees_paid": fee,
        "dividends_received": dividends,
        "ending_cash": cash,
        "ending_position_value": curve[-1]["position_value"],
        "equity_curve": curve,
    }


def _write_outputs(
    output_dir: Path,
    ticker: str,
    result: dict[str, Any],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ticker}_rolling_test_benchmark_{stamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    curve_path = output_dir / f"{stem}_equity.csv"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "strategy", "total_return_pct", "cagr_pct", "max_drawdown_pct",
            "annualized_volatility_pct", "average_position_exposure_pct",
            "fees_paid", "final_equity",
        ])
        writer.writeheader()
        for row in result["strategies"]:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    curve_rows = []
    for row in result["grid_equity_curve"]:
        curve_rows.append({
            "strategy": "rolling_grid",
            "date": row.get("date"),
            "equity": row.get("equity"),
            "cash": row.get("cash"),
            "position_value": row.get("position_value"),
            "exposure_pct": (
                100 * row.get("position_value", 0) / row.get("equity", 1)
                if row.get("equity", 0) else 0
            ),
        })
    for strategy in result["strategies"][1:]:
        for row in strategy["equity_curve"]:
            curve_rows.append({
                "strategy": strategy["strategy"],
                "date": row["date"],
                "equity": row["equity"],
                "cash": row["cash"],
                "position_value": row["position_value"],
                "exposure_pct": row["exposure_pct"],
            })
    curve_fields = ["strategy", "date", "equity", "cash", "position_value", "exposure_pct"]
    with curve_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows)

    lines = [
        f"# {ticker} Rolling Grid vs Passive Exposure Controls (Test Window)",
        "",
        f"- Test window: {result['start_date']} to {result['end_date']} ({result['bars']} bars)",
        f"- Data source: {result['data_source']}",
        f"- Initial equity: ${result['initial_cash']:,.2f}",
        f"- Rolling grid: SMA lookback {result['parameters']['lookback']}; step {result['parameters']['step_pct']:.2f}%; levels {result['parameters']['levels']}; cash/grid ${result['parameters']['cash_per_grid']:,.2f}",
        f"- Cost assumptions: fee {result['parameters']['fee_rate'] * 100:.3f}% per fill, adverse slippage {result['parameters']['slippage_bps']:.1f} bps per side",
        "",
        "## Results",
        "",
        "| Strategy | Total return | CAGR | Max drawdown | Annualized vol | Average exposure | Fees | Final equity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["strategies"]:
        lines.append(
            f"| {row['strategy']} | {row['total_return_pct']:.2f}% | {row['cagr_pct']:.2f}% | "
            f"{row['max_drawdown_pct']:.2f}% | {row['annualized_volatility_pct']:.2f}% | "
            f"{row['average_position_exposure_pct']:.2f}% | ${row['fees_paid']:.2f} | ${row['final_equity']:.2f} |"
        )
    lines.extend([
        "",
        "## Methodology and caveats",
        "",
        "- The rolling grid uses the full CSV only to warm up the trailing reference; its portfolio starts on the test start date.",
        "- Passive controls buy the chosen percentage of initial equity at the first test-day close, then hold the shares and retain cash. They are not rebalanced, so their exposure drifts as prices move.",
        "- Grid and passive controls use the same fee-rate and per-side adverse-slippage assumptions. Grid performs more transactions; passive controls pay entry cost only.",
        "- Dividends from recorded cash-dividend fields are credited after the initial entry day. Data-source dividend completeness is assumed and should be verified.",
        "- Passive allocation controls help contextualize exposure, but are not exactly exposure-matched because the grid's position changes dynamically.",
        "- This is a time-split robustness comparison on a historical sample already examined in earlier experiments, not a pristine blind test or investment advice.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return [md_path, json_path, csv_path, curve_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare rolling-grid OOS results with passive initial-allocation controls.")
    parser.add_argument("--ticker", default="QQQM")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--step-pct", type=_positive_float, default=2.5, help="Grid step in percent, e.g. 2.5 for 2.5%.")
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--cash-per-grid", type=_positive_float, default=1000.0)
    parser.add_argument("--initial-cash", type=_positive_float, default=10000.0)
    parser.add_argument("--fee-rate", type=_nonnegative_float, default=0.001)
    parser.add_argument("--slippage-bps", type=_nonnegative_float, default=5.0)
    parser.add_argument("--max-position-value", type=_positive_float, default=None)
    parser.add_argument("--allocations", type=_allocation, nargs="+", default=[5.0, 7.5, 10.0, 15.0, 20.0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        date.fromisoformat(args.start_date)
        if args.end_date:
            date.fromisoformat(args.end_date)
    except ValueError:
        parser.error("start-date and end-date must use YYYY-MM-DD")
    if args.lookback < 2:
        parser.error("--lookback must be >= 2")
    if args.levels < 1 or args.levels > 20 or args.step_pct * args.levels >= 80:
        parser.error("Require levels 1..20 and step_pct * levels < 80")

    try:
        bars = _read_price_bars(args.db, args.ticker, data_file=args.data_file)
        active = [b for b in bars if b.trade_date >= args.start_date and (not args.end_date or b.trade_date <= args.end_date)]
        if len(active) < 2:
            raise ValueError("At least two price bars must fall inside the test window")
        if not any(b.trade_date < active[0].trade_date for b in bars[-(len(bars)):]):
            raise ValueError("No pre-test rows available for rolling-reference warm-up")

        max_position_value = args.max_position_value or args.initial_cash
        grid_result = run_rolling_backtest(
            bars,
            initial_cash=args.initial_cash,
            lookback=args.lookback,
            step_pct=args.step_pct / 100.0,
            levels=args.levels,
            cash_per_grid=args.cash_per_grid,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            max_position_value=max_position_value,
            trade_start_date=args.start_date,
            trade_end_date=args.end_date,
        )
        summary = grid_result["summary"]
        if summary["start_date"] != active[0].trade_date:
            raise ValueError("Grid and passive controls resolved different test start dates")
        strategies: list[dict[str, Any]] = [{
            "strategy": "rolling_grid",
            "total_return_pct": summary["grid_strategy"]["total_return_pct"],
            "cagr_pct": summary["grid_strategy"]["cagr_pct"] or 0.0,
            "max_drawdown_pct": summary["grid_strategy"]["max_drawdown_pct"],
            "annualized_volatility_pct": summary["grid_strategy"]["annualized_volatility_pct"],
            "average_position_exposure_pct": summary["grid_strategy"]["average_position_exposure_pct"],
            "fees_paid": summary["grid_strategy"]["fees_paid"],
            "final_equity": summary["grid_strategy"]["final_equity"],
        }]
        for allocation in args.allocations:
            passive = _passive_control(
                active,
                initial_cash=args.initial_cash,
                allocation_pct=allocation,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
            )
            strategies.append(passive)

        result = {
            "ticker": args.ticker.upper(),
            "start_date": summary["start_date"],
            "end_date": summary["end_date"],
            "bars": summary["bars"],
            "data_source": summary["data_source"],
            "initial_cash": args.initial_cash,
            "parameters": {
                "lookback": args.lookback,
                "step_pct": args.step_pct,
                "levels": args.levels,
                "cash_per_grid": args.cash_per_grid,
                "fee_rate": args.fee_rate,
                "slippage_bps": args.slippage_bps,
                "max_position_value": max_position_value,
            },
            "grid_equity_curve": grid_result["equity_curve"],
            "strategies": strategies,
        }
        print(f"Ticker: {args.ticker.upper()}")
        print(f"Test data: {summary['start_date']} -> {summary['end_date']} ({summary['bars']} bars; {summary['data_source']})")
        print(f"Grid parameters: step={args.step_pct:.2f}%; levels={args.levels}; cash/grid=${args.cash_per_grid:.2f}")
        print("\nResults (same test window; passive allocation bought at first test-day close and then held)")
        print(f"{'STRATEGY':<34} {'RETURN%':>9} {'CAGR%':>8} {'MAX DD%':>9} {'VOL%':>8} {'AVG EXP%':>10} {'FEES':>9}")
        for row in strategies:
            print(
                f"{row['strategy']:<34} {row['total_return_pct']:>9.2f} {row['cagr_pct']:>8.2f} "
                f"{row['max_drawdown_pct']:>9.2f} {row['annualized_volatility_pct']:>8.2f} "
                f"{row['average_position_exposure_pct']:>10.2f} {row['fees_paid']:>9.2f}"
            )
        print("\nNote: allocation controls are informative but not exactly risk/exposure matched.")
        paths = _write_outputs(args.output_dir, args.ticker.upper(), result)
        print("\nSaved files")
        for path in paths:
            print(f"  {path.resolve()}")
        return 0
    except Exception as exc:
        print(f"[BENCHMARK ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
