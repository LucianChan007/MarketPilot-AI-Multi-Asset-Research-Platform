from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.backtest import DEFAULT_DB, DEFAULT_REPORT_DIR, _read_price_bars
from app.rolling_backtest import run_rolling_backtest, _write_outputs


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


def _iso_date(value: str, label: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format") from exc
    return value


def _candidate_summary(result: dict[str, Any], step: float, levels: int, cash_per_grid: float) -> dict[str, Any]:
    summary = result["summary"]
    grid = summary["grid_strategy"]
    return {
        "step_pct": step * 100.0,
        "levels": levels,
        "cash_per_grid": cash_per_grid,
        "train_start": summary["start_date"],
        "train_end": summary["end_date"],
        "train_bars": summary["bars"],
        "total_return_pct": grid["total_return_pct"],
        "cagr_pct": grid["cagr_pct"],
        "max_drawdown_pct": grid["max_drawdown_pct"],
        "annualized_volatility_pct": grid["annualized_volatility_pct"],
        "average_position_exposure_pct": grid["average_position_exposure_pct"],
        "buy_orders_filled": grid["buy_orders_filled"],
        "sell_orders_filled": grid["sell_orders_filled"],
        "fees_paid": grid["fees_paid"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select rolling-grid parameters on training history, then evaluate once on a later test period."
    )
    parser.add_argument("--ticker", default="QQQM")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--data-file", type=Path, required=True, help="Validated historical CSV, e.g. data/QQQM_tiingo_history.csv")
    parser.add_argument("--train-end", default="2024-12-31", help="Last date available for parameter selection (YYYY-MM-DD).")
    parser.add_argument("--test-start", default="2025-01-01", help="First date on which test portfolio starts trading (YYYY-MM-DD). Prior bars are indicator warm-up only.")
    parser.add_argument("--test-end", default=None, help="Optional last test date (YYYY-MM-DD); defaults to the last CSV date.")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--initial-cash", type=_positive_float, default=10_000.0)
    parser.add_argument("--fee-rate", type=_nonnegative_float, default=0.001)
    parser.add_argument("--slippage-bps", type=_nonnegative_float, default=5.0)
    parser.add_argument("--max-position-value", type=_positive_float, default=None)
    parser.add_argument("--max-train-drawdown", type=_positive_float, default=15.0, help="Maximum training maximum drawdown, in percent, for candidate eligibility.")
    parser.add_argument("--min-train-exposure", type=_nonnegative_float, default=5.0, help="Minimum average stock exposure during training, in percent.")
    parser.add_argument("--max-train-exposure", type=_positive_float, default=30.0, help="Maximum average stock exposure during training, in percent.")
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
    try:
        train_end = _iso_date(args.train_end, "--train-end")
        test_start = _iso_date(args.test_start, "--test-start")
        test_end = _iso_date(args.test_end, "--test-end") if args.test_end else None
    except ValueError as exc:
        parser.error(str(exc))
    if test_start <= train_end:
        parser.error("--test-start must be later than --train-end")
    if test_end is not None and test_end < test_start:
        parser.error("--test-end must be on or after --test-start")
    if args.max_train_exposure <= args.min_train_exposure:
        parser.error("--max-train-exposure must exceed --min-train-exposure")

    max_position_value = args.max_position_value or args.initial_cash

    try:
        bars = _read_price_bars(args.db, ticker, data_file=args.data_file)
        train_bars = [bar for bar in bars if bar.trade_date <= train_end]
        test_bars = [bar for bar in bars if bar.trade_date >= test_start and (test_end is None or bar.trade_date <= test_end)]
        if len(train_bars) <= args.lookback + 2:
            raise ValueError(f"Training period has too few bars: {len(train_bars)}")
        if not test_bars:
            raise ValueError("No price bars fall inside the requested test period")
        if bars.index(test_bars[0]) < args.lookback:
            raise ValueError("Not enough pre-test history to warm up the rolling reference")

        # A small, predeclared candidate grid. It is evaluated on training data only.
        steps = [0.015, 0.025, 0.04]
        levels_list = [3, 4, 6]
        cash_per_grid_list = [500.0, 1000.0]
        candidates: list[dict[str, Any]] = []

        print(f"Ticker: {ticker}")
        print(f"Full data: {bars[0].trade_date} -> {bars[-1].trade_date} ({len(bars)} bars)")
        print(f"Training window: {train_bars[0].trade_date} -> {train_bars[-1].trade_date} ({len(train_bars)} bars)")
        print(f"Test window: {test_bars[0].trade_date} -> {test_bars[-1].trade_date} ({len(test_bars)} bars)")
        print("Training-only parameter selection: 18 candidates")

        for step in steps:
            for levels in levels_list:
                if step * levels >= 0.8:
                    continue
                for cash_per_grid in cash_per_grid_list:
                    result = run_rolling_backtest(
                        train_bars,
                        initial_cash=args.initial_cash,
                        lookback=args.lookback,
                        step_pct=step,
                        levels=levels,
                        cash_per_grid=cash_per_grid,
                        fee_rate=args.fee_rate,
                        slippage_bps=args.slippage_bps,
                        max_position_value=max_position_value,
                    )
                    candidates.append(_candidate_summary(result, step, levels, cash_per_grid))

        eligible = [
            c for c in candidates
            if c["max_drawdown_pct"] <= args.max_train_drawdown
            and args.min_train_exposure <= c["average_position_exposure_pct"] <= args.max_train_exposure
            and c["cagr_pct"] is not None
        ]
        selection_note = (
            f"Selected highest training CAGR among candidates with max drawdown <= {args.max_train_drawdown:.2f}% "
            f"and average exposure between {args.min_train_exposure:.2f}% and {args.max_train_exposure:.2f}%."
        )
        if not eligible:
            eligible = candidates
            selection_note = "WARNING: No candidates met the predeclared training constraints; selected highest training CAGR across all candidates."
            print("WARNING: No candidates met the training risk/exposure constraints; falling back to all candidates.")

        selected = max(
            eligible,
            key=lambda c: (c["cagr_pct"] if c["cagr_pct"] is not None else float("-inf")),
        )

        print("\nTraining results (sorted by CAGR; these are not test results)")
        print(f"{'STEP%':>7} {'LEVELS':>7} {'CASH/GRID':>10} {'CAGR%':>9} {'RETURN%':>9} {'MAX DD%':>9} {'AVG EXP%':>9} {'BUY/SELL':>10}")
        for c in sorted(candidates, key=lambda item: (item["cagr_pct"] or float("-inf")), reverse=True):
            print(
                f"{c['step_pct']:>7.2f} {c['levels']:>7d} {c['cash_per_grid']:>10.0f} "
                f"{(c['cagr_pct'] if c['cagr_pct'] is not None else 0.0):>9.2f} "
                f"{c['total_return_pct']:>9.2f} {c['max_drawdown_pct']:>9.2f} "
                f"{c['average_position_exposure_pct']:>9.2f} "
                f"{c['buy_orders_filled']}/{c['sell_orders_filled']: <5d}"
            )

        print("\nSelected using TRAINING period only:")
        print(f"  Step: {selected['step_pct']:.2f}%")
        print(f"  Levels: {selected['levels']}")
        print(f"  Cash per grid: ${selected['cash_per_grid']:.2f}")
        print(f"  Training CAGR: {selected['cagr_pct']:.2f}%")
        print(f"  Rule: {selection_note}")

        # This uses all earlier bars strictly as indicator warm-up, while the grid
        # portfolio and buy-and-hold benchmark start fresh on the first test date.
        test_result = run_rolling_backtest(
            bars,
            initial_cash=args.initial_cash,
            lookback=args.lookback,
            step_pct=selected["step_pct"] / 100.0,
            levels=int(selected["levels"]),
            cash_per_grid=float(selected["cash_per_grid"]),
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            max_position_value=max_position_value,
            trade_start_date=test_start,
            trade_end_date=test_end,
        )

        paths = _write_outputs(test_result, args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        test_summary = test_result["summary"]
        grid = test_summary["grid_strategy"]
        bh = test_summary["buy_and_hold"]

        print("\nOUT-OF-SAMPLE TEST RESULTS (not used for parameter selection)")
        print(f"  Actual test dates:       {test_summary['start_date']} -> {test_summary['end_date']} ({test_summary['bars']} bars)")
        print(f"  Grid total return:       {grid['total_return_pct']:.2f}%")
        print(f"  Grid CAGR:               {grid['cagr_pct']:.2f}%" if grid["cagr_pct"] is not None else "  Grid CAGR:               N/A")
        print(f"  Grid max drawdown:       {grid['max_drawdown_pct']:.2f}%")
        print(f"  Grid annualized vol:     {grid['annualized_volatility_pct']:.2f}%")
        print(f"  Grid avg stock exposure: {grid['average_position_exposure_pct']:.2f}%")
        print(f"  Grid filled buy/sell:    {grid['buy_orders_filled']} / {grid['sell_orders_filled']}")
        print(f"  Grid fees paid:          ${grid['fees_paid']:,.2f}")
        print(f"  Buy-hold total return:   {bh['total_return_pct']:.2f}%")
        print(f"  Buy-hold max drawdown:   {bh['max_drawdown_pct']:.2f}%")
        print(f"  Buy-hold annualized vol: {bh['annualized_volatility_pct']:.2f}%")
        print(f"  Buy-hold avg exposure:   {bh['average_position_exposure_pct']:.2f}%")

        validation_path = args.output_dir / f"{ticker}_rolling_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ticker": ticker,
            "data_file": str(args.data_file),
            "full_data_start": bars[0].trade_date,
            "full_data_end": bars[-1].trade_date,
            "training_window": {"start": train_bars[0].trade_date, "end": train_bars[-1].trade_date, "bars": len(train_bars)},
            "test_window": {"start": test_summary["start_date"], "end": test_summary["end_date"], "bars": test_summary["bars"]},
            "selection_rule": selection_note,
            "constraints": {
                "max_training_drawdown_pct": args.max_train_drawdown,
                "min_average_exposure_pct": args.min_train_exposure,
                "max_average_exposure_pct": args.max_train_exposure,
            },
            "training_candidates": candidates,
            "selected_parameters": selected,
            "test_summary": test_summary,
        }
        validation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("\nSaved test report files")
        for path in paths:
            print(f"  {path.resolve()}")
        print(f"  {validation_path.resolve()}")
        print("\nIMPORTANT: Because full-period results were already inspected during earlier experiments, treat this as a time-split robustness check, not a pristine blind holdout. Future walk-forward tests should freeze rules before viewing each test period.")
        return 0

    except Exception as exc:
        print(f"[VALIDATION ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
