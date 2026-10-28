from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.backtest import DEFAULT_DB, DEFAULT_REPORT_DIR, _read_price_bars, run_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed exploratory design: every combination is run against the same bars,
# initial capital, fees, slippage, and max-position assumption.
RANGES: tuple[tuple[float, float], ...] = (
    (270.0, 340.0),
    (280.0, 330.0),
    (290.0, 330.0),
    (280.0, 320.0),
)
STEPS: tuple[float, ...] = (2.5, 5.0, 10.0)
CASH_PER_GRID_VALUES: tuple[float, ...] = (500.0, 1000.0)


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _is_pareto_efficient(rows: list[dict[str, Any]]) -> None:
    """Annotate non-dominated parameter sets for return vs. drawdown only.

    This is a descriptive screening metric, not a recommendation. Average
    exposure varies by parameter set, so the points are not exposure-matched.
    """
    for candidate in rows:
        cand_return = candidate["grid_return_pct"]
        cand_dd = candidate["grid_max_drawdown_pct"]
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_return = other["grid_return_pct"]
            other_dd = other["grid_max_drawdown_pct"]
            if (
                other_return >= cand_return
                and other_dd <= cand_dd
                and (other_return > cand_return or other_dd < cand_dd)
            ):
                dominated = True
                break
        candidate["return_drawdown_pareto"] = not dominated


def _run_scan(
    *,
    ticker: str,
    db_path: Path,
    data_file: Path | None,
    initial_cash: float,
    fee_rate: float,
    slippage_bps: float,
    max_position_value: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bars = _read_price_bars(db_path, ticker, data_file=data_file)
    rows: list[dict[str, Any]] = []
    benchmark_reference: dict[str, Any] | None = None
    total_cases = 0
    failed_cases: list[dict[str, str]] = []

    for lower, upper in RANGES:
        for step in STEPS:
            # Validate divisibility before running this parameter set.
            intervals = (upper - lower) / step
            if not math.isclose(intervals, round(intervals), rel_tol=1e-9, abs_tol=1e-8):
                continue
            for cash_per_grid in CASH_PER_GRID_VALUES:
                total_cases += 1
                try:
                    result = run_backtest(
                        bars,
                        initial_cash=initial_cash,
                        lower=lower,
                        upper=upper,
                        step=step,
                        cash_per_grid=cash_per_grid,
                        fee_rate=fee_rate,
                        slippage_bps=slippage_bps,
                        max_position_value=max_position_value,
                    )
                    summary = result["summary"]
                    grid = summary["grid_strategy"]
                    bh = summary["buy_and_hold"]
                    params = summary["parameters"]

                    if benchmark_reference is None:
                        benchmark_reference = {
                            "total_return_pct": bh["total_return_pct"],
                            "max_drawdown_pct": bh["max_drawdown_pct"],
                            "annualized_volatility_pct": bh["annualized_volatility_pct"],
                            "average_position_exposure_pct": bh["average_position_exposure_pct"],
                            "final_equity": bh["final_equity"],
                            "entry_fee_paid": bh["entry_fee_paid"],
                            "entry_slippage_cost_estimate": bh["entry_slippage_cost_estimate"],
                        }

                    rows.append({
                        "ticker": summary["ticker"],
                        "data_source": summary["data_source"],
                        "start_date": summary["start_date"],
                        "end_date": summary["end_date"],
                        "bars": summary["bars"],
                        "initial_cash": params["initial_cash"],
                        "grid_lower": params["grid_lower"],
                        "grid_upper": params["grid_upper"],
                        "grid_step": params["grid_step"],
                        "cash_per_grid_order": params["cash_per_grid_order"],
                        "fee_rate_pct": params["fee_rate"] * 100.0,
                        "slippage_bps_each_side": params["slippage_bps_each_side"],
                        "max_position_value_assumption": params["max_position_value_assumption"],
                        "grid_final_equity": grid["final_equity"],
                        "grid_return_pct": grid["total_return_pct"],
                        "grid_cagr_pct": grid["cagr_pct"],
                        "grid_max_drawdown_pct": grid["max_drawdown_pct"],
                        "grid_annualized_volatility_pct": grid["annualized_volatility_pct"],
                        "grid_avg_stock_exposure_pct": grid["average_position_exposure_pct"],
                        "grid_avg_cash_weight_pct": grid["average_cash_weight_pct"],
                        "buy_orders_filled": grid["buy_orders_filled"],
                        "sell_orders_filled": grid["sell_orders_filled"],
                        "fees_paid": grid["fees_paid"],
                        "dividends_received": grid["dividends_received"],
                        "ending_cash": grid["ending_cash"],
                        "ending_shares": grid["ending_shares"],
                        "ending_position_value": grid["ending_position_value"],
                        "orders_cancelled": grid["orders_cancelled_on_next_open"],
                        "orders_rejected": grid["orders_rejected_for_cash_inventory_or_lot"],
                        "buy_hold_return_pct": bh["total_return_pct"],
                        "buy_hold_max_drawdown_pct": bh["max_drawdown_pct"],
                        "buy_hold_annualized_volatility_pct": bh["annualized_volatility_pct"],
                        "buy_hold_avg_stock_exposure_pct": bh["average_position_exposure_pct"],
                        "buy_hold_final_equity": bh["final_equity"],
                        "buy_hold_entry_fee": bh["entry_fee_paid"],
                        "buy_hold_entry_slippage_cost": bh["entry_slippage_cost_estimate"],
                        "return_gap_vs_buy_hold_pp": grid["total_return_pct"] - bh["total_return_pct"],
                        "exposure_gap_vs_buy_hold_pp": grid["average_position_exposure_pct"] - bh["average_position_exposure_pct"],
                    })
                except (OSError, sqlite3.Error, RuntimeError, ValueError, KeyError) as exc:
                    failed_cases.append({
                        "lower": str(lower),
                        "upper": str(upper),
                        "step": str(step),
                        "cash_per_grid": str(cash_per_grid),
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    if not rows:
        raise RuntimeError("No parameter combinations completed successfully.")
    _is_pareto_efficient(rows)
    rows.sort(key=lambda row: (row["grid_max_drawdown_pct"], -row["grid_return_pct"]))

    metadata = {
        "ticker": ticker.upper(),
        "database": str(db_path.resolve()) if data_file is None else None,
        "data_file": str(data_file.resolve()) if data_file is not None else None,
        "data_source": rows[0]["data_source"],
        "start_date": rows[0]["start_date"],
        "end_date": rows[0]["end_date"],
        "bars": rows[0]["bars"],
        "initial_cash": initial_cash,
        "fee_rate": fee_rate,
        "slippage_bps_each_side": slippage_bps,
        "max_position_value_assumption": max_position_value,
        "range_candidates": [list(x) for x in RANGES],
        "step_candidates": list(STEPS),
        "cash_per_grid_candidates": list(CASH_PER_GRID_VALUES),
        "cases_requested": total_cases,
        "cases_completed": len(rows),
        "cases_failed": len(failed_cases),
        "buy_and_hold_reference": benchmark_reference,
        "failed_cases": failed_cases,
        "limitations": [
            "Parameter sets have different average market exposure; returns are not exposure-matched.",
            "The parameter scan uses the same available date window and may overfit that period.",
            "Use out-of-sample and multiple market regimes before drawing conclusions.",
            "Daily-close signals execute no earlier than the next session open.",
            "Taxes, market impact, dynamic spread and partial fills are not fully modeled.",
        ],
    }
    return rows, metadata


def _write_outputs(
    rows: list[dict[str, Any]], metadata: dict[str, Any], output_dir: Path
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{metadata['ticker']}_parameter_scan_{stamp}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps({"metadata": metadata, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# {metadata['ticker']} 网格参数敏感性扫描",
        "",
        f"- 回测期间：{metadata['start_date']} 至 {metadata['end_date']}",
        f"- 日线数量：{metadata['bars']}",
        f"- 已完成组合：{metadata['cases_completed']} / {metadata['cases_requested']}",
        f"- 失败组合：{metadata['cases_failed']}",
        f"- 初始资金：${metadata['initial_cash']:,.2f}",
        f"- 手续费：{metadata['fee_rate'] * 100:.3f}% / 成交额；单边滑点 {metadata['slippage_bps_each_side']:.1f} bps",
        "",
        "> 参数扫描用于理解敏感性，不是自动推荐器。当前组合的平均股票仓位不同，结果并非等风险或等仓位比较。",
        "",
        "## 结果（按最大回撤升序，其次按收益降序）",
        "",
        "| 区间 | 间距 | 每格金额 | 收益 | 最大回撤 | 年化波动 | 平均股票仓位 | 成交买/卖 | 手续费 | 相对买持收益差 | Pareto* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        range_label = f"{row['grid_lower']:.0f}-{row['grid_upper']:.0f}"
        lines.append(
            f"| {range_label} | ${row['grid_step']:.2f} | ${row['cash_per_grid_order']:.0f} "
            f"| {row['grid_return_pct']:.2f}% | {row['grid_max_drawdown_pct']:.2f}% "
            f"| {row['grid_annualized_volatility_pct']:.2f}% "
            f"| {row['grid_avg_stock_exposure_pct']:.2f}% "
            f"| {row['buy_orders_filled']}/{row['sell_orders_filled']} "
            f"| ${row['fees_paid']:.2f} | {row['return_gap_vs_buy_hold_pp']:.2f} 个百分点 "
            f"| {'是' if row['return_drawdown_pareto'] else '否'} |"
        )
    lines.extend([
        "",
        "*Pareto 标记仅根据收益和最大回撤筛查：其他参数组合若收益不低且最大回撤不高，则该组合被视为被支配。它没有控制平均股票仓位，因此不是风险调整后排名。",
        "",
        "## 买入持有参考",
        "",
        f"- 收益率：{metadata['buy_and_hold_reference']['total_return_pct']:.2f}%",
        f"- 最大回撤：{metadata['buy_and_hold_reference']['max_drawdown_pct']:.2f}%",
        f"- 年化波动率：{metadata['buy_and_hold_reference']['annualized_volatility_pct']:.2f}%",
        f"- 平均股票仓位：{metadata['buy_and_hold_reference']['average_position_exposure_pct']:.2f}%",
        "",
        "## 局限与使用说明",
        "",
    ])
    for item in metadata["limitations"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        f"- 详细 CSV：`{csv_path.name}`",
        f"- 完整 JSON：`{json_path.name}`",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run grid-parameter sensitivity tests against SQLite market_daily or an optional CSV."
    )
    parser.add_argument("--ticker", default="QQQM", help="Ticker symbol (default: QQQM)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--data-file", type=Path, default=None, help="Optional validated daily OHLCV CSV; overrides SQLite as the price source.")
    parser.add_argument("--initial-cash", type=_positive_float, default=10_000.0, help="Initial cash (default: 10000)")
    parser.add_argument("--fee-rate", type=_nonnegative_float, default=0.001, help="Fee as a fraction of notional (default: 0.001)")
    parser.add_argument("--slippage-bps", type=_nonnegative_float, default=5.0, help="Adverse slippage per side in basis points (default: 5)")
    parser.add_argument("--max-position-value", type=_positive_float, default=None, help="Max grid exposure assumption (default: initial cash)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR, help=f"Output directory (default: {DEFAULT_REPORT_DIR})")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("--ticker cannot be empty")

    max_position_value = args.max_position_value or args.initial_cash
    try:
        rows, metadata = _run_scan(
            ticker=ticker,
            db_path=args.db,
            data_file=args.data_file,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            max_position_value=max_position_value,
        )
        paths = _write_outputs(rows, metadata, args.output_dir)
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"[SCAN ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Ticker: {metadata['ticker']}")
    print(f"Data: {metadata['start_date']} -> {metadata['end_date']} ({metadata['bars']} bars; {metadata['data_source']})")
    print(f"Cases completed: {metadata['cases_completed']} / {metadata['cases_requested']}; failed: {metadata['cases_failed']}")
    print("\nResults (sorted by lower drawdown; exposure is not matched)")
    print(f"{'RANGE':<11} {'STEP':>6} {'CASH/GRID':>10} {'RETURN%':>9} {'MAX DD%':>9} {'AVG EXP%':>10} {'BUY/SELL':>10} {'FEES':>9} {'PARETO':>7}")
    for row in rows:
        label = f"{row['grid_lower']:.0f}-{row['grid_upper']:.0f}"
        print(
            f"{label:<11} {row['grid_step']:>6.2f} {row['cash_per_grid_order']:>10.0f} "
            f"{row['grid_return_pct']:>9.2f} {row['grid_max_drawdown_pct']:>9.2f} "
            f"{row['grid_avg_stock_exposure_pct']:>10.2f} "
            f"{row['buy_orders_filled']:>4}/{row['sell_orders_filled']:<5} "
            f"{row['fees_paid']:>9.2f} {('YES' if row['return_drawdown_pareto'] else 'NO'):>7}"
        )
    print("\nSaved files")
    for path in paths:
        print(f"  {path.resolve()}")
    print("\nWarning: do not choose parameters solely because they performed best on this same 1-year sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
