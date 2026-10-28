from __future__ import annotations

"""Daily QQQM forecast feedback and retraining cycle.

Normal mode:
  1. Evaluate previously saved forecasts against newly available adjusted closes.
  2. Merge newer SQLite sessions into long Tiingo history after source alignment.
  3. Refit the baseline/Ridge candidate models on known labels.
  4. Save a timestamped forecast JSON and record it in forecast_feedback.

--evaluate-only performs step 1 only. This is appropriate on weekends, market
holidays, or runs where the QQQM market date did not advance.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.forecast_context_archive import archive_forecast_context  # noqa: E402
from app.forecast_feedback import (  # noqa: E402
    evaluate_pending_forecasts,
    get_feedback_summary,
    record_forecast_payload,
)
from app.forecast_pipeline import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    build_daily_forecast,
)


def load_adjusted_price_history(db_path: str | Path, ticker: str) -> list[dict[str, Any]]:
    """Read actual adjusted closing prices for scoring stored forecasts."""
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"行情数据库不存在：{path}")

    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT trade_date, close, adj_close
            FROM market_daily
            WHERE UPPER(ticker) = UPPER(?)
            ORDER BY trade_date
            """,
            (ticker.upper(),),
        ).fetchall()
    finally:
        connection.close()

    history: list[dict[str, Any]] = []
    for row in rows:
        adjusted = row["adj_close"]
        if adjusted is None or float(adjusted or 0) <= 0:
            adjusted = row["close"]
        try:
            price = float(adjusted)
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        history.append({"date": str(row["trade_date"])[:10], "adjusted_close": price})

    if not history:
        raise ValueError(f"market_daily 中没有 {ticker} 可用于预测复盘的有效价格。")
    return history


def save_forecast(payload: dict[str, Any], output_dir: str | Path, db_path: str | Path) -> Path:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    ticker = str(payload.get("ticker", "QQQM")).upper()
    output_path = folder / f"{ticker}_return_forecast_latest_{stamp}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)

    result = record_forecast_payload(
        payload,
        db_path=db_path,
        source_file=str(output_path.resolve()),
        ticker=ticker,
    )
    print(f"新预测文件：{output_path.resolve()}")
    print(f"预测记录：{json.dumps(result, ensure_ascii=False)}")

    # Archive the explanatory context next to the forecast. The first capture is
    # marked live; if an older prediction lacks a snapshot, reconstructing it is
    # allowed but its news features remain explicitly ineligible for training.
    capture_mode = "live_prediction" if result.get("inserted") else "reconstructed"
    context_archive = archive_forecast_context(
        payload,
        db_path=db_path,
        source_file=str(output_path.resolve()),
        capture_mode=capture_mode,
    )
    print(f"预测时点上下文归档：{json.dumps(context_archive, ensure_ascii=False)}")
    return output_path


def print_cycle_summary(db_path: str | Path, ticker: str) -> None:
    summary = get_feedback_summary(db_path=db_path, ticker=ticker, last_n=30)
    print("最近预测反馈汇总（最近最多 30 个已结算预测）:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("提示：小样本指标波动很大。反馈表用于监控模型，不会仅凭单日结果自动更换模型或触发交易。")


def main() -> int:
    parser = argparse.ArgumentParser(description="QQQM 每日预测复盘、模型拟合与预测记录")
    parser.add_argument("--ticker", default="QQQM", help="默认 QQQM")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="行情 SQLite 数据库路径")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="长期历史调整后行情 CSV 路径")
    parser.add_argument("--forecast-dir", type=Path, default=ROOT / "reports" / "forecast")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="只结算已有预测，不生成新预测；适用于行情日期没有更新的运行",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    try:
        actual_prices = load_adjusted_price_history(args.db, ticker)
        evaluated = evaluate_pending_forecasts(actual_prices, db_path=args.db, ticker=ticker)
        print(f"已结算预测数：{len(evaluated)}")
        for item in evaluated:
            print(
                f"  {item['as_of_date']} -> {item['target_date']}: "
                f"实际收盘 ${item['actual_adjusted_close']:.4f}, "
                f"实际收益 {item['actual_return_pct']:+.4f}%"
            )

        if not args.evaluate_only:
            payload = build_daily_forecast(
                csv_path=args.csv,
                db_path=args.db,
                ticker=ticker,
                ridge_alpha=args.ridge_alpha,
            )
            payload["generated_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
            payload["actual_price_history_latest_date"] = actual_prices[-1]["date"]
            save_forecast(payload, args.forecast_dir, args.db)
            print("最新预测（仅为研究估计）:")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("已按 --evaluate-only 跳过新预测生成。")

        print_cycle_summary(args.db, ticker)
        return 0
    except Exception as exc:
        print(f"预测反馈周期失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
