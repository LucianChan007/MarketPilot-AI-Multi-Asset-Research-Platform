
from __future__ import annotations

import argparse
import json
import sys

from app.ai_report import generate_report
from app.config import get_tickers
from app.db import database_status, initialize_database, load_history
from app.indicators import build_summary
from app.market_data import sync_ticker
from app.macro_data import (
    get_macro_snapshot,
    macro_status_rows,
    sync_macro_data,
)


def command_sync(args: argparse.Namespace) -> int:
    initialize_database()

    tickers = (
        [item.upper() for item in args.tickers]
        if args.tickers
        else get_tickers()
    )

    errors = 0

    for ticker in tickers:
        try:
            sync_ticker(ticker, period=args.period)
        except Exception as exc:
            errors += 1
            print(f"[失败] {ticker}: {exc}")

    return 1 if errors == len(tickers) else 0


def command_status(_: argparse.Namespace) -> int:
    rows = database_status()

    if not rows:
        print(
            "数据库中还没有行情。"
            "先执行：python main.py sync --period 1y --tickers QQQM"
        )
        return 0

    print("本地行情数据库状态：")

    for row in rows:
        print(
            f"{row['ticker']:6} | "
            f"{row['rows_count']:5} 条 | "
            f"{row['first_date']} -> {row['last_date']} | "
            f"最近同步：{row['last_fetched_at']}"
        )

    return 0


def command_macro_sync(args: argparse.Namespace) -> int:
    """更新 FRED 宏观数据。"""

    try:
        result = sync_macro_data(years=args.years)
    except Exception as exc:
        print(f"[宏观同步失败] {type(exc).__name__}: {exc}")
        return 1

    for series_id, item in result["results"].items():
        print(
            f"[完成] {series_id}："
            f"本次获取 {item['rows_fetched']} 条，"
            f"最新观测日期 {item['latest_date']}"
        )

    for series_id, message in result["errors"].items():
        print(f"[失败] {series_id}：{message}")

    if not result["results"]:
        return 1

    return 1 if result["errors"] else 0


def command_macro_status(_: argparse.Namespace) -> int:
    """查看本地 FRED 数据覆盖情况。"""

    rows = macro_status_rows()

    print("本地宏观数据状态：")

    for row in rows:
        print(
            f"{row['series_id']:8} | "
            f"{row['row_count']:5} 条 | "
            f"{row['latest_date'] or '暂无数据'} | "
            f"{row['name']}"
        )

    if not any(row["row_count"] > 0 for row in rows):
        print(
            "当前没有宏观数据。"
            "请先执行：python main.py macro-sync --years 10"
        )

    return 0


def command_analyze(args: argparse.Namespace) -> int:
    ticker = args.ticker.upper()

    try:
        history = load_history(ticker)
        summary = build_summary(ticker, history)
    except Exception as exc:
        print(f"[行情分析失败] {type(exc).__name__}: {exc}")
        return 1

    print(f"行情最新日期：{summary['latest_date']}")
    print(f"最新收盘价：{summary['metrics']['close']}")
    print(f"5 日收益率：{summary['metrics']['return_5d_pct']}%")
    print(f"20 日收益率：{summary['metrics']['return_20d_pct']}%")
    print(f"RSI(14)：{summary['metrics']['rsi_14']}")

    if args.with_macro:
        try:
            macro = get_macro_snapshot()
        except Exception as exc:
            print(f"[宏观数据读取失败] {type(exc).__name__}: {exc}")
            return 1

        summary["macro"] = macro

        valid_series = {
            series_id: item
            for series_id, item in macro["series"].items()
            if item.get("latest_value") is not None
        }

        print("\nFRED 宏观数据：")

        for series_id, item in macro["series"].items():
            if item.get("latest_value") is None:
                print(
                    f"[缺失] {series_id}: "
                    f"{item.get('error', '暂无有效观测值')}"
                )
                continue

            print(
                f"{series_id} | "
                f"{item['name']} | "
                f"{item['latest_value']} {item['unit']} | "
                f"观测日期：{item['latest_observation_date']}"
            )

        if not valid_series:
            print(
                "[提示] 尚无可用的 FRED 数据。"
                "请先执行 python main.py macro-sync --years 10"
            )
            return 1

        if args.no_ai:
            print("\n本次不调用 DeepSeek。宏观数据摘要如下：")
            print(json.dumps(macro, ensure_ascii=False, indent=2))
            return 0

    elif args.no_ai:
        print("已完成指标摘要。由于指定 --no-ai，本次不调用 DeepSeek。")
        return 0

    try:
        path = generate_report(ticker, summary)
    except Exception as exc:
        print(f"[AI 报告失败] {exc}")
        return 1

    print(f"[完成] AI 研究报告已保存：{path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "QQQM 个人投资研究助手："
            "历史行情、FRED 宏观数据、SQLite 与 DeepSeek 分析。"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sync_parser = subparsers.add_parser(
        "sync",
        help="下载并更新历史日线",
    )
    sync_parser.add_argument(
        "--period",
        default="1y",
        help="行情范围，例如 1y、5y、max",
    )
    sync_parser.add_argument(
        "--tickers",
        nargs="+",
        help="可选标的列表，例如 QQQM QQQ SPY TLT",
    )
    sync_parser.set_defaults(func=command_sync)

    status_parser = subparsers.add_parser(
        "status",
        help="查看本地行情数据库状态",
    )
    status_parser.set_defaults(func=command_status)

    macro_sync_parser = subparsers.add_parser(
        "macro-sync",
        help="同步 FRED 宏观数据",
    )
    macro_sync_parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="首次同步的历史年数，默认10年",
    )
    macro_sync_parser.set_defaults(func=command_macro_sync)

    macro_status_parser = subparsers.add_parser(
        "macro-status",
        help="查看本地 FRED 宏观数据状态",
    )
    macro_status_parser.set_defaults(func=command_macro_status)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="计算指标并生成 AI 研究报告",
    )
    analyze_parser.add_argument(
        "--ticker",
        default="QQQM",
        help="默认 QQQM",
    )
    analyze_parser.add_argument(
        "--no-ai",
        action="store_true",
        help="只计算指标；结合 --with-macro 可以检查宏观数据",
    )
    analyze_parser.add_argument(
        "--with-macro",
        action="store_true",
        help="把本地 FRED 宏观数据一起纳入分析",
    )
    analyze_parser.set_defaults(func=command_analyze)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
