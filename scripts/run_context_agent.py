from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.context_agent import ContextAgentError, run_context_agent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a context-aware QQQM analysis from saved market, news, macro, and forecast data."
    )
    parser.add_argument("--ticker", default="QQQM", help="Ticker symbol (default: QQQM)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Generate a deterministic local briefing without calling DeepSeek.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    try:
        result = run_context_agent(root, ticker=args.ticker.upper(), offline=args.offline)
    except ContextAgentError as exc:
        print(f"上下文 Agent 运行失败：{exc}")
        return 1

    print("========== QQQM 上下文 Agent ==========")
    print(json.dumps({k: v for k, v in result.items() if k != "analysis"}, ensure_ascii=False, indent=2))
    print("\n========== 分析内容 ==========")
    print(result["analysis"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
