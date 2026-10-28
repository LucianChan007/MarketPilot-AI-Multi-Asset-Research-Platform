from __future__ import annotations

"""Run the V2 multi-asset data sync, forecast archiving and due-review workflow."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.multi_asset_engine import REPORT_ROOT, analyze_news_with_deepseek, load_instruments, run_cycle


def main() -> int:
    parser = argparse.ArgumentParser(description="QQQM Research Assistant V2 multi-asset workflow")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols; default uses enabled config")
    parser.add_argument("--skip-refresh", action="store_true", help="Use local cached data only")
    parser.add_argument("--skip-news", action="store_true", help="Skip RSS news collection")
    parser.add_argument("--analyze-news", default="", help="Only run optional DeepSeek news analysis for one symbol")
    args = parser.parse_args()
    try:
        if args.analyze_news:
            result = analyze_news_with_deepseek(args.analyze_news.strip().upper())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else [x["symbol"] for x in load_instruments()]
        result = run_cycle(symbols, refresh=not args.skip_refresh, sync_news_flag=not args.skip_news)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("overall_success") else 1
    except Exception as exc:
        error = {"overall_success": False, "error": f"{type(exc).__name__}: {str(exc)[:800]}", "status_file": str(REPORT_ROOT / "latest_status.json")}
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
