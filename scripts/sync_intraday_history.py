from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.intraday_data import collect_intraday_history  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="将 Alpaca 历史盘中 K 线保存到独立 SQLite 表。")
    parser.add_argument("--symbol", default="QQQM", help="股票代码，默认 QQQM")
    parser.add_argument("--timeframe", default="5Min", help="K 线间隔，默认 5Min")
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex", help="Alpaca feed，默认 iex")
    parser.add_argument("--lookback-days", type=int, default=90, help="回看自然日数，默认 90")
    parser.add_argument("--db", default=str(ROOT / "data" / "market_data.db"), help="SQLite 数据库路径")
    args = parser.parse_args()

    try:
        result = collect_intraday_history(
            symbol=args.symbol,
            timeframe=args.timeframe,
            feed=args.feed,
            lookback_days=args.lookback_days,
            db_path=Path(args.db),
            env_path=ROOT / ".env",
        )
    except Exception as exc:
        print(f"同步失败：{type(exc).__name__}: {exc}")
        return 1

    print("========== Alpaca 盘中历史同步 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("提示：本次只同步并单独标记 Alpaca feed；没有与 EODHD 日线混合，也没有训练预测模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
