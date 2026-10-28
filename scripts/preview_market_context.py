from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.market_context_features import build_market_context_snapshot, write_context_snapshot


def main() -> int:
    snapshot = build_market_context_snapshot()
    print("========== QQQM 市场与新闻上下文预览 ==========")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    latest_path = write_context_snapshot()
    print(f"\n已输出上下文快照：{latest_path}")
    print("提示：当前只生成解释用上下文，不会将未经时间点核验的新闻/宏观数据加入机器学习训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
