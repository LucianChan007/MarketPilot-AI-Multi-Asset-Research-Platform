from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.news_event_features import DEFAULT_DB_PATH, classify_pending_news  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 DeepSeek 将未标注财经新闻转为缓存的结构化事件特征。")
    parser.add_argument("--limit", type=int, default=10, help="本次最多处理多少篇未标注新闻，默认 10")
    parser.add_argument("--batch-size", type=int, default=8, help="每次 API 请求处理多少篇，范围 1-20")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite 数据库路径")
    parser.add_argument("--model", type=str, default=None, help="覆盖 .env 中的 DEEPSEEK_MODEL")
    args = parser.parse_args()

    result = classify_pending_news(
        db_path=args.db,
        limit=args.limit,
        batch_size=args.batch_size,
        model_override=args.model,
    )
    print("========== 新闻事件结构化分类 ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
