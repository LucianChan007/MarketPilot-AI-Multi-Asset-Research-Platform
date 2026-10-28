from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.news_event_features import DEFAULT_DB_PATH, FEATURE_VERSION, init_schema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="审计结构化新闻事件特征，不会训练模型。")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    init_schema(args.db)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        source_total = conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0]
        feature_total = conn.execute(
            "SELECT COUNT(*) FROM news_event_features WHERE feature_version=?", (FEATURE_VERSION,)
        ).fetchone()[0]
        categories = conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM news_event_features WHERE feature_version=? GROUP BY event_type ORDER BY n DESC",
            (FEATURE_VERSION,),
        ).fetchall()
        biases = conn.execute(
            "SELECT nasdaq_bias, COUNT(*) AS n FROM news_event_features WHERE feature_version=? GROUP BY nasdaq_bias ORDER BY n DESC",
            (FEATURE_VERSION,),
        ).fetchall()
        samples = conn.execute(
            "SELECT news_id, published_at, classified_at_utc, classifier_model, event_type, nasdaq_bias, impact_score, relevance_score, confidence, facts_summary FROM news_event_features WHERE feature_version=? ORDER BY published_at DESC LIMIT 5",
            (FEATURE_VERSION,),
        ).fetchall()
        print("========== 新闻特征库审计 ==========")
        print(json.dumps({
            "source_news_total": int(source_total),
            "classified_total": int(feature_total),
            "unclassified_total": max(0, int(source_total) - int(feature_total)),
            "feature_version": FEATURE_VERSION,
            "event_types": [dict(row) for row in categories],
            "nasdaq_bias_counts": [dict(row) for row in biases],
            "recent_samples": [dict(row) for row in samples],
            "historical_ml_training_allowed": False,
            "reason": "当前分类是在原新闻发布时间之后生成的；只有分类结果在某次未来预测前已可用并被当时归档，才可作为该预测的点时特征。旧新闻分类不得回填进历史回测。",
        }, ensure_ascii=False, indent=2))
    finally:
        conn.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
