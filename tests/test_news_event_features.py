from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.news_event_features import (
    FEATURE_VERSION,
    init_schema,
    normalize_classification,
    save_classification,
    load_unclassified_news,
)


class NewsEventFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE news_items (news_id TEXT PRIMARY KEY, source_id TEXT, source_name TEXT, category TEXT, title TEXT, summary TEXT, url TEXT, published_at TEXT, fetched_at TEXT)")
        conn.execute("INSERT INTO news_items VALUES ('n1','fed','Federal Reserve','monetary_policy','FOMC statement','Rates unchanged','https://example.test/1','2026-10-13T18:00:00+00:00','2026-10-14T00:00:00+00:00')")
        conn.commit()
        conn.close()
        init_schema(self.db)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_normalize_clamps_values_and_unknown_categories(self) -> None:
        result = normalize_classification({
            "event_type": "made_up", "nasdaq_bias": "wild", "impact_score": 99,
            "relevance_score": 1.3, "confidence": -0.2, "time_horizon": "soon",
            "facts_summary": "Fact", "evidence": ["A", "B"],
        })
        self.assertEqual(result["event_type"], "other")
        self.assertEqual(result["nasdaq_bias"], "unknown")
        self.assertEqual(result["impact_score"], 2.0)
        self.assertEqual(result["relevance_score"], 1.0)
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["time_horizon"], "unknown")

    def test_classification_is_cached_idempotently(self) -> None:
        article = load_unclassified_news(self.db, limit=5)[0]
        prediction = {
            "news_id": "n1", "event_type": "monetary_policy", "nasdaq_bias": "neutral",
            "impact_score": 0, "relevance_score": 0.8, "confidence": 0.7,
            "time_horizon": "1_5_sessions", "facts_summary": "The statement reports no rate change.",
            "evidence": ["Rates unchanged"],
        }
        first = save_classification(self.db, article, prediction, "unit-test-model", classified_at_utc="2026-10-14T00:00:00+00:00")
        second = save_classification(self.db, article, prediction, "unit-test-model", classified_at_utc="2026-10-15T00:00:00+00:00")
        self.assertTrue(first)
        self.assertFalse(second)
        conn = sqlite3.connect(self.db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM news_event_features WHERE feature_version=?", (FEATURE_VERSION,)).fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()
        self.assertEqual(load_unclassified_news(self.db, limit=5), [])

    def test_does_not_overwrite_existing_feature_version(self) -> None:
        article = load_unclassified_news(self.db, limit=1)[0]
        a = {"event_type": "employment", "nasdaq_bias": "mixed", "impact_score": 0.2,
             "relevance_score": 0.4, "confidence": 0.5, "time_horizon": "unknown",
             "facts_summary": "First classification", "evidence": []}
        save_classification(self.db, article, a, "m1", classified_at_utc="2026-10-14T00:00:00+00:00")
        b = dict(a, facts_summary="Attempted overwrite")
        save_classification(self.db, article, b, "m2", classified_at_utc="2026-10-15T00:00:00+00:00")
        conn = sqlite3.connect(self.db)
        try:
            row = conn.execute("SELECT facts_summary, classifier_model FROM news_event_features WHERE news_id='n1'").fetchone()
            self.assertEqual(row, ("First classification", "m1"))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
