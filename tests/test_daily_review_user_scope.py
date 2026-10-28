from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.daily_review import load_recent_human_feedback, save_human_feedback


class DailyReviewUserScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.context = {
            "target_date": "2026-10-15",
            "as_of_date": "2026-10-14",
            "forecast_generated_at_utc": "2026-10-14T20:00:00Z",
            "forecast_model_name": "test model",
            "forecast_model_family": "test",
            "summary_target_date": "2026-10-15",
            "actual_points_found": 0,
            "expected_points": 26,
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_feedback_is_saved_to_user_specific_files(self) -> None:
        path_one = save_human_feedback(self.root, self.context, {"notes": "Alice private note"}, user_id=1)
        path_two = save_human_feedback(self.root, self.context, {"notes": "Bob private note"}, user_id=2)
        self.assertNotEqual(path_one, path_two)
        self.assertTrue(path_one.exists())
        self.assertTrue(path_two.exists())
        self.assertEqual(load_recent_human_feedback(self.root, user_id=1)[0]["user_notes"], "Alice private note")
        self.assertEqual(load_recent_human_feedback(self.root, user_id=2)[0]["user_notes"], "Bob private note")

    def test_user_specific_view_does_not_show_legacy_shared_feedback(self) -> None:
        save_human_feedback(self.root, self.context, {"notes": "legacy shared row"})
        self.assertEqual(load_recent_human_feedback(self.root, user_id=1), [])
        self.assertEqual(load_recent_human_feedback(self.root)[0]["user_notes"], "legacy shared row")

    def test_invalid_user_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            save_human_feedback(self.root, self.context, {"notes": "test"}, user_id="not-an-id")


if __name__ == "__main__":
    unittest.main()
