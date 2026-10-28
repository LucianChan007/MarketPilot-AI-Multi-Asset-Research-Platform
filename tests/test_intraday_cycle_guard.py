from __future__ import annotations
import unittest
from datetime import datetime, time
from zoneinfo import ZoneInfo

from app.services.intraday_cycle_guard import evaluate_catchup_gate, evaluate_cycle_gate

NY = ZoneInfo("America/New_York")

class IntradayCycleGuardTests(unittest.TestCase):
    def test_after_close_fresh_data_runs_if_latest_forecast_is_older(self):
        proceed, _ = evaluate_catchup_gate(
            datetime(2026, 10, 14, 20, 50, tzinfo=NY), True,
            "2026-10-14", "2026-10-13", True, "2026-10-15")
        self.assertTrue(proceed)

    def test_pre_market_catches_up_when_scheduled_time_was_missed(self):
        proceed, reason = evaluate_catchup_gate(
            datetime(2026, 10, 15, 8, 0, tzinfo=NY), True,
            "2026-10-14", "2026-10-13", True, "2026-10-15")
        self.assertTrue(proceed)
        self.assertIn("补跑目标", reason)

    def test_skips_if_latest_data_already_has_forecast(self):
        proceed, reason = evaluate_catchup_gate(
            datetime(2026, 10, 15, 8, 0, tzinfo=NY), True,
            "2026-10-14", "2026-10-14", True, "2026-10-15")
        self.assertFalse(proceed)
        self.assertIn("避免重复", reason)

    def test_skips_during_regular_market(self):
        proceed, reason = evaluate_catchup_gate(
            datetime(2026, 10, 15, 11, 0, tzinfo=NY), True,
            "2026-10-15", "2026-10-14", True, "2026-10-16")
        self.assertFalse(proceed)
        self.assertIn("交易时段", reason)

    def test_skips_after_close_if_today_data_is_stale(self):
        proceed, reason = evaluate_catchup_gate(
            datetime(2026, 10, 15, 17, 0, tzinfo=NY), True,
            "2026-10-14", "2026-10-13", True, "2026-10-15")
        self.assertFalse(proceed)
        self.assertIn("可能尚未同步完整", reason)

    def test_weekend_catches_up_for_future_session_if_needed(self):
        proceed, reason = evaluate_catchup_gate(
            datetime(2026, 10, 17, 12, 0, tzinfo=NY), False,
            "2026-10-16", "2026-10-15", True, "2026-10-19")
        self.assertTrue(proceed)
        self.assertIn("非交易日", reason)

    def test_skips_stale_data_after_a_session_has_passed(self):
        proceed, reason = evaluate_catchup_gate(
            datetime(2026, 10, 16, 8, 0, tzinfo=NY), True,
            "2026-10-14", "2026-10-13", True, "2026-10-15")
        self.assertFalse(proceed)
        self.assertIn("过期", reason)

    def test_requires_timezone(self):
        proceed, reason = evaluate_catchup_gate(
            datetime(2026, 10, 15, 8, 0), True,
            "2026-10-14", "2026-10-13", True, "2026-10-15")
        self.assertFalse(proceed)
        self.assertIn("时区", reason)

    def test_old_strict_gate_remains_compatible(self):
        proceed, reason = evaluate_cycle_gate(
            datetime(2026, 10, 14, 20, 50, tzinfo=NY), True, "2026-10-14")
        self.assertTrue(proceed)
        self.assertIn("继续", reason)

if __name__ == '__main__':
    unittest.main()
