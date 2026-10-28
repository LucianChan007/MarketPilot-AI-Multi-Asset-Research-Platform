from __future__ import annotations
import unittest
from datetime import date, datetime
from types import SimpleNamespace

from app.services.market_calendar import is_nasdaq_session, next_nasdaq_session

class FakeCalendar:
    def __init__(self, sessions):
        self.sessions = [date.fromisoformat(x) for x in sessions]
    def schedule(self, start_date, end_date):
        start = date.fromisoformat(str(start_date))
        end = date.fromisoformat(str(end_date))
        available = [x for x in self.sessions if start <= x <= end]
        return SimpleNamespace(empty=not available, index=available)

class FakeDatetimeCalendar:
    """Mimic a pandas DatetimeIndex whose entry is midnight datetime."""
    def schedule(self, start_date, end_date):
        return SimpleNamespace(empty=False, index=[datetime(2026, 11, 30, 0, 0)])


class MarketCalendarTests(unittest.TestCase):
    def test_next_session_skips_market_holiday(self):
        # The day after 2026-11-27 is a weekend; next valid session is Monday 11/30.
        calendar = FakeCalendar(["2026-11-30", "2026-12-01"])
        self.assertEqual(next_nasdaq_session(date(2026, 11, 27), calendar), date(2026, 11, 30))

    def test_next_session_normalizes_datetime_index_to_plain_date(self):
        result = next_nasdaq_session(date(2026, 11, 27), FakeDatetimeCalendar())
        self.assertEqual(result, date(2026, 11, 30))
        self.assertIs(type(result), date)

    def test_is_session_true(self):
        calendar = FakeCalendar(["2026-10-14"])
        self.assertTrue(is_nasdaq_session(date(2026, 10, 14), calendar))

    def test_is_session_false_on_holiday(self):
        calendar = FakeCalendar(["2026-10-14"])
        self.assertFalse(is_nasdaq_session(date(2026, 10, 13), calendar))

if __name__ == "__main__":
    unittest.main()
