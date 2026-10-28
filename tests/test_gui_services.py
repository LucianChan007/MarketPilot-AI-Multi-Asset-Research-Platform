"""Lightweight smoke checks for the delivered GUI service files."""
from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

from app.services.market_clock import get_market_clock
from app.services.portfolio import calculate_position


class PortfolioTests(unittest.TestCase):
    def test_pnl_and_fees(self):
        result = calculate_position(shares=2, avg_cost=310, current_price=306.16, fees_paid=1.0)
        self.assertEqual(result["market_value"], 612.32)
        self.assertEqual(result["total_cost"], 621.0)
        self.assertEqual(result["profit"], -8.68)
        self.assertLess(result["profit_pct"], 0)

    def test_zero_cost(self):
        result = calculate_position(shares=0, avg_cost=310, current_price=306.16)
        self.assertEqual(result["profit_pct"], 0.0)

    def test_reject_negative(self):
        with self.assertRaises(ValueError):
            calculate_position(shares=-1, avg_cost=310, current_price=306.16)


class MarketClockTests(unittest.TestCase):
    def test_clock_fields_exist(self):
        sample = datetime(2026, 10, 14, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        result = get_market_clock(sample)
        self.assertIn("local_time", result)
        self.assertIn("new_york_time", result)
        self.assertIn("market_status", result)
        self.assertIn("countdown", result)
        self.assertIn("calendar_is_authoritative", result)


if __name__ == "__main__":
    unittest.main()
