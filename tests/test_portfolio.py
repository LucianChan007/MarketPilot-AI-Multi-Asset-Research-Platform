import unittest

import pandas as pd

from app.portfolio import (
    PortfolioValidationError,
    build_scenario_table,
    calculate_portfolio,
    validate_transactions,
)


class PortfolioTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([
            {"trade_date": "2025-01-10", "ticker": "QQQM", "side": "BUY", "quantity": 10, "price": 100, "fee": 1, "currency": "USD", "amount": None},
            {"trade_date": "2025-02-10", "ticker": "QQQM", "side": "BUY", "quantity": 5, "price": 110, "fee": 1, "currency": "USD", "amount": None},
            {"trade_date": "2025-03-10", "ticker": "QQQM", "side": "SELL", "quantity": 3, "price": 120, "fee": 1, "currency": "USD", "amount": None},
            {"trade_date": "2025-04-10", "ticker": "QQQM", "side": "DIVIDEND", "quantity": None, "price": None, "fee": 0, "currency": "USD", "amount": 5},
        ])

    def test_weighted_average_cost_pnl_and_new_metrics(self):
        result = calculate_portfolio(self.sample(), {"QQQM": 115})
        position = result.positions.iloc[0]

        self.assertAlmostEqual(position["quantity"], 12.0, places=6)
        self.assertAlmostEqual(position["remaining_cost_basis"], 1241.6, places=3)
        self.assertAlmostEqual(position["average_cost"], 1241.6 / 12, places=5)
        self.assertAlmostEqual(position["realized_pnl"], 48.6, places=3)
        self.assertAlmostEqual(position["dividend_income"], 5.0, places=6)
        self.assertAlmostEqual(position["market_value"], 1380.0, places=6)
        self.assertAlmostEqual(position["unrealized_pnl"], 138.4, places=3)
        self.assertAlmostEqual(position["total_fees"], 3.0, places=6)
        self.assertAlmostEqual(position["cumulative_net_pnl"], 192.0, places=3)
        self.assertAlmostEqual(position["equivalent_cost_per_share"], 99.0, places=3)

    def test_oversell_is_rejected(self):
        frame = pd.DataFrame([
            {"trade_date": "2025-01-10", "ticker": "QQQM", "side": "SELL", "quantity": 1, "price": 100, "fee": 0, "currency": "USD"},
        ])
        with self.assertRaises(PortfolioValidationError):
            calculate_portfolio(frame, {"QQQM": 100})

    def test_scenario_table(self):
        result = calculate_portfolio(self.sample(), {"QQQM": 115})
        scenarios = build_scenario_table(result.positions, [-10, 0, 10])
        self.assertEqual(len(scenarios), 3)
        row = scenarios[scenarios["price_change_pct"] == 10].iloc[0]
        self.assertAlmostEqual(row["scenario_price"], 126.5, places=5)
        self.assertAlmostEqual(row["scenario_market_value"], 1518.0, places=5)
        self.assertAlmostEqual(row["current_cumulative_net_pnl"], 192.0, places=3)
        self.assertAlmostEqual(row["scenario_cumulative_net_pnl"], 330.0, places=3)
        self.assertAlmostEqual(row["cumulative_pnl_change"], 138.0, places=3)
        self.assertAlmostEqual(row["realized_pnl"], 48.6, places=3)
        self.assertAlmostEqual(row["dividend_income"], 5.0, places=6)

    def test_scenario_at_minus_100_percent_uses_zero_price(self):
        result = calculate_portfolio(self.sample(), {"QQQM": 115})
        scenarios = build_scenario_table(result.positions, [-100, -101])
        self.assertEqual(len(scenarios), 1)
        row = scenarios.iloc[0]
        self.assertAlmostEqual(row["scenario_price"], 0.0, places=6)
        self.assertAlmostEqual(row["scenario_market_value"], 0.0, places=6)
        self.assertAlmostEqual(row["scenario_cumulative_net_pnl"], -1188.0, places=3)

    def test_invalid_transaction_date_is_rejected(self):
        frame = pd.DataFrame([
            {"trade_date": "not-a-date", "ticker": "QQQM", "side": "BUY", "quantity": 1, "price": 100, "fee": 0},
        ])
        with self.assertRaises(PortfolioValidationError):
            validate_transactions(frame)

    def test_closed_position_retains_realized_pnl_and_dividend(self):
        frame = pd.DataFrame([
            {"trade_date": "2025-01-10", "ticker": "QQQM", "side": "BUY", "quantity": 2, "price": 100, "fee": 1, "currency": "USD"},
            {"trade_date": "2025-02-10", "ticker": "QQQM", "side": "SELL", "quantity": 2, "price": 110, "fee": 1, "currency": "USD"},
            {"trade_date": "2025-03-10", "ticker": "QQQM", "side": "DIVIDEND", "quantity": None, "price": None, "fee": 0, "currency": "USD", "amount": 4},
        ])
        position = calculate_portfolio(frame, {"QQQM": 110}).positions.iloc[0]

        self.assertAlmostEqual(position["quantity"], 0.0, places=6)
        self.assertFalse(position["open_position"])
        self.assertAlmostEqual(position["realized_pnl"], 18.0, places=6)
        self.assertAlmostEqual(position["dividend_income"], 4.0, places=6)
        self.assertAlmostEqual(position["cumulative_net_pnl"], 22.0, places=6)
        self.assertTrue(pd.isna(position["equivalent_cost_per_share"]))
        self.assertAlmostEqual(position["unrealized_pnl"], 0.0, places=6)

    def test_sell_fee_and_buy_fee_are_not_double_counted(self):
        frame = pd.DataFrame([
            {"trade_date": "2025-01-10", "ticker": "QQQM", "side": "BUY", "quantity": 1, "price": 100, "fee": 2, "currency": "USD"},
            {"trade_date": "2025-01-11", "ticker": "QQQM", "side": "SELL", "quantity": 1, "price": 110, "fee": 3, "currency": "USD"},
        ])
        position = calculate_portfolio(frame, {"QQQM": 110}).positions.iloc[0]
        # Net result: 110 - 3 - (100 + 2) = 5, not 0 after another fee subtraction.
        self.assertAlmostEqual(position["realized_pnl"], 5.0, places=6)
        self.assertAlmostEqual(position["cumulative_net_pnl"], 5.0, places=6)
        self.assertAlmostEqual(position["total_fees"], 5.0, places=6)

    def test_currencies_are_not_combined(self):
        frame = pd.DataFrame([
            {"trade_date": "2025-01-10", "ticker": "QQQM", "side": "BUY", "quantity": 1, "price": 100, "fee": 0, "currency": "USD"},
            {"trade_date": "2025-01-10", "ticker": "QQQM", "side": "BUY", "quantity": 1, "price": 700, "fee": 0, "currency": "HKD"},
        ])
        result = calculate_portfolio(frame, {("QQQM", "USD"): 110, ("QQQM", "HKD"): 710})
        self.assertEqual(set(result.positions["currency"]), {"USD", "HKD"})
        self.assertEqual(len(result.positions), 2)

    def test_non_numeric_fee_is_rejected(self):
        frame = pd.DataFrame([
            {"trade_date": "2025-01-10", "ticker": "QQQM", "side": "BUY", "quantity": 1, "price": 100, "fee": "not-a-number"},
        ])
        with self.assertRaises(PortfolioValidationError):
            validate_transactions(frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
