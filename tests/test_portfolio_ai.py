import unittest
import pandas as pd

from app.portfolio_ai import build_portfolio_payload


class PortfolioAIPayloadTests(unittest.TestCase):
    def test_payload_aggregates_positions_and_excludes_private_transaction_fields(self):
        positions = pd.DataFrame([
            {
                "ticker": "QQQM", "currency": "USD", "quantity": 2,
                "average_cost": 100.5, "remaining_cost_basis": 201.0,
                "current_price": 110.0, "market_value": 220.0,
                "unrealized_pnl": 19.0, "unrealized_pnl_pct": 9.45,
                "realized_pnl": 5.0, "dividend_income": 1.0,
                "total_fees": 2.0, "cumulative_net_pnl": 25.0,
                "equivalent_cost_per_share": 97.5, "open_position": True,
            }
        ])
        transactions = pd.DataFrame([
            {
                "trade_date": "2025-01-02", "ticker": "QQQM", "currency": "USD",
                "side": "BUY", "quantity": 2, "fee": 2, "amount": None,
                "trade_id": "private-id", "notes": "private note", "realized_pnl": 0,
            },
            {
                "trade_date": "2025-02-02", "ticker": "QQQM", "currency": "USD",
                "side": "SELL", "quantity": 1, "fee": 1, "amount": None,
                "trade_id": "private-id-2", "notes": "another private note", "realized_pnl": 5,
            },
        ])

        payload = build_portfolio_payload(positions, transactions)
        serialized = str(payload)

        self.assertEqual(payload["positions"][0]["ticker"], "QQQM")
        self.assertEqual(len(payload["transaction_aggregates"]), 2)
        self.assertNotIn("private-id", serialized)
        self.assertNotIn("private note", serialized)
        self.assertNotIn("notes", payload["transaction_aggregates"][0])
        self.assertNotIn("trade_date", payload["transaction_aggregates"][0])
        self.assertTrue(payload["price_scenarios"])

    def test_currency_totals_are_separate(self):
        positions = pd.DataFrame([
            {"ticker": "QQQM", "currency": "USD", "quantity": 1,
             "remaining_cost_basis": 100, "current_price": 110, "market_value": 110,
             "realized_pnl": 2, "unrealized_pnl": 10, "dividend_income": 0,
             "total_fees": 1, "open_position": True},
            {"ticker": "XYZ", "currency": "HKD", "quantity": 1,
             "remaining_cost_basis": 100, "current_price": 90, "market_value": 90,
             "realized_pnl": 0, "unrealized_pnl": -10, "dividend_income": 0,
             "total_fees": 0, "open_position": True},
        ])
        payload = build_portfolio_payload(positions, pd.DataFrame())
        self.assertEqual({row["currency"] for row in payload["currency_totals"]}, {"USD", "HKD"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
