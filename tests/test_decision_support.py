import unittest
import pandas as pd

from app.decision_support import build_decision_assessment


class DecisionSupportTests(unittest.TestCase):
    def positions(self):
        return pd.DataFrame([
            {
                "ticker": "QQQM",
                "currency": "USD",
                "quantity": 2.0,
                "average_cost": 311.375,
                "remaining_cost_basis": 622.75,
                "current_price": 307.19,
                "market_value": 614.38,
                "unrealized_pnl": -8.37,
                "realized_pnl": 23.44,
                "dividend_income": 0.43,
                "open_position": True,
            }
        ])

    def test_drawdown_scenario_is_correct(self):
        result = build_decision_assessment(
            self.positions(), "1–5 年", 10, 0
        )
        item = result["items"][0]
        self.assertAlmostEqual(item["review_scenario_price"], 276.471, places=3)
        self.assertAlmostEqual(item["review_scenario_market_value"], 552.942, places=3)
        self.assertAlmostEqual(item["review_scenario_value_decrease"], 61.438, places=3)
        self.assertAlmostEqual(item["review_scenario_cumulative_net_pnl"], -45.938, places=3)

    def test_does_not_treat_drawdown_as_sell_signal(self):
        result = build_decision_assessment(
            self.positions(), "1–5 年", 10, 0
        )
        self.assertIn("不是价格预测", result["conclusion"])
        self.assertIn("不能单独证明应该卖出或加仓", result["items"][0]["interpretation"])

    def test_multi_currency_is_separate(self):
        frame = pd.concat([
            self.positions(),
            pd.DataFrame([{
                "ticker": "ABC", "currency": "HKD", "quantity": 3,
                "average_cost": 10, "remaining_cost_basis": 30,
                "current_price": 9, "market_value": 27,
                "unrealized_pnl": -3, "realized_pnl": 0,
                "dividend_income": 0, "open_position": True,
            }]),
        ], ignore_index=True)
        result = build_decision_assessment(frame, "5 年以上", 5, 100)
        self.assertEqual({x["currency"] for x in result["items"]}, {"USD", "HKD"})
        self.assertEqual(len(result["items"]), 2)

    def test_no_open_positions(self):
        frame = self.positions()
        frame.loc[0, "open_position"] = False
        result = build_decision_assessment(frame, "少于 1 年", 10, 0)
        self.assertEqual(result["items"], [])
        self.assertIn("没有未平仓持仓", result["conclusion"])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            build_decision_assessment(self.positions(), "unknown", 10, 0)
        with self.assertRaises(ValueError):
            build_decision_assessment(self.positions(), "1–5 年", 0, 0)
        with self.assertRaises(ValueError):
            build_decision_assessment(self.positions(), "1–5 年", 10, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
