import unittest

from app.portfolio_ai import build_plain_language_summary


class PlainLanguageSummaryTests(unittest.TestCase):
    def sample_payload(self):
        return {
            "currency_totals": [{
                "currency": "USD",
                "open_position_count": 1,
                "remaining_cost_basis": 622.75,
                "market_value": 614.38,
                "realized_pnl": 23.44,
                "unrealized_pnl": -8.37,
                "net_dividend_income": 0.43,
                "total_recorded_fees": 11.98,
                "cumulative_net_pnl": 15.5,
            }],
            "positions": [{
                "ticker": "QQQM", "currency": "USD", "quantity": 2.0,
                "average_cost": 311.375, "remaining_cost_basis": 622.75,
                "current_price": 307.19, "market_value": 614.38,
                "unrealized_pnl": -8.37, "unrealized_pnl_pct": -1.344,
                "realized_pnl": 23.44, "dividend_income": 0.43,
                "total_fees": 11.98, "cumulative_net_pnl": 15.5,
                "equivalent_cost_per_share": 299.44, "open_position": True,
            }],
            "price_scenarios": [{
                "ticker": "QQQM", "currency": "USD", "price_change_pct": -5,
                "scenario_price": 291.8305, "scenario_market_value": 583.661,
                "scenario_unrealized_pnl": -39.089,
                "current_cumulative_net_pnl": 15.5,
                "scenario_cumulative_net_pnl": -15.219,
                "cumulative_pnl_change": -30.719,
            }],
        }

    def test_plain_language_summary_states_total_and_current_float_loss(self):
        text = build_plain_language_summary(self.sample_payload(), {
            "qqqm_market_summary": {"latest_date": "2026-10-12"}
        })
        self.assertIn("累计净盈利 +15.50 USD", text)
        self.assertIn("当前持仓浮动盈亏：−8.37 USD", text)
        self.assertIn("历史已实现盈利和净分红抵消了这部分浮亏", text)
        self.assertIn("行情数据日期：** QQQM 2026-10-12", text)

    def test_scenario_summary_is_clear_and_does_not_claim_prediction(self):
        text = build_plain_language_summary(self.sample_payload())
        self.assertIn("累计净盈亏 −15.22 USD", text)
        self.assertIn("假设，不是预测", text)

    def test_missing_totals_does_not_invent_pnl(self):
        text = build_plain_language_summary({"currency_totals": [], "positions": [], "price_scenarios": []})
        self.assertIn("无法可靠展示累计盈亏", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
