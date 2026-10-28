from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.services.auth import AuthService
from app.services.portfolio_ledger import PortfolioLedger, TradeValidationError


class PortfolioLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "users.db"
        self.auth = AuthService(self.db_path)
        self.user1 = self.auth.register("alice_01", "Unique-Password-2026-Alpha!")
        self.user2 = None
        self.ledger = PortfolioLedger(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def other_user(self) -> dict:
        if self.user2 is None:
            self.user2 = self.auth.register("bob_0001", "Unique-Password-2026-Beta!")
        return self.user2

    def trade(self, side: str, quantity: float, price: float, *, fees: float = 0.0,
              at: str = "2026-10-01T10:00:00+08:00", user_id: int | None = None,
              notes: str = "") -> dict:
        return self.ledger.record_trade(
            user_id=user_id or self.user1["user_id"],
            symbol="QQQM", side=side, quantity=quantity, price=price,
            fees=fees, executed_at=at, notes=notes,
        )

    def test_empty_account_has_zero_shares_and_no_transactions(self) -> None:
        result = self.ledger.get_position_summary(user_id=self.user1["user_id"], symbol="QQQM", reference_price=300)
        self.assertEqual(result["shares"], 0.0)
        self.assertEqual(result["transaction_count"], 0)
        self.assertEqual(result["market_value"], 0.0)
        self.assertEqual(self.ledger.list_transactions(user_id=self.user1["user_id"]), [])

    def test_opening_balance_is_separate_from_a_real_buy(self) -> None:
        saved = self.trade("OPENING", 2, 310, at="2026-10-01T10:00:00+08:00", notes="initial holdings")
        self.assertEqual(saved["side"], "OPENING")
        summary = self.ledger.get_position_summary(user_id=self.user1["user_id"], reference_price=300)
        self.assertEqual(summary["shares"], 2.0)
        self.assertEqual(summary["average_cost"], 310.0)
        self.assertEqual(summary["market_value"], 600.0)
        self.assertEqual(summary["unrealized_pnl"], -20.0)
        self.assertEqual(summary["realized_pnl_total"], 0.0)

    def test_buy_uses_weighted_average_and_includes_buy_fee(self) -> None:
        self.trade("OPENING", 2, 310, at="2026-10-01T10:00:00+08:00")
        self.trade("BUY", 1, 300, fees=1, at="2026-10-02T10:00:00+08:00")
        summary = self.ledger.get_position_summary(user_id=self.user1["user_id"])
        self.assertEqual(summary["shares"], 3.0)
        # (2*310 + 1*300 + $1 fee) / 3 shares
        self.assertEqual(summary["average_cost"], 307.0)
        self.assertEqual(summary["cost_basis"], 921.0)

    def test_sell_calculates_realized_profit_after_sell_fee(self) -> None:
        self.trade("OPENING", 2, 310, at="2026-10-01T10:00:00+08:00")
        saved = self.trade("SELL", 1, 320, fees=1, at="2026-10-02T10:00:00+08:00")
        self.assertEqual(saved["realized_pnl"], 9.0)
        summary = self.ledger.get_position_summary(user_id=self.user1["user_id"], reference_price=315)
        self.assertEqual(summary["shares"], 1.0)
        self.assertEqual(summary["average_cost"], 310.0)
        self.assertEqual(summary["realized_pnl_total"], 9.0)
        self.assertEqual(summary["unrealized_pnl"], 5.0)

    def test_cannot_sell_more_than_current_holding_and_no_row_is_written(self) -> None:
        self.trade("OPENING", 1, 300)
        with self.assertRaisesRegex(TradeValidationError, "超过当前持有数量"):
            self.trade("SELL", 1.5, 320, at="2026-10-02T10:00:00+08:00")
        rows = self.ledger.list_transactions(user_id=self.user1["user_id"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["side"], "OPENING")

    def test_users_cannot_see_each_others_positions_or_transactions(self) -> None:
        self.trade("OPENING", 2, 310)
        another = self.other_user()
        other_summary = self.ledger.get_position_summary(user_id=another["user_id"], reference_price=300)
        self.assertEqual(other_summary["shares"], 0.0)
        self.assertEqual(other_summary["transaction_count"], 0)
        self.assertEqual(self.ledger.list_transactions(user_id=another["user_id"]), [])

    def test_cannot_initialize_same_symbol_twice(self) -> None:
        self.trade("OPENING", 2, 310)
        with self.assertRaisesRegex(TradeValidationError, "不能再次初始化"):
            self.trade("OPENING", 1, 300, at="2026-10-02T10:00:00+08:00")

    def test_transactions_must_be_entered_in_chronological_order(self) -> None:
        self.trade("BUY", 1, 300, at="2026-10-02T10:00:00+08:00")
        with self.assertRaisesRegex(TradeValidationError, "早于已有的最后一笔记录"):
            self.trade("BUY", 1, 290, at="2026-10-01T10:00:00+08:00")

    def test_rejects_invalid_quantity_price_fee_and_opening_fee(self) -> None:
        with self.assertRaises(TradeValidationError):
            self.trade("BUY", 0, 300)
        with self.assertRaises(TradeValidationError):
            self.trade("BUY", 1, 0)
        with self.assertRaises(TradeValidationError):
            self.trade("BUY", 1, 300, fees=-1)
        with self.assertRaises(TradeValidationError):
            self.trade("OPENING", 1, 300, fees=1)

    def test_inactive_user_cannot_write_or_read_personal_ledger(self) -> None:
        self.assertTrue(self.auth.set_user_active(self.user1["user_id"], False))
        with self.assertRaises(TradeValidationError):
            self.trade("BUY", 1, 300)
        with self.assertRaises(TradeValidationError):
            self.ledger.get_position_summary(user_id=self.user1["user_id"])

    def test_history_is_scoped_to_user_and_returns_newest_first(self) -> None:
        self.trade("BUY", 1, 290, at="2026-10-01T10:00:00+08:00", notes="first")
        self.trade("BUY", 1, 300, at="2026-10-02T10:00:00+08:00", notes="second")
        rows = self.ledger.list_transactions(user_id=self.user1["user_id"], limit=10)
        self.assertEqual([row["notes"] for row in rows], ["second", "first"])
        self.assertTrue(all(row["user_id"] == self.user1["user_id"] for row in rows))


    def test_dividend_is_saved_and_does_not_change_share_balance_or_average_cost(self) -> None:
        self.trade("OPENING", 2, 310, at="2026-10-01T10:00:00+08:00")
        before = self.ledger.get_position_summary(user_id=self.user1["user_id"], reference_price=300)
        saved = self.ledger.record_dividend(
            user_id=self.user1["user_id"],
            symbol="QQQM",
            eligible_shares=2,
            dividend_per_share=0.25,
            tax_withheld=0.10,
            fees=0,
            paid_at="2026-10-03T10:00:00+08:00",
            notes="quarterly dividend",
        )
        after = self.ledger.get_position_summary(user_id=self.user1["user_id"], reference_price=300)
        self.assertEqual(saved["gross_amount"], 0.50)
        self.assertEqual(saved["net_amount"], 0.40)
        self.assertEqual(after["shares"], before["shares"])
        self.assertEqual(after["average_cost"], before["average_cost"])
        self.assertEqual(after["dividend_gross_total"], 0.50)
        self.assertEqual(after["dividend_tax_total"], 0.10)
        self.assertEqual(after["dividend_income_total"], 0.40)
        self.assertEqual(after["realized_plus_dividends_total"], 0.40)

    def test_dividend_can_be_recorded_even_if_current_position_is_empty(self) -> None:
        saved = self.ledger.record_dividend(
            user_id=self.user1["user_id"], symbol="QQQM", eligible_shares=1,
            dividend_per_share=0.20, paid_at="2026-09-01T10:00:00+08:00",
        )
        summary = self.ledger.get_position_summary(user_id=self.user1["user_id"])
        self.assertEqual(saved["net_amount"], 0.20)
        self.assertEqual(summary["shares"], 0.0)
        self.assertEqual(summary["dividend_income_total"], 0.20)

    def test_dividend_entries_are_isolated_between_users(self) -> None:
        self.ledger.record_dividend(
            user_id=self.user1["user_id"], symbol="QQQM", eligible_shares=2,
            dividend_per_share=0.25, paid_at="2026-10-03T10:00:00+08:00",
        )
        another = self.other_user()
        other_summary = self.ledger.get_position_summary(user_id=another["user_id"])
        self.assertEqual(other_summary["dividend_income_total"], 0.0)
        self.assertEqual(self.ledger.list_dividends(user_id=another["user_id"]), [])

    def test_dividend_cannot_have_tax_or_fees_above_gross_amount(self) -> None:
        with self.assertRaisesRegex(TradeValidationError, "不能超过股息总额"):
            self.ledger.record_dividend(
                user_id=self.user1["user_id"], symbol="QQQM", eligible_shares=1,
                dividend_per_share=0.25, tax_withheld=0.20, fees=0.10,
                paid_at="2026-10-03T10:00:00+08:00",
            )

    def test_activity_history_merges_dividends_and_trades_newest_first(self) -> None:
        self.trade("BUY", 2, 300, at="2026-10-01T10:00:00+08:00")
        self.ledger.record_dividend(
            user_id=self.user1["user_id"], symbol="QQQM", eligible_shares=2,
            dividend_per_share=0.25, paid_at="2026-10-03T10:00:00+08:00",
        )
        activities = self.ledger.list_activity(user_id=self.user1["user_id"], symbol="QQQM")
        self.assertEqual([row["side"] for row in activities], ["DIVIDEND", "BUY"])
        self.assertEqual(activities[0]["net_amount"], 0.50)

    def test_void_latest_buy_recalculates_position_and_keeps_audit_row(self) -> None:
        self.trade("OPENING", 2, 310, at="2026-10-01T10:00:00+08:00")
        bought = self.trade("BUY", 1, 300, fees=1, at="2026-10-02T10:00:00+08:00")
        result = self.ledger.void_transaction(
            user_id=self.user1["user_id"], transaction_id=bought["transaction_id"],
            reason="测试：买入数量录错",
        )
        self.assertEqual(result["transaction_id"], bought["transaction_id"])
        summary = self.ledger.get_position_summary(user_id=self.user1["user_id"])
        self.assertEqual(summary["shares"], 2.0)
        self.assertEqual(summary["average_cost"], 310.0)
        self.assertEqual(summary["transaction_count"], 1)
        rows = self.ledger.list_transactions(user_id=self.user1["user_id"], limit=10)
        voided = next(row for row in rows if row["transaction_id"] == bought["transaction_id"])
        self.assertEqual(voided["is_voided"], 1)
        self.assertEqual(voided["void_reason"], "测试：买入数量录错")

    def test_void_sell_recalculates_realized_profit(self) -> None:
        self.trade("OPENING", 2, 310, at="2026-10-01T10:00:00+08:00")
        sold = self.trade("SELL", 1, 320, fees=1, at="2026-10-02T10:00:00+08:00")
        self.assertEqual(self.ledger.get_position_summary(user_id=self.user1["user_id"])["realized_pnl_total"], 9.0)
        self.ledger.void_transaction(
            user_id=self.user1["user_id"], transaction_id=sold["transaction_id"],
            reason="测试：误录卖出",
        )
        summary = self.ledger.get_position_summary(user_id=self.user1["user_id"])
        self.assertEqual(summary["shares"], 2.0)
        self.assertEqual(summary["realized_pnl_total"], 0.0)

    def test_void_earlier_buy_is_rejected_if_later_sale_would_oversell(self) -> None:
        self.trade("BUY", 2, 300, at="2026-10-01T10:00:00+08:00")
        second = self.trade("BUY", 1, 310, at="2026-10-02T10:00:00+08:00")
        self.trade("SELL", 2.5, 320, at="2026-10-03T10:00:00+08:00")
        with self.assertRaisesRegex(TradeValidationError, "后续卖出会超过"):
            self.ledger.void_transaction(
                user_id=self.user1["user_id"], transaction_id=second["transaction_id"],
                reason="测试：错误记录",
            )
        rows = self.ledger.list_transactions(user_id=self.user1["user_id"], limit=10)
        item = next(row for row in rows if row["transaction_id"] == second["transaction_id"])
        self.assertEqual(item["is_voided"], 0)
        self.assertEqual(self.ledger.get_position_summary(user_id=self.user1["user_id"])["shares"], 0.5)

    def test_void_dividend_removes_it_from_totals_but_keeps_audit_row(self) -> None:
        dividend = self.ledger.record_dividend(
            user_id=self.user1["user_id"], symbol="QQQM", eligible_shares=2,
            dividend_per_share=0.25, paid_at="2026-10-03T10:00:00+08:00",
        )
        self.assertEqual(self.ledger.get_position_summary(user_id=self.user1["user_id"])["dividend_income_total"], 0.5)
        self.ledger.void_dividend(
            user_id=self.user1["user_id"], dividend_id=dividend["dividend_id"],
            reason="测试：股息金额录错",
        )
        summary = self.ledger.get_position_summary(user_id=self.user1["user_id"])
        self.assertEqual(summary["dividend_income_total"], 0.0)
        rows = self.ledger.list_dividends(user_id=self.user1["user_id"])
        self.assertEqual(rows[0]["is_voided"], 1)

    def test_cannot_void_another_users_transaction(self) -> None:
        saved = self.trade("BUY", 1, 300)
        another = self.other_user()
        with self.assertRaisesRegex(TradeValidationError, "不属于当前账号"):
            self.ledger.void_transaction(
                user_id=another["user_id"], transaction_id=saved["transaction_id"],
                reason="not mine",
            )
        self.assertEqual(self.ledger.get_position_summary(user_id=self.user1["user_id"])["shares"], 1.0)

    def test_trade_has_foreign_key_to_users(self) -> None:
        # sqlite3.Connection context managers commit/rollback but do not close the
        # connection. Explicit closing is required on Windows before TemporaryDirectory
        # removes the database file.
        with closing(sqlite3.connect(self.db_path)) as connection:
            fk_rows = connection.execute("PRAGMA foreign_key_list(portfolio_transactions)").fetchall()
        self.assertTrue(any(row[2] == "users" and row[3] == "user_id" for row in fk_rows))


if __name__ == "__main__":
    unittest.main()
