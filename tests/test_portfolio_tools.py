from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.services.auth import AuthService
from app.services.portfolio_ledger import PortfolioLedger
from app.services.portfolio_tools import backup_sqlite_database, export_user_activity_csv


class PortfolioToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "users.db"
        self.auth = AuthService(self.db)
        self.user_a = self.auth.register("alice_01", "Unique-Password-2026-Alpha!")
        self.user_b = self.auth.register("bob_0001", "Unique-Password-2026-Beta!")
        self.ledger = PortfolioLedger(self.db)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_csv_export_contains_only_current_users_rows(self) -> None:
        self.ledger.record_trade(
            user_id=self.user_a["user_id"], symbol="QQQM", side="BUY", quantity=2,
            price=300, executed_at="2026-10-01T10:00:00+08:00", notes="Alice record",
        )
        self.ledger.record_dividend(
            user_id=self.user_b["user_id"], symbol="QQQM", eligible_shares=2,
            dividend_per_share=0.2, paid_at="2026-10-02T10:00:00+08:00", notes="Bob private",
        )
        payload = export_user_activity_csv(self.ledger, user_id=self.user_a["user_id"], symbol="QQQM")
        text = payload.decode("utf-8-sig")
        self.assertIn("Alice record", text)
        self.assertNotIn("Bob private", text)

    def test_backup_sqlite_database_has_valid_integrity(self) -> None:
        destination_dir = self.root / "backups"
        result = backup_sqlite_database(self.db, destination_dir)
        self.assertTrue(result.is_file())
        with closing(sqlite3.connect(result)) as connection:
            check = connection.execute("PRAGMA integrity_check").fetchone()[0]
            users = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self.assertEqual(check, "ok")
        self.assertEqual(users, 2)

    def test_backup_fails_for_missing_source(self) -> None:
        with self.assertRaises(FileNotFoundError):
            backup_sqlite_database(self.root / "missing.db", self.root / "backups")


if __name__ == "__main__":
    unittest.main()
