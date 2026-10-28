from __future__ import annotations

import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

from app.services.auth import (
    AuthError,
    AuthService,
    PASSWORD_ITERATIONS,
    UsernameAlreadyExistsError,
)


class AuthServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "users.db"
        self.auth = AuthService(self.db_path)
        self.password = "Example-Long-Unique-Passphrase-2026!"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_register_and_authenticate_case_insensitive_username(self) -> None:
        user = self.auth.register("Lucian_01", self.password)
        self.assertGreater(user["user_id"], 0)
        self.assertEqual(user["username"], "Lucian_01")
        self.assertNotIn("password_hash", user)
        logged_in = self.auth.authenticate("lucian_01", self.password)
        self.assertIsNotNone(logged_in)
        self.assertEqual(logged_in["user_id"], user["user_id"])

    def test_password_is_hashed_with_salt_not_stored_in_plaintext(self) -> None:
        self.auth.register("alice_01", self.password)
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT password_hash, password_salt, password_iterations, password_scheme FROM users"
            ).fetchone()
        self.assertIsNotNone(row)
        password_hash, salt, iterations, scheme = row
        self.assertNotEqual(password_hash, self.password)
        self.assertEqual(len(salt), 32)  # 16 random bytes, represented as hex
        self.assertEqual(iterations, PASSWORD_ITERATIONS)
        self.assertEqual(scheme, "pbkdf2_sha256")

    def test_wrong_password_and_unknown_user_return_none(self) -> None:
        self.auth.register("alice_01", self.password)
        self.assertIsNone(self.auth.authenticate("alice_01", "wrong-password-000"))
        self.assertIsNone(self.auth.authenticate("unknown_01", self.password))

    def test_duplicate_username_is_rejected_case_insensitively(self) -> None:
        self.auth.register("Alice_01", self.password)
        with self.assertRaises(UsernameAlreadyExistsError):
            self.auth.register("alice_01", self.password)

    def test_username_and_password_requirements(self) -> None:
        with self.assertRaises(AuthError):
            self.auth.register("ab", self.password)
        with self.assertRaises(AuthError):
            self.auth.register("valid_user", "short")

    def test_disabled_account_cannot_log_in_or_be_loaded_for_a_session(self) -> None:
        user = self.auth.register("alice_01", self.password)
        self.assertTrue(self.auth.set_user_active(user["user_id"], False))
        self.assertIsNone(self.auth.authenticate("alice_01", self.password))
        self.assertIsNone(self.auth.get_user_by_id(user["user_id"]))

    def test_separate_accounts_have_separate_ids(self) -> None:
        one = self.auth.register("alice_01", self.password)
        two = self.auth.register("bob_0001", self.password)
        self.assertNotEqual(one["user_id"], two["user_id"])


if __name__ == "__main__":
    unittest.main()
