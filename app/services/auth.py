"""Local user authentication service for QQQM Investment Assistant.

Passwords are stored as PBKDF2-HMAC-SHA256 hashes with per-user random salts.
This module is intended for the local Streamlit prototype. Before public hosting,
add a production session/security layer, rate limiting, HTTPS, password recovery,
and a server database such as PostgreSQL.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 600_000
PASSWORD_SALT_BYTES = 16
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_DUMMY_SALT = hashlib.sha256(b"qqqm-auth-dummy-salt-v1").digest()[:PASSWORD_SALT_BYTES]


class AuthError(ValueError):
    """Expected user-facing authentication or registration error."""


class UsernameAlreadyExistsError(AuthError):
    """Raised when a username is already registered."""


class AuthService:
    """SQLite-backed account registry. Each method opens its own connection."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_scheme TEXT NOT NULL,
                    password_iterations INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)"
            )
            connection.commit()

    @staticmethod
    def _validate_username(username: str) -> str:
        if not isinstance(username, str):
            raise AuthError("用户名格式不正确。")
        cleaned = username.strip()
        if not USERNAME_PATTERN.fullmatch(cleaned):
            raise AuthError("用户名需为 3–32 位，只能使用英文字母、数字、点、下划线或连字符。")
        return cleaned

    @staticmethod
    def _validate_password(password: str) -> None:
        if not isinstance(password, str):
            raise AuthError("密码格式不正确。")
        if len(password) < PASSWORD_MIN_LENGTH:
            raise AuthError(f"密码至少需要 {PASSWORD_MIN_LENGTH} 个字符。建议使用独特且不重复的长密码。")
        if len(password) > PASSWORD_MAX_LENGTH:
            raise AuthError(f"密码不能超过 {PASSWORD_MAX_LENGTH} 个字符。")

    @staticmethod
    def _hash_password(password: str, salt: bytes, iterations: int = PASSWORD_ITERATIONS) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        ).hex()

    @staticmethod
    def _public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": int(row["user_id"]),
            "username": str(row["username"]),
            "created_at_utc": str(row["created_at_utc"]),
            "is_active": bool(row["is_active"]),
        }

    def register(self, username: str, password: str) -> dict[str, Any]:
        cleaned = self._validate_username(username)
        self._validate_password(password)
        salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
        password_hash = self._hash_password(password, salt)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (
                        username, password_hash, password_salt, password_scheme,
                        password_iterations, created_at_utc, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        cleaned,
                        password_hash,
                        salt.hex(),
                        PASSWORD_SCHEME,
                        PASSWORD_ITERATIONS,
                        created,
                    ),
                )
                connection.commit()
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise UsernameAlreadyExistsError("这个用户名已被注册，请换一个用户名。") from exc
        user = self.get_user_by_id(user_id)
        if user is None:  # Defensive guard in case the row cannot be read back.
            raise RuntimeError("账号已创建，但无法读取账号信息。")
        return user

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        """Return a safe public user record or None for any invalid credentials."""
        if not isinstance(username, str) or not isinstance(password, str):
            return None
        cleaned = username.strip()
        # Avoid significantly cheaper failures for unknown usernames.
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE LIMIT 1",
                (cleaned,),
            ).fetchone()

        if row is None:
            self._hash_password(password, _DUMMY_SALT)
            return None

        try:
            stored_salt = bytes.fromhex(str(row["password_salt"]))
            iterations = int(row["password_iterations"])
            scheme = str(row["password_scheme"])
            stored_hash = str(row["password_hash"])
            active = bool(row["is_active"])
        except (ValueError, TypeError, KeyError):
            self._hash_password(password, _DUMMY_SALT)
            return None

        if scheme != PASSWORD_SCHEME or iterations < 100_000 or iterations > 2_000_000:
            self._hash_password(password, _DUMMY_SALT)
            return None
        candidate = self._hash_password(password, stored_salt, iterations)
        if not hmac.compare_digest(candidate, stored_hash) or not active:
            return None
        return self._public_user(row)

    def get_user_by_id(self, user_id: int | str) -> dict[str, Any] | None:
        try:
            normalized_id = int(user_id)
        except (TypeError, ValueError):
            return None
        if normalized_id <= 0:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT user_id, username, created_at_utc, is_active "
                "FROM users WHERE user_id = ? LIMIT 1",
                (normalized_id,),
            ).fetchone()
        if row is None or not bool(row["is_active"]):
            return None
        return self._public_user(row)

    def set_user_active(self, user_id: int | str, active: bool) -> bool:
        """Administrative primitive for later account-management work."""
        try:
            normalized_id = int(user_id)
        except (TypeError, ValueError):
            return False
        if normalized_id <= 0:
            return False
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE users SET is_active = ? WHERE user_id = ?",
                (1 if active else 0, normalized_id),
            )
            connection.commit()
            return cursor.rowcount == 1
