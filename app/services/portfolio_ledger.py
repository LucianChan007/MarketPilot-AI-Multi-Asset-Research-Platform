"""Per-user QQQM transaction ledger with moving weighted-average cost basis.

This local prototype stores its tables in data/users.db alongside the auth table.
Every user-facing query and mutation is scoped by the authenticated user_id.
Corrections are audit-preserving voids; active trade state is recalculated after a void.
"""
from __future__ import annotations

import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_QTY_STEP = Decimal("0.0001")
_PRICE_STEP = Decimal("0.0001")
_FEE_STEP = Decimal("0.01")
_MONEY_STEP = Decimal("0.01")
_COST_STEP = Decimal("0.0001")


class TradeValidationError(ValueError):
    """An invalid transaction or transaction sequence supplied by the user."""


def _positive_user_id(value: Any) -> int:
    try:
        user_id = int(value)
    except (TypeError, ValueError) as exc:
        raise TradeValidationError("当前账号信息无效，请退出后重新登录。") from exc
    if user_id <= 0:
        raise TradeValidationError("当前账号信息无效，请退出后重新登录。")
    return user_id


def _decimal_value(value: Any, label: str, step: Decimal, *, allow_zero: bool) -> Decimal:
    try:
        result = Decimal(str(value)).quantize(step, rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TradeValidationError(f"{label}不是有效数字。") from exc
    if not result.is_finite():
        raise TradeValidationError(f"{label}必须是有限数字。")
    if result < 0 or (not allow_zero and result <= 0):
        op = "不能为负数" if allow_zero else "必须大于 0"
        raise TradeValidationError(f"{label}{op}。")
    return result


def _iso_to_utc(value: str | datetime | None) -> str:
    if value is None:
        current = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        current = value
    elif isinstance(value, str):
        try:
            current = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise TradeValidationError("成交时间格式无效，请重新选择日期和时间。") from exc
    else:
        raise TradeValidationError("成交时间格式无效。")

    # Naive datetimes are interpreted as local machine time rather than silently as UTC.
    if current.tzinfo is None:
        current = current.astimezone()
    return current.astimezone(timezone.utc).isoformat(timespec="seconds")


def _to_float(value: Decimal | int | float | None) -> float | None:
    if value is None:
        return None
    return float(value)


class PortfolioLedger:
    """SQLite-backed account-scoped transaction ledger."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            # AuthService creates this table. Fail clearly if the auth database was not initialized.
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='users'"
            ).fetchone()
            if exists is None:
                raise RuntimeError(
                    "账号数据库尚未初始化。请先通过 Dashboard 注册账号，再初始化持仓账本。"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('OPENING', 'BUY', 'SELL')),
                    quantity REAL NOT NULL CHECK (quantity > 0),
                    price REAL NOT NULL CHECK (price > 0),
                    gross_amount REAL NOT NULL CHECK (gross_amount >= 0),
                    fees REAL NOT NULL CHECK (fees >= 0),
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    shares_before REAL NOT NULL DEFAULT 0,
                    average_cost_before REAL NOT NULL DEFAULT 0,
                    shares_after REAL NOT NULL DEFAULT 0,
                    average_cost_after REAL NOT NULL DEFAULT 0,
                    executed_at_utc TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE RESTRICT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_user_symbol_time "
                "ON portfolio_transactions(user_id, symbol, executed_at_utc, transaction_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_user_symbol_side "
                "ON portfolio_transactions(user_id, symbol, side)"
            )
            # Cash dividends live in a separate table so the existing transaction
            # table's BUY/SELL/OPENING CHECK constraint remains backward compatible.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_dividends (
                    dividend_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    eligible_shares REAL NOT NULL CHECK (eligible_shares > 0),
                    dividend_per_share REAL NOT NULL CHECK (dividend_per_share > 0),
                    gross_amount REAL NOT NULL CHECK (gross_amount > 0),
                    tax_withheld REAL NOT NULL DEFAULT 0 CHECK (tax_withheld >= 0),
                    fees REAL NOT NULL DEFAULT 0 CHECK (fees >= 0),
                    net_amount REAL NOT NULL CHECK (net_amount >= 0),
                    paid_at_utc TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE RESTRICT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_dividends_user_symbol_time "
                "ON portfolio_dividends(user_id, symbol, paid_at_utc, dividend_id)"
            )
            # Corrections are recorded as voids, not destructive deletes. This preserves
            # an audit trail while allowing the active ledger to be recalculated.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_transaction_voids (
                    transaction_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    voided_at_utc TEXT NOT NULL,
                    FOREIGN KEY (transaction_id) REFERENCES portfolio_transactions(transaction_id) ON DELETE RESTRICT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE RESTRICT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_dividend_voids (
                    dividend_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    voided_at_utc TEXT NOT NULL,
                    FOREIGN KEY (dividend_id) REFERENCES portfolio_dividends(dividend_id) ON DELETE RESTRICT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE RESTRICT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tx_voids_user ON portfolio_transaction_voids(user_id, transaction_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_div_voids_user ON portfolio_dividend_voids(user_id, dividend_id)"
            )
            connection.commit()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        cleaned = str(symbol or "").strip().upper()
        if not _SYMBOL_RE.fullmatch(cleaned):
            raise TradeValidationError("标的代码格式不正确，例如 QQQM。")
        return cleaned

    @staticmethod
    def _get_state(connection: sqlite3.Connection, user_id: int, symbol: str) -> dict[str, Decimal | int | str]:
        latest = connection.execute(
            """
            SELECT shares_after, average_cost_after, executed_at_utc, transaction_id
            FROM portfolio_transactions
            WHERE user_id = ? AND symbol = ?
              AND NOT EXISTS (
                  SELECT 1 FROM portfolio_transaction_voids v
                  WHERE v.transaction_id = portfolio_transactions.transaction_id
                    AND v.user_id = portfolio_transactions.user_id
              )
            ORDER BY executed_at_utc DESC, transaction_id DESC
            LIMIT 1
            """,
            (user_id, symbol),
        ).fetchone()
        realized_row = connection.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) AS total "
            "FROM portfolio_transactions WHERE user_id = ? AND symbol = ? AND side = 'SELL' "
            "AND NOT EXISTS (SELECT 1 FROM portfolio_transaction_voids v "
            "WHERE v.transaction_id = portfolio_transactions.transaction_id "
            "AND v.user_id = portfolio_transactions.user_id)",
            (user_id, symbol),
        ).fetchone()
        count_row = connection.execute(
            "SELECT COUNT(*) AS n FROM portfolio_transactions WHERE user_id = ? AND symbol = ? "
            "AND NOT EXISTS (SELECT 1 FROM portfolio_transaction_voids v "
            "WHERE v.transaction_id = portfolio_transactions.transaction_id "
            "AND v.user_id = portfolio_transactions.user_id)",
            (user_id, symbol),
        ).fetchone()
        if latest is None:
            return {
                "shares": Decimal("0"),
                "average_cost": Decimal("0"),
                "latest_executed_at_utc": "",
                "realized_pnl_total": Decimal("0"),
                "transaction_count": int(count_row["n"]),
            }
        return {
            "shares": Decimal(str(latest["shares_after"])),
            "average_cost": Decimal(str(latest["average_cost_after"])),
            "latest_executed_at_utc": str(latest["executed_at_utc"]),
            "realized_pnl_total": Decimal(str(realized_row["total"] or 0)),
            "transaction_count": int(count_row["n"]),
        }

    def record_trade(
        self,
        *,
        user_id: int | str,
        symbol: str = "QQQM",
        side: str,
        quantity: Any,
        price: Any,
        fees: Any = 0,
        executed_at: str | datetime | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Append an opening-balance, buy, or sell transaction for one user.

        OPENING is an initial balance (not a claim that a real buy happened). Its
        price is the existing per-share cost including allocated past buy fees.
        Transactions must be entered in chronological order for each symbol.
        """
        normalized_user_id = _positive_user_id(user_id)
        normalized_symbol = self._normalize_symbol(symbol)
        normalized_side = str(side or "").strip().upper()
        if normalized_side not in {"OPENING", "BUY", "SELL"}:
            raise TradeValidationError("交易类型必须是 OPENING（期初持仓）、BUY（买入）或 SELL（卖出）。")

        qty = _decimal_value(quantity, "股数", _QTY_STEP, allow_zero=False)
        unit_price = _decimal_value(price, "每股价格", _PRICE_STEP, allow_zero=False)
        fee = _decimal_value(fees, "手续费", _FEE_STEP, allow_zero=True)
        executed_utc = _iso_to_utc(executed_at)
        note = str(notes or "").strip()[:500]
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if normalized_side == "OPENING" and fee != 0:
            raise TradeValidationError("初始化现有持仓时，每股成本应已包含历史买入手续费，手续费栏必须为 0。")

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            user_row = connection.execute(
                "SELECT is_active FROM users WHERE user_id = ?",
                (normalized_user_id,),
            ).fetchone()
            if user_row is None or int(user_row["is_active"]) != 1:
                connection.rollback()
                raise TradeValidationError("账号不存在或已停用，请重新登录。")

            state = self._get_state(connection, normalized_user_id, normalized_symbol)
            previous_timestamp = str(state["latest_executed_at_utc"] or "")
            if previous_timestamp and executed_utc < previous_timestamp:
                connection.rollback()
                raise TradeValidationError(
                    "这笔交易的时间早于已有的最后一笔记录。为保证平均成本和盈亏计算正确，"
                    "请按成交时间从早到晚录入；不要在现有账本之后补录更早的交易。"
                )

            shares_before = state["shares"]
            avg_before = state["average_cost"]
            assert isinstance(shares_before, Decimal) and isinstance(avg_before, Decimal)
            existing_count = int(state["transaction_count"])
            gross = (qty * unit_price).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)
            realized = Decimal("0.00")

            if normalized_side == "OPENING":
                if existing_count > 0:
                    connection.rollback()
                    raise TradeValidationError(
                        "这个标的已经有交易记录，不能再次初始化。若要记录历史买卖，请先确定一套完整的时间顺序账本。"
                    )
                shares_after = qty
                avg_after = unit_price
            elif normalized_side == "BUY":
                shares_after = (shares_before + qty).quantize(_QTY_STEP, rounding=ROUND_HALF_UP)
                total_cost_before = shares_before * avg_before
                avg_after = ((total_cost_before + gross + fee) / shares_after).quantize(
                    _COST_STEP, rounding=ROUND_HALF_UP
                )
            else:  # SELL
                if qty > shares_before:
                    connection.rollback()
                    raise TradeValidationError(
                        f"卖出数量超过当前持有数量。当前只有 {shares_before:.4f} 股，不能卖出 {qty:.4f} 股。"
                    )
                cost_basis_sold = (qty * avg_before).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)
                net_proceeds = gross - fee
                realized = (net_proceeds - cost_basis_sold).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)
                shares_after = (shares_before - qty).quantize(_QTY_STEP, rounding=ROUND_HALF_UP)
                if shares_after <= 0:
                    shares_after = Decimal("0")
                    avg_after = Decimal("0")
                else:
                    # Moving-average cost per share stays the same for remaining shares.
                    avg_after = avg_before

            cursor = connection.execute(
                """
                INSERT INTO portfolio_transactions (
                    user_id, symbol, side, quantity, price, gross_amount, fees,
                    realized_pnl, shares_before, average_cost_before,
                    shares_after, average_cost_after, executed_at_utc,
                    created_at_utc, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_user_id,
                    normalized_symbol,
                    normalized_side,
                    float(qty),
                    float(unit_price),
                    float(gross),
                    float(fee),
                    float(realized),
                    float(shares_before),
                    float(avg_before),
                    float(shares_after),
                    float(avg_after),
                    executed_utc,
                    now_utc,
                    note,
                ),
            )
            transaction_id = int(cursor.lastrowid)
            connection.commit()

        return {
            "transaction_id": transaction_id,
            "user_id": normalized_user_id,
            "symbol": normalized_symbol,
            "side": normalized_side,
            "quantity": float(qty),
            "price": float(unit_price),
            "gross_amount": float(gross),
            "fees": float(fee),
            "realized_pnl": float(realized),
            "shares_after": float(shares_after),
            "average_cost_after": float(avg_after),
            "executed_at_utc": executed_utc,
        }

    def void_transaction(
        self,
        *,
        user_id: int | str,
        transaction_id: int,
        reason: str,
    ) -> dict[str, Any]:
        """Void one user's transaction and replay the remaining ledger chronologically.

        A void is permanent and auditable. It is rejected if removing the row would
        make any later sale exceed the remaining position. In that case later sales
        must be voided first, newest-to-oldest, then the erroneous record can be voided.
        """
        normalized_user_id = _positive_user_id(user_id)
        try:
            tx_id = int(transaction_id)
        except (TypeError, ValueError) as exc:
            raise TradeValidationError("交易记录编号无效。") from exc
        clean_reason = str(reason or "").strip()[:500]
        if not clean_reason:
            raise TradeValidationError("请填写冲正原因，例如：成交数量录入错误。")
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            user = connection.execute(
                "SELECT is_active FROM users WHERE user_id = ?", (normalized_user_id,)
            ).fetchone()
            if user is None or int(user["is_active"]) != 1:
                connection.rollback()
                raise TradeValidationError("账号不存在或已停用，请重新登录。")
            tx = connection.execute(
                "SELECT * FROM portfolio_transactions WHERE transaction_id = ? AND user_id = ?",
                (tx_id, normalized_user_id),
            ).fetchone()
            if tx is None:
                connection.rollback()
                raise TradeValidationError("找不到这笔交易，或它不属于当前账号。")
            if connection.execute(
                "SELECT 1 FROM portfolio_transaction_voids WHERE transaction_id = ? AND user_id = ?",
                (tx_id, normalized_user_id),
            ).fetchone():
                connection.rollback()
                raise TradeValidationError("这笔交易已经冲正，不能重复操作。")

            connection.execute(
                "INSERT INTO portfolio_transaction_voids(transaction_id, user_id, reason, voided_at_utc) "
                "VALUES (?, ?, ?, ?)",
                (tx_id, normalized_user_id, clean_reason, now_utc),
            )

            active_rows = connection.execute(
                """
                SELECT * FROM portfolio_transactions
                WHERE user_id = ? AND symbol = ?
                  AND NOT EXISTS (
                    SELECT 1 FROM portfolio_transaction_voids v
                    WHERE v.transaction_id = portfolio_transactions.transaction_id
                      AND v.user_id = portfolio_transactions.user_id
                  )
                ORDER BY executed_at_utc, transaction_id
                """,
                (normalized_user_id, tx["symbol"]),
            ).fetchall()

            shares = Decimal("0")
            average_cost = Decimal("0")
            processed = 0
            for row in active_rows:
                side = str(row["side"])
                qty = Decimal(str(row["quantity"]))
                price = Decimal(str(row["price"]))
                gross = Decimal(str(row["gross_amount"]))
                fee = Decimal(str(row["fees"]))
                shares_before = shares
                cost_before = average_cost
                realized = Decimal("0.00")

                if side == "OPENING":
                    if processed != 0 or shares != 0:
                        connection.rollback()
                        raise TradeValidationError(
                            "冲正会让期初持仓顺序失效，已取消本次操作；请先冲正该标的后续交易。"
                        )
                    shares = qty
                    average_cost = price
                elif side == "BUY":
                    next_shares = (shares + qty).quantize(_QTY_STEP, rounding=ROUND_HALF_UP)
                    if next_shares <= 0:
                        connection.rollback()
                        raise TradeValidationError("重新计算后的持仓数量无效，本次冲正已取消。")
                    average_cost = ((shares * average_cost + gross + fee) / next_shares).quantize(
                        _COST_STEP, rounding=ROUND_HALF_UP
                    )
                    shares = next_shares
                elif side == "SELL":
                    if qty > shares:
                        connection.rollback()
                        raise TradeValidationError(
                            "不能冲正这笔较早交易，因为后续卖出会超过重新计算后的持仓。"
                            "请先按时间倒序冲正造成冲突的后续卖出，再重试。"
                        )
                    basis = (qty * average_cost).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)
                    realized = (gross - fee - basis).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)
                    shares = (shares - qty).quantize(_QTY_STEP, rounding=ROUND_HALF_UP)
                    if shares <= 0:
                        shares = Decimal("0")
                        average_cost = Decimal("0")
                else:
                    connection.rollback()
                    raise TradeValidationError(f"未知交易类型 {side}，无法安全重算账本。")

                connection.execute(
                    """
                    UPDATE portfolio_transactions
                    SET realized_pnl = ?, shares_before = ?, average_cost_before = ?,
                        shares_after = ?, average_cost_after = ?
                    WHERE transaction_id = ? AND user_id = ?
                    """,
                    (
                        float(realized), float(shares_before), float(cost_before),
                        float(shares), float(average_cost), row["transaction_id"], normalized_user_id,
                    ),
                )
                processed += 1
            connection.commit()

        return {
            "transaction_id": tx_id,
            "user_id": normalized_user_id,
            "symbol": str(tx["symbol"]),
            "reason": clean_reason,
            "recomputed_active_transactions": processed,
            "voided_at_utc": now_utc,
        }

    def void_dividend(
        self,
        *,
        user_id: int | str,
        dividend_id: int,
        reason: str,
    ) -> dict[str, Any]:
        """Void a cash-dividend record while retaining its audit trail."""
        normalized_user_id = _positive_user_id(user_id)
        try:
            div_id = int(dividend_id)
        except (TypeError, ValueError) as exc:
            raise TradeValidationError("股息记录编号无效。") from exc
        clean_reason = str(reason or "").strip()[:500]
        if not clean_reason:
            raise TradeValidationError("请填写冲正原因，例如：分红金额录入错误。")
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            user = connection.execute(
                "SELECT is_active FROM users WHERE user_id = ?", (normalized_user_id,)
            ).fetchone()
            if user is None or int(user["is_active"]) != 1:
                connection.rollback()
                raise TradeValidationError("账号不存在或已停用，请重新登录。")
            row = connection.execute(
                "SELECT dividend_id, symbol FROM portfolio_dividends WHERE dividend_id = ? AND user_id = ?",
                (div_id, normalized_user_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise TradeValidationError("找不到这笔股息记录，或它不属于当前账号。")
            if connection.execute(
                "SELECT 1 FROM portfolio_dividend_voids WHERE dividend_id = ? AND user_id = ?",
                (div_id, normalized_user_id),
            ).fetchone():
                connection.rollback()
                raise TradeValidationError("这笔股息已经冲正，不能重复操作。")
            connection.execute(
                "INSERT INTO portfolio_dividend_voids(dividend_id, user_id, reason, voided_at_utc) VALUES (?, ?, ?, ?)",
                (div_id, normalized_user_id, clean_reason, now_utc),
            )
            connection.commit()
        return {"dividend_id": div_id, "user_id": normalized_user_id, "symbol": str(row["symbol"]), "reason": clean_reason}

    def record_dividend(
        self,
        *,
        user_id: int | str,
        symbol: str = "QQQM",
        eligible_shares: Any,
        dividend_per_share: Any,
        tax_withheld: Any = 0,
        fees: Any = 0,
        paid_at: str | datetime | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Append a cash dividend receipt for one user without changing share balance.

        `eligible_shares` is the number of shares entitled to the distribution,
        which may differ from current holdings if shares were bought/sold around
        the ex-dividend date. `dividend_per_share` is the gross cash amount per share.
        """
        normalized_user_id = _positive_user_id(user_id)
        normalized_symbol = self._normalize_symbol(symbol)
        shares = _decimal_value(eligible_shares, "参与分红的股数", _QTY_STEP, allow_zero=False)
        per_share = _decimal_value(dividend_per_share, "每股股息", _PRICE_STEP, allow_zero=False)
        tax = _decimal_value(tax_withheld, "预扣税款", _FEE_STEP, allow_zero=True)
        fee = _decimal_value(fees, "其他费用", _FEE_STEP, allow_zero=True)
        paid_utc = _iso_to_utc(paid_at)
        note = str(notes or "").strip()[:500]
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        gross = (shares * per_share).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)
        if gross <= 0:
            raise TradeValidationError("按美分四舍五入后，股息总额小于 $0.01，无法记账。")
        if tax + fee > gross:
            raise TradeValidationError("预扣税款与其他费用不能超过股息总额。")
        net = (gross - tax - fee).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            user_row = connection.execute(
                "SELECT is_active FROM users WHERE user_id = ?", (normalized_user_id,)
            ).fetchone()
            if user_row is None or int(user_row["is_active"]) != 1:
                connection.rollback()
                raise TradeValidationError("账号不存在或已停用，请重新登录。")
            cursor = connection.execute(
                """
                INSERT INTO portfolio_dividends (
                    user_id, symbol, eligible_shares, dividend_per_share,
                    gross_amount, tax_withheld, fees, net_amount,
                    paid_at_utc, created_at_utc, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_user_id, normalized_symbol, float(shares), float(per_share),
                    float(gross), float(tax), float(fee), float(net),
                    paid_utc, now_utc, note,
                ),
            )
            dividend_id = int(cursor.lastrowid)
            connection.commit()

        return {
            "dividend_id": dividend_id,
            "user_id": normalized_user_id,
            "symbol": normalized_symbol,
            "side": "DIVIDEND",
            "eligible_shares": float(shares),
            "dividend_per_share": float(per_share),
            "gross_amount": float(gross),
            "tax_withheld": float(tax),
            "fees": float(fee),
            "net_amount": float(net),
            "paid_at_utc": paid_utc,
            "notes": note,
        }

    def list_dividends(
        self,
        *,
        user_id: int | str,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List dividend receipts scoped to the currently authenticated user."""
        normalized_user_id = _positive_user_id(user_id)
        safe_limit = max(1, min(int(limit), 1000))
        params: list[Any] = [normalized_user_id]
        where = "portfolio_dividends.user_id = ?"
        if symbol is not None:
            where += " AND portfolio_dividends.symbol = ?"
            params.append(self._normalize_symbol(symbol))
        params.append(safe_limit)
        with closing(self._connect()) as connection:
            user_row = connection.execute(
                "SELECT is_active FROM users WHERE user_id = ?", (normalized_user_id,)
            ).fetchone()
            if user_row is None or int(user_row["is_active"]) != 1:
                raise TradeValidationError("账号不存在或已停用，请重新登录。")
            rows = connection.execute(
                f"SELECT portfolio_dividends.*, "
                "CASE WHEN v.dividend_id IS NULL THEN 0 ELSE 1 END AS is_voided, "
                "v.reason AS void_reason, v.voided_at_utc "
                f"FROM portfolio_dividends LEFT JOIN portfolio_dividend_voids v "
                "ON v.dividend_id = portfolio_dividends.dividend_id AND v.user_id = portfolio_dividends.user_id "
                f"WHERE {where} ORDER BY paid_at_utc DESC, portfolio_dividends.dividend_id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_activity(
        self,
        *,
        user_id: int | str,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Merge trade and dividend records into one newest-first activity list."""
        safe_limit = max(1, min(int(limit), 1000))
        trades = self.list_transactions(user_id=user_id, symbol=symbol, limit=safe_limit)
        dividends = self.list_dividends(user_id=user_id, symbol=symbol, limit=safe_limit)
        merged: list[dict[str, Any]] = []
        for row in trades:
            merged.append({**row, "activity_type": row.get("side"), "activity_id": f"T-{row['transaction_id']}"})
        for row in dividends:
            merged.append({
                **row,
                "transaction_id": None,
                "side": "DIVIDEND",
                "quantity": row.get("eligible_shares"),
                "price": row.get("dividend_per_share"),
                "realized_pnl": None,
                "shares_before": None,
                "average_cost_before": None,
                "shares_after": None,
                "average_cost_after": None,
                "executed_at_utc": row.get("paid_at_utc"),
                "activity_type": "DIVIDEND",
                "activity_id": f"D-{row['dividend_id']}",
            })
        merged.sort(
            key=lambda row: (str(row.get("executed_at_utc") or ""), str(row.get("activity_id") or "")),
            reverse=True,
        )
        return merged[:safe_limit]

    def get_position_summary(
        self,
        *,
        user_id: int | str,
        symbol: str = "QQQM",
        reference_price: Any | None = None,
    ) -> dict[str, Any]:
        normalized_user_id = _positive_user_id(user_id)
        normalized_symbol = self._normalize_symbol(symbol)
        price = None if reference_price is None else _decimal_value(
            reference_price, "参考行情价", _PRICE_STEP, allow_zero=False
        )
        with closing(self._connect()) as connection:
            user_row = connection.execute(
                "SELECT is_active FROM users WHERE user_id = ?",
                (normalized_user_id,),
            ).fetchone()
            if user_row is None or int(user_row["is_active"]) != 1:
                raise TradeValidationError("账号不存在或已停用，请重新登录。")
            state = self._get_state(connection, normalized_user_id, normalized_symbol)
            dividend_row = connection.execute(
                """
                SELECT
                    COALESCE(SUM(gross_amount), 0) AS gross_total,
                    COALESCE(SUM(tax_withheld), 0) AS tax_total,
                    COALESCE(SUM(fees), 0) AS fee_total,
                    COALESCE(SUM(net_amount), 0) AS net_total,
                    COUNT(*) AS dividend_count
                FROM portfolio_dividends
                WHERE user_id = ? AND symbol = ?
                  AND NOT EXISTS (
                    SELECT 1 FROM portfolio_dividend_voids v
                    WHERE v.dividend_id = portfolio_dividends.dividend_id
                      AND v.user_id = portfolio_dividends.user_id
                  )
                """,
                (normalized_user_id, normalized_symbol),
            ).fetchone()

        shares = state["shares"]
        avg_cost = state["average_cost"]
        realized = state["realized_pnl_total"]
        dividend_gross = Decimal(str(dividend_row["gross_total"] or 0))
        dividend_tax = Decimal(str(dividend_row["tax_total"] or 0))
        dividend_fees = Decimal(str(dividend_row["fee_total"] or 0))
        dividend_net = Decimal(str(dividend_row["net_total"] or 0))
        assert isinstance(shares, Decimal) and isinstance(avg_cost, Decimal) and isinstance(realized, Decimal)
        cost_basis = (shares * avg_cost).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP)
        market_value = (shares * price).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP) if price is not None else None
        unrealized = (market_value - cost_basis).quantize(_MONEY_STEP, rounding=ROUND_HALF_UP) if market_value is not None else None
        unrealized_pct = float(unrealized / cost_basis * Decimal("100")) if unrealized is not None and cost_basis > 0 else (0.0 if unrealized is not None else None)
        return {
            "user_id": normalized_user_id,
            "symbol": normalized_symbol,
            "shares": float(shares),
            "average_cost": float(avg_cost),
            "cost_basis": float(cost_basis),
            "reference_price": float(price) if price is not None else None,
            "market_value": float(market_value) if market_value is not None else None,
            "unrealized_pnl": float(unrealized) if unrealized is not None else None,
            "unrealized_pnl_pct": unrealized_pct,
            "realized_pnl_total": float(realized),
            "dividend_gross_total": float(dividend_gross),
            "dividend_tax_total": float(dividend_tax),
            "dividend_fees_total": float(dividend_fees),
            "dividend_income_total": float(dividend_net),
            "realized_plus_dividends_total": float(realized + dividend_net),
            "dividend_count": int(dividend_row["dividend_count"]),
            "transaction_count": int(state["transaction_count"]),
            "latest_executed_at_utc": str(state["latest_executed_at_utc"] or ""),
        }

    def list_transactions(
        self,
        *,
        user_id: int | str,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        normalized_user_id = _positive_user_id(user_id)
        safe_limit = max(1, min(int(limit), 1000))
        params: list[Any] = [normalized_user_id]
        where = "portfolio_transactions.user_id = ?"
        if symbol is not None:
            where += " AND portfolio_transactions.symbol = ?"
            params.append(self._normalize_symbol(symbol))
        params.append(safe_limit)

        with closing(self._connect()) as connection:
            user_row = connection.execute(
                "SELECT is_active FROM users WHERE user_id = ?",
                (normalized_user_id,),
            ).fetchone()
            if user_row is None or int(user_row["is_active"]) != 1:
                raise TradeValidationError("账号不存在或已停用，请重新登录。")
            rows = connection.execute(
                "SELECT portfolio_transactions.*, "
                "CASE WHEN v.transaction_id IS NULL THEN 0 ELSE 1 END AS is_voided, "
                "v.reason AS void_reason, v.voided_at_utc "
                "FROM portfolio_transactions LEFT JOIN portfolio_transaction_voids v "
                "ON v.transaction_id = portfolio_transactions.transaction_id AND v.user_id = portfolio_transactions.user_id "
                f"WHERE {where} ORDER BY executed_at_utc DESC, portfolio_transactions.transaction_id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]
