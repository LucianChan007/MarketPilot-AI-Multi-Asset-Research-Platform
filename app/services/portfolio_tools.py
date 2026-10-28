"""CSV export and safe local backup helpers for the multi-user portfolio ledger."""
from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from app.services.portfolio_ledger import PortfolioLedger, TradeValidationError


def export_user_activity_csv(
    ledger: PortfolioLedger,
    *,
    user_id: int | str,
    symbol: str | None = None,
    limit: int = 1000,
) -> bytes:
    """Export only the requested account's ledger activity as UTF-8-SIG CSV."""
    activities = ledger.list_activity(user_id=user_id, symbol=symbol, limit=limit)
    output = StringIO(newline="")
    columns = [
        "activity_id", "activity_type", "symbol", "executed_at_utc", "quantity_or_eligible_shares",
        "price_or_dividend_per_share", "gross_amount", "fees", "tax_withheld", "net_amount_or_realized_pnl",
        "shares_after", "average_cost_after", "is_voided", "void_reason", "notes",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in activities:
        is_dividend = row.get("activity_type") == "DIVIDEND"
        writer.writerow({
            "activity_id": row.get("activity_id", ""),
            "activity_type": row.get("activity_type", row.get("side", "")),
            "symbol": row.get("symbol", ""),
            "executed_at_utc": row.get("executed_at_utc", row.get("paid_at_utc", "")),
            "quantity_or_eligible_shares": row.get("eligible_shares") if is_dividend else row.get("quantity"),
            "price_or_dividend_per_share": row.get("dividend_per_share") if is_dividend else row.get("price"),
            "gross_amount": row.get("gross_amount", ""),
            "fees": row.get("fees", ""),
            "tax_withheld": row.get("tax_withheld", ""),
            "net_amount_or_realized_pnl": row.get("net_amount") if is_dividend else row.get("realized_pnl", ""),
            "shares_after": "" if is_dividend else row.get("shares_after", ""),
            "average_cost_after": "" if is_dividend else row.get("average_cost_after", ""),
            "is_voided": bool(row.get("is_voided", 0)),
            "void_reason": row.get("void_reason") or "",
            "notes": row.get("notes") or "",
        })
    return output.getvalue().encode("utf-8-sig")


def default_private_backup_dir() -> Path:
    """Keep backups outside the project folder, reducing accidental Git commits."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base = Path(local_app_data)
    else:
        base = Path.home() / "AppData" / "Local"
    return base / "QQQMInvestmentAssistant" / "backups"


def backup_sqlite_database(
    source_path: str | Path,
    destination_dir: str | Path | None = None,
) -> Path:
    """Create a consistent SQLite backup with SQLite's backup API and verify integrity."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到需要备份的数据库：{source}")
    destination_root = Path(destination_dir).expanduser().resolve() if destination_dir else default_private_backup_dir()
    destination_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = destination_root / f"users_db_backup_{stamp}.sqlite3"
    suffix = 1
    while destination.exists():
        destination = destination_root / f"users_db_backup_{stamp}_{suffix}.sqlite3"
        suffix += 1

    src_conn: sqlite3.Connection | None = None
    dst_conn: sqlite3.Connection | None = None
    try:
        src_conn = sqlite3.connect(source, timeout=15)
        dst_conn = sqlite3.connect(destination, timeout=15)
        src_conn.backup(dst_conn)
        check = dst_conn.execute("PRAGMA integrity_check").fetchone()
        if not check or check[0] != "ok":
            raise sqlite3.DatabaseError(f"备份完整性检查失败：{check[0] if check else 'no result'}")
    except Exception:
        try:
            if dst_conn is not None:
                dst_conn.close()
        finally:
            if destination.exists():
                destination.unlink()
        raise
    finally:
        if src_conn is not None:
            src_conn.close()
        if dst_conn is not None:
            dst_conn.close()
    return destination
