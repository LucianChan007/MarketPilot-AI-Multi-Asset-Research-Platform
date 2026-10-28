r"""Create and integrity-check a local backup of data/users.db.

Run from the project root:
    .\.venv\Scripts\python.exe -m scripts.backup_user_database
The backup is stored outside the repository under the current Windows user's LocalAppData.
It contains all local accounts and their password hashes. Never upload or share it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.portfolio_tools import backup_sqlite_database  # noqa: E402


def main() -> int:
    source = ROOT / "data" / "users.db"
    try:
        destination = backup_sqlite_database(source)
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "success": True,
        "source_database": str(source),
        "backup_path": str(destination),
        "note": "Backup includes all local user accounts, password hashes and portfolio records. Keep it private; do not commit it to GitHub.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
