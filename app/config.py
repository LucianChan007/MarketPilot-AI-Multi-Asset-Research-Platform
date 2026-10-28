from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
DB_PATH = DATA_DIR / "market_data.db"

load_dotenv(ROOT_DIR / ".env")


def get_tickers() -> list[str]:
    raw = os.getenv("TICKERS", "QQQM,QQQ,SPY,TLT")
    tickers = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(tickers))


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
