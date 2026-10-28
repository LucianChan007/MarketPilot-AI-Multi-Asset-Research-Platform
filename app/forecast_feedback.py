from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "market_data.db"
DEFAULT_FORECAST_DIR = PROJECT_ROOT / "reports" / "forecast"

_POINT_FORECAST_KEYS = (
    "zero_return_baseline_pct",
    "historical_mean_return_pct",
    "ridge_full_return_pct",
    "ridge_returns_only_return_pct",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    source_file TEXT NOT NULL,
    current_price REAL NOT NULL,
    zero_return_baseline_pct REAL,
    historical_mean_return_pct REAL,
    ridge_full_return_pct REAL,
    ridge_returns_only_return_pct REAL,
    historical_return_bands_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    actual_target_date TEXT,
    actual_adjusted_close REAL,
    actual_return_pct REAL,
    error_zero_return_bps REAL,
    error_historical_mean_bps REAL,
    error_ridge_full_bps REAL,
    error_ridge_returns_only_bps REAL,
    direction_correct_zero_return INTEGER,
    direction_correct_historical_mean INTEGER,
    direction_correct_ridge_full INTEGER,
    direction_correct_ridge_returns_only INTEGER,
    interval_20_covered INTEGER,
    interval_60_covered INTEGER,
    interval_252_covered INTEGER,
    evaluated_at_utc TEXT,
    UNIQUE (ticker, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_forecast_feedback_status_date
ON forecast_feedback(status, as_of_date);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _validate_date(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty YYYY-MM-DD string")
    try:
        parsed = datetime.strptime(value[:10], "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format: {value!r}") from exc
    return parsed.strftime("%Y-%m-%d")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


@contextmanager
def _managed_connection(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection, commit or rollback, and always close it.

    sqlite3.Connection's built-in context manager handles transactions but does
    not close the file handle. Explicitly closing is important on Windows, where
    an open SQLite handle can prevent temporary database files from being deleted.
    """
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_feedback_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the feedback table in the given SQLite database if needed."""
    with _managed_connection(db_path) as connection:
        connection.executescript(_SCHEMA)


def record_forecast_payload(
    payload: Mapping[str, Any],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_file: str = "<in-memory>",
    ticker: str = "QQQM",
) -> dict[str, Any]:
    """Persist a forecast without overwriting a forecast already logged for that date.

    Returns a status dictionary. The first forecast for (ticker, as_of_date) is
    immutable by default; rerunning a script cannot rewrite the record later.
    """
    as_of_date = _validate_date(payload.get("as_of_date"), "as_of_date")
    current_price = _as_float(payload.get("latest_adjusted_close"))
    if current_price is None or current_price <= 0:
        raise ValueError("latest_adjusted_close must be a positive number")

    point_forecasts = payload.get("point_forecasts")
    if not isinstance(point_forecasts, Mapping):
        raise ValueError("point_forecasts must be a JSON object")

    bands = payload.get("historical_return_bands")
    if not isinstance(bands, Mapping):
        raise ValueError("historical_return_bands must be a JSON object")

    values = {key: _as_float(point_forecasts.get(key)) for key in _POINT_FORECAST_KEYS}
    generated_at = _utc_now()
    initialize_feedback_db(db_path)

    with _managed_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO forecast_feedback (
                ticker, as_of_date, generated_at_utc, source_file, current_price,
                zero_return_baseline_pct, historical_mean_return_pct,
                ridge_full_return_pct, ridge_returns_only_return_pct,
                historical_return_bands_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.upper(),
                as_of_date,
                generated_at,
                str(source_file),
                current_price,
                values["zero_return_baseline_pct"],
                values["historical_mean_return_pct"],
                values["ridge_full_return_pct"],
                values["ridge_returns_only_return_pct"],
                json.dumps(bands, ensure_ascii=False, sort_keys=True),
            ),
        )
        inserted = cursor.rowcount == 1
        row = connection.execute(
            "SELECT id, ticker, as_of_date, status, source_file FROM forecast_feedback WHERE ticker=? AND as_of_date=?",
            (ticker.upper(), as_of_date),
        ).fetchone()

    return {
        "inserted": inserted,
        "message": "forecast recorded" if inserted else "forecast already recorded for this ticker/date; original kept",
        "id": int(row["id"]),
        "ticker": row["ticker"],
        "as_of_date": row["as_of_date"],
        "status": row["status"],
        "source_file": row["source_file"],
    }


def record_latest_forecast(
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    ticker: str = "QQQM",
) -> dict[str, Any]:
    """Record the newest timestamped forecast JSON from a forecast directory."""
    folder = Path(forecast_dir)
    if not folder.exists():
        raise FileNotFoundError(f"forecast directory does not exist: {folder}")
    files = list(folder.glob(f"{ticker.upper()}_return_forecast_latest_*.json"))
    if not files:
        raise FileNotFoundError(
            f"no timestamped forecast file found in {folder} matching "
            f"{ticker.upper()}_return_forecast_latest_*.json"
        )
    latest = max(files, key=lambda path: (path.stat().st_mtime, path.name))
    with latest.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"forecast JSON top level must be an object: {latest.name}")
    return record_forecast_payload(
        payload,
        db_path=db_path,
        source_file=str(latest.resolve()),
        ticker=ticker,
    )


def evaluate_pending_forecasts(
    price_history: Iterable[Mapping[str, Any]],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    ticker: str = "QQQM",
) -> list[dict[str, Any]]:
    """Settle pending forecasts using the first later trading date in price_history.

    price_history must contain normalized records with `date` and
    `adjusted_close`. The target date is inferred from the actual trading data,
    so weekends and exchange holidays are naturally skipped.
    """
    prices: list[tuple[str, float]] = []
    for record in price_history:
        date_value = _validate_date(record.get("date"), "price_history.date")
        close_value = _as_float(record.get("adjusted_close"))
        if close_value is None or close_value <= 0:
            continue
        prices.append((date_value, close_value))
    prices.sort(key=lambda item: item[0])
    if not prices:
        return []

    initialize_feedback_db(db_path)
    updates: list[dict[str, Any]] = []
    with _managed_connection(db_path) as connection:
        pending = connection.execute(
            "SELECT * FROM forecast_feedback WHERE ticker=? AND status='pending' ORDER BY as_of_date, id",
            (ticker.upper(),),
        ).fetchall()

        for row in pending:
            target = next((item for item in prices if item[0] > row["as_of_date"]), None)
            if target is None:
                continue
            target_date, actual_close = target
            current_price = float(row["current_price"])
            actual_return_pct = (actual_close / current_price - 1.0) * 100.0

            prediction_columns = {
                "zero_return_baseline": row["zero_return_baseline_pct"],
                "historical_mean": row["historical_mean_return_pct"],
                "ridge_full": row["ridge_full_return_pct"],
                "ridge_returns_only": row["ridge_returns_only_return_pct"],
            }
            errors: dict[str, float | None] = {}
            directions: dict[str, int | None] = {}
            actual_direction = 1 if actual_return_pct > 0 else 0
            for name, prediction in prediction_columns.items():
                if prediction is None:
                    errors[name] = None
                    directions[name] = None
                    continue
                # Values are expressed in percent points. Multiplying their
                # difference by 100 converts percentage points to basis points.
                errors[name] = (actual_return_pct - float(prediction)) * 100.0
                pred_direction = 1 if float(prediction) > 0 else 0
                directions[name] = int(pred_direction == actual_direction)

            try:
                bands = json.loads(row["historical_return_bands_json"])
            except (TypeError, json.JSONDecodeError):
                bands = {}
            coverage: dict[int, int | None] = {}
            for window in (20, 60, 252):
                band = bands.get(f"last_{window}_sessions", {}) if isinstance(bands, dict) else {}
                lower = _as_float(band.get("p10_return_pct")) if isinstance(band, dict) else None
                upper = _as_float(band.get("p90_return_pct")) if isinstance(band, dict) else None
                coverage[window] = int(lower <= actual_return_pct <= upper) if lower is not None and upper is not None else None

            now = _utc_now()
            connection.execute(
                """
                UPDATE forecast_feedback SET
                    status='evaluated', actual_target_date=?, actual_adjusted_close=?, actual_return_pct=?,
                    error_zero_return_bps=?, error_historical_mean_bps=?, error_ridge_full_bps=?,
                    error_ridge_returns_only_bps=?, direction_correct_zero_return=?,
                    direction_correct_historical_mean=?, direction_correct_ridge_full=?,
                    direction_correct_ridge_returns_only=?, interval_20_covered=?,
                    interval_60_covered=?, interval_252_covered=?, evaluated_at_utc=?
                WHERE id=? AND status='pending'
                """,
                (
                    target_date, actual_close, actual_return_pct,
                    errors["zero_return_baseline"], errors["historical_mean"],
                    errors["ridge_full"], errors["ridge_returns_only"],
                    directions["zero_return_baseline"], directions["historical_mean"],
                    directions["ridge_full"], directions["ridge_returns_only"],
                    coverage[20], coverage[60], coverage[252], now, row["id"],
                ),
            )
            updates.append({
                "id": int(row["id"]),
                "ticker": row["ticker"],
                "as_of_date": row["as_of_date"],
                "target_date": target_date,
                "current_price": current_price,
                "actual_adjusted_close": actual_close,
                "actual_return_pct": actual_return_pct,
                "errors_bps": errors,
                "interval_coverage": coverage,
            })

    return updates


def get_feedback_summary(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    ticker: str = "QQQM",
    last_n: int | None = None,
) -> dict[str, Any]:
    """Summarize realized sample-out-of-sample performance for the logged forecasts."""
    initialize_feedback_db(db_path)
    with _managed_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM forecast_feedback WHERE ticker=? AND status='evaluated' ORDER BY actual_target_date, id",
            (ticker.upper(),),
        ).fetchall()

    if last_n is not None:
        if last_n <= 0:
            raise ValueError("last_n must be positive or None")
        rows = rows[-last_n:]

    model_error_columns = {
        "zero_return_baseline": "error_zero_return_bps",
        "historical_mean_baseline": "error_historical_mean_bps",
        "ridge_full": "error_ridge_full_bps",
        "ridge_returns_only": "error_ridge_returns_only_bps",
    }
    model_direction_columns = {
        "zero_return_baseline": "direction_correct_zero_return",
        "historical_mean_baseline": "direction_correct_historical_mean",
        "ridge_full": "direction_correct_ridge_full",
        "ridge_returns_only": "direction_correct_ridge_returns_only",
    }
    models: dict[str, Any] = {}
    for model_name, error_col in model_error_columns.items():
        errors = [float(row[error_col]) for row in rows if row[error_col] is not None]
        directions_col = model_direction_columns[model_name]
        directions = [int(row[directions_col]) for row in rows if row[directions_col] is not None]
        if errors:
            models[model_name] = {
                "n": len(errors),
                "mae_bps": sum(abs(value) for value in errors) / len(errors),
                "rmse_bps": math.sqrt(sum(value * value for value in errors) / len(errors)),
                "direction_accuracy": sum(directions) / len(directions) if directions else None,
            }
        else:
            models[model_name] = {"n": 0, "mae_bps": None, "rmse_bps": None, "direction_accuracy": None}

    coverage_summary: dict[str, float | None] = {}
    for window in (20, 60, 252):
        values = [int(row[f"interval_{window}_covered"]) for row in rows if row[f"interval_{window}_covered"] is not None]
        coverage_summary[f"last_{window}_sessions_p10_p90_coverage"] = sum(values) / len(values) if values else None

    return {
        "ticker": ticker.upper(),
        "evaluated_forecasts": len(rows),
        "first_target_date": rows[0]["actual_target_date"] if rows else None,
        "last_target_date": rows[-1]["actual_target_date"] if rows else None,
        "models": models,
        "interval_coverage": coverage_summary,
        "note": "Small samples are noisy. Compare against simple baselines and do not use this report alone as a trading signal.",
    }


def list_forecasts(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    ticker: str = "QQQM",
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Return recent forecast ledger records, newest first, for a future GUI table."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    initialize_feedback_db(db_path)
    with _managed_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM forecast_feedback WHERE ticker=? ORDER BY as_of_date DESC, id DESC LIMIT ?",
            (ticker.upper(), limit),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["historical_return_bands"] = json.loads(item.pop("historical_return_bands_json"))
        except (TypeError, json.JSONDecodeError):
            item["historical_return_bands"] = {}
        result.append(item)
    return result
