
from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any

import requests
from dotenv import load_dotenv

from app.config import DB_PATH, ROOT_DIR


load_dotenv(ROOT_DIR / ".env")

FRED_API_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)

SERIES = {
    "EFFR": {
        "name": "有效联邦基金利率",
        "unit": "%",
        "frequency": "daily",
        "refresh_days": 120,
    },
    "DGS10": {
        "name": "美国10年期国债收益率",
        "unit": "%",
        "frequency": "daily",
        "refresh_days": 120,
    },
    "DFII10": {
        "name": "美国10年期实际国债收益率",
        "unit": "%",
        "frequency": "daily",
        "refresh_days": 120,
    },
    "CPIAUCSL": {
        "name": "美国CPI消费者价格指数",
        "unit": "指数，1982-1984年=100",
        "frequency": "monthly",
        "refresh_days": 800,
    },
}


@contextmanager
def database():
    """打开本地行情数据库，并确保宏观数据表存在。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS macro_observations (
                series_id TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                value REAL NOT NULL,
                retrieved_at TEXT NOT NULL,
                PRIMARY KEY (series_id, observation_date)
            )
            """
        )
        conn.commit()

        yield conn

        conn.commit()
    finally:
        conn.close()


def _years_ago(today: date, years: int) -> date:
    """计算若干年前的日期，兼容闰日。"""
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(
            year=today.year - years,
            month=2,
            day=28,
        )


def _fetch_observations(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """从 FRED 读取一个时间序列的历史观测值。"""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
        "limit": 100000,
    }

    try:
        response = requests.get(
            FRED_API_URL,
            params=params,
            timeout=30,
        )
    except requests.RequestException as exc:
        # 不直接打印完整请求 URL，避免意外泄露 API Key。
        raise RuntimeError(
            f"网络请求失败，错误类型：{type(exc).__name__}"
        ) from None

    if not response.ok:
        # FRED 的错误信息不需要包含请求密钥。
        detail = response.text[:300].replace(api_key, "***")
        raise RuntimeError(
            f"FRED HTTP {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("FRED 返回的内容不是有效 JSON。") from None

    observations = payload.get("observations")

    if not isinstance(observations, list):
        raise RuntimeError("FRED 响应缺少 observations 字段。")

    results = []

    for item in observations:
        raw_value = item.get("value")

        # FRED 使用 "." 表示缺失值。
        if raw_value in (None, "", "."):
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        results.append(
            {
                "date": item["date"],
                "value": value,
            }
        )

    return results


def sync_macro_data(years: int = 10) -> dict[str, Any]:
    """
    首次同步最近若干年的数据。
    后续同步会重新读取近期区间，并更新数据库中的同日期记录。
    """
    if not 1 <= years <= 50:
        raise ValueError("years 必须在 1 到 50 之间。")

    api_key = os.getenv("FRED_API_KEY", "").strip()

    if (
        not api_key
        or api_key == "replace_with_your_fred_api_key"
    ):
        raise RuntimeError(
            "缺少有效的 FRED_API_KEY，请检查项目根目录的 .env。"
        )

    today = date.today()
    end_date = today.isoformat()
    fetched_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    with database() as conn:
        for series_id, meta in SERIES.items():
            try:
                row = conn.execute(
                    """
                    SELECT MAX(observation_date) AS latest_date
                    FROM macro_observations
                    WHERE series_id = ?
                    """,
                    (series_id,),
                ).fetchone()

                latest_date = row["latest_date"]

                if latest_date:
                    # 回看近期数据，兼顾补齐遗漏与数据修订。
                    start = (
                        date.fromisoformat(latest_date)
                        - timedelta(days=meta["refresh_days"])
                    )
                else:
                    start = _years_ago(today, years)

                observations = _fetch_observations(
                    series_id=series_id,
                    api_key=api_key,
                    start_date=start.isoformat(),
                    end_date=end_date,
                )

                conn.executemany(
                    """
                    INSERT INTO macro_observations (
                        series_id,
                        observation_date,
                        value,
                        retrieved_at
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(series_id, observation_date)
                    DO UPDATE SET
                        value = excluded.value,
                        retrieved_at = excluded.retrieved_at
                    """,
                    [
                        (
                            series_id,
                            item["date"],
                            item["value"],
                            fetched_at,
                        )
                        for item in observations
                    ],
                )

                conn.commit()

                latest = conn.execute(
                    """
                    SELECT MAX(observation_date) AS latest_date
                    FROM macro_observations
                    WHERE series_id = ?
                    """,
                    (series_id,),
                ).fetchone()["latest_date"]

                results[series_id] = {
                    "name": meta["name"],
                    "rows_fetched": len(observations),
                    "latest_date": latest,
                }

            except Exception as exc:
                errors[series_id] = (
                    f"{type(exc).__name__}: "
                    f"{str(exc).replace(api_key, '***')}"
                )

    return {
        "source": "FRED",
        "results": results,
        "errors": errors,
    }


def _latest_on_or_before(
    conn: sqlite3.Connection,
    series_id: str,
    target_date: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT observation_date, value
        FROM macro_observations
        WHERE series_id = ?
          AND observation_date <= ?
        ORDER BY observation_date DESC
        LIMIT 1
        """,
        (series_id, target_date),
    ).fetchone()


def get_macro_snapshot() -> dict[str, Any]:
    """读取数据库中的最新宏观数据，供报告模块使用。"""
    retrieved_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    result: dict[str, Any] = {
        "source": "FRED",
        "retrieved_at": retrieved_at,
        "series": {},
        "notes": [
            "每个指标均使用自身最新的有效观测日期，不代表所有数据同日发布。",
            "部分观测值可能在后续被数据发布机构修订。",
            "宏观指标只能提供分析背景，不能单独证明其导致了QQQM涨跌。",
        ],
    }

    with database() as conn:
        for series_id, meta in SERIES.items():
            latest = conn.execute(
                """
                SELECT observation_date, value
                FROM macro_observations
                WHERE series_id = ?
                ORDER BY observation_date DESC
                LIMIT 1
                """,
                (series_id,),
            ).fetchone()

            if latest is None:
                result["series"][series_id] = {
                    "name": meta["name"],
                    "unit": meta["unit"],
                    "error": "数据库中没有数据，请先执行宏观数据同步。",
                }
                continue

            latest_date = date.fromisoformat(
                latest["observation_date"]
            )
            latest_value = float(latest["value"])

            recent_rows = conn.execute(
                """
                SELECT observation_date, value
                FROM macro_observations
                WHERE series_id = ?
                ORDER BY observation_date DESC
                LIMIT 6
                """,
                (series_id,),
            ).fetchall()

            item: dict[str, Any] = {
                "name": meta["name"],
                "unit": meta["unit"],
                "frequency": meta["frequency"],
                "latest_observation_date": latest["observation_date"],
                "latest_value": round(latest_value, 4),
                "recent_observations": [
                    {
                        "date": row["observation_date"],
                        "value": round(float(row["value"]), 4),
                    }
                    for row in reversed(recent_rows)
                ],
            }

            if meta["frequency"] == "daily":
                target_date = (
                    latest_date - timedelta(days=30)
                ).isoformat()

                previous = _latest_on_or_before(
                    conn,
                    series_id,
                    target_date,
                )

                if previous is not None:
                    item["comparison_date"] = previous["observation_date"]
                    item["change_vs_30_days_ago_pp"] = round(
                        latest_value - float(previous["value"]),
                        4,
                    )
                    item["change_unit"] = "percentage_points"

            if series_id == "CPIAUCSL":
                try:
                    year_ago_date = latest_date.replace(
                        year=latest_date.year - 1
                    )
                except ValueError:
                    year_ago_date = latest_date.replace(
                        year=latest_date.year - 1,
                        day=28,
                    )

                year_ago = conn.execute(
                    """
                    SELECT observation_date, value
                    FROM macro_observations
                    WHERE series_id = ?
                      AND observation_date = ?
                    """,
                    (series_id, year_ago_date.isoformat()),
                ).fetchone()

                if year_ago is not None and float(year_ago["value"]) != 0:
                    item["year_ago_date"] = year_ago["observation_date"]
                    item["year_over_year_pct"] = round(
                        (
                            latest_value / float(year_ago["value"])
                            - 1
                        ) * 100,
                        3,
                    )
                else:
                    item["year_over_year_pct"] = None
                    item["year_over_year_note"] = (
                        "没有找到去年同月的观测值，暂无法计算同比。"
                    )

                previous_month = _latest_on_or_before(
                    conn,
                    series_id,
                    (
                        latest_date - timedelta(days=25)
                    ).isoformat(),
                )

                if previous_month is not None and float(
                    previous_month["value"]
                ) != 0:
                    item["month_over_month_pct"] = round(
                        (
                            latest_value / float(previous_month["value"])
                            - 1
                        ) * 100,
                        3,
                    )

            result["series"][series_id] = item

    return result


def macro_status_rows() -> list[dict[str, Any]]:
    """返回每个宏观指标在本地数据库中的记录数量和最新日期。"""
    rows = []

    with database() as conn:
        for series_id, meta in SERIES.items():
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS row_count,
                    MAX(observation_date) AS latest_date
                FROM macro_observations
                WHERE series_id = ?
                """,
                (series_id,),
            ).fetchone()

            rows.append(
                {
                    "series_id": series_id,
                    "name": meta["name"],
                    "row_count": row["row_count"],
                    "latest_date": row["latest_date"],
                }
            )

    return rows


def main() -> int:
    """提供独立的宏观数据同步和检查命令。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="FRED 宏观经济数据采集模块"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sync_parser = subparsers.add_parser(
        "sync",
        help="同步历史宏观数据",
    )
    sync_parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="首次同步的历史年数，默认10年",
    )

    subparsers.add_parser(
        "status",
        help="查看本地宏观数据状态",
    )

    subparsers.add_parser(
        "snapshot",
        help="查看最新宏观指标摘要",
    )

    args = parser.parse_args()

    if args.command == "sync":
        try:
            result = sync_macro_data(args.years)
        except Exception as exc:
            print(f"[宏观同步失败] {type(exc).__name__}: {exc}")
            return 1

        for series_id, item in result["results"].items():
            print(
                f"[完成] {series_id}："
                f"本次获取 {item['rows_fetched']} 条，"
                f"最新观测日期 {item['latest_date']}"
            )

        for series_id, message in result["errors"].items():
            print(f"[失败] {series_id}：{message}")

        return 0 if result["results"] else 1

    if args.command == "status":
        rows = macro_status_rows()

        print("本地宏观数据状态：")
        for row in rows:
            print(
                f"{row['series_id']:8} | "
                f"{row['row_count']:5} 条 | "
                f"{row['latest_date'] or '暂无数据'} | "
                f"{row['name']}"
            )

        return 0

    if args.command == "snapshot":
        print(
            json.dumps(
                get_macro_snapshot(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
