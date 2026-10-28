from __future__ import annotations

import argparse
import hashlib
import html
import logging
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv 是可选依赖
    load_dotenv = None


# app/news_data.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

HTTP_TIMEOUT = max(5, int(os.getenv("NEWS_SYNC_TIMEOUT", "30")))
MAX_SUMMARY_CHARS = max(200, int(os.getenv("NEWS_MAX_SUMMARY_CHARS", "6000")))
NEWS_CONTACT_EMAIL = os.getenv("NEWS_CONTACT_EMAIL", "").strip()
USER_AGENT = (
    f"QQQMInvestmentAssistant/1.0 (mailto:{NEWS_CONTACT_EMAIL})"
    if NEWS_CONTACT_EMAIL
    else "QQQMInvestmentAssistant/1.0"
)
BLS_API_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BLS_API_YEARS_BACK = max(1, min(9, int(os.getenv("BLS_API_YEARS_BACK", "2"))))

# 只将真实新闻放入 news_items。BLS 新闻 RSS 在当前网络环境返回 403，
# 因此不继续尝试绕过其访问控制；宏观统计指标通过 BLS 官方 API 单独采集。
FEEDS: list[dict[str, str]] = [
    {
        "source_id": "federal_reserve_monetary",
        "source_name": "Federal Reserve / Monetary Policy",
        "category": "monetary_policy",
        "url": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    },
]

BLS_SERIES: list[dict[str, str]] = [
    {
        "series_id": "CUSR0000SA0",
        "series_name": "U.S. CPI, All Urban Consumers (Seasonally Adjusted)",
        "category": "inflation",
        "unit": "index (1982-84=100)",
        "source_url": "https://data.bls.gov/timeseries/CUSR0000SA0",
    },
    {
        "series_id": "LNS14000000",
        "series_name": "U.S. Unemployment Rate (Seasonally Adjusted)",
        "category": "employment",
        "unit": "percent",
        "source_url": "https://data.bls.gov/timeseries/LNS14000000",
    },
    {
        "series_id": "CES0000000001",
        "series_name": "U.S. Total Nonfarm Employment (Seasonally Adjusted)",
        "category": "employment",
        "unit": "thousands of jobs",
        "source_url": "https://data.bls.gov/timeseries/CES0000000001",
    },
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

NEWS_COLUMNS = {
    "news_id",
    "source_id",
    "source_name",
    "category",
    "title",
    "summary",
    "url",
    "published_at",
    "fetched_at",
    "guid",
    "raw_pub_date",
}
MACRO_COLUMNS = {
    "series_id", "series_name", "category", "year", "period", "period_name",
    "value", "unit", "footnotes", "source_name", "source_url", "fetched_at",
}
MACRO_SYNC_COLUMNS = {
    "series_id", "series_name", "category", "source_url", "last_attempt_at",
    "last_success_at", "last_observation_count", "last_error",
}


class _TextExtractor(HTMLParser):
    """将 RSS 描述中的 HTML 转成可读纯文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"p", "div", "br", "li", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "div", "li", "tr"}:
            self.parts.append(" ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    # 表名来自 sqlite_master，不接受用户输入；双引号转义仍保留，避免特殊名字破坏 SQL。
    safe_name = table_name.replace('"', '""')
    rows = conn.execute(f'PRAGMA table_info("{safe_name}")').fetchall()
    return {str(row[1]).lower() for row in rows}


def _looks_like_news_database(path: Path) -> bool:
    """识别项目内已有的新闻 SQLite 库，尽量复用旧库而非让旧数据看起来消失。"""
    try:
        conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=2)
        try:
            for table in _table_names(conn):
                cols = _table_columns(conn, table)
                has_title = bool(cols & {"title", "headline", "news_title"})
                has_link = bool(cols & {"url", "link", "article_url", "source_url"})
                if has_title and has_link:
                    return True
        finally:
            conn.close()
    except (sqlite3.Error, OSError, ValueError):
        return False
    return False


def _resolve_db_path() -> Path:
    configured = os.getenv("NEWS_DB_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    # 若旧版模块把新闻存入 data/ 下的其他 SQLite 文件，优先复用该数据库。
    # 忽略虚拟环境、Git 目录等，避免扫描无关的第三方数据库。
    excluded_parts = {".venv", "venv", ".git", "node_modules", "__pycache__"}
    candidates: list[Path] = []
    try:
        for current_dir, dir_names, file_names in os.walk(PROJECT_ROOT):
            # 直接剪掉虚拟环境等目录，不然扫描一个带 .venv 的项目能把人等到怀疑人生。
            dir_names[:] = [name for name in dir_names if name not in excluded_parts]
            folder = Path(current_dir)
            for filename in file_names:
                path = folder / filename
                if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
                    candidates.append(path)
    except OSError:
        pass

    matching = [path for path in candidates if _looks_like_news_database(path)]
    if matching:
        # 最新修改的候选库优先；若同项目有多个新闻库，用户仍可用 NEWS_DB_PATH 指定。
        return max(matching, key=lambda item: item.stat().st_mtime)

    return DATA_DIR / "news.db"


DB_PATH = _resolve_db_path()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _column_value(row: sqlite3.Row | dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    keys = {str(key).lower(): key for key in row.keys()}
    for name in names:
        key = keys.get(name.lower())
        if key is not None:
            value = row[key]
            if value is not None and str(value).strip():
                return value
    return default


def _stable_id(source_id: str, external_id: str) -> str:
    raw = f"{source_id}|{external_id.strip()}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()


def _clean_html(value: Any, limit: int = MAX_SUMMARY_CHARS) -> str:
    if value is None:
        return ""
    raw = html.unescape(str(value)).replace("\x00", " ")
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        text = " ".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", html.unescape(text)).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text[:limit]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1].lower()


def _child_text(element: ET.Element, names: set[str]) -> str:
    for child in list(element):
        if _local_name(child.tag) in names:
            # Atom <link> 可能将 URL 存在 href 属性里。
            if _local_name(child.tag) == "link":
                href = child.attrib.get("href", "").strip()
                rel = child.attrib.get("rel", "alternate").lower()
                if href and rel in {"alternate", ""}:
                    return href
                if href and not child.text:
                    return href
            text = " ".join(part.strip() for part in child.itertext() if part.strip())
            if text:
                return text
    return ""


def _normalize_date(raw_value: str) -> str:
    """将 RSS RFC 822 日期或 Atom ISO 日期规范化为 UTC ISO-8601 字符串。"""
    raw_value = (raw_value or "").strip()
    if not raw_value:
        return ""

    dt: datetime | None = None
    try:
        dt = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError, OverflowError):
        pass

    if dt is None:
        normalized = raw_value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            # 部分源可能只给出 YYYY-MM-DD。
            try:
                dt = datetime.strptime(raw_value[:10], "%Y-%m-%d")
            except ValueError:
                return ""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_feed(xml_content: str | bytes, feed: dict[str, str]) -> list[dict[str, Any]]:
    """解析 RSS 2.0 / Atom；优先接收 response.content 原始字节以正确处理 BOM。"""
    if isinstance(xml_content, str):
        # 对兼容旧调用的 str 输入做 BOM 清理；正常路径应传 bytes。
        xml_content = xml_content.lstrip("\ufeff").encode("utf-8")
    elif xml_content.startswith(b"\xef\xbb\xbf"):
        # ElementTree 通常可以自行处理 UTF-8 BOM，这里显式剥掉，兼容边缘响应。
        xml_content = xml_content[3:]

    root = ET.fromstring(xml_content)
    entries = [node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}]
    fetched_at = _utc_now()
    parsed: list[dict[str, Any]] = []

    for entry in entries:
        title = _clean_html(_child_text(entry, {"title"}), 1000)
        if not title:
            continue

        url = _child_text(entry, {"link"}).strip()
        guid = _child_text(entry, {"guid", "id"}).strip()
        parsed_url = urlparse(url) if url else None
        if parsed_url and parsed_url.scheme.lower() not in {"http", "https"}:
            url = ""
        if not url and guid.startswith(("http://", "https://")):
            url = guid
        # 没有可访问链接的记录无法用于后续溯源分析，直接跳过。
        if not url:
            continue

        raw_pub_date = _child_text(entry, {"pubdate", "published", "updated", "date", "issued"})
        published_at = _normalize_date(raw_pub_date)
        summary_raw = _child_text(
            entry,
            {"description", "summary", "encoded", "content", "content:encoded"},
        )
        summary = _clean_html(summary_raw)
        if not guid:
            guid = url

        source_id = feed["source_id"]
        parsed.append(
            {
                "news_id": _stable_id(source_id, guid or url),
                "source_id": source_id,
                "source_name": feed["source_name"],
                "category": feed["category"],
                "title": title,
                "summary": summary,
                "url": url,
                "published_at": published_at,
                "fetched_at": fetched_at,
                "guid": guid,
                "raw_pub_date": raw_pub_date,
            }
        )

    # 同一份 RSS 中若出现重复 GUID，只留第一条。
    unique: dict[str, dict[str, Any]] = {}
    for item in parsed:
        unique.setdefault(item["news_id"], item)
    return list(unique.values())


def _migrate_legacy_news(conn: sqlite3.Connection) -> int:
    """把旧库中名称/字段不同但可识别的新闻表复制到统一表，原表不会删除。"""
    copied = 0
    tables = _table_names(conn)
    for table in tables:
        if table in {"news_items", "feed_sync_state", "news_migration_log", "bls_macro_observations", "bls_macro_sync_state"} or table.startswith("news_items_legacy"):
            continue
        columns = _table_columns(conn, table)
        title_col = columns & {"title", "headline", "news_title"}
        url_col = columns & {"url", "link", "article_url", "source_url"}
        if not title_col or not url_col:
            continue

        # 要求表名像新闻表，或至少同时具备标题、链接以及日期/来源字段，避免误读普通业务表。
        table_looks_like_news = "news" in table.lower() or "feed" in table.lower()
        looks_like_record = bool(columns & {"published_at", "pub_date", "published", "date", "source", "source_name", "source_id"})
        if not table_looks_like_news and not looks_like_record:
            continue

        safe_table = table.replace('"', '""')
        try:
            rows = conn.execute(f'SELECT * FROM "{safe_table}"').fetchall()
        except sqlite3.Error:
            continue

        for row in rows:
            title = _clean_html(_column_value(row, ["title", "headline", "news_title"], ""), 1000)
            url = str(_column_value(row, ["url", "link", "article_url", "source_url"], "") or "").strip()
            if not title or not url:
                continue

            old_source_name = str(_column_value(row, ["source_name", "source", "publisher"], "Legacy source"))
            old_source_id = str(_column_value(row, ["source_id"], "legacy"))
            if old_source_id == "legacy":
                old_source_id = re.sub(r"[^a-z0-9]+", "_", old_source_name.lower()).strip("_") or "legacy"
            category = str(_column_value(row, ["category", "topic", "type"], "general"))
            raw_date = str(_column_value(row, ["raw_pub_date", "pub_date", "published_at", "published", "date"], "") or "")
            published_at = _normalize_date(raw_date)
            guid = str(_column_value(row, ["guid", "external_id", "id", "news_id"], url) or url)
            summary = _clean_html(_column_value(row, ["summary", "description", "content", "excerpt"], ""))
            item = {
                "news_id": _stable_id(old_source_id, guid or url),
                "source_id": old_source_id,
                "source_name": old_source_name,
                "category": category,
                "title": title,
                "summary": summary,
                "url": url,
                "published_at": published_at,
                "fetched_at": str(_column_value(row, ["fetched_at", "created_at", "updated_at"], _utc_now()) or _utc_now()),
                "guid": guid,
                "raw_pub_date": raw_date,
            }
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO news_items
                    (news_id, source_id, source_name, category, title, summary, url,
                     published_at, fetched_at, guid, raw_pub_date)
                    VALUES (:news_id, :source_id, :source_name, :category, :title, :summary,
                            :url, :published_at, :fetched_at, :guid, :raw_pub_date)""",
                    item,
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    copied += 1
            except sqlite3.Error as exc:
                LOGGER.debug("迁移旧表 %s 的记录失败：%s", table, exc)

    return copied


def _next_legacy_name(conn: sqlite3.Connection, base: str) -> str:
    """返回一个尚未使用的旧表名，确保迁移只重命名、不覆盖旧数据。"""
    names = set(_table_names(conn))
    candidate = base
    suffix = 1
    while candidate in names:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _migrate_legacy_bls_macro_observations(
    conn: sqlite3.Connection, legacy_table: str
) -> int:
    """尽可能把旧宏观表的数据映射到新结构；无法识别的记录仍留在旧表。"""
    safe_table = legacy_table.replace('"', '""')
    try:
        rows = conn.execute(f'SELECT * FROM "{safe_table}"').fetchall()
    except sqlite3.Error as exc:
        LOGGER.warning("读取旧宏观表 %s 失败，原表已保留：%s", legacy_table, exc)
        return 0

    copied = 0
    now = _utc_now()
    month_map = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    for row in rows:
        series_id = str(_column_value(row, ["series_id", "seriesid", "series", "id"], "") or "").strip()
        value = str(_column_value(row, ["value", "observation_value", "amount", "data_value"], "") or "").strip()
        raw_period = str(_column_value(row, ["period", "period_name", "observation_date", "date", "date_time", "time_period"], "") or "").strip()
        year_value = _column_value(row, ["year", "observation_year"], None)
        year_match = re.search(r"(?:19|20)\d{2}", f"{year_value or ''} {raw_period}")
        if not series_id or not value or not year_match:
            continue
        year = int(year_match.group(0))

        period = str(_column_value(row, ["period", "period_code", "month"], "") or "").strip().upper()
        period_match = re.fullmatch(r"M?(0?[1-9]|1[0-2])", period)
        if period_match:
            period = f"M{int(period_match.group(1)):02d}"
        elif re.fullmatch(r"M(?:0[1-9]|1[0-2])", period):
            pass
        else:
            date_month = re.search(r"(?:19|20)\d{2}[-/](\d{1,2})", raw_period)
            month_num = int(date_month.group(1)) if date_month else 0
            if not month_num:
                lower = raw_period.lower()
                month_num = next((number for name, number in month_map.items() if re.search(rf"\b{name}\b", lower)), 0)
            if not 1 <= month_num <= 12:
                continue
            period = f"M{month_num:02d}"

        series_name = str(_column_value(row, ["series_name", "name", "title"], series_id) or series_id)
        category = str(_column_value(row, ["category", "topic", "type"], "other") or "other")
        period_name = str(_column_value(row, ["period_name", "date_label", "observation_date", "date"], raw_period or period) or period)
        unit = str(_column_value(row, ["unit", "units", "measurement_unit"], "") or "")
        footnotes = str(_column_value(row, ["footnotes", "notes", "note"], "") or "")
        source_name = str(_column_value(row, ["source_name", "source", "publisher"], "U.S. Bureau of Labor Statistics") or "U.S. Bureau of Labor Statistics")
        source_url = str(_column_value(row, ["source_url", "url", "link"], "") or "")
        fetched_at = str(_column_value(row, ["fetched_at", "updated_at", "created_at"], now) or now)
        conn.execute(
            """INSERT OR IGNORE INTO bls_macro_observations
               (series_id, series_name, category, year, period, period_name, value,
                unit, footnotes, source_name, source_url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (series_id, series_name, category, year, period, period_name, value,
             unit, footnotes, source_name, source_url, fetched_at),
        )
        copied += int(conn.execute("SELECT changes()").fetchone()[0] > 0)
    return copied


def _initialize_db() -> None:
    with _connect() as conn:
        table_names = set(_table_names(conn))

        # 仅检查 BLS 专属的 bls_macro_observations 表，绝不修改 FRED 使用的 macro_observations 表。CREATE TABLE IF NOT EXISTS
        # 不会修正旧表结构，因此先检测并重命名旧表，再创建当前版本结构。
        legacy_macro_table: str | None = None
        if "bls_macro_observations" in table_names:
            if not MACRO_COLUMNS.issubset(_table_columns(conn, "bls_macro_observations")):
                # 旧表的同名索引会随表重命名保留，先删掉当前版本将使用的索引名。
                conn.execute("DROP INDEX IF EXISTS idx_bls_macro_observations_series_period")
                legacy_macro_table = _next_legacy_name(conn, "bls_macro_observations_legacy")
                conn.execute(f'ALTER TABLE bls_macro_observations RENAME TO "{legacy_macro_table}"')
                LOGGER.warning(
                    "检测到旧版宏观数据表字段不兼容，已保留为 %s，并将创建新版数据表。",
                    legacy_macro_table,
                )

        # BLS 专属宏观同步状态表也做同样的版本检查，避免之后在 UPSERT 时再次触发缺列错误。
        if "bls_macro_sync_state" in _table_names(conn):
            if not MACRO_SYNC_COLUMNS.issubset(_table_columns(conn, "bls_macro_sync_state")):
                old_state = _next_legacy_name(conn, "bls_macro_sync_state_legacy")
                conn.execute(f'ALTER TABLE bls_macro_sync_state RENAME TO "{old_state}"')
                LOGGER.warning("旧版宏观同步状态表已保留为 %s。", old_state)

        # 若旧版新闻表字段不同，重命名保留记录，再迁移到统一结构。
        if "news_items" in _table_names(conn):
            existing_cols = _table_columns(conn, "news_items")
            if not NEWS_COLUMNS.issubset(existing_cols):
                conn.execute("DROP INDEX IF EXISTS idx_news_items_published_at")
                conn.execute("DROP INDEX IF EXISTS idx_news_items_source_id")
                legacy_name = _next_legacy_name(conn, "news_items_legacy")
                conn.execute(f'ALTER TABLE news_items RENAME TO "{legacy_name}"')
                LOGGER.warning("旧版新闻表已保留为 %s。", legacy_name)

        conn.execute(
            """CREATE TABLE IF NOT EXISTS news_items (
                news_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL,
                published_at TEXT NOT NULL DEFAULT '',
                fetched_at TEXT NOT NULL,
                guid TEXT NOT NULL DEFAULT '',
                raw_pub_date TEXT NOT NULL DEFAULT ''
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS feed_sync_state (
                source_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                feed_url TEXT NOT NULL,
                last_attempt_at TEXT NOT NULL DEFAULT '',
                last_success_at TEXT NOT NULL DEFAULT '',
                last_item_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT ''
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS news_migration_log (
                migration_key TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS bls_macro_observations (
                series_id TEXT NOT NULL,
                series_name TEXT NOT NULL,
                category TEXT NOT NULL,
                year INTEGER NOT NULL,
                period TEXT NOT NULL,
                period_name TEXT NOT NULL DEFAULT '',
                value TEXT NOT NULL,
                unit TEXT NOT NULL DEFAULT '',
                footnotes TEXT NOT NULL DEFAULT '',
                source_name TEXT NOT NULL DEFAULT 'U.S. Bureau of Labor Statistics',
                source_url TEXT NOT NULL DEFAULT '',
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (series_id, year, period)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS bls_macro_sync_state (
                series_id TEXT PRIMARY KEY,
                series_name TEXT NOT NULL,
                category TEXT NOT NULL,
                source_url TEXT NOT NULL,
                last_attempt_at TEXT NOT NULL DEFAULT '',
                last_success_at TEXT NOT NULL DEFAULT '',
                last_observation_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT ''
            )"""
        )

        # 现在 BLS 专属新表已经存在，可以尽力迁移旧 BLS 数据；FRED 的 macro_observations 表不会被本模块改名或写入。
        if legacy_macro_table:
            copied_macro = _migrate_legacy_bls_macro_observations(conn, legacy_macro_table)
            LOGGER.info("从旧宏观数据表迁移 %s 条可识别记录（旧表保留）", copied_macro)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_items_published_at ON news_items(published_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_items_source_id ON news_items(source_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bls_macro_observations_series_period "
            "ON bls_macro_observations(series_id, year DESC, period DESC)"
        )

        migrated = _migrate_legacy_news(conn)
        if migrated:
            LOGGER.info("已从旧新闻表迁移 %s 条记录（旧表保留）", migrated)


def _record_feed_result(
    feed: dict[str, str], item_count: int, error: str = ""
) -> None:
    now = _utc_now()
    with _connect() as conn:
        previous = conn.execute(
            "SELECT last_success_at FROM feed_sync_state WHERE source_id = ?",
            (feed["source_id"],),
        ).fetchone()
        last_success = previous[0] if previous else ""
        if not error:
            last_success = now
        conn.execute(
            """INSERT INTO feed_sync_state
                (source_id, source_name, feed_url, last_attempt_at, last_success_at,
                 last_item_count, last_error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                source_name=excluded.source_name,
                feed_url=excluded.feed_url,
                last_attempt_at=excluded.last_attempt_at,
                last_success_at=excluded.last_success_at,
                last_item_count=excluded.last_item_count,
                last_error=excluded.last_error""",
            (
                feed["source_id"], feed["source_name"], feed["url"], now,
                last_success, item_count, error[:2000],
            ),
        )


def _upsert_news(items: list[dict[str, Any]]) -> tuple[int, int]:
    inserted = 0
    updated = 0
    with _connect() as conn:
        for item in items:
            existed = conn.execute(
                "SELECT 1 FROM news_items WHERE news_id = ?", (item["news_id"],)
            ).fetchone() is not None
            conn.execute(
                """INSERT INTO news_items
                    (news_id, source_id, source_name, category, title, summary, url,
                     published_at, fetched_at, guid, raw_pub_date)
                VALUES (:news_id, :source_id, :source_name, :category, :title, :summary,
                        :url, :published_at, :fetched_at, :guid, :raw_pub_date)
                ON CONFLICT(news_id) DO UPDATE SET
                    source_name=excluded.source_name,
                    category=excluded.category,
                    title=excluded.title,
                    summary=CASE WHEN excluded.summary <> '' THEN excluded.summary ELSE news_items.summary END,
                    url=excluded.url,
                    published_at=CASE WHEN excluded.published_at <> '' THEN excluded.published_at ELSE news_items.published_at END,
                    fetched_at=excluded.fetched_at,
                    raw_pub_date=CASE WHEN excluded.raw_pub_date <> '' THEN excluded.raw_pub_date ELSE news_items.raw_pub_date END""",
                item,
            )
            if existed:
                updated += 1
            else:
                inserted += 1
    return inserted, updated


def _fetch_feed(feed: dict[str, str]) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-US,en;q=0.8",
    }
    response = requests.get(feed["url"], headers=headers, timeout=HTTP_TIMEOUT)
    if response.status_code == 403:
        preview = _clean_html(response.text, 300)
        raise requests.HTTPError(
            f"HTTP 403 Forbidden；响应片段：{preview or '(空响应)'}",
            response=response,
        )
    response.raise_for_status()
    if not response.content:
        raise ValueError("服务器返回了空响应")
    return _parse_feed(response.content, feed)


def _bls_period_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    """用于对 BLS 年/月统计期排序。"""
    period = str(item.get("period", ""))
    try:
        month_num = int(period[1:]) if period.startswith("M") else 0
    except ValueError:
        month_num = 0
    try:
        year_num = int(item.get("year", 0))
    except (TypeError, ValueError):
        year_num = 0
    return year_num, month_num


def _get_bls_data() -> dict[str, list[dict[str, Any]]]:
    """通过 BLS 官方 Public Data API 一次获取三条月度序列。"""
    current_year = datetime.now(timezone.utc).year
    start_year = current_year - BLS_API_YEARS_BACK
    payload = {
        "seriesid": [series["series_id"] for series in BLS_SERIES],
        "startyear": str(start_year),
        "endyear": str(current_year),
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = requests.post(
        BLS_API_URL,
        json=payload,
        headers=headers,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    try:
        body = response.json()
    except ValueError as exc:
        raise ValueError(f"BLS API 返回的不是有效 JSON：{response.text[:300]}") from exc

    status = str(body.get("status", ""))
    if status != "REQUEST_SUCCEEDED":
        messages = body.get("message") or []
        detail = "; ".join(str(message) for message in messages) or status or "未知 API 错误"
        raise RuntimeError(f"BLS API 请求未成功：{detail}")

    result_map: dict[str, list[dict[str, Any]]] = {}
    results = body.get("Results") or {}
    for series_data in results.get("series", []) or []:
        series_id = str(series_data.get("seriesID", ""))
        raw_items = series_data.get("data", []) or []
        # 本项目使用月度指标；过滤年平均、季度或其他非月度数据。
        monthly = [
            item for item in raw_items
            if re.fullmatch(r"M(?:0[1-9]|1[0-2])", str(item.get("period", "")))
        ]
        result_map[series_id] = sorted(monthly, key=_bls_period_sort_key, reverse=True)
    return result_map


def _record_macro_result(series: dict[str, str], count: int, error: str = "") -> None:
    now = _utc_now()
    with _connect() as conn:
        previous = conn.execute(
            "SELECT last_success_at FROM bls_macro_sync_state WHERE series_id = ?",
            (series["series_id"],),
        ).fetchone()
        last_success = previous[0] if previous else ""
        if not error:
            last_success = now
        conn.execute(
            """INSERT INTO bls_macro_sync_state
                (series_id, series_name, category, source_url, last_attempt_at,
                 last_success_at, last_observation_count, last_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(series_id) DO UPDATE SET
                series_name=excluded.series_name,
                category=excluded.category,
                source_url=excluded.source_url,
                last_attempt_at=excluded.last_attempt_at,
                last_success_at=excluded.last_success_at,
                last_observation_count=excluded.last_observation_count,
                last_error=excluded.last_error""",
            (
                series["series_id"], series["series_name"], series["category"],
                series["source_url"], now, last_success, count, error[:2000],
            ),
        )


def _upsert_bls_macro_observations(
    series: dict[str, str], raw_items: list[dict[str, Any]]
) -> tuple[int, int]:
    inserted = 0
    updated = 0
    now = _utc_now()
    with _connect() as conn:
        for raw in raw_items:
            year = str(raw.get("year", "")).strip()
            period = str(raw.get("period", "")).strip()
            value = str(raw.get("value", "")).strip()
            if not year.isdigit() or not period or not value:
                continue
            try:
                year_int = int(year)
            except ValueError:
                continue
            footnotes_raw = raw.get("footnotes") or []
            footnotes = "; ".join(
                str(note.get("text", "")).strip()
                for note in footnotes_raw
                if isinstance(note, dict) and str(note.get("text", "")).strip()
            )
            exists = conn.execute(
                "SELECT 1 FROM bls_macro_observations WHERE series_id = ? AND year = ? AND period = ?",
                (series["series_id"], year_int, period),
            ).fetchone() is not None
            conn.execute(
                """INSERT INTO bls_macro_observations
                    (series_id, series_name, category, year, period, period_name,
                     value, unit, footnotes, source_name, source_url, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_id, year, period) DO UPDATE SET
                    series_name=excluded.series_name,
                    category=excluded.category,
                    period_name=excluded.period_name,
                    value=excluded.value,
                    unit=excluded.unit,
                    footnotes=excluded.footnotes,
                    source_url=excluded.source_url,
                    fetched_at=excluded.fetched_at""",
                (
                    series["series_id"], series["series_name"], series["category"],
                    year_int, period, str(raw.get("periodName", "")), value,
                    series["unit"], footnotes, "U.S. Bureau of Labor Statistics",
                    series["source_url"], now,
                ),
            )
            if exists:
                updated += 1
            else:
                inserted += 1
    return inserted, updated


def sync_bls_macro_data() -> dict[str, Any]:
    """同步 CPI、失业率和非农就业数据；这些是宏观观测值而非新闻。"""
    _initialize_db()
    print(f"BLS API：{BLS_API_URL}")
    inserted_total = 0
    updated_total = 0
    failed = 0

    try:
        data_by_series = _get_bls_data()
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        for series in BLS_SERIES:
            _record_macro_result(series, 0, message)
            print(f"[宏观数据失败] {series['series_name']}：{message}")
        return {
            "failed": len(BLS_SERIES), "inserted": 0, "updated": 0,
            "details": [{"series_id": s["series_id"], "ok": False, "error": message} for s in BLS_SERIES],
        }

    details: list[dict[str, Any]] = []
    for series in BLS_SERIES:
        series_id = series["series_id"]
        if series_id not in data_by_series:
            failed += 1
            message = "BLS API 响应中缺少该序列；请检查序列 ID 或 API 返回内容"
            _record_macro_result(series, 0, message)
            print(f"[宏观数据失败] {series['series_name']}：{message}")
            details.append({"series_id": series_id, "ok": False, "error": message})
            continue
        raw_items = data_by_series[series_id]
        inserted, updated = _upsert_bls_macro_observations(series, raw_items)
        _record_macro_result(series, len(raw_items))
        inserted_total += inserted
        updated_total += updated
        latest = raw_items[0] if raw_items else {}
        latest_period = f"{latest.get('year', '')} {latest.get('periodName', latest.get('period', ''))}".strip()
        latest_value = str(latest.get("value", "暂无"))
        print(
            f"[宏观数据完成] {series['series_name']}："
            f"获取 {len(raw_items)} 期，新增 {inserted} 期，更新 {updated} 期；"
            f"最新值 {latest_value}（{latest_period}，单位：{series['unit']}）"
        )
        details.append({
            "series_id": series_id, "ok": True, "observations": len(raw_items),
            "inserted": inserted, "updated": updated, "latest_value": latest_value,
            "latest_period": latest_period,
        })

    return {
        "failed": failed, "inserted": inserted_total, "updated": updated_total,
        "details": details,
    }


def sync_news() -> dict[str, Any]:
    """同步美联储新闻和 BLS 官方宏观数据。"""
    _initialize_db()
    total_news_inserted = 0
    total_news_updated = 0
    total_failed = 0
    details: list[dict[str, Any]] = []

    print(f"数据库：{DB_PATH}")
    for feed in FEEDS:
        try:
            items = _fetch_feed(feed)
            inserted, updated = _upsert_news(items)
            _record_feed_result(feed, len(items))
            total_news_inserted += inserted
            total_news_updated += updated
            print(
                f"[新闻完成] {feed['source_name']}：RSS {len(items)} 条，"
                f"新增 {inserted} 条，更新 {updated} 条"
            )
            details.append({
                "source_id": feed["source_id"], "ok": True, "items": len(items),
                "inserted": inserted, "updated": updated, "error": "",
            })
        except Exception as exc:
            total_failed += 1
            message = f"{type(exc).__name__}: {exc}"
            try:
                _record_feed_result(feed, 0, message)
            except sqlite3.Error:
                LOGGER.exception("无法记录 RSS 同步错误")
            print(f"[新闻失败] {feed['source_name']}：{message}")
            details.append({
                "source_id": feed["source_id"], "ok": False, "items": 0,
                "inserted": 0, "updated": 0, "error": message,
            })

    macro_result = sync_bls_macro_data()
    total_failed += int(macro_result["failed"])

    with _connect() as conn:
        total_news = int(conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0])
        total_macro = int(conn.execute("SELECT COUNT(*) FROM bls_macro_observations").fetchone()[0])
    print("\n同步汇总")
    print(f"新闻新增：{total_news_inserted} 条")
    print(f"新闻更新：{total_news_updated} 条")
    print(f"宏观数据新增：{macro_result['inserted']} 期")
    print(f"宏观数据更新：{macro_result['updated']} 期")
    print(f"失败来源/序列：{total_failed}/{len(FEEDS) + len(BLS_SERIES)}")
    print(f"数据库新闻总数：{total_news} 条")
    print(f"数据库宏观观测总数：{total_macro} 期")
    return {
        "inserted": total_news_inserted,
        "updated": total_news_updated,
        "macro_inserted": macro_result["inserted"],
        "macro_updated": macro_result["updated"],
        "failed": total_failed,
        "total": total_news,
        "macro_total": total_macro,
        "details": details + macro_result["details"],
    }


def get_recent_news(
    limit: int = 20,
    source_id: str | None = None,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """供报告生成器调用，返回最新官方新闻记录。"""
    _initialize_db()
    limit = max(1, min(int(limit), 500))
    conditions: list[str] = []
    params: list[Any] = []

    if source_id:
        conditions.append("source_id = ?")
        params.append(source_id)
    if days is not None:
        days = max(1, min(int(days), 3650))
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat(timespec="seconds")
        conditions.append("(published_at = '' OR published_at >= ?)")
        params.append(cutoff_iso)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    with _connect() as conn:
        rows = conn.execute(
            f"""SELECT news_id, source_id, source_name, category, title, summary, url,
                       published_at, fetched_at, guid
                FROM news_items {where}
                ORDER BY CASE WHEN published_at = '' THEN 1 ELSE 0 END,
                         published_at DESC, fetched_at DESC
                LIMIT ?""",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_latest_bls_macro_observations(limit_per_series: int = 6) -> list[dict[str, Any]]:
    """返回每条 BLS 序列最近的若干个月度观测值。"""
    _initialize_db()
    limit_per_series = max(1, min(int(limit_per_series), 120))
    result: list[dict[str, Any]] = []
    with _connect() as conn:
        for series in BLS_SERIES:
            rows = conn.execute(
                """SELECT series_id, series_name, category, year, period, period_name,
                          value, unit, footnotes, source_name, source_url, fetched_at
                   FROM bls_macro_observations WHERE series_id = ?
                   ORDER BY year DESC, period DESC LIMIT ?""",
                (series["series_id"], limit_per_series),
            ).fetchall()
            result.extend(dict(row) for row in rows)
    return result


def get_macro_data_for_analysis(limit_per_series: int = 6) -> str:
    """将官方 BLS 月度数据整理成 LLM 可用文本，明确单位和原始来源。"""
    items = get_latest_bls_macro_observations(limit_per_series=limit_per_series)
    if not items:
        return "暂无已同步的 BLS 宏观数据。请先运行：python -m app.news_data sync"

    lines = [
        "以下为美国劳工统计局（BLS）Public Data API 返回的官方月度统计数据，不是新闻文章。",
        "不得将单个月份的数值变化直接解释为因果关系；引用时保留序列 ID、统计期、单位及官方链接。",
    ]
    by_series: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_series.setdefault(item["series_id"], []).append(item)
    for series in BLS_SERIES:
        rows = by_series.get(series["series_id"], [])
        if not rows:
            lines.append(f"\n{series['series_name']}（{series['series_id']}）：暂无数据；官方链接：{series['source_url']}")
            continue
        lines.append(
            f"\n{series['series_name']}（序列 {series['series_id']}；单位：{series['unit']}）"
        )
        for row in rows:
            period = row.get("period_name") or row.get("period")
            note = f"；注释：{row['footnotes']}" if row.get("footnotes") else ""
            lines.append(f"- {row['year']} {period}: {row['value']}{note}")
        lines.append(f"官方数据：{series['source_url']}")
    return "\n".join(lines)


def get_news_for_analysis(limit: int = 15, days: int = 45) -> str:
    """将新闻压缩为可放进 LLM 提示词的纯文本上下文；不替新闻编造结论。"""
    items = get_recent_news(limit=limit, days=days)
    if not items:
        return "暂无已同步的官方经济新闻。请先运行：python -m app.news_data sync"

    lines = ["以下是从官方 RSS 同步的经济新闻。请区分新闻事实与推断，并在引用时保留原始链接："]
    for index, item in enumerate(items, start=1):
        date = item.get("published_at") or "发布时间未知"
        summary = item.get("summary") or "（RSS 未提供摘要，请查看原文）"
        lines.append(
            f"{index}. [{item.get('source_name', 'Unknown source')}] "
            f"{item.get('title', '')}\n"
            f"   发布时间：{date}\n"
            f"   摘要：{summary}\n"
            f"   原文：{item.get('url', '')}"
        )
    return "\n\n".join(lines)


def show_status() -> None:
    _initialize_db()
    with _connect() as conn:
        total_news = int(conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0])
        total_macro = int(conn.execute("SELECT COUNT(*) FROM bls_macro_observations").fetchone()[0])
        news_states = conn.execute(
            "SELECT * FROM feed_sync_state ORDER BY source_name"
        ).fetchall()
        macro_states = conn.execute(
            "SELECT * FROM bls_macro_sync_state ORDER BY series_name"
        ).fetchall()
        news_counts = {
            row["source_id"]: int(row["n"])
            for row in conn.execute(
                "SELECT source_id, COUNT(*) AS n FROM news_items GROUP BY source_id"
            ).fetchall()
        }
        macro_counts = {
            row["series_id"]: int(row["n"])
            for row in conn.execute(
                "SELECT series_id, COUNT(*) AS n FROM bls_macro_observations GROUP BY series_id"
            ).fetchall()
        }

    print(f"数据库：{DB_PATH}")
    print(f"新闻总数：{total_news} 条")
    print(f"宏观观测总数：{total_macro} 期\n")
    news_by_id = {row["source_id"]: row for row in news_states}
    for feed in FEEDS:
        state = news_by_id.get(feed["source_id"])
        print(f"新闻来源：{feed['source_name']}")
        print(f"  RSS：{feed['url']}")
        print(f"  本地记录：{news_counts.get(feed['source_id'], 0)} 条")
        print(f"  最近尝试：{state['last_attempt_at'] if state else '尚未同步'}")
        print(f"  最近成功：{state['last_success_at'] if state else '尚未成功'}")
        if state and state["last_error"]:
            print(f"  最近错误：{state['last_error']}")
        print()

    macro_by_id = {row["series_id"]: row for row in macro_states}
    for series in BLS_SERIES:
        state = macro_by_id.get(series["series_id"])
        print(f"BLS 宏观指标：{series['series_name']}")
        print(f"  序列 ID：{series['series_id']}")
        print(f"  官方数据：{series['source_url']}")
        print(f"  本地观测：{macro_counts.get(series['series_id'], 0)} 期")
        print(f"  最近尝试：{state['last_attempt_at'] if state else '尚未同步'}")
        print(f"  最近成功：{state['last_success_at'] if state else '尚未成功'}")
        if state and state["last_error"]:
            print(f"  最近错误：{state['last_error']}")
        print()


def show_macro(limit: int = 6) -> None:
    items = get_latest_bls_macro_observations(limit_per_series=limit)
    print(f"数据库：{DB_PATH}")
    if not items:
        print("当前没有 BLS 宏观数据。先运行：python -m app.news_data sync")
        return
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item["series_id"], []).append(item)
    for series in BLS_SERIES:
        print(f"\n{series['series_name']} [{series['series_id']}]，单位：{series['unit']}")
        rows = grouped.get(series["series_id"], [])
        if not rows:
            print("  暂无数据")
            continue
        for row in rows:
            period = row.get("period_name") or row.get("period")
            suffix = f"；注释：{row['footnotes']}" if row.get("footnotes") else ""
            print(f"  {row['year']} {period}: {row['value']}{suffix}")
        print(f"  来源：{series['source_url']}")


def show_news(limit: int = 20) -> None:
    items = get_recent_news(limit=limit)
    print(f"数据库：{DB_PATH}")
    if not items:
        print("当前没有新闻记录。先运行：python -m app.news_data sync")
        return
    for index, item in enumerate(items, start=1):
        published = item.get("published_at") or "发布时间未知"
        print(f"{index}. [{item['source_name']}] {item['title']}")
        print(f"   发布时间：{published}")
        if item.get("summary"):
            print(f"   摘要：{item['summary'][:260]}")
        print(f"   链接：{item['url']}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="同步美联储官方新闻与 BLS 官方宏观经济指标到 SQLite 数据库"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sync", help="同步美联储新闻与 BLS 宏观数据")
    subparsers.add_parser("status", help="查看新闻库、宏观数据和同步状态")
    list_parser = subparsers.add_parser("list", help="查看最近的新闻记录")
    list_parser.add_argument("--limit", type=int, default=20, help="最多显示多少条，默认 20")
    macro_parser = subparsers.add_parser("macro", help="查看最近的 BLS 宏观数据")
    macro_parser.add_argument("--limit", type=int, default=6, help="每项指标显示多少期，默认 6")
    args = parser.parse_args(argv)

    if args.command == "sync":
        result = sync_news()
        return 1 if result["failed"] else 0
    if args.command == "status":
        show_status()
        return 0
    if args.command == "list":
        show_news(args.limit)
        return 0
    if args.command == "macro":
        show_macro(args.limit)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
