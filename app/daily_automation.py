from __future__ import annotations

"""Daily automation for QQQM Investment Assistant.

Runs existing project entry points in sequence:
1) refresh daily market bars;
2) sync Fed/BLS news and macro observations;
3) sync FRED macro series;
4) settle saved next-session forecasts against newly available actual closes;
5) when QQQM's market date advances, refit the current return-model candidates,
   save a new forecast, and record it for later evaluation;
6) generate DeepSeek reports only when market/news/macro data changed.

Run from the repository root:
    python -m app.daily_automation --dry-run
    python -m app.daily_automation
    python -m app.daily_automation --force-ai
    python -m app.daily_automation --status
"""

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market_data.db"
PORTFOLIO_CSV = ROOT / "data" / "portfolio_transactions.csv"
AUTOMATION_DIR = ROOT / "reports" / "automation"
STATUS_PATH = AUTOMATION_DIR / "latest_status.json"
DEFAULT_AI_TICKERS = ["QQQM"]
FORECAST_DIR = ROOT / "reports" / "forecast"
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.^=_-]{0,24}$")


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_tickers(values: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = str(value or "").strip().upper()
        if not ticker or not TICKER_PATTERN.fullmatch(ticker):
            continue
        if ticker not in seen:
            cleaned.append(ticker)
            seen.add(ticker)
    return cleaned


def configured_tickers() -> list[str]:
    """Read the project's existing TICKERS config, if available."""
    try:
        from app.config import get_tickers

        values = get_tickers()
        if isinstance(values, (list, tuple, set)):
            return normalize_tickers(values)
    except Exception:
        pass
    raw = os.getenv("TICKERS", "")
    return normalize_tickers(raw.split(","))


def portfolio_tickers() -> list[str]:
    """Read symbols from the local portfolio ledger without printing trade data."""
    if not PORTFOLIO_CSV.exists():
        return []
    try:
        with PORTFOLIO_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            ticker_column = next(
                (name for name in reader.fieldnames if name.strip().casefold() in {"ticker", "symbol"}),
                None,
            )
            if not ticker_column:
                return []
            return normalize_tickers(row.get(ticker_column, "") for row in reader)
    except (OSError, csv.Error):
        return []


def resolve_tickers(extra_tickers: Iterable[str] | None = None) -> list[str]:
    """Union watchlist, recorded holdings, configured AI tickers and CLI additions."""
    candidates: list[str] = []
    candidates.extend(configured_tickers())
    candidates.extend(portfolio_tickers())
    candidates.extend(DEFAULT_AI_TICKERS)
    if extra_tickers:
        candidates.extend(extra_tickers)
    return normalize_tickers(candidates) or ["QQQM"]


def _connect_readonly() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    try:
        uri = DB_PATH.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error:
        return None


def read_market_dates() -> dict[str, str]:
    """Return each ticker's newest stored daily bar date."""
    connection = _connect_readonly()
    if connection is None:
        return {}
    try:
        rows = connection.execute(
            "SELECT UPPER(ticker) AS ticker, MAX(trade_date) AS latest_date "
            "FROM market_daily GROUP BY UPPER(ticker)"
        ).fetchall()
        return {
            str(row["ticker"]): str(row["latest_date"])
            for row in rows
            if row["ticker"] and row["latest_date"]
        }
    except sqlite3.Error:
        return {}
    finally:
        connection.close()


def read_latest_market_state() -> dict[str, dict[str, str]]:
    """Return latest date/close/source; detects same-date snapshot-to-provider upgrades."""
    connection = _connect_readonly()
    if connection is None:
        return {}
    try:
        rows = connection.execute(
            """
            SELECT UPPER(m.ticker) AS ticker, m.trade_date, m.close, m.adj_close, m.source
            FROM market_daily m
            INNER JOIN (
                SELECT UPPER(ticker) AS ticker, MAX(trade_date) AS latest_date
                FROM market_daily GROUP BY UPPER(ticker)
            ) latest ON UPPER(m.ticker)=latest.ticker AND m.trade_date=latest.latest_date
            """
        ).fetchall()
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            ticker = str(row["ticker"] or "").upper()
            if not ticker:
                continue
            price = row["adj_close"] if row["adj_close"] is not None else row["close"]
            result[ticker] = {
                "trade_date": str(row["trade_date"] or "")[:10],
                "price": f"{float(price):.6f}" if price is not None else "",
                "source": str(row["source"] or ""),
            }
        return result
    except (sqlite3.Error, TypeError, ValueError):
        return {}
    finally:
        connection.close()


def read_table_keys(table: str, columns: tuple[str, ...]) -> set[tuple[str, ...]]:
    """Read stable keys from known local tables for change detection."""
    allowed = {
        "news_items": ("news_id",),
        "macro_observations": ("series_id", "observation_date"),
        "bls_macro_observations": ("series_id", "year", "period"),
    }
    if table not in allowed or allowed[table] != columns:
        raise ValueError("Unsupported snapshot table/columns")

    connection = _connect_readonly()
    if connection is None:
        return set()
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return set()
        selected = ", ".join(columns)
        rows = connection.execute(f"SELECT {selected} FROM {table}").fetchall()
        return {tuple(str(row[column]) for column in columns) for row in rows}
    except sqlite3.Error:
        return set()
    finally:
        connection.close()


def snapshot_data() -> dict[str, Any]:
    return {
        "market_dates": read_market_dates(),
        "market_state": read_latest_market_state(),
        "news_ids": read_table_keys("news_items", ("news_id",)),
        "fred_macro_keys": read_table_keys(
            "macro_observations", ("series_id", "observation_date")
        ),
        "bls_macro_keys": read_table_keys(
            "bls_macro_observations", ("series_id", "year", "period")
        ),
    }


def latest_forecast_as_of_date() -> str | None:
    """Read the as-of date in the newest timestamped production forecast file."""
    files = list(FORECAST_DIR.glob("QQQM_return_forecast_latest_*.json")) if FORECAST_DIR.exists() else []
    if not files:
        return None
    latest_file = max(files, key=lambda path: (path.stat().st_mtime, path.name))
    try:
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("as_of_date") if isinstance(payload, dict) else None
    return str(value)[:10] if value else None


def build_logger(log_path: Path) -> logging.Logger:
    AUTOMATION_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("qqqm_daily_automation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


class AutomationRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started_at = local_timestamp()
        self.run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        AUTOMATION_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = AUTOMATION_DIR / f"daily_{self.run_id}.log"
        self.logger = build_logger(self.log_path)
        self.stages: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def run_command(self, name: str, command: list[str], timeout: int = 600) -> dict[str, Any]:
        self.logger.info("========== %s =========", name)
        self.logger.info("执行命令：%s", subprocess.list2cmdline(command))
        started = time.monotonic()
        record: dict[str, Any] = {
            "name": name,
            "command": command,
            "started_at": local_timestamp(),
            "returncode": None,
            "success": False,
            "duration_seconds": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }

        if self.args.dry_run:
            record.update({"success": True, "dry_run": True, "returncode": 0})
            record["duration_seconds"] = 0.0
            self.stages.append(record)
            self.logger.info("DRY RUN：未执行。")
            return record

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            record["returncode"] = int(result.returncode)
            record["success"] = result.returncode == 0
            record["stdout_tail"] = stdout[-6000:]
            record["stderr_tail"] = stderr[-3000:]

            for line in stdout.splitlines():
                self.logger.info("%s", line)
            for line in stderr.splitlines():
                self.logger.warning("%s", line)

            # main.py may return zero when only some ticker syncs fail; detect its per-symbol markers.
            failed_symbols = re.findall(r"\[失败\]\s*([A-Z0-9.^=_-]+)\s*:", stdout)
            if failed_symbols:
                record["failed_symbols"] = sorted(set(failed_symbols))
                record["success"] = False
                self.logger.warning("以下标的同步失败：%s", ", ".join(record["failed_symbols"]))

            if result.returncode != 0:
                self.logger.error("任务失败，退出码：%s", result.returncode)
                self.errors.append(f"{name} exited with code {result.returncode}")
        except subprocess.TimeoutExpired as exc:
            record["success"] = False
            record["error"] = f"命令超过 {timeout} 秒，已超时中止"
            self.logger.error("%s", record["error"])
            if exc.stdout:
                record["stdout_tail"] = str(exc.stdout)[-6000:]
            if exc.stderr:
                record["stderr_tail"] = str(exc.stderr)[-3000:]
            self.errors.append(f"{name} timeout")
        except OSError as exc:
            record["success"] = False
            record["error"] = f"无法启动命令：{type(exc).__name__}: {exc}"
            self.logger.exception("%s", record["error"])
            self.errors.append(f"{name}: {exc}")
        finally:
            record["finished_at"] = local_timestamp()
            record["duration_seconds"] = round(time.monotonic() - started, 2)
            self.stages.append(record)
        return record

    def write_status(self, payload: dict[str, Any]) -> None:
        payload["log_path"] = str(self.log_path.relative_to(ROOT))
        payload["status_path"] = str(STATUS_PATH.relative_to(ROOT))
        temp_path = STATUS_PATH.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temp_path.replace(STATUS_PATH)

    def run(self) -> int:
        extra_tickers = self.args.tickers or []
        tickers = resolve_tickers(extra_tickers)
        ai_tickers = normalize_tickers(self.args.ai_tickers or DEFAULT_AI_TICKERS)
        ai_tickers = [ticker for ticker in ai_tickers if ticker]
        tickers = normalize_tickers([*tickers, *ai_tickers])

        self.logger.info("QQQM Investment Assistant 每日自动任务启动")
        self.logger.info("项目目录：%s", ROOT)
        self.logger.info("行情标的：%s", ", ".join(tickers))
        self.logger.info("AI 报告标的：%s", ", ".join(ai_tickers) if ai_tickers else "无")
        self.logger.info("本次模式：%s", "演练，不执行 API 请求" if self.args.dry_run else "正式执行")

        before = snapshot_data()
        snapshot_bootstrap: dict[str, Any] = {"applied": False, "added_dates": []}
        if self.args.dry_run:
            self.logger.info("DRY RUN：跳过公开网页快照写入，数据库不会因演练模式而修改。")
        else:
            try:
                from app.public_market_snapshot import bootstrap_public_snapshot
                snapshot_bootstrap = bootstrap_public_snapshot(DB_PATH)
                if snapshot_bootstrap.get("applied"):
                    self.logger.warning(
                        "本地行情缺少近期数据，补入公开网页校验快照（不是实时 API 同步）：%s；来源 %s",
                        ", ".join(snapshot_bootstrap.get("added_dates", [])),
                        snapshot_bootstrap.get("source_url", "未知"),
                    )
            except Exception as exc:
                self.logger.warning("公开行情快照引导步骤未执行：%s: %s", type(exc).__name__, exc)

        market_command = [
            sys.executable, "main.py", "sync", "--period", self.args.period,
            "--tickers", *tickers,
        ]
        self.run_command("行情同步", market_command, timeout=self.args.timeout)

        if not self.args.skip_news:
            self.run_command(
                "美联储新闻与 BLS 数据同步",
                [sys.executable, "-m", "app.news_data", "sync"],
                timeout=self.args.timeout,
            )
        else:
            self.logger.info("按参数跳过新闻同步。")

        if not self.args.skip_macro:
            self.run_command(
                "FRED 宏观数据同步",
                [sys.executable, "main.py", "macro-sync"],
                timeout=self.args.timeout,
            )
        else:
            self.logger.info("按参数跳过 FRED 宏观数据同步。")

        # Classify pending news after syncing sources but before the new forecast
        # is generated. The context archive then includes only articles whose
        # publication AND classification timestamps precede forecast generation.
        if not self.args.skip_news:
            self.run_command(
                "新闻事件结构化分类（待处理新闻，最多 10 条）",
                [
                    sys.executable,
                    "scripts/classify_news_events.py",
                    "--limit", "10",
                    "--batch-size", "8",
                ],
                timeout=self.args.timeout,
            )

        after = snapshot_data()
        before_dates = before["market_dates"]
        after_dates = after["market_dates"]
        before_state = before.get("market_state", {})
        after_state = after.get("market_state", {})
        updated_market_tickers = sorted(
            ticker for ticker, latest_date in after_dates.items()
            if ticker in tickers and latest_date and (
                latest_date != before_dates.get(ticker)
                or after_state.get(ticker, {}).get("source") != before_state.get(ticker, {}).get("source")
                or after_state.get(ticker, {}).get("price") != before_state.get(ticker, {}).get("price")
            )
        )
        new_news_count = len(after["news_ids"] - before["news_ids"])
        new_fred_count = len(after["fred_macro_keys"] - before["fred_macro_keys"])
        new_bls_count = len(after["bls_macro_keys"] - before["bls_macro_keys"])
        new_macro_count = new_fred_count + new_bls_count

        self.logger.info("行情日期发生变化的标的：%s", ", ".join(updated_market_tickers) or "无")
        self.logger.info("新增新闻：%d 条；新增宏观观测：%d 条", new_news_count, new_macro_count)

        # Run forecast feedback on every invocation. Besides detecting a new market
        # date, repair the common state where the DB is newer than the saved forecast
        # (or the forecast file is missing/corrupt). Otherwise only settle outcomes.
        latest_saved_forecast_date = latest_forecast_as_of_date()
        latest_qqqm_date = after_dates.get("QQQM")
        forecast_is_behind = bool(
            latest_qqqm_date and (
                not latest_saved_forecast_date or latest_saved_forecast_date < latest_qqqm_date
            )
        )
        if self.args.dry_run or "QQQM" in updated_market_tickers or forecast_is_behind:
            forecast_cycle_args: list[str] = []
            if forecast_is_behind and "QQQM" not in updated_market_tickers:
                self.logger.info(
                    "发现预测文件缺失或落后于行情数据库（预测=%s，行情=%s），将重新生成最新预测。",
                    latest_saved_forecast_date or "无",
                    latest_qqqm_date or "未知",
                )
        else:
            forecast_cycle_args = ["--evaluate-only"]
        self.run_command(
            "预测反馈与每日模型更新",
            [sys.executable, "scripts/run_forecast_daily_cycle.py", *forecast_cycle_args],
            timeout=self.args.timeout,
        )

        if not self.args.skip_ai:
            if self.args.force_ai:
                ai_targets = [ticker for ticker in ai_tickers if ticker in after_dates]
                trigger_reason = "用户指定强制生成"
            else:
                ai_targets = [ticker for ticker in ai_tickers if ticker in updated_market_tickers]
                if new_news_count > 0 or new_macro_count > 0:
                    ai_targets = normalize_tickers([*ai_targets, *[t for t in ai_tickers if t in after_dates]])
                trigger_reason = "行情日期更新或发现新增新闻/宏观数据"

            if self.args.dry_run:
                # Dry-run reports planned AI targets without issuing model requests.
                if self.args.force_ai:
                    ai_targets = ai_tickers
                elif not ai_targets:
                    ai_targets = []
                for ticker in ai_targets:
                    self.run_command(
                        f"DeepSeek 分析（演练）：{ticker}",
                        [sys.executable, "main.py", "analyze", "--ticker", ticker, "--with-macro"],
                        timeout=self.args.timeout,
                    )
            elif not ai_targets:
                self.logger.info("没有检测到新行情/新新闻/新宏观观测，跳过 DeepSeek，避免重复消耗额度。")
            else:
                self.logger.info("触发 AI 分析原因：%s", trigger_reason)
                for ticker in ai_targets:
                    self.run_command(
                        f"DeepSeek 分析：{ticker}",
                        [sys.executable, "main.py", "analyze", "--ticker", ticker, "--with-macro"],
                        timeout=self.args.timeout,
                    )
        else:
            self.logger.info("按参数跳过 AI 分析。")

        failed_stages = [stage["name"] for stage in self.stages if not stage.get("success", False)]
        overall_success = not failed_stages and not self.errors
        finished_at = local_timestamp()
        status = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "overall_success": overall_success,
            "dry_run": bool(self.args.dry_run),
            "tickers": tickers,
            "ai_tickers": ai_tickers,
            "before_market_dates": before_dates,
            "after_market_dates": after_dates,
            "updated_market_tickers": updated_market_tickers,
            "public_market_snapshot_bootstrap": snapshot_bootstrap,
            "latest_saved_forecast_as_of_date_before_run": latest_saved_forecast_date,
            "new_news_count": new_news_count,
            "new_fred_macro_observations": new_fred_count,
            "new_bls_macro_observations": new_bls_count,
            "stages": self.stages,
            "errors": self.errors,
            "failed_stages": failed_stages,
        }
        self.write_status(status)
        self.logger.info("自动任务结束：%s", "成功" if overall_success else "部分步骤失败，请查看日志")
        self.logger.info("状态文件：%s", STATUS_PATH)
        self.logger.info("日志文件：%s", self.log_path)
        return 0 if overall_success else 1


def show_status() -> int:
    if not STATUS_PATH.exists():
        print("尚无自动化运行记录。请先运行：python -m app.daily_automation --dry-run")
        return 0
    try:
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"无法读取状态文件：{exc}")
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("overall_success") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动同步行情与宏观信息、复盘并更新下一交易日预测、按需生成 DeepSeek 报告。"
    )
    parser.add_argument("--period", default="30d", help="日常同步行情范围，默认 30d；也支持 1y/5y/max")
    parser.add_argument("--tickers", nargs="+", help="额外同步标的；默认读取 app.config、持仓 CSV 并包含 QQQM")
    parser.add_argument("--ai-tickers", nargs="+", default=DEFAULT_AI_TICKERS,
                        help="允许自动生成 AI 报告的标的，默认仅 QQQM，控制 API 成本")
    parser.add_argument("--force-ai", action="store_true", help="无论数据是否变化，都为 AI 标的生成报告")
    parser.add_argument("--skip-ai", action="store_true", help="跳过 AI 报告")
    parser.add_argument("--skip-news", action="store_true", help="跳过 Fed/BLS 新闻和数据同步")
    parser.add_argument("--skip-macro", action="store_true", help="跳过 FRED 宏观数据同步")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划命令，不发起同步或模型请求")
    parser.add_argument("--status", action="store_true", help="查看最近一次自动任务的状态 JSON")
    parser.add_argument("--timeout", type=int, default=600, help="每个子任务超时秒数，默认 600")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.status:
        return show_status()
    if args.timeout < 1:
        print("--timeout 必须大于 0", file=sys.stderr)
        return 2
    return AutomationRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
