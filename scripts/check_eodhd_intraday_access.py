from __future__ import annotations

import importlib
import json
import os
import sqlite3
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import requests
except ImportError:
    print("缺少 requests。请先运行：& .\\.venv\\Scripts\\python.exe -m pip install requests")
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "market_data.db"
ENV_PATH = ROOT / ".env"
API_BASE = "https://eodhd.com/api/intraday/QQQM.US"


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            values[key] = value
    return values


def _find_api_token() -> tuple[str | None, str | None]:
    candidates: list[tuple[str, str]] = []

    for name, value in os.environ.items():
        upper = name.upper()
        if "EODHD" in upper and any(part in upper for part in ("TOKEN", "KEY", "API")) and value.strip():
            candidates.append((name, value.strip()))

    for name, value in _parse_env_file(ENV_PATH).items():
        upper = name.upper()
        if "EODHD" in upper and any(part in upper for part in ("TOKEN", "KEY", "API")) and value.strip():
            candidates.append((name, value.strip()))

    # Try the project's existing config module without displaying secret values.
    try:
        module = importlib.import_module("app.config")
        for name in dir(module):
            upper = name.upper()
            if "EODHD" not in upper or not any(part in upper for part in ("TOKEN", "KEY", "API")):
                continue
            value = getattr(module, name, None)
            if isinstance(value, str) and value.strip():
                candidates.append((f"app.config.{name}", value.strip()))
    except Exception:
        pass

    # Prefer common names in a predictable order, without ever printing the value.
    preferred = {"EODHD_API_TOKEN": 0, "EODHD_API_KEY": 1, "EODHD_TOKEN": 2, "EODHD_APIKEY": 3}
    candidates.sort(key=lambda item: preferred.get(item[0].split(".")[-1].upper(), 10))
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, value in candidates:
        if value not in seen:
            unique.append((name, value))
            seen.add(value)
    if not unique:
        return None, None
    return unique[0][1], unique[0][0]


def _latest_market_date() -> date:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"找不到行情数据库：{DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT MAX(trade_date) FROM market_daily WHERE ticker = ?", ("QQQM",)
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        raise RuntimeError("market_daily 中没有 QQQM 行情记录。")
    return date.fromisoformat(row[0])


def _safe_response_body(response: requests.Response, token: str | None = None) -> str:
    text = (response.text or "").strip().replace("\r", " ").replace("\n", " ")
    if token:
        text = text.replace(token, "[REDACTED]")
    if len(text) > 500:
        text = text[:500] + "…"
    return text


def main() -> int:
    print("========== EODHD 5分钟行情权限检查 ==========")
    print(f"项目目录：{ROOT}")

    token, source_name = _find_api_token()
    if not token:
        print("没有从环境变量、.env 或 app.config 中识别到 EODHD API token。")
        print("请检查本地已有密钥变量名是否包含 EODHD 和 TOKEN/KEY/API；不要把密钥值发给任何人。")
        return 2

    print(f"API 密钥来源：{source_name}（密钥值不会显示）")

    try:
        latest_date = _latest_market_date()
    except Exception as exc:
        print(f"读取最新 QQQM 行情日期失败：{exc}")
        return 2

    ny = ZoneInfo("America/New_York")
    session_open = datetime.combine(latest_date, time(9, 30), tzinfo=ny)
    session_close = datetime.combine(latest_date, time(16, 0), tzinfo=ny)
    params = {
        "api_token": token,
        "interval": "5m",
        "from": int(session_open.timestamp()),
        "to": int(session_close.timestamp()),
        "fmt": "json",
    }

    print(f"检查标的：QQQM.US")
    print(f"检查交易日：{latest_date.isoformat()}（美东时间）")
    print("请求间隔：5 分钟；只发起一次小范围查询。")

    try:
        response = requests.get(API_BASE, params=params, timeout=25)
    except requests.RequestException as exc:
        print(f"请求未完成：{type(exc).__name__}。为避免泄露 API 密钥，不显示底层请求详情。")
        return 3

    print(f"HTTP 状态码：{response.status_code}")
    if response.status_code != 200:
        body = _safe_response_body(response, token)
        print(f"返回信息：{body or '(无响应正文)'}")
        if response.status_code in (401, 403):
            print("判断：可能是套餐权限、API token 或访问授权问题。请先检查 EODHD 账户权限，不要重复请求。")
        elif response.status_code == 429:
            print("判断：触发限流。请等待限制解除，不要立即反复请求。")
        return 4

    try:
        payload: Any = response.json()
    except ValueError:
        print(f"响应不是 JSON：{_safe_response_body(response, token)}")
        return 5

    if not isinstance(payload, list):
        print("接口已响应，但返回内容不是预期的 K 线数组：")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:1200])
        print("这通常表示接口错误信息被包在 JSON 中，而不是成功返回分钟线。")
        return 5

    rows = payload
    print(f"返回 5 分钟 K 线数量：{len(rows)}")
    if not rows:
        print("接口可访问，但这个日期没有返回分钟线。可能是数据权限、日期范围或该交易日数据可用性问题。")
        return 6

    first = rows[0]
    last = rows[-1]
    keys = sorted({str(key) for row in rows[:3] if isinstance(row, dict) for key in row.keys()})
    print(f"返回字段：{', '.join(keys)}")
    print(f"第一根 K 线（简要）：{json.dumps(first, ensure_ascii=False)[:400]}")
    print(f"最后一根 K 线（简要）：{json.dumps(last, ensure_ascii=False)[:400]}")
    print("结果：已成功获取分钟级数据，可以开始设计次日开盘预测与盘中轨迹模型。")
    print("注意：分钟线历史数据可用不代表未来路径能被准确预测；模型仍需滚动样本外验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
