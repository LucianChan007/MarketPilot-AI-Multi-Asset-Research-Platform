from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
SYMBOL = "QQQM"
TRADE_DATE = "2026-10-13"  # US Eastern trading date


def load_dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and value:
            values[key] = value
    return values


def get_credential(name: str, dotenv: dict[str, str]) -> str | None:
    value = os.environ.get(name) or dotenv.get(name)
    if value:
        return value.strip()
    return None


def main() -> int:
    dotenv = load_dotenv_values(ENV_FILE)
    key = get_credential("ALPACA_API_KEY_ID", dotenv)
    secret = get_credential("ALPACA_API_SECRET_KEY", dotenv)

    print("========== Alpaca 5分钟行情权限检查 ==========")
    print(f"项目目录：{ROOT}")
    print(f"检查标的：{SYMBOL}")
    print(f"检查美东交易日：{TRADE_DATE}")
    print("数据源：Alpaca Basic feed=iex；免费层数据可能不覆盖美国所有交易所。")
    print("凭据检查：只报告是否存在，不会输出密钥值。")

    if not key or not secret:
        print("结果：未找到 Alpaca API 凭据。")
        print("如果你决定使用 Alpaca 免费数据，请先注册账户并创建 Market Data API key。")
        print("将以下两项加入项目根目录 .env（不要把密钥发到聊天或提交到 GitHub）：")
        print("ALPACA_API_KEY_ID=你的KeyID")
        print("ALPACA_API_SECRET_KEY=你的SecretKey")
        return 2

    params = {
        "timeframe": "5Min",
        "start": f"{TRADE_DATE}T13:30:00Z",
        "end": f"{TRADE_DATE}T20:00:00Z",
        "limit": "10000",
        "feed": "iex",
        "sort": "asc",
    }
    url = "https://data.alpaca.markets/v2/stocks/" + SYMBOL + "/bars?" + urlencode(params)
    request = Request(
        url,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
            "User-Agent": "QQQM-Investment-Assistant-intraday-check/1.0",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=25) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError) as exc:
        print(f"请求失败：{type(exc).__name__}: {exc}")
        return 3

    print(f"HTTP 状态码：{status}")
    if status != 200:
        try:
            payload = json.loads(body)
            message = payload.get("message") or payload.get("error") or body[:1000]
        except json.JSONDecodeError:
            message = body[:1000]
        print(f"返回信息：{message}")
        if status in (401, 403):
            print("判断：凭据无效或该请求/数据源权限不可用。请勿连续重试；核对 Alpaca Market Data API 设置。")
        elif status == 429:
            print("判断：触发频率限制，请稍后再试。")
        return 1

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print("返回内容不是有效 JSON，暂时无法确认权限。")
        print(body[:500])
        return 4

    bars = payload.get("bars", [])
    if not isinstance(bars, list):
        bars = []
    print(f"返回 5分钟K线数量：{len(bars)}")
    if not bars:
        print("结果：API 请求成功，但该标的在所选 IEX feed/时段没有返回 K 线。")
        print("这不等于有完整盘中历史数据；继续建模前先确认另一个交易日是否有数据或选择有完整历史覆盖的行情源。")
        return 5

    print("前3条K线（时间为 UTC）：")
    for bar in bars[:3]:
        print({field: bar.get(field) for field in ("t", "o", "h", "l", "c", "v")})
    print("最后1条K线（时间为 UTC）：")
    last = bars[-1]
    print({field: last.get(field) for field in ("t", "o", "h", "l", "c", "v")})
    print("判断：已成功读取有限的 5分钟行情。免费 IEX 数据可能并非全市场完整成交记录，先用于原型验证；不要和 EODHD 的另一种 feed 混在同一训练序列中而不做数据源标记。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
