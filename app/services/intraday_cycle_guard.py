"""Safe gate for scheduled runs and catch-up runs when the user opens the app."""
from __future__ import annotations

from datetime import date, datetime, time


MARKET_OPEN = time(9, 30)
CLOSE_READY = time(16, 30)


def _as_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    # Accept YYYY-MM-DD and ISO datetime strings like YYYY-MM-DDT00:00:00.
    return date.fromisoformat(raw[:10])


def evaluate_catchup_gate(
    now_ny: datetime,
    current_day_is_session: bool,
    latest_data_date: str | date | datetime | None,
    latest_forecast_as_of_date: str | date | datetime | None,
    latest_data_day_is_session: bool,
    next_session_date: str | date | datetime | None,
    market_open_time: time = MARKET_OPEN,
    close_ready_time: time = CLOSE_READY,
) -> tuple[bool, str]:
    """Decide if one catch-up cycle can run without creating a duplicate or stale forecast.

    Rules:
    - Never create a new pre-market-style forecast while the regular session is active.
    - Before market open, allow a missed forecast only when latest data is from the
      immediately previous session and is newer than the latest forecast snapshot.
    - After 16:30 ET, require the current session's bars to be present.
    - On a weekend/holiday, allow a catch-up only if the next scheduled session is
      still in the future and there is no forecast yet for the latest data date.
    """
    if now_ny.tzinfo is None or now_ny.utcoffset() is None:
        return False, "纽约当前时间缺少时区信息；为避免错误日期，跳过本次流程。"

    data_day = _as_date(latest_data_date)
    forecast_day = _as_date(latest_forecast_as_of_date)
    next_day = _as_date(next_session_date)
    today = now_ny.date()
    current_time = now_ny.timetz().replace(tzinfo=None)

    if data_day is None:
        return False, "没有找到 QQQM 15 分钟行情日期；跳过本次补跑。"
    if not latest_data_day_is_session:
        return False, f"最新行情日期 {data_day.isoformat()} 不是有效 NASDAQ 交易日；跳过补跑。"
    if next_day is None:
        return False, "无法确认下一次 NASDAQ 交易日；跳过补跑。"

    if forecast_day is not None and forecast_day >= data_day:
        return False, (
            f"最新行情日期 {data_day.isoformat()} 已有预测快照（as_of {forecast_day.isoformat()}）；"
            "为避免重复生成，本次不再运行预测。"
        )

    if current_day_is_session:
        if market_open_time <= current_time < close_ready_time:
            return False, (
                "当前处于美股交易时段或收盘数据等待窗口；不补做盘前预测，"
                "避免把盘中信息伪装成开盘前预测。"
            )

        if current_time < market_open_time:
            if next_day < today:
                return False, (
                    f"最新行情 {data_day.isoformat()} 之后已有交易日未同步；"
                    "行情可能过期，跳过补跑。"
                )
            if next_day > today or data_day >= today:
                return False, (
                    "盘前数据日期不是一个已完成的上一交易日；"
                    "为避免使用不完整行情，跳过本次补跑。"
                )
            # next_day == today and data_day < today: missed pre-market forecast is eligible.
            return True, (
                f"检测到盘前启动且尚无基于 {data_day.isoformat()} 收盘数据的预测；"
                f"补跑目标为 {today.isoformat()}。"
            )

        # At or after close_ready_time on a regular NASDAQ session.
        if data_day != today:
            return False, (
                f"纽约今天是交易日 {today.isoformat()}，但最新行情日期为 {data_day.isoformat()}；"
                "当天行情可能尚未同步完整，跳过补跑。"
            )
        return True, f"纽约交易日 {today.isoformat()} 已收盘且数据已更新，运行预测与复盘。"

    # Weekend or holiday: no session is active. Only catch up if the next session
    # remains in the future; if a session should already have happened, data is stale.
    if next_day <= today:
        return False, (
            f"最新行情 {data_day.isoformat()} 之后应已有交易日，但尚未找到更新行情；"
            "跳过补跑，避免使用过期数据。"
        )
    return True, (
        f"当前为非交易日；最新完整行情为 {data_day.isoformat()}，"
        f"下一个交易日为 {next_day.isoformat()}，允许补做一次预测。"
    )


def evaluate_cycle_gate(
    now_ny: datetime,
    is_market_session: bool,
    latest_data_date: str | None,
    close_ready_time: time = CLOSE_READY,
) -> tuple[bool, str]:
    """Backward-compatible strict gate for older callers/tests."""
    today = now_ny.date().isoformat()
    if now_ny.tzinfo is None or now_ny.utcoffset() is None:
        return False, "纽约当前时间缺少时区信息；为避免错误日期，跳过本次流程。"
    if not is_market_session:
        return False, f"纽约日期 {today} 不是 NASDAQ 交易日；跳过预测生成。"
    if now_ny.timetz().replace(tzinfo=None) < close_ready_time:
        return False, f"尚未到收盘后数据检查时间（纽约时间 {close_ready_time.strftime('%H:%M')}）；跳过预测生成。"
    if not latest_data_date:
        return False, "行情表中没有找到 QQQM 15 分钟数据；跳过后续步骤。"
    if str(latest_data_date)[:10] != today:
        return False, f"最新 15 分钟数据日期为 {latest_data_date}，纽约今天是 {today}；数据可能尚未同步完整，跳过新预测。"
    return True, "交易日已结束且当天 15 分钟数据已同步，继续评估和生成预测。"
