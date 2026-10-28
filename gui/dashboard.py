r"""Streamlit Dashboard for QQQM Investment Assistant.

Run from project root:
    .\.venv\Scripts\python.exe -m streamlit run gui/dashboard.py

The clock updates once per second without re-running the rest of the dashboard
when the installed Streamlit version supports st.fragment(). This page reads
local reports only; it does not call market APIs or DeepSeek on each refresh.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pandas as pd
from html import escape
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.market_clock import get_market_clock  # noqa: E402
from app.services.market_calendar import next_nasdaq_session  # noqa: E402
from app.services.daily_review import (  # noqa: E402
    build_daily_review_context,
    load_recent_human_feedback,
    save_human_feedback,
)
from gui.components.chart import render_intraday_chart  # noqa: E402
from app.services.auth import AuthService, AuthError  # noqa: E402
from app.services.portfolio_ledger import PortfolioLedger, TradeValidationError  # noqa: E402
from app.services.portfolio_tools import export_user_activity_csv, backup_sqlite_database  # noqa: E402
from app.services.system_health import inspect_project, format_time, status_label  # noqa: E402

FORECAST_PATH = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
VIDEO_FORECAST_PATH = ROOT / "reports" / "intraday" / "QQQM_video_forecast_latest.json"
SUMMARY_PATH = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_vs_actual_summary_latest.json"
CONTEXT_REPORT_PATH = ROOT / "reports" / "context_agent" / "QQQM_context_analysis_latest.md"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _money(value: Any, digits: int = 2) -> str:
    try:
        return f"${float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _utc_to_local(value: Any, timezone_name: str = "Asia/Hong_Kong") -> str:
    if not isinstance(value, str) or not value.strip():
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, KeyError):
        return value


def _clock_styles() -> None:
    """Keep the live clock compact without changing the size of other metrics."""
    st.markdown(
        """
        <style>
        .qqqm-clock-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.55rem;
            width: 100%;
            margin: 0.25rem 0 0.45rem 0;
        }
        .qqqm-clock-card {
            min-width: 0;
            box-sizing: border-box;
            padding: 0.65rem 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.26);
            border-radius: 0.7rem;
            background: var(--secondary-background-color, rgba(128, 128, 128, 0.06));
        }
        .qqqm-clock-label {
            font-size: 0.76rem;
            line-height: 1.25;
            opacity: 0.78;
            margin-bottom: 0.35rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .qqqm-clock-value {
            font-size: clamp(0.95rem, 1.2vw, 1.18rem);
            font-weight: 650;
            line-height: 1.3;
            letter-spacing: -0.015em;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .qqqm-clock-meta {
            font-size: 0.70rem;
            line-height: 1.3;
            opacity: 0.68;
            margin-top: 0.18rem;
            min-height: 0.9rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        @media (max-width: 1000px) {
            .qqqm-clock-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 480px) {
            .qqqm-clock-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.4rem; }
            .qqqm-clock-card { padding: 0.55rem 0.6rem; }
            .qqqm-clock-value { font-size: 0.98rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _split_datetime(value: str) -> tuple[str, str, str]:
    """Return date, time and optional zone abbreviation for compact cards."""
    parts = str(value or "—").split()
    if len(parts) >= 2:
        zone = parts[2] if len(parts) >= 3 else ""
        return parts[0], parts[1], zone
    return "", str(value or "—"), ""


def _short_market_status(raw_status: str) -> tuple[str, str]:
    """Use a short headline while retaining the full explanation underneath."""
    raw = str(raw_status or "状态未知")
    if "正常交易中" in raw:
        return "🟢 交易中", "美股常规交易时段"
    if "盘前" in raw:
        return "🟡 盘前", "常规交易尚未开市"
    if "盘后" in raw:
        return "🟠 盘后", "常规交易已经收市"
    if "休市" in raw:
        if "非交易日" in raw:
            return "🔴 休市", "周末或交易所假日"
        return "🔴 休市", "当前不在常规交易时段"
    return "⚪ 未知", raw


def _short_countdown_label(raw_label: str) -> str:
    raw = str(raw_label or "市场倒计时")
    if "收盘" in raw:
        return "距常规收盘"
    if "开盘" in raw and "下次" in raw:
        return "距下次开盘"
    if "开盘" in raw:
        return "距常规开盘"
    return "市场倒计时"


def _render_clock_content() -> None:
    clock = get_market_clock()
    local_date, local_time, _ = _split_datetime(clock.get("local_time", "—"))
    ny_date, ny_time, ny_zone = _split_datetime(clock.get("new_york_time", "—"))
    local_zone = str(clock.get("local_timezone", "Asia/Hong_Kong"))
    zone_abbreviations = {
        "Asia/Hong_Kong": "HKT",
        "Asia/Tokyo": "JST",
        "Asia/Shanghai": "CST",
        "Asia/Singapore": "SGT",
        "Asia/Taipei": "Taipei",
    }
    local_zone_label = zone_abbreviations.get(local_zone, local_zone.rsplit("/", 1)[-1].replace("_", " "))
    status_value, status_detail = _short_market_status(clock.get("market_status", "状态未知"))
    countdown_label = _short_countdown_label(clock.get("countdown_label", "市场倒计时"))
    countdown = str(clock.get("countdown") or "—")

    cards = [
        (f"本地时间 · {local_zone_label}", local_time, local_date),
        (f"美东时间{(' · ' + ny_zone) if ny_zone else ''}", ny_time, ny_date),
        ("美股市场状态", status_value, status_detail),
        (countdown_label, countdown, "按交易日历计算"),
    ]
    card_html = []
    for label, value, meta in cards:
        card_html.append(
            '<div class="qqqm-clock-card">'
            f'<div class="qqqm-clock-label">{escape(str(label))}</div>'
            f'<div class="qqqm-clock-value" title="{escape(str(value))}">{escape(str(value))}</div>'
            f'<div class="qqqm-clock-meta" title="{escape(str(meta))}">{escape(str(meta))}</div>'
            '</div>'
        )

    st.markdown(
        '<div class="qqqm-clock-grid">' + "".join(card_html) + '</div>',
        unsafe_allow_html=True,
    )

    source = clock.get("calendar_source", "未知")
    note = clock.get("calendar_note", "")
    st.caption(f"时间按秒刷新 · 时区：{local_zone} · 状态依据：{source}。{note}")


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def _live_market_clock() -> None:
        _render_clock_content()

    def _show_live_clock() -> None:
        _live_market_clock()
else:
    # Compatibility fallback for older Streamlit versions.
    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        st_autorefresh = None

    def _show_live_clock() -> None:
        if st_autorefresh is not None:
            st_autorefresh(interval=1000, key="market_clock_autorefresh")
        _render_clock_content()
        if st_autorefresh is None:
            st.warning(
                "当前 Streamlit 版本不支持 st.fragment，且未安装 streamlit-autorefresh，"
                "所以时间不会自动跳秒。请升级 Streamlit，或安装兼容刷新组件。"
            )


AUTH_DB_PATH = ROOT / "data" / "users.db"
AUTH_SESSION_KEY = "qqqm_authenticated_user"


@st.cache_resource(show_spinner=False)
def _get_auth_service(db_path: str) -> AuthService:
    """Share the stateless service object; each database operation opens a connection."""
    return AuthService(db_path)


@st.cache_resource(show_spinner=False)
def _get_portfolio_ledger(db_path: str) -> PortfolioLedger:
    """Share the ledger service; every ledger call opens an independent SQLite connection."""
    return PortfolioLedger(db_path)


def _render_auth_gate() -> bool:
    """Reuse the central auth gate when this page is opened via dashboard.py."""
    if st.session_state.get("_qqqm_unified_navigation"):
        return isinstance(st.session_state.get(AUTH_SESSION_KEY), dict)
    try:
        auth_service = _get_auth_service(str(AUTH_DB_PATH))
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        st.error(f"无法初始化本地账号数据库：{exc}")
        st.stop()

    principal = st.session_state.get(AUTH_SESSION_KEY)
    if isinstance(principal, dict):
        user = auth_service.get_user_by_id(principal.get("user_id"))
        if user:
            # Revalidate every full app rerun so disabled accounts lose access.
            st.session_state[AUTH_SESSION_KEY] = {
                "user_id": user["user_id"],
                "username": user["username"],
            }
            with st.sidebar:
                st.caption(f"已登录：{user['username']}")
                if st.button("退出登录", key="qqqm_logout_button", use_container_width=True):
                    st.session_state.pop(AUTH_SESSION_KEY, None)
                    st.rerun()
            return True
        st.session_state.pop(AUTH_SESSION_KEY, None)

    st.title("🔐 QQQM Investment Assistant")
    st.caption("请登录或创建本地账号后使用投资 Dashboard。")
    st.info(
        "账号与密码已经单独保存；公共行情和预测报告仍然共享。个人持仓、买卖流水和股息收入会按当前账号写入本地账本。"
    )
    st.warning(
        "当前版本适合本机开发验证，不要直接开放到公网。正式多人部署前还需要 HTTPS、"
        "登录限流、密码找回、服务器级会话保护和生产数据库。"
    )

    login_tab, register_tab = st.tabs(["登录", "注册新账号"])
    with login_tab:
        with st.form("qqqm_login_form", clear_on_submit=True):
            login_username = st.text_input("用户名", max_chars=32, key="qqqm_login_username")
            login_password = st.text_input("密码", type="password", key="qqqm_login_password")
            login_submitted = st.form_submit_button("登录", type="primary", use_container_width=True)
        if login_submitted:
            user = auth_service.authenticate(login_username, login_password)
            if user is None:
                st.error("用户名或密码不正确，或账号暂不可用。")
            else:
                st.session_state[AUTH_SESSION_KEY] = {
                    "user_id": user["user_id"],
                    "username": user["username"],
                }
                st.rerun()

    with register_tab:
        st.caption("用户名 3–32 位，只能包含英文字母、数字、点、下划线或连字符。密码至少 12 个字符。")
        with st.form("qqqm_register_form", clear_on_submit=True):
            register_username = st.text_input("新用户名", max_chars=32, key="qqqm_register_username")
            register_password = st.text_input("设置密码", type="password", key="qqqm_register_password")
            register_confirm = st.text_input("再次输入密码", type="password", key="qqqm_register_confirm")
            register_submitted = st.form_submit_button("创建账号并登录", type="primary", use_container_width=True)
        if register_submitted:
            if register_password != register_confirm:
                st.error("两次输入的密码不一致。")
            else:
                try:
                    user = auth_service.register(register_username, register_password)
                except AuthError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[AUTH_SESSION_KEY] = {
                        "user_id": user["user_id"],
                        "username": user["username"],
                    }
                    st.success("账号创建成功，正在进入 Dashboard。")
                    st.rerun()
    return False


if not _render_auth_gate():
    st.stop()


_clock_styles()

st.title("📈 QQQM 每日总览与复盘")
st.caption("每天先看昨天预测得怎么样，再看今天的数据和下一交易日情景。这里只做研究，不会自动买卖。")
st.info("使用顺序建议：① 看上一交易日实际涨跌；② 对照已保存的预测与真实价格；③ 检查模型评估；④ 再去‘多平台视频脚本与配图’制作视频内容。页面显示的价格可能是最近收盘价，不一定是实时行情。")

with st.container(border=True):
    st.subheader("🕒 市场时间与交易状态")
    _show_live_clock()

# Data health is intentionally read-only. Opening the home page does not sync any provider.
_health = inspect_project(ROOT)
_expected = _health.get("expected_last_completed_session")
_daily_health = _health.get("daily", {})
_intraday_health = _health.get("intraday", {})
_news_health = _health.get("news", {})
_video_health = _health.get("video", {})
_daily_label, daily_note = status_label(_daily_health.get("latest_date"), _expected)
_intraday_label, intraday_note = status_label(_intraday_health.get("latest_complete_date"), _expected)
with st.container(border=True):
    st.subheader("🩺 数据更新状态")
    st.caption("这里展示本地数据库里实际保存到哪一天，不代表页面刚刚联网刷新。打开首页不会自动下载行情、生成预测或训练模型。")
    hs = st.columns(4)
    hs[0].metric("应有的最近完整交易日", _expected or "暂无法确认")
    hs[1].metric("日线实际最新日", _daily_health.get("latest_date") or "缺失", help=f"状态：{_daily_label}。{daily_note}")
    hs[2].metric("完整分时最新日", _intraday_health.get("latest_complete_date") or "缺失", help=f"状态：{_intraday_label}。{intraday_note}")
    hs[3].metric("最新视频包", _video_health.get("folder") or "尚未生成")
    if _expected and _daily_health.get("latest_date") and _daily_health.get("latest_date") < _expected:
        st.warning(f"日线缓存滞后：本地最新为 {_daily_health.get('latest_date')}，按{_health.get('calendar_source', '交易日历')}应已到 {_expected}。先到‘多平台视频脚本与配图’刷新公开行情，再重新生成预测。")
    if _expected and _intraday_health.get("latest_complete_date") and _intraday_health.get("latest_complete_date") < _expected:
        st.warning(f"完整分时缓存滞后：最近完整交易日为 {_intraday_health.get('latest_complete_date')}。盘中路径模型不能用日线四个价格点替代分时数据。")
    if _news_health.get("latest_published_date"):
        st.caption(f"新闻：最新已收录发布时间 {_news_health.get('latest_published_date')}；最近抓取时间 {format_time(_news_health.get('latest_fetched_at'))}。抓取得近不等于新闻发布时间就是最新。")
    with st.expander("查看各数据源、密钥配置与流程状态"):
        rows = [
            {"数据类别": "日线行情", "本地数据最新日期": _daily_health.get("latest_date") or "缺失", "最后抓取时间": format_time(_daily_health.get("fetched_at")), "说明": f"来源：{_daily_health.get('source') or '未知'}；{daily_note}"},
            {"数据类别": "完整十五分钟分时", "本地数据最新日期": _intraday_health.get("latest_complete_date") or "缺失", "最后抓取时间": format_time(_intraday_health.get("fetched_at")), "说明": f"来源：{_intraday_health.get('feed') or '未知'}；{intraday_note}"},
            {"数据类别": "新闻", "本地数据最新日期": _news_health.get("latest_published_date") or "缺失", "最后抓取时间": format_time(_news_health.get("latest_fetched_at")), "说明": f"本地累计 {_news_health.get('count', 0)} 条；发布日期和抓取时间是两回事"},
            {"数据类别": "宏观数据", "本地数据最新日期": _health.get("macro", {}).get("latest_observation_date") or "缺失", "最后抓取时间": format_time(_health.get("macro", {}).get("latest_retrieved_at")), "说明": "宏观指标按各自发布周期更新，不一定每日都有新观测"},
            {"数据类别": "模型评估", "本地数据最新日期": _health.get("model_evaluation", {}).get("data_latest_date") or "未生成成绩单", "最后抓取时间": format_time(_health.get("model_evaluation", {}).get("created_at")), "说明": f"滚动验证样本 {_health.get('model_evaluation', {}).get('rolling_test_samples', '—')} 条；前瞻已结算 {_health.get('model_evaluation', {}).get('live_evaluated_samples', '—')} 条"},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        _cred = _health.get("credentials", {})
        st.write("**密钥检查（只显示是否配置，不显示密钥内容）**")
        st.write(f"Alpaca：{'已检测到 Key 与 Secret' if _cred.get('alpaca_configured') else '未配置完整的 Key/Secret'}；EODHD：{'已配置' if _cred.get('eodhd_configured') else '未配置'}；DeepSeek：{'已配置' if _cred.get('deepseek_configured') else '未配置'}。")
        _cycle_health = _health.get("intraday_cycle") or {}
        if _cycle_health:
            st.write(f"原有盘中学习流程最近运行：{format_time(_cycle_health.get('finished_at') or _cycle_health.get('started_at'))}；结果：{'成功' if _cycle_health.get('overall_success') else '失败/未完成'}。这是上次运行记录，不代表视频流程失败。")
        else:
            st.write("原有盘中学习流程：尚无本地运行记录。")
        st.page_link("app/pages/system_status.py", label="打开完整数据更新与系统诊断", icon="🩺")

forecast = _read_json(FORECAST_PATH)
video_forecast = _read_json(VIDEO_FORECAST_PATH)
summary = _read_json(SUMMARY_PATH) or {}

if forecast is None and not (video_forecast and video_forecast.get("available") is True):
    st.warning(
        "目前没有可供前瞻展示的新鲜预测文件。首页将继续显示数据状态、持仓和历史复盘；"
        "这不会阻断其他功能。请前往‘多平台视频脚本与配图’刷新公开行情后再生成新情景。"
    )
if forecast is None:
    forecast = video_forecast or {}

open_data = forecast.get("estimated_open") or {}
close_data = forecast.get("estimated_close_from_path") or {}
target_date = str(forecast.get("forecast_date_weekday_estimate") or "待确认")
as_of_date = str(forecast.get("as_of_date") or "未知")
previous_close = forecast.get("previous_session_close")
_latest_actual_close = None
try:
    with sqlite3.connect(ROOT / "data" / "market_data.db") as _conn:
        _latest_row = _conn.execute(
            "SELECT trade_date, close FROM market_daily WHERE UPPER(ticker)='QQQM' ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
    _latest_actual_date = str(_latest_row[0])[:10] if _latest_row and _latest_row[0] else None
    _latest_actual_close = float(_latest_row[1]) if _latest_row and _latest_row[1] is not None else None
except (sqlite3.Error, TypeError, ValueError):
    _latest_actual_date = None
    _latest_actual_close = None
_ny_now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
_ny_closed = _ny_now.time().replace(tzinfo=None) >= time(16, 15)
legacy_stale = False
if _latest_actual_date and as_of_date not in ("", "未知") and _latest_actual_date > as_of_date:
    legacy_stale = True
if target_date != "待确认":
    try:
        _target_day = date.fromisoformat(target_date[:10])
        if _target_day < _ny_now.date() or (_target_day == _ny_now.date() and _ny_closed):
            legacy_stale = True
    except ValueError:
        pass

# Only use the fresh video curve as a forward-looking display after its target is
# later than both today's completed session and the latest actual daily bar.
chart_forecast = None
if video_forecast and video_forecast.get("available") is True and video_forecast.get("curve"):
    _video_target = str(video_forecast.get("forecast_date_weekday_estimate") or "")
    _video_asof = str(video_forecast.get("as_of_date") or "")
    if _video_target and (_latest_actual_date is None or (_video_asof >= _latest_actual_date and _video_target > _latest_actual_date)):
        try:
            _video_target_day = date.fromisoformat(_video_target[:10])
            if _video_target_day > _ny_now.date() or (_video_target_day == _ny_now.date() and not _ny_closed):
                chart_forecast = video_forecast
        except ValueError:
            pass

if legacy_stale:
    st.error(
        f"旧预测目标日为 {target_date}，预测基准为 {as_of_date}。"
        + (f"本地真实日线已到 {_latest_actual_date}。" if _latest_actual_date else "")
        + "旧曲线仅供历史复盘，不再作为下一交易日预测展示。请在‘多平台视频脚本与配图’刷新行情并生成新情景。"
    )
    if _latest_actual_close is not None:
        previous_close = _latest_actual_close

if chart_forecast:
    st.info(f"盘中曲线图将显示新生成的 {chart_forecast.get('forecast_date_weekday_estimate')} 情景参考。")

main_path_obj = forecast.get("main_path") if isinstance(forecast.get("main_path"), dict) else {}
main_open = main_path_obj.get("predicted_open_price", summary.get("main_path_open_price"))
main_close = main_path_obj.get("predicted_close_price", summary.get("main_path_close_price"))
main_model_name = str(main_path_obj.get("model_name") or "主路径尚未生成")
main_news_applied = bool(main_path_obj.get("news_adjustment_applied", summary.get("news_adjustment_applied", False)))

st.divider()
st.subheader("📌 下一交易日主预测" if not legacy_stale else "📌 上次预测记录（已过期）")
info_a, info_b, info_c, info_d = st.columns(4)
info_a.metric("最近可用收盘参考价", _money(previous_close))
info_b.metric("主路径预计开盘", _money(main_open) if not legacy_stale else "已停用")
info_c.metric("主路径预计收盘", _money(main_close) if not legacy_stale else "已停用")
info_d.metric("新闻实验调整", ("已应用" if main_news_applied else "未应用") if not legacy_stale else "不适用")
main_is_independent = bool(main_path_obj.get("independent_of_quantile_curves", False))
if legacy_stale:
    st.info("过期预测的开盘价、收盘价和新闻调整状态不再展示为当前估计。新鲜曲线若已生成，会显示在下面的盘中曲线图中。")
elif main_is_independent:
    st.info(
        f"预测依据：截至 {as_of_date} 的已同步数据；目标日期：{target_date}。"
        "粗线是系统独立算出的主预测，不是把 P50 当答案。它仍是实验预测，"
        "不保证实际价格会照着它走。顶部参考价是最近一次已同步收盘价，不是实时成交价。"
    )
else:
    st.warning(
        f"目标日期：{target_date}。当前没有可用的独立主预测模型结果，"
        "页面可能正在显示旧方法生成的参考路径。请先重新运行预测，再把这条线当作主模型结果查看。"
    )

# Account-scoped portfolio ledger. All reads/writes use the authenticated user_id.
current_user = st.session_state.get(AUTH_SESSION_KEY, {})
user_id = int(current_user["user_id"])
username = str(current_user.get("username") or "当前用户")
portfolio_ledger = _get_portfolio_ledger(str(AUTH_DB_PATH))

local_timezone_name = os.getenv("QQQM_LOCAL_TIMEZONE", "Asia/Hong_Kong")
try:
    portfolio_tz = ZoneInfo(local_timezone_name)
except Exception:
    portfolio_tz = ZoneInfo("Asia/Hong_Kong")
    local_timezone_name = "Asia/Hong_Kong"

st.divider()
st.subheader("💼 我的持仓与交易账本")
st.caption(f"当前账号：{username} · 交易记录只属于当前账号，并保存在本地账号数据库中。")
st.caption("行情参考价使用最近已同步收盘价，不是实时成交价；此模块不会连接券商，也不会自动下单。")

try:
    ledger_summary = portfolio_ledger.get_position_summary(
        user_id=user_id,
        symbol="QQQM",
        reference_price=float(previous_close) if previous_close is not None else None,
    )
except (OSError, sqlite3.Error, ValueError) as exc:
    st.error(f"读取个人持仓失败：{exc}")
    ledger_summary = {
        "symbol": "QQQM", "shares": 0.0, "average_cost": 0.0,
        "cost_basis": 0.0, "reference_price": None, "market_value": None,
        "unrealized_pnl": None, "unrealized_pnl_pct": None,
        "realized_pnl_total": 0.0, "transaction_count": 0,
    }

portfolio_overview_tab, portfolio_trade_tab, portfolio_history_tab, portfolio_safety_tab = st.tabs(
    ["持仓概览", "记录买入 / 卖出 / 股息", "交易与股息记录", "备份与纠错"]
)

with portfolio_overview_tab:
    summary_cols = st.columns(5)
    summary_cols[0].metric("当前持有股数", _number(ledger_summary.get("shares"), 4))
    summary_cols[1].metric("平均持仓成本", _money(ledger_summary.get("average_cost")))
    summary_cols[2].metric("已实现盈亏（不含股息）", _money(ledger_summary.get("realized_pnl_total")))
    summary_cols[3].metric("累计净股息", _money(ledger_summary.get("dividend_income_total", 0.0)))
    if ledger_summary.get("market_value") is not None:
        summary_cols[4].metric("按最近收盘估算市值", _money(ledger_summary.get("market_value")))
    else:
        summary_cols[4].metric("估算市值", "暂无收盘价")

    detail_cols = st.columns(4)
    detail_cols[0].metric("剩余持仓成本", _money(ledger_summary.get("cost_basis")))
    if ledger_summary.get("unrealized_pnl") is not None:
        detail_cols[1].metric("未实现盈亏估算", _money(ledger_summary.get("unrealized_pnl")))
        pct = ledger_summary.get("unrealized_pnl_pct")
        detail_cols[2].metric("未实现盈亏率", f"{float(pct):+.2f}%" if pct is not None else "—")
    else:
        detail_cols[1].metric("未实现盈亏估算", "暂无收盘价")
        detail_cols[2].metric("未实现盈亏率", "—")
    detail_cols[3].metric(
        "已实现盈亏 + 净股息",
        _money(ledger_summary.get("realized_plus_dividends_total", ledger_summary.get("realized_pnl_total", 0.0))),
    )

    if int(ledger_summary.get("transaction_count", 0)) == 0:
        st.info(
            "这个账号目前还没有买卖流水。若你已经持有 QQQM，可以选择“初始化现有持仓”，输入当前剩余股数和每股成本；"
            "如果要计算历史卖出的盈亏，则不要初始化，应该从最早一笔真实买入开始，按时间顺序录入所有买卖。"
            "股息可以单独记录，不会改变持有股数。"
        )
    st.caption(
        "买入手续费计入持仓成本；卖出按移动加权平均成本计算已实现盈亏，并扣除卖出手续费。"
        "股息按参与分红的股数 × 每股股息计算毛额，再扣除预扣税与费用得到净收入。"
        "持仓估值使用最近已同步收盘价，不是实时成交价。"
    )

with portfolio_trade_tab:
    transaction_count = int(ledger_summary.get("transaction_count", 0))
    if transaction_count == 0:
        st.info(
            "第一次使用可选择一种起步方式：完整重建历史买卖，或初始化当前持仓。不要先初始化再录入更早的历史买卖，否则历史盈亏无法正确重算。"
        )
        type_options = ["初始化现有持仓", "买入", "卖出", "股息收入 / 分红到账"]
    else:
        type_options = ["买入", "卖出", "股息收入 / 分红到账"]
        st.caption("买卖记录请按成交时间从早到晚录入，以保持平均成本和已实现盈亏计算正确。股息记录可按实际派发日期补录，不会改变持仓成本。")

    # Keep the type selector outside the form so Streamlit reruns and swaps the
    # fields immediately when the user switches between trade and dividend.
    action_label = st.radio(
        "记录类型", type_options, horizontal=True, key="qqqm_portfolio_action_type"
    )
    with st.form("qqqm_portfolio_transaction_form", clear_on_submit=True):
        if action_label == "股息收入 / 分红到账":
            st.caption("请按券商分红明细填写有资格收取本次股息的股数，而不是一定填写当前持仓数量。")
            q_col, price_col = st.columns(2)
            quantity = q_col.number_input(
                "参与分红的股数", min_value=0.0, value=0.0, step=1.0, format="%.4f",
                help="以券商分红明细为准；除息日附近买卖时，它可能不同于现在持有的股数。",
            )
            price = price_col.number_input(
                "每股股息（美元）", min_value=0.0, value=0.0, step=0.001, format="%.4f",
                help="填写税前每股分红金额，不是整笔股息的总金额。",
            )
            gross_preview = round(float(quantity) * float(price) + 1e-9, 2)
            st.write(f"预计税前股息：**{_money(gross_preview)}**")
            tax_col, fee_col = st.columns(2)
            tax_withheld = tax_col.number_input("预扣税款（美元）", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            fees = fee_col.number_input("其他费用（美元）", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            st.caption(f"预计净到账：**{_money(gross_preview - tax_withheld - fees)}**（税前股息 − 预扣税 − 其他费用）")
        else:
            q_col, price_col = st.columns(2)
            quantity = q_col.number_input(
                "股数", min_value=0.0, value=0.0, step=1.0, format="%.4f",
                help="支持小数股；请按实际成交记录填写。",
            )
            if action_label == "初始化现有持仓":
                price_label = "每股现有成本（含已分摊买入手续费）"
            else:
                price_label = "成交价格（美元 / 股）"
            price = price_col.number_input(
                price_label, min_value=0.0, value=0.0, step=0.01, format="%.4f"
            )
            if action_label == "初始化现有持仓":
                fees = 0.0
                tax_withheld = 0.0
                st.caption("初始化是期初余额，不是一笔真实成交；每股成本需要已包含过去买入手续费，手续费栏固定为 0。")
            else:
                fees = st.number_input("本笔手续费（美元）", min_value=0.0, value=0.0, step=0.01, format="%.2f")
                tax_withheld = 0.0

        now_local = datetime.now(portfolio_tz)
        dt_col, tm_col = st.columns(2)
        date_label = "股息派发/到账日期" if action_label == "股息收入 / 分红到账" else "成交日期（本地时间）"
        trade_date = dt_col.date_input(date_label, value=now_local.date())
        trade_time = tm_col.time_input("时间（本地时间）", value=now_local.time().replace(second=0, microsecond=0))
        trade_notes = st.text_input("备注（选填）", max_chars=500, placeholder="例如：券商分红明细、交易单号或手续费说明")
        transaction_submitted = st.form_submit_button("保存记录", type="primary", use_container_width=True)

    if transaction_submitted:
        local_dt = datetime.combine(trade_date, trade_time).replace(tzinfo=portfolio_tz)
        try:
            if action_label == "股息收入 / 分红到账":
                saved = portfolio_ledger.record_dividend(
                    user_id=user_id,
                    symbol="QQQM",
                    eligible_shares=quantity,
                    dividend_per_share=price,
                    tax_withheld=tax_withheld,
                    fees=fees,
                    paid_at=local_dt.isoformat(),
                    notes=trade_notes,
                )
                st.success(
                    f"股息记录已保存。税前 {_money(saved['gross_amount'])}，"
                    f"预扣税 {_money(saved['tax_withheld'])}，净股息 {_money(saved['net_amount'])}。持仓股数和平均成本未改变。"
                )
            else:
                side_map = {"初始化现有持仓": "OPENING", "买入": "BUY", "卖出": "SELL"}
                saved = portfolio_ledger.record_trade(
                    user_id=user_id,
                    symbol="QQQM",
                    side=side_map[action_label],
                    quantity=quantity,
                    price=price,
                    fees=fees,
                    executed_at=local_dt.isoformat(),
                    notes=trade_notes,
                )
                labels = {"OPENING": "期初持仓已初始化", "BUY": "买入记录已保存", "SELL": "卖出记录已保存"}
                st.success(
                    f"{labels.get(saved['side'], '交易记录已保存')}。当前持有 {saved['shares_after']:.4f} 股，"
                    f"平均成本 {_money(saved['average_cost_after'])}。"
                )
            st.rerun()
        except TradeValidationError as exc:
            st.error(str(exc))
        except (OSError, sqlite3.Error) as exc:
            st.error(f"保存记录失败：{exc}")

with portfolio_history_tab:
    try:
        activities = portfolio_ledger.list_activity(user_id=user_id, symbol="QQQM", limit=1000)
    except (OSError, sqlite3.Error, TradeValidationError) as exc:
        activities = []
        st.error(f"读取交易与股息记录失败：{exc}")

    if not activities:
        st.info("当前账号还没有买卖或股息记录。请在上一栏记录第一笔交易，或录入一笔历史股息。")
    else:
        import pandas as pd

        side_labels = {"OPENING": "期初持仓", "BUY": "买入", "SELL": "卖出", "DIVIDEND": "股息收入 / 分红到账"}
        display_rows = []
        for tx in activities:
            is_dividend = tx.get("side") == "DIVIDEND"
            activity_label = side_labels.get(tx.get("side"), tx.get("side", "—"))
            if tx.get("is_voided"):
                activity_label = f"已冲正 · {activity_label}"
            display_rows.append({
                "日期（本地）": _utc_to_local(tx.get("executed_at_utc"), local_timezone_name),
                "类型": activity_label,
                "股数 / 分红股数": float(tx.get("quantity", 0.0) or 0.0),
                "成交价 / 每股股息": _money(tx.get("price")),
                "税前金额": _money(tx.get("gross_amount")),
                "预扣税": _money(tx.get("tax_withheld")) if is_dividend else "—",
                "手续费 / 其他费用": _money(tx.get("fees")),
                "净股息 / 已实现盈亏": _money(tx.get("net_amount")) if is_dividend else (_money(tx.get("realized_pnl")) if tx.get("side") == "SELL" else "—"),
                "交易后股数": "—" if is_dividend or tx.get("shares_after") is None else float(tx.get("shares_after", 0.0)),
                "交易后平均成本": "—" if is_dividend or tx.get("average_cost_after") is None else _money(tx.get("average_cost_after")),
                "冲正原因": tx.get("void_reason") or "",
                "备注": tx.get("notes", ""),
            })
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
        safe_username = "".join(ch for ch in username if ch.isalnum() or ch in "_-.")[:40] or "user"
        try:
            csv_payload = export_user_activity_csv(
                portfolio_ledger, user_id=user_id, symbol="QQQM", limit=1000
            )
            st.download_button(
                "下载我的账本 CSV",
                data=csv_payload,
                file_name=f"qqqm_ledger_{safe_username}.csv",
                mime="text/csv",
                key="qqqm_download_personal_ledger_csv",
                help="只导出当前登录账号的交易和股息记录，包括已冲正记录及原因。",
            )
        except (OSError, sqlite3.Error, TradeValidationError) as exc:
            st.error(f"导出个人账本失败：{exc}")
        st.caption(
            "买卖流水按成交时间录入；股息以税前每股金额 × 符合分红资格的股数计算，再扣除预扣税及费用。"
            "已冲正记录会保留在历史中作为审计线索，但不再计入当前持仓和汇总金额。"
        )

with portfolio_safety_tab:
    st.subheader("账本备份")
    st.warning(
        "完整数据库备份包含本机所有账号、密码哈希和个人账本，不要上传 GitHub、发给他人或放到公开网盘。"
    )
    if st.button("立即备份本地账号与账本数据库", key="qqqm_backup_local_users_db", type="primary"):
        try:
            backup_path = backup_sqlite_database(AUTH_DB_PATH)
            st.success(f"备份已完成：{backup_path}")
        except (OSError, sqlite3.Error, RuntimeError) as exc:
            st.error(f"数据库备份失败：{exc}")

    st.divider()
    st.subheader("纠正录错的流水")
    st.caption(
        "纠错不会物理删除旧记录，而是标记为‘已冲正’，保存原因，并重新计算该标的后续有效交易。"
        "这样能留下审计记录，也能避免直接改数据库造成成本和盈亏错乱。"
    )
    try:
        all_activities_for_void = portfolio_ledger.list_activity(user_id=user_id, symbol="QQQM", limit=1000)
        active_activities = [row for row in all_activities_for_void if not bool(row.get("is_voided", 0))]
    except (OSError, sqlite3.Error, TradeValidationError) as exc:
        all_activities_for_void = []
        active_activities = []
        st.error(f"读取可纠错记录失败：{exc}")

    activity_lookup = {str(row.get("activity_id")): row for row in active_activities}
    if not activity_lookup:
        st.info("当前账号没有可以冲正的有效交易或股息记录。")
    else:
        type_labels = {"OPENING": "期初持仓", "BUY": "买入", "SELL": "卖出", "DIVIDEND": "股息"}
        def _activity_option_label(activity_key: str) -> str:
            row = activity_lookup[activity_key]
            is_dividend = row.get("activity_type") == "DIVIDEND"
            date_text = _utc_to_local(row.get("executed_at_utc"), local_timezone_name)
            side_text = type_labels.get(str(row.get("activity_type") or row.get("side")), "记录")
            amount = row.get("net_amount") if is_dividend else row.get("gross_amount")
            return f"{activity_key} · {date_text} · {side_text} · {_money(amount)}"

        selected_activity_id = st.selectbox(
            "选择需要纠正的记录",
            options=list(activity_lookup),
            format_func=_activity_option_label,
            key="qqqm_void_activity_selector",
        )
        void_reason = st.text_input(
            "冲正原因（必填）",
            max_chars=500,
            placeholder="例如：成交股数录错，应为 1 股",
            key="qqqm_void_activity_reason",
        )
        st.caption(
            "如果要冲正较早的买入，导致后续卖出超过剩余持仓，系统会拒绝此次操作。"
            "请先按时间倒序冲正冲突的后续卖出，再纠正较早记录。"
        )
        if st.button(
            "确认冲正所选记录",
            key="qqqm_confirm_void_activity",
            disabled=not bool(selected_activity_id and void_reason.strip()),
        ):
            selected = activity_lookup[selected_activity_id]
            try:
                if selected.get("activity_type") == "DIVIDEND":
                    result = portfolio_ledger.void_dividend(
                        user_id=user_id,
                        dividend_id=int(selected["dividend_id"]),
                        reason=void_reason,
                    )
                    st.success(f"股息记录已冲正：D-{result['dividend_id']}。旧记录仍保留，可在历史中查看。")
                else:
                    result = portfolio_ledger.void_transaction(
                        user_id=user_id,
                        transaction_id=int(selected["transaction_id"]),
                        reason=void_reason,
                    )
                    st.success(
                        f"交易记录已冲正：T-{result['transaction_id']}。"
                        f"系统重新计算了 {result['recomputed_active_transactions']} 笔有效交易。"
                    )
                st.rerun()
            except TradeValidationError as exc:
                st.error(str(exc))
            except (OSError, sqlite3.Error) as exc:
                st.error(f"冲正失败：{exc}")

st.divider()

st.divider()
_chart_target_date = str((chart_forecast or {}).get("forecast_date_weekday_estimate") or target_date)
st.subheader(f"📊 {_chart_target_date} 盘中走势参考")
show_scenarios = st.checkbox(
    "同时显示 P10 / P50 / P90 情景参考线",
    value=False,
    help="默认只突出显示独立主预测路径。勾选后，额外显示历史分位情景供比较。",
)
if chart_forecast:
    render_intraday_chart(st, ROOT, show_scenarios=show_scenarios,
                          forecast_override=chart_forecast, summary_override={})
elif not legacy_stale:
    render_intraday_chart(st, ROOT, show_scenarios=show_scenarios)
else:
    st.warning("目前没有与最新实际行情日期对齐的新曲线。请先在视频内容页面刷新行情；旧曲线已隐藏。")

with st.expander("P10 / P50 / P90 是什么？什么时候有用？"):
    if legacy_stale:
        st.info("这份旧预测的开盘和收盘情景已过期，暂不显示具体价格，避免误认为是下一交易日的区间。")
    else:
        st.markdown(
            """
            **它们是风险情景参考，不是买卖指令，也不是三种未来概率。**

            - **P10：** 每个十五分钟时间点上，历史候选价格分布较低一端的参考值。
            - **P50：** 每个时间点历史候选价格的中位数，不代表完整走势一定会沿着它运行。
            - **P90：** 每个时间点历史候选价格分布较高一端的参考值。

            这些分位数是逐时间点分别计算的。把它们连接成线，不代表那条完整曲线曾经真实发生过，也不保证未来落在这些线之间。
            """
        )
        st.subheader("开盘情景参考")
        o1, o2, o3 = st.columns(3)
        o1.metric("偏低参考 · P10", _money(open_data.get("p10_price")))
        o2.metric("中位参考 · P50", _money(open_data.get("p50_price")))
        o3.metric("偏高参考 · P90", _money(open_data.get("p90_price")))
        st.subheader("收盘情景参考")
        c1, c2, c3 = st.columns(3)
        c1.metric("偏低参考 · P10", _money(close_data.get("p10_price")))
        c2.metric("中位参考 · P50", _money(close_data.get("p50_price")))
        c3.metric("偏高参考 · P90", _money(close_data.get("p90_price")))

if main_path_obj:
    with st.expander("查看主预测是怎么生成的（进阶说明）"):
        st.write("模型名称：", main_model_name)
        st.write("用于训练的历史记录数：", main_path_obj.get("training_rows", "—"))
        st.write("是否独立于 P10/P50/P90：", "是" if main_path_obj.get("independent_of_quantile_curves") else "否，可能使用旧方法回退")
        if main_path_obj.get("news_adjustment_note"):
            st.write(str(main_path_obj["news_adjustment_note"]))

st.divider()
st.subheader("📰 新闻有没有改变这次预测？")
if legacy_stale:
    st.caption("以下新闻记录、候选模型诊断和历史回测摘要来自旧预测快照，仅作为历史研究，不代表刚生成的新情景。")
path_diagnostics = forecast.get("path_selection_diagnostics") or {}
news = path_diagnostics.get("news_signal") or summary.get("news_signal") or {}
forecast_news_context = forecast.get("news_context") or {}
news_count = forecast_news_context.get("included_article_count", summary.get("news_article_count_at_forecast_cutoff"))
qualified_news = news.get("qualified_directional_articles", 0)
news_applied = bool(path_diagnostics.get("news_adjustment_applied", summary.get("news_adjustment_applied", False)))
news_signal = news.get("signal", 0)

n1, n2, n3 = st.columns(3)
n1.metric("预测时系统看到的新闻", str(news_count if news_count is not None else "—"))
n2.metric("方向较明确的新闻", str(qualified_news))
n3.metric("新闻调整这次主预测了吗？", "有（实验）" if news_applied else "没有")
if not news_applied:
    st.info(
        "这次没有足够可靠、方向明确的新闻，因此系统没有因为新闻去改变主预测线。"
        "这比硬把新闻解释成利好或利空更稳妥。"
    )
else:
    st.warning(
        "这次主预测线做了小幅新闻实验调整，但还没有足够的历史实测结果证明这样会更准。"
    )
with st.expander("查看新闻判断的技术细节"):
    st.write(f"方向性新闻计数：{qualified_news}")
    st.write(f"实验信号数值：{_number(news_signal, 2)}")
    st.write("提醒：这个信号不是上涨概率，也不是已经验证的因果关系。")

btest = forecast.get("backtest_summary") or {}
if not btest:
    btest = (forecast.get("backtest") or {}) if isinstance(forecast.get("backtest"), dict) else {}

st.subheader("🧪 过去的预测表现怎么样？")

# 用普通语言解释模型历史验证，不让缩写成为页面主角。
test_days = btest.get("walk_forward_test_sessions")
ridge_mae = btest.get("ridge_ml_path_mae_bps")
cluster_mae = btest.get("path_mae_bps_conditional_neighbor")
flat_mae = btest.get("path_mae_bps_flat_after_open_baseline")
ridge_open_mae = btest.get("ridge_ml_opening_mae_bps")

valid_a, valid_b, valid_c, valid_d = st.columns(4)
valid_a.metric("过去模拟检查了多少个交易日", str(test_days if test_days is not None else "—"))
valid_b.metric("主预测线平均偏差", f"{float(ridge_mae) / 100:.2f}%" if ridge_mae is not None else "暂时没有数据")
valid_c.metric("简单参考方法平均偏差", f"{float(flat_mae) / 100:.2f}%" if flat_mae is not None else "暂时没有数据")
valid_d.metric("主预测开盘价平均偏差", f"{float(ridge_open_mae) / 100:.2f}%" if ridge_open_mae is not None else "暂时没有数据")

if ridge_mae is not None and flat_mae is not None:
    delta_pct = (float(flat_mae) - float(ridge_mae)) / float(flat_mae) * 100 if float(flat_mae) else None
    if delta_pct is not None and delta_pct > 0:
        st.success(f"在这组历史模拟中，主预测线的平均偏差比简单参考方法低 {delta_pct:.1f}%。这只是当前样本的结果，不保证未来继续如此。")
    elif delta_pct is not None and delta_pct < 0:
        st.warning(f"在这组历史模拟中，主预测线的平均偏差比简单参考方法高 {abs(delta_pct):.1f}%。目前还看不出复杂模型有优势。")
    else:
        st.info("在这组历史模拟中，主预测线和简单参考方法的平均偏差相同。")

if test_days is not None:
    if int(test_days) < 30:
        st.warning("目前用于历史模拟检查的交易日较少，结果容易受个别行情影响，暂时不能判断模型是否稳定。")
    else:
        st.caption("这里是过去交易日的模拟检验，不是未来准确率。测试数量继续增加、并覆盖不同市场环境后，结论才会更有参考价值。")

with st.expander("“过去模拟检查”是什么意思？（新手版）"):
    st.markdown(
        """
        可以把它理解成**用过去的交易日做模拟考试**：

        1. 先选一个过去的交易日，只让模型看到那一天之前已经知道的资料。
        2. 让模型预测接下来的开盘和盘中走势。
        3. 再把预测与后来真实发生的价格比较，计算平均偏差。
        4. 把这个过程在多个交易日重复进行，观察模型有没有持续表现得更好。

        **“平均偏差”不是准确率。** 它表示预测价格平均离真实价格有多远，百分比越小通常越好。简单参考方法例如“开盘之后假设价格基本不变”，有时候也会胜过复杂模型。

        这种检查能减少“偷看未来数据”的问题，但不能保证模型未来继续有效。当前历史数据不多，结果只能作为观察依据。
        """
    )
    detail_a, detail_b, detail_c = st.columns(3)
    detail_a.metric("相似历史走势方法的平均偏差", f"{float(cluster_mae) / 100:.2f}%" if cluster_mae is not None else "暂时没有数据")
    detail_b.metric("简单参考方法的平均偏差", f"{float(flat_mae) / 100:.2f}%" if flat_mae is not None else "暂时没有数据")
    detail_c.metric("主预测线的平均偏差", f"{float(ridge_mae) / 100:.2f}%" if ridge_mae is not None else "暂时没有数据")
    if btest.get("first_test_date") or btest.get("last_test_date"):
        st.caption(f"这次模拟覆盖的日期：{btest.get('first_test_date', '—')} 至 {btest.get('last_test_date', '—')}。")

daily_review_ctx = build_daily_review_context(forecast, summary)
matched_summary = daily_review_ctx["matched_summary"]
actual_points = daily_review_ctx["actual_points_found"]
main_path_mae = matched_summary.get("main_path_curve_mae_bps_vs_actual")
p50_path_mae = matched_summary.get("p50_curve_mae_bps_vs_actual")
if int(actual_points or 0) > 0:
    st.markdown("**最近一次预测与真实走势的对照**")
    a1, a2, a3 = st.columns(3)
    a1.metric("已匹配的真实盘中价格点", str(actual_points))
    a2.metric("主预测线与实际价格的平均偏差", f"{float(main_path_mae) / 100:.2f}%" if main_path_mae is not None else "等待完整结算")
    a3.metric("中位参考线与实际价格的平均偏差", f"{float(p50_path_mae) / 100:.2f}%" if p50_path_mae is not None else "等待完整结算")
    st.caption("只有预测日期与实际数据日期相同，以上价格和误差才会对应起来；数据未完整同步时，不宜急着判断模型好坏。")
else:
    st.info(daily_review_ctx["explanation"])

with st.container(border=True):
    st.subheader("📝 每日预测复盘")
    st.caption("先看实际结果，再记录你的观察。人工反馈会单独保存，不会自动加入机器学习训练。")
    st.info(daily_review_ctx["explanation"])

    if daily_review_ctx["summary_matches_target"] and actual_points > 0:
        point_ratio = actual_points / max(1, daily_review_ctx["expected_points"])
        review_a, review_b, review_c, review_d = st.columns(4)
        review_a.metric("目标交易日", daily_review_ctx["target_date"] or "—")
        review_b.metric("实际价格点", f'{actual_points}/{daily_review_ctx["expected_points"]}')
        review_c.metric("实际开盘价", _money(daily_review_ctx["actual_open_price"]))
        review_d.metric("最后可用收盘价", _money(daily_review_ctx["actual_close_price"]))

        price_a, price_b = st.columns(2)
        open_err = daily_review_ctx["open_error_usd_actual_minus_predicted"]
        close_err = daily_review_ctx["close_error_usd_actual_minus_predicted"]
        with price_a:
            st.markdown("**开盘：预测 vs 实际**")
            st.write(f'预测：{_money(daily_review_ctx["predicted_open_price"])}')
            st.write(f'实际：{_money(daily_review_ctx["actual_open_price"])}')
            if open_err is not None:
                st.write(f'实际比预测：{open_err:+.2f} 美元')
        with price_b:
            st.markdown("**收盘：预测 vs 实际**")
            st.write(f'预测：{_money(daily_review_ctx["predicted_close_price"])}')
            st.write(f'最后可用实际价：{_money(daily_review_ctx["actual_close_price"])}')
            if close_err is not None:
                st.write(f'实际比预测：{close_err:+.2f} 美元')

        if point_ratio < 0.90:
            st.warning("这次还有部分 15 分钟价格点没匹配到。先把它当作初步复盘，数据同步完整后再比较模型。")
        if daily_review_ctx["main_path_mae_pct"] is not None:
            st.metric("这次主预测线的平均偏差", f'{daily_review_ctx["main_path_mae_pct"]:.2f}%')
            st.caption("这个百分比描述预测线和真实盘中价格平均相差多少，不是准确率，也不能单独代表长期水平。")
    elif daily_review_ctx["previous_completed_summary"]:
        previous = daily_review_ctx["previous_completed_summary"]
        st.caption(
            f'最近一次已有真实结果的交易日是 {previous["target_date"]}，'
            f'匹配到 {previous["actual_points_found"]} 个价格点。它属于旧预测，不能拿来评价当前目标日。'
        )

    with st.expander("记录我的观察（会保存到本地，方便以后改进）", expanded=True):
        with st.form(key=f'human_review_{daily_review_ctx["target_date"] or "unknown"}'):
            overall_match = st.selectbox(
                "看完实际走势后，你觉得这次主预测整体怎么样？",
                ["还没有实际结果，暂不判断", "走势比较接近", "有些地方接近", "走势差异较大"],
                index=0 if actual_points <= 0 else 1,
            )
            biggest_miss = st.selectbox(
                "你觉得最值得改进的部分是？",
                ["暂时看不出来", "开盘价偏差", "上午走势", "午后走势", "收盘方向或价格", "全天波动幅度", "新闻影响判断", "数据缺失或时间对不齐"],
            )
            news_helpfulness = st.selectbox(
                "这次新闻信息对理解预测有没有帮助？",
                ["暂时无法判断", "有帮助", "帮助不明显", "可能造成误导", "当时没有足够明确的新闻"],
            )
            notes = st.text_area(
                "补充观察（选填）",
                placeholder="例如：开盘比预测低，但午后反弹幅度没有被主路径体现。",
                max_chars=4000,
            )
            submitted = st.form_submit_button("保存本次复盘意见", type="primary")

        if submitted:
            try:
                current_user = st.session_state.get(AUTH_SESSION_KEY, {})
                feedback_path = save_human_feedback(
                    ROOT,
                    daily_review_ctx,
                    {
                        "overall_match": overall_match,
                        "biggest_miss": biggest_miss,
                        "news_helpfulness": news_helpfulness,
                        "notes": notes,
                    },
                    user_id=current_user.get("user_id"),
                )
                st.success(f"已保存到 {feedback_path.relative_to(ROOT)}。这条意见目前只作为人工复盘资料，不会自动训练模型。")
            except OSError as exc:
                st.error(f"保存失败：{exc}")

    current_user = st.session_state.get(AUTH_SESSION_KEY, {})
    recent_feedback = load_recent_human_feedback(ROOT, limit=5, user_id=current_user.get("user_id"))
    if recent_feedback:
        with st.expander("查看最近保存的复盘意见"):
            for item in recent_feedback:
                st.markdown(f'**{item.get("target_date") or "日期未知"} · {item.get("user_overall_match") or "未填写"}**')
                st.caption(
                    f'重点：{item.get("user_biggest_miss") or "未填写"}；'
                    f'新闻帮助：{item.get("user_news_helpfulness") or "未填写"}'
                )
                if item.get("user_notes"):
                    st.write(item["user_notes"])
                st.divider()

with st.expander("查看数据来源、模型细节和注意事项（进阶）"):
    generated_at = forecast.get("generated_at_utc")
    st.write("预测生成时间（香港时间）：", _utc_to_local(generated_at, "Asia/Hong_Kong"))
    st.write("数据源：", f"{forecast.get('feed', '—')} / {forecast.get('timeframe', '—')}")
    st.write("相似历史交易日数量：", forecast.get("similar_session_count", "—"))
    st.write("主路径模型名称（技术字段）：", main_model_name)
    st.write("主路径训练记录数（技术字段）：", main_path_obj.get("training_rows", "—"))
    st.write("主路径是否独立于 P10/P50/P90：", "是" if main_path_obj.get("independent_of_quantile_curves") else "否，可能使用旧方法回退")
    st.write("当前历史训练是否把新闻当作训练输入：没有。因为还缺少足够完整的历史记录，无法确认每条新闻当时是否已经发布并完成分类。当前新闻只在条件满足时做可检查的实验性调整。")
    caveats = forecast.get("caveats", [])
    if caveats:
        st.markdown("**模型局限性**")
        for caveat in caveats:
            st.write(f"- {caveat}")

if CONTEXT_REPORT_PATH.exists():
    with st.expander("查看最新 AI 市场上下文报告"):
        try:
            report_text = CONTEXT_REPORT_PATH.read_text(encoding="utf-8-sig")
            st.markdown(report_text)
        except OSError as exc:
            st.warning(f"读取 AI 报告失败：{exc}")

st.caption("本页面仅用于研究和复盘，不构成投资建议，也不会自动下单。")
