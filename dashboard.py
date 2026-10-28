from __future__ import annotations

"""Unified Streamlit entry point for QQQM Investment Assistant.

All existing dashboards and analysis pages are presented in one application
with a single navigation and authentication gate. Forecasting is never run just
by opening the app; users start expensive/syncing workflows explicitly.
"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.auth import AuthError, AuthService  # noqa: E402

st.set_page_config(
    page_title="MarketPilot AI · 美股多资产研究台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

AUTH_DB = ROOT / "data" / "users.db"
AUTH_KEY = "qqqm_authenticated_user"


@st.cache_resource(show_spinner=False)
def get_auth_service(path: str) -> AuthService:
    return AuthService(path)


def render_login_gate() -> bool:
    """Gate every page, instead of protecting only the account page."""
    try:
        auth = get_auth_service(str(AUTH_DB))
    except Exception as exc:
        st.error(f"账号数据库暂时无法打开：{type(exc).__name__}: {exc}")
        st.stop()

    principal = st.session_state.get(AUTH_KEY)
    if isinstance(principal, dict):
        user = auth.get_user_by_id(principal.get("user_id"))
        if user:
            st.session_state[AUTH_KEY] = {
                "user_id": user["user_id"],
                "username": user["username"],
            }
            with st.sidebar:
                st.caption(f"当前本地账号：{user['username']}")
                if st.button("退出登录", key="global_logout_button", use_container_width=True):
                    st.session_state.pop(AUTH_KEY, None)
                    st.rerun()
            return True
        st.session_state.pop(AUTH_KEY, None)

    st.title("📈 MarketPilot AI · 美股多资产研究台")
    st.caption("MarketPilot AI · Multi-Asset US Market Research, Analysis and Review")
    st.info("这是本机研究版。账号保存在本地 SQLite 数据库中；不是公开网站的生产级身份系统。")
    login_tab, register_tab = st.tabs(["登录", "创建本地账号"])
    with login_tab:
        with st.form("unified_login_form", clear_on_submit=True):
            username = st.text_input("用户名", max_chars=32)
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", type="primary", use_container_width=True)
        if submitted:
            user = auth.authenticate(username, password)
            if not user:
                st.error("用户名或密码不正确，或账号暂不可用。")
            else:
                st.session_state[AUTH_KEY] = {"user_id": user["user_id"], "username": user["username"]}
                st.rerun()
    with register_tab:
        st.caption("用户名 3–32 位，只能使用英文字母、数字、点、下划线或连字符；密码至少 12 位。")
        with st.form("unified_register_form", clear_on_submit=True):
            new_username = st.text_input("新用户名", max_chars=32)
            new_password = st.text_input("设置密码", type="password")
            confirm = st.text_input("再次输入密码", type="password")
            submitted = st.form_submit_button("创建账号并登录", type="primary", use_container_width=True)
        if submitted:
            if new_password != confirm:
                st.error("两次输入的密码不一致。")
            else:
                try:
                    user = auth.register(new_username, new_password)
                except AuthError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[AUTH_KEY] = {"user_id": user["user_id"], "username": user["username"]}
                    st.rerun()
    st.warning("所有页面都经过同一登录入口。若要公开给多人使用，需先迁移到生产级数据库与会话/限流方案，不能直接把本地 SQLite 版开放到公网。")
    return False


if not render_login_gate():
    st.stop()

# The distributable excludes local SQLite files. Bootstrap the bundled historical
# QQQM CSV locally (no network call) so legacy pages never open with an empty DB.
try:
    from app.services.local_data_bootstrap import ensure_bundled_qqqm_history
    st.session_state["_local_market_bootstrap_status"] = ensure_bundled_qqqm_history()
except Exception as exc:
    # Keep the app usable even if local seeding fails; the home page reports the issue.
    st.session_state["_local_market_bootstrap_error"] = f"{type(exc).__name__}: {exc}"

st.session_state["_qqqm_unified_navigation"] = True

# Keep the primary navigation intentionally small. Experimental/legacy tools remain
# in the source tree for compatibility, but are not exposed in the demo navigation.
st.markdown("""
<style>
[data-testid="stSidebar"] {background:linear-gradient(180deg, rgba(30,58,138,.045), transparent 44%);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {font-size:1rem;letter-spacing:-.025em;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {font-size:.76rem;line-height:1.5;}
[data-testid="stSidebarNav"] {padding-top:.35rem;}
[data-testid="stSidebarNav"] a {border-radius:10px;margin:3px 0;}
[data-testid="stSidebarNav"] a[aria-current="page"] {background:rgba(37,99,235,.12);font-weight:700;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📈 MarketPilot AI")
    st.caption("美股多资产研究台")
    st.divider()

pages = {
    "核心工作台": [
        st.Page("app/pages/research_home.py", title="研究驾驶舱", icon="🏠", url_path="research-home", default=True),
        st.Page("app/dashboard.py", title="市场与持仓", icon="📊", url_path="market-analysis"),
        st.Page("app/pages/next_day_forecast.py", title="预测情景", icon="🔭", url_path="next-day-forecast"),
        st.Page("app/pages/model_learning_review.py", title="预测复盘", icon="🧠", url_path="model-learning-review"),
    ],
    "多资产研究与内容工厂": [
        st.Page("app/pages/multi_asset_research.py", title="多资产研究与复盘", icon="🧭", url_path="multi-asset-research"),
        st.Page("app/pages/video_script_studio.py", title="多平台视频脚本与配图", icon="🎬", url_path="video-script-studio"),
    ],
    "账本与系统诊断": [
        st.Page(
            "app/pages/portfolio_planner.py",
            title="个人投资组合与交易账本",
            icon="🧾",
            url_path="portfolio-planner",
        ),
        st.Page(
            "app/pages/system_status.py",
            title="数据更新与系统诊断",
            icon="🩺",
            url_path="system-status",
        ),
    ],
}

page = st.navigation(pages, position="sidebar", expanded=True)
page.run()
