from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.services.system_health import inspect_project, format_time  # noqa: E402
DB = ROOT / "data" / "market_data.db"
VIDEO_FORECAST = ROOT / "reports" / "intraday" / "QQQM_video_forecast_latest.json"
LEGACY_FORECAST = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
STATUS = ROOT / "reports" / "intraday_learning_cycle" / "latest_status.json"
WORKFLOW_STATUS = ROOT / "reports" / "unified_daily_workflow" / "latest_status.json"
OUT = ROOT / "reports" / "video_scripts"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def latest_market_date() -> str | None:
    try:
        with sqlite3.connect(DB) as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
        return str(row[0])[:10] if row and row[0] else None
    except (sqlite3.Error, OSError):
        return None


def latest_intraday_date() -> str | None:
    try:
        with sqlite3.connect(DB) as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='intraday_bars'").fetchone()
            if not exists:
                return None
            rows = conn.execute("""SELECT trade_date_ny, LOWER(feed) AS feed,
                COUNT(DISTINCT session_time_ny), MIN(session_time_ny), MAX(session_time_ny)
                FROM intraday_bars
                WHERE UPPER(symbol)='QQQM' AND LOWER(timeframe)='15min'
                AND session_time_ny >= '09:30' AND session_time_ny < '16:00'
                GROUP BY trade_date_ny, LOWER(feed) ORDER BY trade_date_ny DESC""").fetchall()
        # A full session must exist within one provider; never stitch one provider's
        # opening bar to another provider's closing bar just to pass the completeness check.
        completed = [str(day)[:10] for day, feed, count, first_time, last_time in rows
                     if int(count or 0) >= 24 and str(first_time) == '09:30' and str(last_time) >= '15:45']
        return max(completed) if completed else None
    except (sqlite3.Error, OSError):
        return None


def run(command: list[str], timeout: int = 900) -> dict[str, Any]:
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
        return {"returncode": completed.returncode, "stdout": completed.stdout or "", "stderr": completed.stderr or ""}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def parse_last_json(text: str) -> dict[str, Any] | None:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None
    return None


st.title("🎬 多平台视频脚本与配图")
st.caption("一键刷新公开行情，结算同源历史预测，训练并滚动验证盘中岭回归候选，再生成下一交易日曲线、30–60 秒中文口播稿和四个平台发布文案。打开页面不会自动联网。模型不会自动晋级。")
st.info("所有生成结果都会自动保存在项目内 reports\video_scripts\日期 文件夹，不依赖你把下载文件放在哪里。用于后续学习的是项目归档的预测 JSON + 后续真实行情数据；单独下载或移动 PNG 不会自动进入训练集。")
st.code(str(OUT), language=None)

health = inspect_project(ROOT)
market_date = latest_market_date()
intraday_date = latest_intraday_date()
expected_date = health.get("expected_last_completed_session")
news_health = health.get("news") or {}
video_health = health.get("video") or {}
hstatus = st.columns(4)
hstatus[0].metric("本地日线最新日", market_date or "缺失")
hstatus[1].metric("完整十五分钟分时", intraday_date or "缺失")
hstatus[2].metric("最新新闻发布日期", news_health.get("latest_published_date") or "缺失")
hstatus[3].metric("最近生成的视频包", video_health.get("folder") or "尚未生成")
if expected_date and not market_date:
    st.warning("本地没有 QQQM 日线行情记录，不能生成新的日线预测。请先尝试刷新公开行情。")
elif expected_date and market_date and market_date < expected_date:
    st.warning(f"本地日线截至 {market_date}，按{health.get('calendar_source', '交易日历')}最近完整交易日应为 {expected_date}。打开页面不会自动更新，请使用下方刷新按钮。")
if expected_date and not intraday_date:
    st.warning("目前没有通过完整度校验的十五分钟交易日；第一张真实分时复盘图和下一交易日盘中情景暂不可用。")
elif expected_date and intraday_date and intraday_date < expected_date:
    st.warning(f"完整十五分钟分时截至 {intraday_date}，落后于最近完整交易日 {expected_date}；新鲜分时未补齐之前，下一交易日盘中情景不可用。")
if news_health.get("latest_published_date"):
    st.caption(f"新闻实际发布时间最新到 {news_health.get('latest_published_date')}；最近抓取时间 {format_time(news_health.get('latest_fetched_at'))}。抓取时间新，不代表来源一定有新新闻。")
st.info("本页一键流程会先执行公开行情机器学习闭环：用同一数据源结算历史前瞻预测、重新拟合盘中岭回归候选并做滚动回测，再生成视频。数据或样本不足时，可回退生成明确标注的历史统计情景，但不会冒充机器学习输出，也不会自动晋级模型。原有 Alpaca 流程仍作为独立高级功能保留。")
last_pipeline_summary = st.session_state.get("last_public_pipeline_summary")
if isinstance(last_pipeline_summary, dict):
    learning_success = bool(last_pipeline_summary.get("learning_success"))
    video_success = bool(last_pipeline_summary.get("video_success"))
    if learning_success:
        st.success(f"最近一次更新：盘中候选训练与滚动回测完成，数据基准 {last_pipeline_summary.get('as_of') or '待确认'}，目标 {last_pipeline_summary.get('target') or '待确认'}。模型仍处于候选状态。")
    else:
        st.warning("最近一次更新：公开行情机器学习闭环没有完成。" + str(last_pipeline_summary.get("learning_error") or "请检查公开行情源和完整分时样本。"))
    if video_success:
        kind = last_pipeline_summary.get("source_kind")
        if kind == "trained_intraday_candidate":
            st.caption("对应视频包已生成，图二使用本轮机器学习候选主路径。")
        elif last_pipeline_summary.get("forecast_available"):
            st.caption("对应视频包已生成，但图二使用的是历史统计情景参考，不是本轮机器学习主路径。")
        else:
            st.caption("对应视频包已生成，但没有可确认有效的新预测曲线。")
    else:
        st.caption("最近一次视频内容包没有生成成功，详情见本页下方运行日志或重新执行。")
video_forecast = read_json(VIDEO_FORECAST)
legacy_forecast = read_json(LEGACY_FORECAST)
forecast = video_forecast if video_forecast is not None else legacy_forecast
forecast_asof = str((forecast or {}).get("as_of_date") or "")
forecast_target = str((forecast or {}).get("forecast_date_weekday_estimate") or "")
forecast_available_flag = bool((forecast or {}).get("available", True))
latest_known = max([d for d in (market_date, intraday_date) if d], default="")
forecast_fresh = bool(
    forecast and forecast_available_flag and forecast_target and latest_known
    and forecast_target > latest_known and forecast_asof and forecast_asof >= latest_known
)

curve_kind = str((forecast or {}).get("video_source_kind") or "")
if not forecast_fresh:
    curve_status = "待刷新/不可用"
elif curve_kind == "trained_intraday_candidate":
    curve_status = "机器学习候选"
elif curve_kind == "historical_profile_only":
    curve_status = "历史统计情景"
else:
    curve_status = "已保存曲线，来源待核验"

cols = st.columns(4)
cols[0].metric("日线最新日期", market_date or "缺失")
cols[1].metric("完整分时最新日期", intraday_date or "缺失")
cols[2].metric("预测基准日期", forecast_asof or "缺失")
cols[3].metric("下一交易日曲线类型", curve_status)

if forecast_fresh:
    st.info(f"当前目标日为 {forecast_target}。曲线类型：{curve_status}。生成时仍会检查分时日期、目标日和数据来源是否一致。")
else:
    st.warning("目前没有可确认有效的未来走势曲线。旧的十月十六日快照不会被当作下一交易日预测；请先刷新公开行情。")
if forecast and forecast.get("available") is False and forecast.get("reason_zh"):
    st.info(str(forecast["reason_zh"]))

st.subheader("第一步：更新行情、复盘历史预测、训练候选模型并生成视频")
st.write("主要流程不需要 Alpaca 密钥。它会尝试用公开行情源刷新日线和十五分钟分时、按原预测的数据源结算历史档案、拟合盘中 Ridge 候选并执行滚动回测。只有数据日期与模型快照一致时，第二张图才会标记机器学习候选主路径。")
if st.button("🔄 更新行情 → 复盘历史预测 → 训练盘中候选 → 生成曲线和视频包", type="primary", use_container_width=True):
    with st.spinner("第 1/2 步：刷新公开行情与新闻、结算历史预测、训练盘中候选并运行滚动回测……"):
        learn_result = run([sys.executable, "scripts/run_public_intraday_learning_cycle.py"], timeout=3600)
    st.session_state["public_intraday_learning_result"] = learn_result
    learn_payload = parse_last_json(learn_result.get("stdout", ""))
    learning_ok = bool(learn_result.get("returncode") == 0 and learn_payload and learn_payload.get("overall_success"))
    if learning_ok:
        st.success("公开行情盘中学习闭环完成。候选模型已重新拟合并完成滚动回测，但仍处于研究候选状态，不会自动晋级。")
        metric_cols = st.columns(4)
        metric_cols[0].metric("完整分时最新日", learn_payload.get("latest_complete_intraday_date") or "缺失")
        metric_cols[1].metric("训练交易日", str(learn_payload.get("completed_training_sessions") or 0))
        metric_cols[2].metric("候选模型", learn_payload.get("model_family") or "未标记")
        metric_cols[3].metric("预测目标日", learn_payload.get("forecast_target_date") or "未生成")
        backtest = learn_payload.get("walk_forward_backtest") or {}
        with st.expander("查看盘中滚动回测成绩", expanded=True):
            bcols = st.columns(3)
            bcols[0].metric("测试交易日", str(backtest.get("ridge_ml_path_test_sessions", 0)))
            bcols[1].metric("岭回归路径 MAE（基点）", str(backtest.get("ridge_ml_path_mae_bps") if backtest.get("ridge_ml_path_mae_bps") is not None else "暂无"))
            bcols[2].metric("相对历史统计基准改善", f"{backtest.get('ridge_ml_improvement_pct_vs_historical_profile')}%" if backtest.get("ridge_ml_improvement_pct_vs_historical_profile") is not None else "暂无")
            st.caption("MAE 越低越好。只有足够的样本外证据证明候选持续胜过基准，才考虑晋级；本次重新拟合本身不等于模型变准。")
        if learn_payload.get("refresh_errors"):
            with st.expander("公开行情刷新中的非致命提示"):
                for item in learn_payload.get("refresh_errors", []):
                    st.write("• " + str(item))
    else:
        st.warning("盘中机器学习候选没有得到完整确认。现在继续尝试用最新可用的本地缓存生成视频包；若数据过期或样本不足，第二张图会明确标注历史统计参考或不可用，不会伪装成机器学习预测。")
        detail = (learn_payload or {}).get("error") or (learn_result.get("stderr") or learn_result.get("stdout") or "没有返回详细错误")
        st.code(str(detail)[-4500:])
        with st.expander("查看公开行情学习流程完整输出"):
            st.code((learn_result.get("stdout", "") + "\n" + learn_result.get("stderr", ""))[-9000:])

    with st.spinner("第 2/2 步：使用本轮已更新的数据与模型快照生成两张图、精简口播稿和发布文案……"):
        video_result = run([sys.executable, "scripts/generate_video_script.py"], timeout=1200)
    video_payload = parse_last_json(video_result.get("stdout", ""))
    if video_result.get("returncode") == 0 and video_payload:
        source_kind = video_payload.get("forecast_source_kind")
        if video_payload.get("next_forecast_available") and source_kind == "trained_intraday_candidate":
            st.success("视频包已生成，第二张图使用与当前数据日期一致的盘中机器学习候选主路径，并叠加历史波动参考。候选仍未自动晋级。")
        elif video_payload.get("next_forecast_available"):
            st.warning("视频包已生成，但第二张图当前是历史统计情景参考，不是机器学习主路径。可用于展示不确定性，不能宣称模型完成了本次预测。")
        else:
            st.warning("视频内容包已生成，但目前没有有效的下一交易日曲线。请不要把图二当作新预测发布。")
        status_cols = st.columns(4)
        status_cols[0].metric("日线最新日", video_payload.get("daily_latest_date") or "暂不可确认")
        status_cols[1].metric("完整分时最新日", video_payload.get("last_completed_intraday_date") or "暂不可确认")
        status_cols[2].metric("曲线目标日", video_payload.get("forecast_target") or "不可用")
        status_cols[3].metric("预测来源", "机器学习候选" if source_kind == "trained_intraday_candidate" else "历史统计/暂不可用")
        st.caption(f"预测基准日：{video_payload.get('forecast_as_of_date') or '缺失'}；数据源：{video_payload.get('forecast_feed') or '未确认'}；模型：{video_payload.get('forecast_model_family') or '未使用机器学习主路径'}。")
        if video_payload.get("content_dir"):
            st.success("本次完整内容包已保存在项目目录：" + str(video_payload["content_dir"]))
        st.caption("口播 TXT 现在同时包含推荐标题、备用标题、四个平台标题、视频简介和可直接照念的正文。")
        if video_payload.get("refresh_errors"):
            with st.expander("查看视频流程数据状态与警告"):
                for item in video_payload.get("refresh_errors", []):
                    st.write("• " + str(item))
    else:
        st.error("视频内容包生成失败，请展开日志检查具体步骤。")
        st.code((video_result.get("stderr") or video_result.get("stdout") or "没有日志")[-6000:])
    st.session_state["last_public_pipeline_summary"] = {
        "learning_success": learning_ok,
        "learning_error": ((learn_payload or {}).get("error") or (learn_result.get("stderr") or learn_result.get("stdout") or ""))[-700:],
        "as_of": (learn_payload or {}).get("forecast_as_of_date"),
        "target": (learn_payload or {}).get("forecast_target_date"),
        "video_success": bool(video_result.get("returncode") == 0 and video_payload),
        "forecast_available": bool((video_payload or {}).get("next_forecast_available")),
        "source_kind": (video_payload or {}).get("forecast_source_kind"),
    }
    # Refresh top-line dates/diagnostics immediately after data writes, while keeping
    # the concise outcome in session state so a Streamlit rerun does not erase it.
    st.rerun()

with st.expander("只用本地缓存生成内容包（不联网刷新）", expanded=False):
    st.write("适合网络不可用时检查版式。若本地分时数据落后于日线，程序会停止生成新的未来曲线，并在图中写明原因。")
    if st.button("生成本地缓存版本", use_container_width=True):
        with st.spinner("正在整理本地数据并输出脚本与配图……"):
            result = run([sys.executable, "scripts/generate_video_script.py"], timeout=600)
        payload = parse_last_json(result["stdout"])
        if result["returncode"] == 0 and payload:
            st.success("本地版本已生成。请检查内容包中的数据日期与状态说明。")
        else:
            st.error("本地版本生成失败。")
            st.code((result["stderr"] or result["stdout"])[-3500:])

st.divider()
st.subheader("第二步：预览与下载")
latest_zips = sorted(OUT.glob("*/每日行情视频脚本与两张配图.zip"), key=lambda path: path.stat().st_mtime, reverse=True) if OUT.exists() else []
if latest_zips:
    archive = latest_zips[0]
    folder = archive.parent
    st.caption(f"最新内容包生成目录：{folder.name}")
    markdown_path = folder / "每日行情视频口播稿与发布文案.md"
    text_path = folder / "每日视频口播稿_可直接照念.txt"
    image1 = folder / "01_上一完整交易日真实走势与预测对比.png"
    image2 = folder / "02_下一交易日走势情景参考.png"
    c1, c2 = st.columns(2)
    if image1.exists():
        c1.image(str(image1), caption="图一：完整分时走势与预测参考对照", use_container_width=True)
    if image2.exists():
        c2.image(str(image2), caption="图二：下一交易日走势情景参考", use_container_width=True)
    if text_path.exists():
        voiceover = text_path.read_text(encoding="utf-8")
        with st.expander("预览口播稿（可直接照念）", expanded=True):
            st.text(voiceover)
        st.download_button("下载纯口播稿文本", data=voiceover, file_name=text_path.name, mime="text/plain", use_container_width=True)
    if markdown_path.exists():
        content = markdown_path.read_text(encoding="utf-8")
        with st.expander("预览发布文案与制作说明", expanded=False):
            st.markdown(content)
        st.download_button("下载完整文案与制作说明", data=content, file_name=markdown_path.name, mime="text/markdown", use_container_width=True)
    st.download_button("下载完整视频内容包（脚本、两张图与数据记录）", data=archive.read_bytes(), file_name=archive.name, mime="application/zip", use_container_width=True)
else:
    st.info("还没有生成视频内容包。点击上方刷新按钮即可开始。")

with st.expander("高级功能：模型复盘与原有盘中学习闭环", expanded=False):
    st.markdown("**先分清三件事，避免把生成视频当成训练模型：**")
    st.write("① 视频流程：刷新可用行情、用历史分时统计生成情景、生成口播稿和图片；不训练模型。")
    st.write("② 日线复盘：结算之前保存的下一交易日预测，并使用当前本地日线重新拟合/生成日线候选预测；不需要 Alpaca，但必须先把日线更新到最近完整交易日。")
    st.write("③ 盘中闭环：同步十五分钟分时、结算保存的盘中预测、生成学习数据集并运行候选盘中模型；第一步依赖有效的 Alpaca Key/Secret。仅仅保存复盘或点击视频生成，不会自动让模型从错误中学习。")

    credentials = (inspect_project(ROOT).get("credentials") or {})
    alpaca_ready = bool(credentials.get("alpaca_configured"))
    if alpaca_ready:
        st.success("已检测到 Alpaca Key 与 Secret 均已配置。此项只确认配置存在，不代表接口认证或分时权限一定有效。")
    else:
        st.warning("当前项目没有配置完整的 Alpaca 凭据，所以你刚才的盘中闭环在‘同步最新 15 分钟行情’这第一步就停止了；后面的预测、结算和模型步骤没有运行。这不是模型训练失败，而是数据还没进来。")
        st.code("# 在项目根目录 .env 添加两行（不要把真实值发给任何人）\nALPACA_API_KEY_ID=你的KeyID\nALPACA_API_SECRET_KEY=你的SecretKey", language="text")

    st.markdown("**本页默认推荐的公开行情学习闭环（无需 Alpaca）**")
    public_status_path = ROOT / "reports" / "public_intraday_learning_cycle" / "latest_status.json"
    public_status = read_json(public_status_path)
    if public_status:
        pcols = st.columns(4)
        pcols[0].metric("上次闭环结果", "成功" if public_status.get("overall_success") else "未完成")
        pcols[1].metric("最新训练样本日", public_status.get("latest_complete_intraday_date") or "缺失")
        pcols[2].metric("候选模型基准日", public_status.get("forecast_as_of_date") or "缺失")
        pcols[3].metric("候选目标日", public_status.get("forecast_target_date") or "缺失")
        st.caption(f"数据源：{public_status.get('feed', '未确认')}；训练完整交易日：{public_status.get('completed_training_sessions', 0)}；正式训练标记：{'是' if public_status.get('model_training_performed') else '否'}；自动晋级：否。")
        if not public_status.get("overall_success") and public_status.get("error"):
            st.warning(str(public_status.get("error")))
    else:
        st.caption("尚无公开行情机器学习闭环运行记录。点击页面上方的一键流程后，这里会显示最近结果。")

    st.markdown("**A. 日线预测反馈与重新拟合（无需 Alpaca）**")
    st.write("先成功刷新行情，确认本地日线日期已到上方显示的最近完整交易日，再执行此操作。它会结算此前待验证的预测、按当前数据生成一条新的日线候选预测并存档；这不等于候选模型已经通过晋级评估。")
    local_daily_ready = bool(expected_date and market_date and market_date >= expected_date)
    if not local_daily_ready:
        st.info(f"当前日线截至 {market_date or '缺失'}，预期至少要到 {expected_date or '暂无法确认'}。请先刷新行情；为防止生成目标日已经过去的预测，此按钮暂时禁用。")
    if st.button("结算日线预测反馈并生成下一交易日候选", disabled=not local_daily_ready, use_container_width=True):
        with st.spinner("正在结算过往日线预测并拟合新的候选模型……"):
            result = run([sys.executable, "scripts/run_forecast_daily_cycle.py"], timeout=1800)
        if result["returncode"] == 0:
            st.success("日线预测流程已完成：已尝试结算旧预测，并把新候选预测写入历史记录。请到‘模型学习与预测复盘’查看样本数、误差和晋级条件。")
        else:
            st.error("日线预测流程未能完成。以下是最后一段诊断日志。")
        with st.expander("查看日线反馈流程日志", expanded=result["returncode"] != 0):
            st.code((result.get("stdout") or "")[-6000:] + ("\n" + (result.get("stderr") or "")[-3000:] if result.get("stderr") else ""))

    st.markdown("**B. 本地滚动验证与模型晋级评估（无需密钥）**")
    st.write("只用现有数据库和历史文件，比较简单基准与 Ridge 候选模型；不联网、不修改生产模型，也不自动晋级。数据本身过期时，成绩单也只代表旧数据。")
    if st.button("运行本地滚动验证与晋级评估", use_container_width=True):
        with st.spinner("正在运行时序滚动验证与候选模型对照……"):
            result = run([sys.executable, "scripts/run_model_evaluation_suite.py"], timeout=1800)
        if result["returncode"] == 0:
            st.success("滚动验证已完成。结果只用于评估，生产模型不会自动替换。")
        else:
            st.error("本地模型评估未能完成。")
        with st.expander("查看本地评估日志", expanded=result["returncode"] != 0):
            st.code((result.get("stdout") or "")[-6000:] + ("\n" + (result.get("stderr") or "")[-3000:] if result.get("stderr") else ""))
    st.page_link("app/pages/model_learning_review.py", label="打开模型学习与预测复盘页面", icon="🧠")

    st.markdown("**C. 原有盘中学习闭环（需要有效的 Alpaca 凭据）**")
    if not alpaca_ready:
        st.button("运行原有盘中学习流程", disabled=True, use_container_width=True,
                  help="先在项目根目录 .env 填写 ALPACA_API_KEY_ID 和 ALPACA_API_SECRET_KEY，再重新打开页面。")
    elif st.button("运行原有盘中学习流程", use_container_width=True):
        with st.spinner("运行盘中同步、旧预测结算、学习数据集整理与候选路径回测……"):
            result = run([sys.executable, "scripts/run_intraday_learning_cycle.py"], timeout=3600)
        current_status = read_json(STATUS) or {}
        if result["returncode"] == 0 and current_status.get("skipped"):
            st.info("流程根据交易日历或重复运行保护规则跳过。本次没有生成新预测，也没有执行正式训练或模型晋级。")
        elif result["returncode"] == 0 and current_status.get("overall_success"):
            st.success("盘中闭环各步骤报告成功。请再检查生成的前瞻记录和滚动回测指标；这不代表模型已晋级。")
        else:
            failed = current_status.get("failed_stages") or []
            st.error("盘中闭环没有完整跑完。" + ("失败步骤：" + "、".join(map(str, failed)) if failed else "请查看运行日志。"))
            stages = current_status.get("stages") or []
            failed_stage = next((item for item in stages if not item.get("success")), {})
            detail = str(failed_stage.get("error") or failed_stage.get("stderr_tail") or failed_stage.get("stdout_tail") or "")
            if "401" in detail or "凭据" in detail or "API Key" in detail:
                st.warning("失败原因属于数据源鉴权/权限，不是模型精度问题。先修复凭据并成功同步新分时，再讨论模型训练和预测误差。")
            if detail:
                st.code(detail[-3000:])
        if result.get("stdout"):
            with st.expander("查看完整流程输出"):
                st.code(result["stdout"][-7000:])
    cycle_status = read_json(STATUS)
    if cycle_status:
        last_run = cycle_status.get("finished_at") or cycle_status.get("started_at")
        failed_stages = cycle_status.get("failed_stages") or []
        st.caption(f"上次盘中闭环记录：{format_time(last_run)}；结果：{'成功' if cycle_status.get('overall_success') else '失败/未完成'}；本次正式训练标记：{'是' if cycle_status.get('model_training_performed') else '否'}；自动晋级：{'是' if cycle_status.get('model_promotion_performed') else '否'}。")
        if failed_stages:
            st.caption("上次失败步骤：" + "、".join(map(str, failed_stages)) + "。这是独立的盘中流程状态，不代表上方视频内容流程失败。")

st.caption("所有预测仅为历史数据生成的情景参考，不保证准确，不构成投资建议。公开行情源可能限流或暂时不可用；日期、来源和错误不会被伪装成成功。")
