from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.system_health import inspect_project, format_time, status_label  # noqa: E402


def _run_description(status: dict[str, Any] | None) -> tuple[str, str, str]:
    if not status:
        return "尚无记录", "尚无状态文件", ""
    success = bool(status.get("overall_success"))
    skipped = bool(status.get("skipped"))
    finished = format_time(status.get("finished_at") or status.get("started_at"))
    if skipped:
        return "按规则跳过", finished, str(status.get("skip_reason") or "流程守门规则阻止了重复或过期运行。")
    failed = status.get("failed_stages") or []
    if success:
        return "成功", finished, "流程各阶段报告成功。"
    stages = status.get("stages") or []
    failure_detail = ""
    for stage in stages:
        if not stage.get("success"):
            failure_detail = str(stage.get("error") or stage.get("stderr_tail") or stage.get("stdout_tail") or "")
            break
    failure_detail = " ".join(failure_detail.split())[-700:]
    label = "失败：" + "、".join(str(x) for x in failed) if failed else "失败/未完成"
    return label, finished, failure_detail or "打开状态文件查看阶段详情。"


st.title("🩺 数据更新与系统诊断")
st.caption("本页只读取本地数据库、配置和运行报告，不会自动发起联网请求，也不会触发预测或训练。")
if st.button("🔄 重新读取本地状态", use_container_width=False):
    st.cache_data.clear()
    st.rerun()

health = inspect_project(ROOT)
expected = health.get("expected_last_completed_session")
daily = health.get("daily") or {}
intraday = health.get("intraday") or {}
news = health.get("news") or {}
macro = health.get("macro") or {}
video = health.get("video") or {}
forecast = health.get("forecast") or {}
eval_data = health.get("model_evaluation") or {}
credentials = health.get("credentials") or {}

st.subheader("一、核心数据是否跟上了最近的交易日")
cols = st.columns(4)
cols[0].metric("按交易日历应到", expected or "无法确认")
cols[1].metric("日线实际最新日", daily.get("latest_date") or "缺失")
cols[2].metric("完整十五分钟分时", intraday.get("latest_complete_date") or "缺失")
cols[3].metric("最新视频内容包", video.get("folder") or "尚未生成")

daily_status, daily_detail = status_label(daily.get("latest_date"), expected)
intra_status, intra_detail = status_label(intraday.get("latest_complete_date"), expected)
if daily.get("latest_date") and expected and daily.get("latest_date") < expected:
    st.warning(f"日线数据有滞后：{daily_detail}。这表示本地缓存较旧，不一定代表程序崩溃；请在视频页刷新公开行情。")
elif daily.get("latest_date"):
    st.success(f"日线状态：{daily_status}。最后记录抓取时间：{format_time(daily.get('fetched_at'))}。")
else:
    st.error("本地没有 QQQM 日线记录，无法生成新预测。")
if intraday.get("latest_complete_date") and expected and intraday.get("latest_complete_date") < expected:
    st.warning(f"十五分钟分时有滞后：{intra_detail}。盘中路径评估需要完整分时，不能用开高低收四个日线价格点替代。")
elif intraday.get("latest_complete_date"):
    st.success(f"分时状态：{intra_status}。来源 {intraday.get('feed') or '未知'}，有效点数 {intraday.get('slots', '—')}。")
else:
    st.error("目前没有通过完整度检查的十五分钟交易日；盘中路径模型和真实曲线复盘无法可靠运行。")

st.markdown("**各数据源快照**")
rows = [
    {"类别": "日线行情", "本地最新日期": daily.get("latest_date") or "缺失", "最后抓取/读取时间": format_time(daily.get("fetched_at")), "状态/说明": f"{daily_status}；来源 {daily.get('source') or '未知'}"},
    {"类别": "十五分钟分时", "本地最新日期": intraday.get("latest_complete_date") or "缺失", "最后抓取/读取时间": format_time(intraday.get("fetched_at")), "状态/说明": f"{intra_status}；来源 {intraday.get('feed') or '未知'}；有效点 {intraday.get('slots', '—')}"},
    {"类别": "新闻条目", "本地最新日期": news.get("latest_published_date") or "缺失", "最后抓取/读取时间": format_time(news.get("latest_fetched_at")), "状态/说明": f"本地共 {news.get('count', 0)} 条；注意区分新闻发布时间和抓取时间"},
    {"类别": "宏观数据", "本地最新日期": macro.get("latest_observation_date") or "缺失", "最后抓取/读取时间": format_time(macro.get("latest_retrieved_at")), "状态/说明": "各宏观指标有自己的发布周期，不要求每日都新增观测"},
    {"类别": "模型评估", "本地最新日期": eval_data.get("data_latest_date") or "尚无成绩单", "最后运行时间": format_time(eval_data.get("created_at")), "状态/说明": f"滚动测试 {eval_data.get('rolling_test_samples', '—')} 条；真实已结算 {eval_data.get('live_evaluated_samples', '—')} 条；待验证 {eval_data.get('live_pending_samples', '—')} 条"},
]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
if news.get("latest_published_date") and news.get("latest_fetched_at"):
    st.caption(f"新闻记录最新发布时间：{news.get('latest_published_date')}；最近抓取时间：{format_time(news.get('latest_fetched_at'))}。如果抓取时间更新但发布时间没变，说明抓取器运行过，但来源并未提供更新的新闻条目。")

st.divider()
st.subheader("二、预测和模型状态")
video_fc = forecast.get("video_forecast") or {}
intraday_fc = forecast.get("intraday_forecast") or {}
fc1, fc2 = st.columns(2)
with fc1:
    st.markdown("**视频情景曲线**")
    st.write(f"状态：{'有可用缓存' if video_fc.get('available') else '不可用/未生成'}")
    st.write(f"基准日期：{video_fc.get('as_of_date') or '缺失'}")
    st.write(f"目标日期：{video_fc.get('target_date') or '缺失'}")
    if video_fc.get("model"):
        st.caption(f"方法：{video_fc.get('model')}")
    st.info("视频生成使用近 20 个完整交易日的十五分钟历史路径统计生成情景，不进行机器学习训练。")
with fc2:
    st.markdown("**原有盘中模型快照**")
    st.write(f"状态：{'有缓存（仍须检查目标日是否过期）' if intraday_fc else '缺失'}")
    st.write(f"基准日期：{intraday_fc.get('as_of_date') or '缺失'}")
    st.write(f"目标日期：{intraday_fc.get('target_date') or '缺失'}")
    if intraday_fc.get("model"):
        st.caption(f"模型/方法：{intraday_fc.get('model')}")
    st.info("原有盘中流程会尝试结算旧预测、整理学习数据并重新运行候选路径模型；是否真的训练、是否晋级，必须以流程状态和模型评估成绩单为准。")

st.write("**当前模型评估**")
st.write(f"最近成绩单：{format_time(eval_data.get('created_at'))}；评估数据截止：{eval_data.get('data_latest_date') or '未知'}；晋级状态：{eval_data.get('gate_status') or '尚无状态'}。")
if eval_data.get("gate_reason"):
    st.caption(str(eval_data["gate_reason"]))
st.page_link("app/pages/model_learning_review.py", label="进入模型学习与预测复盘", icon="🧠")
st.page_link("app/pages/video_script_studio.py", label="进入视频生成与行情刷新", icon="🎬")

st.divider()
st.subheader("三、数据源凭据检查")
st.caption("这里只检查有没有配置，不显示任何密钥内容。检测到密钥不代表密钥有效，最终仍要以接口返回为准。")
cred_cols = st.columns(3)
cred_cols[0].metric("Alpaca 盘中接口", "配置完整" if credentials.get("alpaca_configured") else "缺少 Key/Secret")
cred_cols[1].metric("EODHD 备用接口", "已配置" if credentials.get("eodhd_configured") else "未配置")
cred_cols[2].metric("DeepSeek 文本分析", "已配置" if credentials.get("deepseek_configured") else "未配置")
if not credentials.get("alpaca_configured"):
    st.warning("未检测到完整的 ALPACA_API_KEY_ID 与 ALPACA_API_SECRET_KEY。原有盘中同步按钮应保持禁用；这不会阻止使用公开行情的独立视频流程。")
    st.code("# 项目根目录 .env 示例\nALPACA_API_KEY_ID=你的KeyID\nALPACA_API_SECRET_KEY=你的SecretKey", language="text")

st.divider()
st.subheader("四、最近一次自动化流程结果")
for title, key in (("原有盘中学习闭环", "intraday_cycle"), ("统一每日工作流", "daily_workflow")):
    status = health.get(key)
    label, finished, detail = _run_description(status)
    with st.expander(f"{title}：{label} · {finished}", expanded=bool(status and not status.get("overall_success"))):
        if not status:
            st.info("尚无运行记录。")
        else:
            st.write(f"运行开始：{format_time(status.get('started_at'))}")
            st.write(f"运行结束：{finished}")
            if status.get("failed_stages"):
                st.write("失败步骤：" + "、".join(str(x) for x in status.get("failed_stages", [])))
            if detail:
                st.warning(detail if len(detail) < 1000 else detail[-1000:])
            st.caption(f"是否训练模型：{'是' if status.get('model_training_performed') else '否/未记录为正式训练'}；是否自动晋级：{'是' if status.get('model_promotion_performed') else '否'}。")
            st.caption("此处显示的是报告里记录的上一次运行，不会因为打开页面就自动重试。")

st.caption("健康检查只读本地信息，不会调用行情服务。要更新日线、分时或新闻，请主动进入视频页刷新；要评估模型，请进入模型学习与预测复盘。")
