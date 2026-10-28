from __future__ import annotations

"""Product-facing overview and guided demo page for MarketPilot AI."""

import streamlit as st
import pandas as pd

st.title("📈 MarketPilot AI · 美股多资产研究台")
st.caption("MarketPilot AI · 面向投资新手的多资产研究、解释与复盘工具")

st.markdown(
    """
    <style>
    .qqqm-hero {padding: 1.35rem 1.5rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.28); background: linear-gradient(120deg, rgba(45,91,130,.14), rgba(48,150,130,.08)); margin-bottom: 1rem;}
    .qqqm-hero h2 {margin: 0 0 .55rem 0; font-size: 1.55rem;}
    .qqqm-hero p {margin: .25rem 0; line-height: 1.65;}
    .qqqm-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 1rem; min-height: 150px;}
    .qqqm-card h3 {font-size: 1.05rem; margin: .2rem 0 .55rem 0;}
    .qqqm-muted {color: var(--text-color); opacity: .78; font-size: .92rem; line-height: 1.6;}
    </style>
    <div class="qqqm-hero">
      <h2>让分散的金融信息，变成可理解、可追溯、可复盘的研究流程。</h2>
      <p>面向投资新手，将行情、宏观背景、市场新闻和个人交易记录放在同一工作台中，辅助理解持仓与市场变化。</p>
      <p><b>产品原则：</b>先看数据与证据，再看模型输出；每次预测都应留下记录，并在实际行情出现后接受检验。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("产品解决什么问题")
cols = st.columns(3)
with cols[0]:
    st.markdown('<div class="qqqm-card"><h3>🧾 看清自己的持仓</h3><p class="qqqm-muted">把买入、卖出、手续费和股息放进交易账本，区分已实现与未实现盈亏，减少只看当前价格造成的误判。</p></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown('<div class="qqqm-card"><h3>🧠 理解市场背景</h3><p class="qqqm-muted">结合价格走势、新闻与宏观数据生成通俗解释，并尽可能保留数据日期、来源与结论边界。</p></div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown('<div class="qqqm-card"><h3>🔍 检验预测表现</h3><p class="qqqm-muted">保存事前预测，将它与之后的实际结果比较；同时与简单基准对照，不因单次猜中就宣称模型有效。</p></div>', unsafe_allow_html=True)

st.subheader("三分钟演示路线")
st.markdown(
    """
    1. **持仓与交易记录（约 30 秒）**：展示一笔交易、成本和盈亏的计算逻辑。
    2. **AI 市场分析（约 40 秒）**：打开一份已有报告，指出行情、新闻或宏观数据的来源和日期。
    3. **预测与实际复盘（约 50 秒）**：展示事前预测、实际价格、误差指标，以及模型与简单基准的对比。
    4. **产品下一步（约 30 秒）**：说明先通过校园用户访谈和小规模试用验证价值，再考虑商业化。

    演示前请检查数据日期。若本机数据没有更新，应明确说是“截至某日的历史研究数据”，不要把旧数据称为实时行情。
    """
)

st.subheader("直接进入演示页面")
left, right = st.columns(2)
with left:
    st.page_link("/market-analysis", label="打开：市场、持仓与网格回测", icon="📊", use_container_width=True)
    st.page_link("/next-day-forecast", label="打开：下一交易日价格情景", icon="🔭", use_container_width=True)
    st.page_link("/model-learning-review", label="打开：模型学习与预测复盘", icon="🧠", use_container_width=True)
with right:
    st.page_link("/daily-overview", label="打开：每日总览与复盘", icon="📈", use_container_width=True)
    st.page_link("/video-script-studio", label="打开：多平台视频脚本与配图", icon="🎬", use_container_width=True)
    st.page_link("/system-status", label="打开：数据更新与系统诊断", icon="🩺", use_container_width=True)

st.subheader("从原型到市场验证")
roadmap = [
    {"阶段": "阶段 1 · 校园验证", "目标": "访谈 20–30 名有投资学习需求的学生或年轻投资者；观察报告是否看得懂、账本是否易用、用户是否愿意复用。", "指标": "任务完成率、报告理解度、建议采纳反馈、复用意愿、付费意愿。"},
    {"阶段": "阶段 2 · 个人研究工具", "目标": "只有在需求得到初步验证后，再扩展多 ETF、组合对比、个性化复盘和报告导出。", "指标": "留存、复用频率、数据错误率、单位用户服务成本。"},
    {"阶段": "阶段 3 · 合作探索", "目标": "研究与金融教育、财经内容或合规金融服务机构合作的可能性。", "指标": "真实合作需求、数据授权条件、合规评估与可持续收入。"},
]
st.dataframe(pd.DataFrame(roadmap), hide_index=True, use_container_width=True)

st.warning(
    "研究与教学用途，不提供个性化买卖指令，也不保证投资收益。预测和情景区间存在不确定性；"
    "公开部署前还需要进一步完善隐私保护、账号安全、数据授权和合规审查。"
)
