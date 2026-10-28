from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

DASHBOARD = Path("app/dashboard.py")
if not DASHBOARD.exists():
    raise SystemExit("app/dashboard.py not found. Run this script from the project root.")

source = DASHBOARD.read_text(encoding="utf-8")
marker = "# V0.4.0-C decision support section"
if marker in source:
    raise SystemExit("Decision support section already exists. No changes made.")

anchor = '                    st.subheader("交易历史与已实现盈亏")'
if source.count(anchor) != 1:
    raise SystemExit(
        "Expected portfolio-history anchor not found exactly once; no changes made. "
        "Please inspect app/dashboard.py and update manually if its layout has changed."
    )

import_anchor = "from app.portfolio import ("
if source.count(import_anchor) != 1:
    raise SystemExit("Could not safely locate app.portfolio import block; no changes made.")

# Add the helper import beside the existing app.portfolio import block.
closing = re.search(r"from app\.portfolio import \([\s\S]*?\n\)", source)
if not closing:
    raise SystemExit("Could not identify the full app.portfolio import block; no changes made.")
source = source[:closing.end()] + "\nfrom app.decision_support import build_decision_assessment" + source[closing.end():]

block = r'''                    # V0.4.0-C decision support section
                    st.subheader("下一步怎么做？")
                    st.caption(
                        "根据你自己设定的持有期限、回撤提醒线和预算，生成透明的风险复核计划。"
                        "这是规则化情景分析，不是价格预测、个人化买卖指令或自动交易。"
                    )
                    with st.form("portfolio_decision_support_form"):
                        horizon = st.selectbox(
                            "你计划持有这类投资多久？",
                            options=["少于 1 年", "1–5 年", "5 年以上"],
                            index=1,
                            help="按你自己的计划填写；期限较长并不保证盈利。",
                        )
                        review_drawdown = st.slider(
                            "如果价格从当前估值下跌多少，提醒你重新评估？",
                            min_value=5,
                            max_value=30,
                            value=10,
                            step=5,
                            format="%d%%",
                        )
                        planned_budget = st.number_input(
                            "未来考虑投入的新增预算（参考金额；不代表建议买入）",
                            min_value=0.0,
                            value=0.0,
                            step=50.0,
                            help="填写 0 表示目前不计划追加。多币种持仓时，此预算不自动分配或换汇。",
                        )
                        decision_submitted = st.form_submit_button(
                            "生成我的条件式观察计划",
                            type="primary",
                        )

                    if decision_submitted:
                        try:
                            st.session_state["portfolio_decision_assessment"] = build_decision_assessment(
                                positions=positions,
                                holding_horizon=horizon,
                                review_drawdown_pct=review_drawdown,
                                planned_additional_budget=planned_budget,
                            )
                            st.session_state["portfolio_decision_preferences"] = {
                                "holding_horizon": horizon,
                                "review_drawdown_pct": review_drawdown,
                                "planned_additional_budget": planned_budget,
                            }
                        except Exception as exc:
                            st.error(f"无法生成观察计划：{type(exc).__name__}: {exc}")

                    decision_result = st.session_state.get("portfolio_decision_assessment")
                    if decision_result:
                        st.markdown("#### 你的观察计划")
                        st.write(decision_result.get("conclusion", ""))
                        if decision_result.get("horizon_note"):
                            st.info(decision_result["horizon_note"])
                        if decision_result.get("budget_note"):
                            st.caption(decision_result["budget_note"])
                        for decision_warning in decision_result.get("warnings", []):
                            st.warning(decision_warning)
                        for item in decision_result.get("items", []):
                            with st.container(border=True):
                                st.markdown(f"**{item['ticker']} · {item['currency']}**")
                                metric_cols = st.columns(3)
                                metric_cols[0].metric(
                                    "当前累计净盈亏",
                                    f"{item['current_cumulative_net_pnl']:+,.2f} {item['currency']}",
                                )
                                metric_cols[1].metric(
                                    f"价格下跌 {item['review_scenario_drawdown_pct']:.0f}% 的市值变化",
                                    f"−{item['review_scenario_value_decrease']:,.2f} {item['currency']}",
                                )
                                metric_cols[2].metric(
                                    "该情景下累计净盈亏",
                                    f"{item['review_scenario_cumulative_net_pnl']:+,.2f} {item['currency']}",
                                )
                                st.write(item["review_trigger"])
                                st.caption(item["interpretation"])
                        st.caption(
                            "设置目前保存在当前 Streamlit 会话中。刷新浏览器或重启应用后可能需要重新填写。"
                            "系统不会因浮亏自动建议加仓，也不会发送或执行交易。"
                        )

'''

source = source.replace(anchor, block + anchor, 1)
compile(source, str(DASHBOARD), "exec")
backup = DASHBOARD.with_name(
    f"dashboard.py.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
)
shutil.copy2(DASHBOARD, backup)
DASHBOARD.write_text(source, encoding="utf-8", newline="\n")
print("Decision support section inserted successfully.")
print(f"Backup: {backup}")
print("Python syntax check passed.")
