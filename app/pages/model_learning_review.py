from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.services.ui_workflows import run_project_command

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "market_data.db"
MODEL_LAB_DIR = ROOT / "reports" / "model_lab"
CHALLENGER_REPORT = MODEL_LAB_DIR / "QQQM_champion_challenger_latest.json"
MONTHLY_REPORT = MODEL_LAB_DIR / "QQQM_champion_challenger_monthly_latest.csv"
EVALUATION_SCORECARD = MODEL_LAB_DIR / "QQQM_model_evaluation_scorecard_latest.json"

MODEL_COLUMNS = {
    "零收益基线": ("error_zero_return_bps", "direction_correct_zero_return"),
    "历史均值基线": ("error_historical_mean_bps", "direction_correct_historical_mean"),
    "完整特征 Ridge": ("error_ridge_full_bps", "direction_correct_ridge_full"),
    "收益率特征 Ridge": ("error_ridge_returns_only_bps", "direction_correct_ridge_returns_only"),
}
PREDICTION_COLUMNS = {
    "实际收益率": "actual_return_pct",
    "零收益基线": "zero_return_baseline_pct",
    "历史均值基线": "historical_mean_return_pct",
    "完整特征 Ridge": "ridge_full_return_pct",
    "收益率特征 Ridge": "ridge_returns_only_return_pct",
}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _read_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _load_feedback() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), {"error": f"找不到数据库：{DB_PATH}"}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "forecast_feedback"):
            return pd.DataFrame(), pd.DataFrame(), {"error": "数据库中不存在 forecast_feedback 表。"}

        columns = _read_table_columns(conn, "forecast_feedback")
        wanted = [
            "id", "ticker", "as_of_date", "status", "actual_target_date",
            "actual_adjusted_close", "actual_return_pct", "current_price",
            "zero_return_baseline_pct", "historical_mean_return_pct",
            "ridge_full_return_pct", "ridge_returns_only_return_pct",
            "error_zero_return_bps", "error_historical_mean_bps",
            "error_ridge_full_bps", "error_ridge_returns_only_bps",
            "direction_correct_zero_return", "direction_correct_historical_mean",
            "direction_correct_ridge_full", "direction_correct_ridge_returns_only",
        ]
        select_cols = [col for col in wanted if col in columns]
        query = f"SELECT {', '.join(select_cols)} FROM forecast_feedback WHERE ticker = ? ORDER BY as_of_date DESC, id DESC"
        all_rows = pd.read_sql_query(query, conn, params=("QQQM",))
        evaluated = all_rows[
            (all_rows.get("status", pd.Series(index=all_rows.index, dtype="object")) == "evaluated")
        ].copy() if not all_rows.empty else pd.DataFrame()

        context_summary: dict[str, Any] = {}
        if _table_exists(conn, "forecast_context_snapshots"):
            ctx_cols = _read_table_columns(conn, "forecast_context_snapshots")
            if {"ticker", "as_of_date", "capture_mode"}.issubset(ctx_cols):
                context_summary["snapshot_count"] = conn.execute(
                    "SELECT COUNT(*) FROM forecast_context_snapshots WHERE ticker='QQQM'"
                ).fetchone()[0]
                context_summary["modes"] = pd.read_sql_query(
                    "SELECT capture_mode, COUNT(*) AS count FROM forecast_context_snapshots WHERE ticker='QQQM' GROUP BY capture_mode ORDER BY count DESC",
                    conn,
                )
                context_summary["latest"] = pd.read_sql_query(
                    "SELECT as_of_date, capture_mode FROM forecast_context_snapshots WHERE ticker='QQQM' ORDER BY as_of_date DESC LIMIT 5",
                    conn,
                )
        if _table_exists(conn, "market_daily"):
            context_summary["latest_market_date"] = conn.execute(
                "SELECT MAX(trade_date) FROM market_daily WHERE ticker='QQQM'"
            ).fetchone()[0]
        return all_rows, evaluated, context_summary
    finally:
        conn.close()


def _metric_table(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, (error_col, direction_col) in MODEL_COLUMNS.items():
        if error_col not in evaluated.columns:
            continue
        errors = pd.to_numeric(evaluated[error_col], errors="coerce").dropna()
        direction = (
            pd.to_numeric(evaluated[direction_col], errors="coerce").dropna()
            if direction_col in evaluated.columns
            else pd.Series(dtype=float)
        )
        if errors.empty:
            continue
        rows.append({
            "模型": label,
            "已验证样本数": int(errors.count()),
            "MAE（bps，越低越好）": round(float(errors.abs().mean()), 3),
            "RMSE（bps，越低越好）": round(float((errors.pow(2).mean()) ** 0.5), 3),
            "方向准确率": round(float(direction.mean() * 100), 1) if not direction.empty else None,
        })
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        return result if isinstance(result, dict) else {"content": result}
    except (OSError, json.JSONDecodeError):
        return None


st.title("🧠 模型学习与预测复盘")
st.caption("查看预测记录如何被真实收盘价结算，比较简单基准与候选模型。打开页面只读取本地数据；只有点击按钮才会运行同步、结算或评估任务，不会下单，也不会自动晋级模型。")
st.info("先看‘已结算预测’有多少条，再看模型误差是否优于简单基准。MAE / RMSE 越低通常越好；样本太少时，不能因为一次猜中就认定模型变准。")

try:
    with sqlite3.connect(DB_PATH) as status_conn:
        last_market = status_conn.execute(
            "SELECT trade_date, source, fetched_at FROM market_daily WHERE UPPER(ticker)='QQQM' ORDER BY trade_date DESC LIMIT 1"
        ).fetchone()
except sqlite3.Error:
    last_market = None
status_cols = st.columns(3)
status_cols[0].metric("本地最新行情日期", str(last_market[0])[:10] if last_market and last_market[0] else "未知")
status_cols[1].metric("最新记录来源", str(last_market[1]) if last_market and last_market[1] else "未知")
status_cols[2].metric("记录更新时间", str(last_market[2]) if last_market and last_market[2] else "未知")
if last_market and str(last_market[1] or "").startswith("public_snapshot_"):
    st.warning("行情记录来自公开网页校验快照，不代表本次 API 已同步成功。点击下面的完整同步按钮可以尝试更新供应商数据。")
    st.markdown("[打开原始公开历史行情来源](https://stockanalysis.com/etf/qqqm/history/)")

workflow_col, settle_col = st.columns(2)
with workflow_col:
    if st.button("🔄 同步行情并更新预测反馈", type="primary", use_container_width=True, key="review_run_full_update"):
        with st.spinner("正在同步数据、结算旧预测并按需生成最新预测……"):
            result = run_project_command(["-m", "app.daily_automation"], timeout=2400)
        st.session_state["review_full_update_result"] = result
        st.rerun()
with settle_col:
    if st.button("🧾 仅结算已有预测", use_container_width=True, key="review_settle_only"):
        with st.spinner("正在用本地真实行情结算尚未完成的预测……"):
            result = run_project_command(["scripts/run_forecast_daily_cycle.py", "--evaluate-only"], timeout=1800)
        st.session_state["review_settle_result"] = result
        st.rerun()

for result_key, label in [
    ("review_full_update_result", "完整同步与预测工作流"),
    ("review_settle_result", "历史预测结算"),
]:
    action_result = st.session_state.get(result_key)
    if action_result:
        if action_result.get("returncode") == 0:
            st.success(f"{label}已执行完成。刷新后查看最新状态与已结算记录。")
        else:
            st.warning(f"{label}未完整成功。具体错误见日志，未确认成功的记录不会被标成实时更新。")
        with st.expander(f"查看：{label}日志", expanded=action_result.get("returncode") != 0):
            if action_result.get("stdout"):
                st.code(action_result["stdout"][-10000:])
            if action_result.get("stderr"):
                st.code(action_result["stderr"][-6000:])

if st.button("🔄 只刷新当前页面数据", key="review_refresh_view"):
    st.rerun()

st.subheader("第一步：统一模型评估与晋级检查")
st.write(
    "这一步只在本地运行时序回测：将历史行情 CSV 与 SQLite 中更新的日线合并到独立副本，"
    "比较简单基准和 Ridge，再检查历史留出集与真实前瞻反馈。它不会联网、不会改生产预测，也不会自动晋级模型。"
)
if st.button("🧪 运行滚动验证与晋级评估", type="primary", use_container_width=True):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_model_evaluation_suite.py")],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=1800, check=False,
        )
        st.session_state["model_eval_suite_result"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
        }
    except Exception as exc:
        st.session_state["model_eval_suite_result"] = {
            "returncode": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"
        }

eval_run_result = st.session_state.get("model_eval_suite_result")
if eval_run_result:
    if eval_run_result.get("returncode") == 0:
        st.success("模型评估完成。请同时检查历史留出集和实时前瞻门槛；它们通过也不会自动替换生产模型。")
    else:
        st.error("模型评估未能完成。错误信息见下方。")
    with st.expander("查看本次运行日志", expanded=eval_run_result.get("returncode") != 0):
        if eval_run_result.get("stdout"):
            st.code(eval_run_result["stdout"][-10000:])
        if eval_run_result.get("stderr"):
            st.code(eval_run_result["stderr"][-5000:])

scorecard = _read_json(EVALUATION_SCORECARD) if EVALUATION_SCORECARD.exists() else None
if scorecard:
    history_info = scorecard.get("history", {})
    walk_info = scorecard.get("walk_forward", {}).get("metadata", {})
    tournament_info = scorecard.get("historical_champion_challenger", {})
    holdout_info = tournament_info.get("holdout", {})
    history_gate = tournament_info.get("promotion_gate", {})
    live_gate = scorecard.get("live_shadow", {})
    overall_gate = scorecard.get("overall_gate", {})
    st.markdown("**最近一次统一评估结果**")
    summary_cols = st.columns(4)
    summary_cols[0].metric("评估数据最新日期", str(history_info.get("combined_latest_date") or "未知"))
    summary_cols[1].metric("滚动验证样本", str(walk_info.get("out_of_sample_samples", "未知")))
    summary_cols[2].metric("已结算前瞻样本", str(live_gate.get("evaluated_samples", 0)))
    summary_cols[3].metric("综合晋级状态", str(overall_gate.get("status", "未知")))
    st.caption(
        f"历史留出集：{holdout_info.get('start_target_date', '未知')} 至 "
        f"{holdout_info.get('end_target_date', '未知')}；候选模型："
        f"{holdout_info.get('selected_challenger', '未知')}；历史门槛："
        f"{'通过' if history_gate.get('historical_gate_passed') else '未通过'}；"
        f"真实前瞻门槛：{'通过' if live_gate.get('gate_passed') else '未通过'}。"
    )
    evaluation_metrics = scorecard.get("walk_forward", {}).get("metrics", [])
    if evaluation_metrics:
        with st.expander("展开统一滚动验证成绩单", expanded=True):
            eval_frame = pd.DataFrame(evaluation_metrics)
            display_cols = [c for c in ["experiment", "n", "mae_bps", "rmse_bps", "sign_accuracy", "prediction_actual_correlation"] if c in eval_frame.columns]
            rename_cols = {
                "experiment": "模型", "n": "样本数", "mae_bps": "平均绝对误差（基点）",
                "rmse_bps": "均方根误差（基点）", "sign_accuracy": "方向命中率",
                "prediction_actual_correlation": "预测与实际相关系数",
            }
            st.dataframe(eval_frame[display_cols].rename(columns=rename_cols), hide_index=True, use_container_width=True)
            st.write("实时前瞻门槛：" + str(live_gate.get("reason", "尚无说明。")))
            st.caption("当前流程只评估，不替换生产模型。收益率滚动验证与十五分钟走势模型是不同任务，应分别评价。")
else:
    st.info("尚未生成统一评估成绩单。点击上面的按钮，即可把最新本地日线纳入独立评估副本并生成成绩单。")

all_rows, evaluated, context_summary = _load_feedback()

if "error" in context_summary:
    st.error(context_summary["error"])
    st.stop()

if all_rows.empty:
    st.warning("尚未找到 QQQM 预测记录。先运行 scripts/run_forecast_daily_cycle.py 生成预测和反馈记录。")
    st.stop()

all_rows = all_rows.sort_values([c for c in ["as_of_date", "id"] if c in all_rows.columns], ascending=False)
latest = all_rows.iloc[0]
evaluated_count = len(evaluated)
pending_count = int((all_rows["status"] == "pending").sum()) if "status" in all_rows.columns else 0
latest_date = context_summary.get("latest_market_date", "未知")

col1, col2, col3, col4 = st.columns(4)
col1.metric("最新行情日期", str(latest_date or "未知"))
col2.metric("已结算预测", f"{evaluated_count} 条")
col3.metric("等待验证", f"{pending_count} 条")
col4.metric("最新预测日期", str(latest.get("as_of_date", "未知")))

if evaluated_count < 30:
    st.warning(
        f"目前只有 {evaluated_count} 条已结算预测。样本不足以判定模型已经变准，也不足以自动晋级候选模型。"
        "请把现有模型保持在影子评估状态，持续记录真实的下一交易日结果。"
    )

st.subheader("一、实盘前瞻反馈：预测收益率 vs 实际收益率")
if evaluated.empty:
    st.info("尚无已结算预测，因此暂时无法绘制实际收益对比图。")
else:
    chart_cols = [c for c in PREDICTION_COLUMNS.values() if c in evaluated.columns]
    date_col = "actual_target_date" if "actual_target_date" in evaluated.columns else "as_of_date"
    if chart_cols:
        chart = evaluated[[date_col] + chart_cols].copy()
        chart[date_col] = pd.to_datetime(chart[date_col], errors="coerce")
        chart = chart.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        chart = chart.rename(columns={v: k for k, v in PREDICTION_COLUMNS.items() if v in chart.columns})
        for col in chart.columns:
            chart[col] = pd.to_numeric(chart[col], errors="coerce")
        st.line_chart(chart)
        st.caption("单位为收益率百分比（%）。当前前瞻样本很少，折线仅用于追踪，不代表可靠的统计结论。")
    else:
        st.info("现有反馈表缺少绘图所需的预测字段。")

st.subheader("二、基于真实已结算样本的模型表现")
metrics = _metric_table(evaluated)
if metrics.empty:
    st.info("当前数据库未提供完整误差字段，无法计算模型指标。")
else:
    st.dataframe(metrics, hide_index=True)
    st.caption("MAE/RMSE 使用数据库记录的误差（bps）；方向准确率来自对应的历史结算记录。样本太少时，指标会剧烈波动。")

st.subheader("三、最近预测记录")
show_cols = [c for c in [
    "id", "as_of_date", "status", "actual_target_date", "current_price",
    "actual_adjusted_close", "actual_return_pct", "ridge_full_return_pct",
    "ridge_returns_only_pct", "historical_mean_return_pct",
] if c in all_rows.columns]
# The production schema uses ridge_returns_only_return_pct; keep the actual column visible when available.
if "ridge_returns_only_return_pct" in all_rows.columns and "ridge_returns_only_return_pct" not in show_cols:
    show_cols.append("ridge_returns_only_return_pct")
if show_cols:
    st.dataframe(all_rows[show_cols].head(30), hide_index=True)

st.subheader("四、历史 Champion–Challenger 实验")
report = _read_json(CHALLENGER_REPORT) if CHALLENGER_REPORT.exists() else None
if report:
    st.info(
        "最近一次历史实验只作为研究报告。当前回测未达到晋级门槛，候选模型不应自动替换现有参考模型。"
    )
    with st.expander("查看 Champion–Challenger 完整实验报告"):
        st.json(report)
else:
    st.info("尚未生成 Champion–Challenger 报告。运行 scripts/run_champion_challenger.py 后，此处会显示结果。")

if MONTHLY_REPORT.exists():
    try:
        monthly = pd.read_csv(MONTHLY_REPORT)
        with st.expander("查看按月拆分的历史留出集表现"):
            st.dataframe(monthly, hide_index=True)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"无法读取月度实验报告：{exc}")

st.subheader("五、上下文快照归档状态")
if context_summary.get("snapshot_count") is not None:
    st.metric("已归档快照", f"{context_summary['snapshot_count']} 条")
    modes = context_summary.get("modes")
    if isinstance(modes, pd.DataFrame) and not modes.empty:
        st.dataframe(modes, hide_index=True)
    latest_snapshots = context_summary.get("latest")
    if isinstance(latest_snapshots, pd.DataFrame) and not latest_snapshots.empty:
        st.caption("最近快照")
        st.dataframe(latest_snapshots, hide_index=True)
    st.warning("reconstructed（事后补建）快照不作为新闻预测特征训练样本。只有预测时即时保存且具备可靠时间点可用性的记录，才可进入后续实验。")
else:
    st.info("尚未发现 forecast_context_snapshots 表；确认每日预测周期已接入上下文归档模块。")

st.subheader("六、当前系统状态")
st.markdown(
    "- 预测周期每天重训候选模型，不等于预测准确率每天提升。\n"
    "- 新闻和宏观特征目前用于情境解释；在确认历史发布时间、数据修订和时间点可用性之前，不加入历史训练。\n"
    "- 不根据单日结果自动晋级模型；新模型必须在独立样本外测试中稳定改善。\n"
    "- 本页面不下单，也不生成无数据支撑的买入/卖出指令。"
)
