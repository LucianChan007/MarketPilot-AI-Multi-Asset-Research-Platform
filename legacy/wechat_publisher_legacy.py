from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORECAST_PATH = ROOT / "reports" / "intraday" / "QQQM_intraday_forecast_latest.json"
DB_PATH = ROOT / "data" / "market_data.db"
ARTICLES = ROOT / "reports" / "articles"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def latest_market_date() -> str | None:
    if not DB_PATH.exists():
        return None
    try:
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM market_daily WHERE UPPER(ticker)='QQQM'").fetchone()
        return str(row[0])[:10] if row and row[0] else None
    except Exception:
        return None


def run_command(command: list[str], timeout: int = 2400) -> dict[str, Any]:
    try:
        r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, check=False)
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "command": command}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": -1, "stdout": exc.stdout or "", "stderr": f"超过 {timeout} 秒，已停止等待。", "command": command}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "command": command}


st.title("📝 公众号文章生成与导出")
st.caption("先完成数据与模型流程，再导出可读、可复制的文章草稿。打开页面不会偷偷刷新数据、训练模型或生成新预测。")

market_date = latest_market_date()
forecast = read_json(FORECAST_PATH)
if not forecast:
    st.warning("目前没有可读取的盘中预测快照。请先运行下面的每日闭环。")
    target = None
    as_of = None
else:
    target = str(forecast.get("forecast_date_weekday_estimate") or "") or None
    as_of = str(forecast.get("as_of_date") or "") or None
    main = forecast.get("main_path") or {}
    op = (forecast.get("estimated_open") or {}).get("p50_price")
    cl = (forecast.get("estimated_close_from_path") or {}).get("p50_price")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行情最新日期", market_date or "缺失")
    c2.metric("预测基准日期", as_of or "缺失")
    c3.metric("目标交易日", target or "缺失")
    c4.metric("主路径模型", str(main.get("model_name") or "历史参考路径")[:28])
    if op is not None:
        st.metric("预计开盘参考", f"${float(op):.2f}")
    if cl is not None:
        st.metric("预计收盘参考", f"${float(cl):.2f}")
    if market_date and target and target <= market_date:
        st.error(f"当前保存的预测目标日是 {target}，但本地真实行情已经更新到 {market_date}。这条曲线已过期，不能作为下一交易日公众号预测。")
        st.caption("下面可以运行统一每日流程，更新历史反馈、新闻上下文、模型输入和下一交易日盘中路径。")
    elif target and as_of:
        st.success(f"预测基准为 {as_of}，目标日为 {target}。文章将使用已存档的预测数字，不会让语言模型改写价格。")

st.divider()
st.subheader("第一步：统一每日闭环")
st.markdown(
    "流程会复用原项目的已有模块：同步日线与新闻、按数据日期结算已有预测、更新日线模型反馈、同步盘中 K 线、复盘旧盘中路径、生成下一交易日路径，并在预测日期有效时导出文章。"
)
col_a, col_b = st.columns(2)
with col_a:
    sync_news = st.checkbox("同步并分类新闻", value=True, help="使用已有新闻模块；待分类新闻可能调用 DeepSeek，建议保留，因为预测需要盘前上下文。")
with col_b:
    sync_macro = st.checkbox("同步宏观数据（FRED）", value=False, help="可选，若未配置 FRED 权限可以先不勾选。")
if st.button("▶ 运行统一每日闭环", type="primary", use_container_width=True):
    command = [sys.executable, "scripts/run_unified_daily_workflow.py"]
    if not sync_news:
        command.append("--skip-news")
    if not sync_macro:
        command.append("--skip-macro")
    with st.spinner("正在执行同步、复盘与路径学习流程。这个动作可能需要几分钟，期间会显示真实步骤结果。"):
        result = run_command(command, timeout=3600)
    st.session_state["unified_daily_workflow_result"] = result
    if result["returncode"] == 0:
        st.success("流程已完成。请检查下面的状态摘要，再确认文章目标日期。")
    else:
        st.warning("流程存在失败或部分失败。下面会显示每个步骤的输出，不要忽略数据源权限或行情日期问题。")

last_result = st.session_state.get("unified_daily_workflow_result")
if last_result:
    with st.expander("查看刚才的每日流程日志", expanded=True):
        st.code((last_result.get("stdout") or "")[-12000:] or "没有标准输出")
        if last_result.get("stderr"):
            st.warning((last_result.get("stderr") or "")[-5000:])

st.divider()
st.subheader("第二步：生成 / 下载公众号稿件")
st.caption("此按钮只读取当前已保存的预测 JSON 并导出文件，不会训练模型，也不会生成新的预测。")
fresh = bool(forecast and target and market_date and target > market_date)
if not fresh:
    st.info("当前没有可确认仍然面向未来交易日的预测，暂时禁用新文章导出，以免把过期走势发到公众号。先完成上面的统一每日流程。")
if st.button("生成公众号 HTML + Markdown + 曲线 PNG", disabled=not fresh, use_container_width=True):
    with st.spinner("正在用已保存的预测曲线生成公众号稿件……"):
        result = run_command([
            sys.executable, "scripts/generate_wechat_article.py",
            "--forecast", "reports/intraday/QQQM_intraday_forecast_latest.json",
            "--db", "data/market_data.db",
        ], timeout=300)
    st.session_state["wechat_article_result"] = result
    if result["returncode"] == 0:
        st.success("公众号草稿已导出。它不会自动发布，最终内容仍由你检查。")
        st.rerun()
    else:
        st.error("导出失败。请查看程序输出并检查预测日期、曲线字段或绘图依赖。")
        st.code((result.get("stderr") or result.get("stdout") or "无详细错误信息")[-6000:])

if target:
    stamp = target.replace("-", "")
    html_path = ARTICLES / f"QQQM_forecast_{stamp}_wechat.html"
    md_path = ARTICLES / f"QQQM_forecast_{stamp}_wechat.md"
    png_path = ARTICLES / f"QQQM_forecast_{stamp}_curve.png"
    if html_path.exists() and md_path.exists() and png_path.exists():
        st.success(f"已找到目标日 {target} 的文章草稿。")
        if png_path.exists():
            st.image(str(png_path), caption="公众号曲线图（PNG）", use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.download_button("下载公众号 HTML", data=html_path.read_bytes(), file_name=html_path.name,
                           mime="text/html", use_container_width=True)
        c2.download_button("下载 Markdown", data=md_path.read_bytes(), file_name=md_path.name,
                           mime="text/markdown", use_container_width=True)
        c3.download_button("下载曲线 PNG", data=png_path.read_bytes(), file_name=png_path.name,
                           mime="image/png", use_container_width=True)
        with st.expander("在网页里预览文章"):
            import streamlit.components.v1 as components
            components.html(html_path.read_text(encoding="utf-8"), height=1100, scrolling=True)
    else:
        archive_html = ROOT / "reports" / f"QQQM_forecast_{stamp}_wechat.html"
        if archive_html.exists():
            st.caption(f"发现旧版历史文章：{archive_html.relative_to(ROOT)}。这是历史留档，不代表目前保存的主预测路径。")
            st.download_button("下载这篇历史文章", data=archive_html.read_bytes(), file_name=archive_html.name,
                               mime="text/html", key="download_legacy_article")

st.divider()
with st.expander("进阶：复盘某个旧预测，并实验加入 K 线特征", expanded=False):
    st.markdown("选择一个**已保存的历史预测快照**。该实验会计算开盘、最高、最低、收盘等误差，并比较基础特征与 K 线特征的 Ridge 候选模型；不会生成下一交易日预测。")
    archive_dir = ROOT / "reports" / "forecasts" / "archive"
    candidates = list(archive_dir.glob("*.json")) if archive_dir.exists() else []
    candidates += [p for p in (ROOT / "reports").glob("QQQM_forecast_*.json") if p.is_file()]
    candidates = sorted({p.resolve() for p in candidates if p.is_file()}, key=lambda p: p.name)
    if not candidates:
        st.info("目前没有可选的旧预测 JSON。建议在每天预测时保留快照，然后再做公平复盘。")
    else:
        labels: dict[str, Path] = {f"{p.name} · {p.parent.relative_to(ROOT)}": p for p in candidates}
        chosen_label = st.selectbox("历史预测快照", list(labels.keys()))
        chosen_path = labels[chosen_label]
        chosen_payload = read_json(chosen_path) or {}
        default_day = chosen_payload.get("target_date") or chosen_payload.get("forecast_date_weekday_estimate") or (market_date or date.today().isoformat())
        day = st.text_input("要复盘的交易日（YYYY-MM-DD）", value=str(default_day), key="review_target_date")
        st.caption(f"读取文件：{chosen_path.relative_to(ROOT)}")
        if st.button("运行历史误差复盘 + K 线特征对照", use_container_width=True):
            result = run_command([
                sys.executable, "review_qqqm_day.py", "--target-date", day,
                "--forecast-file", str(chosen_path.relative_to(ROOT)), "--reports-dir", "reports",
            ], timeout=900)
            st.session_state["single_day_review_result"] = result
            if result["returncode"] == 0:
                st.success("历史复盘完成。")
            else:
                st.error("复盘未完全成功，请查看数据源/日期/数据库报错。")
            st.code((result.get("stdout") or "")[-8000:])
            if result.get("stderr"):
                st.warning((result.get("stderr") or "")[-3000:])

st.divider()
st.markdown(
    "**发布提醒：** HTML 是排版草稿，不会自动发到微信公众号。通常可在浏览器打开后复制正文；如果公众号编辑器没有保留内嵌图片，就使用单独的 PNG 上传。发出前先检查行情截止日期、目标交易日、新闻来源和风险提示。"
)
