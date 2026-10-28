from __future__ import annotations

import ast
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "app" / "dashboard.py"

IMPORT_LINE = "from app.portfolio_ai import generate_portfolio_analysis"
INSERT_ANCHOR = '                transaction_results = calculated["transactions"].copy()'
FEATURE_MARKER = '                # --- Personal portfolio AI analysis (explicit consent required) ---'

FEATURE_BLOCK = '''

                # --- Personal portfolio AI analysis (explicit consent required) ---
                if not positions.empty:
                    ai_report_state_key = f"portfolio_ai_report_{file_fingerprint}"
                    ai_consent_key = f"portfolio_ai_consent_{file_fingerprint}"
                    with st.expander("个人持仓 DeepSeek 分析", expanded=False):
                        st.markdown(
                            "生成一份结合实际持仓、交易汇总、价格情景及本地已保存市场/宏观/新闻证据的分析报告。"
                        )
                        st.warning(
                            "隐私说明：勾选同意后，程序会将聚合持仓、按标的/币种/交易类型汇总的交易统计、"
                            "价格情景，以及本地保存的市场/宏观/新闻上下文发送给 DeepSeek。"
                            "不会发送交易 ID、备注或逐笔交易日期。请求由外部模型服务处理。"
                        )
                        result_prices_current = calculated.get("price_signature") == price_signature
                        if not result_prices_current:
                            st.warning("当前估值价格与已计算结果不同。请先点击“计算持仓与盈亏”，再生成 AI 报告。")
                        consent = st.checkbox(
                            "我已了解以上数据范围，并同意将这些汇总信息发送给 DeepSeek。",
                            key=ai_consent_key,
                            value=False,
                        )
                        generate_ai_clicked = st.button(
                            "生成我的持仓分析",
                            type="primary",
                            key=f"generate_portfolio_ai_{file_fingerprint}",
                            disabled=(not consent) or (not result_prices_current),
                        )
                        if generate_ai_clicked:
                            try:
                                with st.spinner("正在整理持仓摘要并请求 DeepSeek……"):
                                    ai_report_path = generate_portfolio_analysis(
                                        positions,
                                        transaction_results,
                                    )
                                st.session_state[ai_report_state_key] = str(ai_report_path)
                                st.success(f"分析报告已保存：{ai_report_path.name}")
                            except Exception as exc:
                                st.error(f"持仓 AI 分析失败：{type(exc).__name__}: {exc}")

                        saved_ai_report = st.session_state.get(ai_report_state_key)
                        if saved_ai_report:
                            ai_path = Path(saved_ai_report)
                            if ai_path.exists():
                                st.download_button(
                                    "下载个人持仓分析报告",
                                    data=ai_path.read_bytes(),
                                    file_name=ai_path.name,
                                    mime="text/markdown",
                                    key=f"download_portfolio_ai_{file_fingerprint}",
                                )
                                with st.expander("查看最近生成的个人持仓分析"):
                                    st.markdown(ai_path.read_text(encoding="utf-8"))
'''


def main() -> int:
    if not DASHBOARD.exists():
        raise SystemExit(f"找不到页面文件：{DASHBOARD}")

    source = DASHBOARD.read_text(encoding="utf-8")
    original = source

    if FEATURE_MARKER in source:
        print("个人持仓 AI 区域已存在，未做修改。")
        return 0

    tree = ast.parse(source, filename=str(DASHBOARD))
    portfolio_import = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "app.portfolio":
            portfolio_import = node
            break

    if portfolio_import is None:
        raise SystemExit("未找到 app.portfolio 导入块，已停止且未修改文件。")
    if INSERT_ANCHOR not in source:
        raise SystemExit("未找到持仓分析插入点，已停止且未修改文件。")

    if IMPORT_LINE not in source:
        lines = source.splitlines(keepends=True)
        insert_index = portfolio_import.end_lineno
        lines.insert(insert_index, IMPORT_LINE + "\n")
        source = "".join(lines)

    if INSERT_ANCHOR not in source:
        raise SystemExit("导入调整后未找到插入锚点，已停止且未修改文件。")

    source = source.replace(INSERT_ANCHOR, INSERT_ANCHOR + FEATURE_BLOCK, 1)
    ast.parse(source, filename=str(DASHBOARD))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DASHBOARD.with_name(f"dashboard.py.{stamp}.bak")
    shutil.copy2(DASHBOARD, backup)
    DASHBOARD.write_text(source, encoding="utf-8", newline="\n")

    print(f"已更新：{DASHBOARD}")
    print(f"备份：{backup}")
    print("Python 语法检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
