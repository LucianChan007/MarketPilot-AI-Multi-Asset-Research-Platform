from __future__ import annotations

"""Patch app/dashboard.py so a previously-calculated portfolio auto-recalculates
when the quote inputs change. Makes a backup and only writes when the exact
expected source block is present.
"""

import ast
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "dashboard.py"

OLD = '''            if calculate_clicked:
                try:
                    with st.spinner("正在校验交易顺序并计算持仓……"):
                        calculation = calculate_portfolio(
                            raw_transactions, current_prices=current_prices
                        )
                    st.session_state[result_key] = {
                        "transactions": calculation.transactions,
                        "positions": calculation.positions,
                        "warnings": calculation.warnings,
                        "price_signature": price_signature,
                    }
                    st.success("持仓计算完成。")
                except PortfolioValidationError as exc:
                    st.error("无法完成持仓计算，请检查交易记录。")
                    for error in exc.errors:
                        st.write(f"- {error}")
                except Exception as exc:
                    st.error(f"持仓计算发生意外错误：{type(exc).__name__}: {exc}")

            calculated = st.session_state.get(result_key)
'''

NEW = '''            calculated_before = st.session_state.get(result_key)
            auto_recalculate = (
                calculated_before is not None
                and calculated_before.get("price_signature") != price_signature
            )

            if calculate_clicked or auto_recalculate:
                try:
                    status_message = (
                        "估值价格已更新，正在自动重算持仓……"
                        if auto_recalculate and not calculate_clicked
                        else "正在校验交易顺序并计算持仓……"
                    )
                    with st.spinner(status_message):
                        calculation = calculate_portfolio(
                            raw_transactions, current_prices=current_prices
                        )
                    st.session_state[result_key] = {
                        "transactions": calculation.transactions,
                        "positions": calculation.positions,
                        "warnings": calculation.warnings,
                        "price_signature": price_signature,
                    }
                    if auto_recalculate and not calculate_clicked:
                        st.success("估值价格已变化，持仓市值与盈亏已自动更新。")
                    else:
                        st.success("持仓计算完成。")
                except PortfolioValidationError as exc:
                    st.error("无法完成持仓计算，请检查交易记录。")
                    for error in exc.errors:
                        st.write(f"- {error}")
                except Exception as exc:
                    st.error(f"持仓计算发生意外错误：{type(exc).__name__}: {exc}")

            calculated = st.session_state.get(result_key)
'''


def main() -> int:
    if not TARGET.exists():
        print(f"找不到目标文件：{TARGET}")
        return 1
    original = TARGET.read_text(encoding="utf-8")
    if NEW in original:
        print("自动重算逻辑似乎已经存在，无需重复修改。")
        return 0
    if original.count(OLD) != 1:
        print("没有找到唯一的预期代码块。为避免误改，未写入任何文件。")
        print("请检查 app/dashboard.py 中 calculate_clicked 附近的代码。")
        return 2

    updated = original.replace(OLD, NEW, 1)
    try:
        ast.parse(updated, filename=str(TARGET))
    except SyntaxError as exc:
        print(f"更新后的 dashboard.py 语法检查失败：{exc}")
        return 3

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"dashboard.py.{stamp}.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(updated, encoding="utf-8", newline="")
    print("已启用行情估值变化后的自动重算。")
    print(f"备份文件：{backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
