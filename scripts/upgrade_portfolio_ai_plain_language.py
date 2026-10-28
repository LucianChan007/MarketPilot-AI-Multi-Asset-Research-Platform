from __future__ import annotations

import ast
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "app" / "dashboard.py"
MARKER = "# >>> QQQM PORTFOLIO AI PLAIN LANGUAGE V0.4.0 >>>"

PATTERN = re.compile(
    r'(?m)^(?P<indent>[ ]*)with st\.expander\("查看最近生成的个人持仓分析"\):\r?\n'
    r'(?P=indent)    st\.markdown\(ai_path\.read_text\(encoding="utf-8"\)\)'
)


def main() -> int:
    if not DASHBOARD.is_file():
        raise SystemExit(f"找不到 {DASHBOARD}。请在项目根目录运行脚本。")

    source = DASHBOARD.read_text(encoding="utf-8")
    if MARKER in source:
        print("个人持仓 AI 简明展示已安装，未重复修改。")
        return 0

    match = PATTERN.search(source)
    if not match:
        raise SystemExit(
            "没有找到预期的个人持仓 AI 报告展示代码，未修改 dashboard.py。\n"
            "请检查 app/dashboard.py 中‘查看最近生成的个人持仓分析’区域。"
        )

    indent = match.group("indent")
    child = indent + "    "
    grandchild = child + "    "
    replacement_lines = [
        f"{indent}{MARKER}",
        f'{indent}report_content = ai_path.read_text(encoding="utf-8")',
        f'{indent}summary_start = report_content.find("## 一眼看懂")',
        f'{indent}ai_section_start = report_content.find("## AI 解读：")',
        f"{indent}if summary_start >= 0:",
        f"{child}summary_end = (",
        f"{child}    ai_section_start",
        f"{child}    if ai_section_start > summary_start",
        f"{child}    else len(report_content)",
        f"{child})",
        f"{child}summary_content = report_content[summary_start:summary_end].strip()",
        f"{child}st.markdown(summary_content)",
        f'{child}with st.expander("查看完整个人持仓分析报告"): ',
        f"{child}    st.markdown(report_content)",
        f"{indent}else:",
        f'{child}with st.expander("查看最近生成的个人持仓分析"): ',
        f"{child}    st.markdown(report_content)",
    ]
    replacement = "\n".join(line.rstrip() for line in replacement_lines)
    updated = source[:match.start()] + replacement + source[match.end():]

    try:
        ast.parse(updated, filename=str(DASHBOARD))
    except SyntaxError as exc:
        raise SystemExit(f"修改后语法检查失败，未写入文件：{exc}") from exc

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DASHBOARD.with_name(f"dashboard.py.{stamp}.bak")
    shutil.copy2(DASHBOARD, backup)
    DASHBOARD.write_text(updated, encoding="utf-8", newline="\n")

    print(f"已更新：{DASHBOARD}")
    print(f"原文件备份：{backup}")
    print("语法检查通过。新报告的‘一眼看懂’部分会优先显示，完整报告放入展开区域。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
