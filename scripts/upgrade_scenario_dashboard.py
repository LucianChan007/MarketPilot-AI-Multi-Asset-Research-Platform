from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

path = Path("app/dashboard.py")
if not path.is_file():
    raise SystemExit(f"找不到 {path}。请从项目根目录运行本脚本。")

source = path.read_text(encoding="utf-8")
original = source

mapping_anchor = '                            "scenario_unrealized_pnl_pct": "情景未实现收益率(%)",'
mapping_additions = '''
                            "realized_pnl": "历史已实现盈亏",
                            "dividend_income": "历史净分红",
                            "current_cumulative_net_pnl": "当前累计净盈亏",
                            "scenario_cumulative_net_pnl": "情景累计净盈亏",
                            "cumulative_pnl_change": "累计盈亏变化",'''
if '"scenario_cumulative_net_pnl": "情景累计净盈亏"' not in source:
    if mapping_anchor not in source:
        raise SystemExit("找不到价格情景字段映射，未修改 dashboard.py。")
    source = source.replace(mapping_anchor, mapping_anchor + mapping_additions, 1)

config_anchor = '                                "情景未实现收益率(%)": st.column_config.NumberColumn(format="%+.2f"),'
config_additions = '''
                                "历史已实现盈亏": st.column_config.NumberColumn(format="%+.2f"),
                                "历史净分红": st.column_config.NumberColumn(format="%+.2f"),
                                "当前累计净盈亏": st.column_config.NumberColumn(format="%+.2f"),
                                "情景累计净盈亏": st.column_config.NumberColumn(format="%+.2f"),
                                "累计盈亏变化": st.column_config.NumberColumn(format="%+.2f"),'''
if '"情景累计净盈亏": st.column_config.NumberColumn' not in source:
    if config_anchor not in source:
        raise SystemExit("找不到价格情景数字格式配置，未修改 dashboard.py。")
    source = source.replace(config_anchor, config_anchor + config_additions, 1)

caption_anchor = '                    st.subheader("交易历史与已实现盈亏")'
caption_block = '''                    st.caption(
                        "情景累计净盈亏 = 历史已实现盈亏 + 情景下未实现盈亏 + 历史净分红。"
                        "累计盈亏变化展示相对于当前估值的变化。"
                        "这是基于假设价格的情景分析，不是未来收益预测；不同币种分开计算。"
                    )

'''
if '情景累计净盈亏 = 历史已实现盈亏' not in source:
    if caption_anchor not in source:
        raise SystemExit("找不到价格情景表格后的说明位置，未修改 dashboard.py。")
    source = source.replace(caption_anchor, caption_block + caption_anchor, 1)

if source == original:
    print("dashboard.py 看起来已经应用了情景分析更新，没有重复修改。")
    raise SystemExit(0)

compile(source, str(path), "exec")
backup = path.with_name(f"dashboard.py.{datetime.now():%Y%m%d_%H%M%S}.bak")
shutil.copy2(path, backup)
path.write_text(source, encoding="utf-8", newline="")
print(f"已更新：{path}")
print(f"原文件备份：{backup}")
print("只修改了情景表格字段映射、数字格式与说明；其余 dashboard 内容保持原样。")
