# 本次复试演示版变更记录

版本：2026-10-23 · QQQM AI Investment Research Assistant

## 新增与调整

- 统一品牌名为“QQQM 智能投资研究助手 / QQQM AI Investment Research Assistant”。
- 新增默认首页“产品介绍与演示路线”，集中说明产品定位、痛点、功能闭环、演示路径、市场验证阶段和风险边界。
- 保留原有统一 Streamlit 入口和现有研究功能，不重写行情、持仓、回测或模型逻辑。
- 新增 `PROJECT_ONE_PAGER.md`，便于复试前快速复习项目逻辑。
- 新增 `DEMO_RECORDING_GUIDE.md`，包含 3 分钟讲解词、Windows/OBS 录制步骤与交付检查表。
- 保留本地行情数据库、已有本地账号及交易账本数据，便于延续原项目状态。

## 验证情况

- Python 文件已通过 `compileall` 语法编译检查。
- 本地 SQLite 用户数据库通过 `PRAGMA integrity_check`，并确认含有既有账号和交易记录。
- 当前执行环境无法访问 Python 包索引，因此未能在此环境安装 Streamlit 等依赖，也未能完成真实浏览器启动或全量 pytest；首次运行需在用户 Windows 环境安装 `requirements.txt` 并完成启动检查。

## 重要说明

这是现有项目的演示体验与文档增强版，不代表预测模型已获得新的准确率提升，也不代表产品已通过市场验证。模型表现以本地实际评估报告为准。
