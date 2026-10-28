# 本次整合记录

## 保留并接入统一导航的核心功能

- `gui/dashboard.py`：每日总览、市场时间、盘中预测 / 实际复盘、用户级反馈与账本入口。
- `app/dashboard.py`：市场数据、技术指标、持仓、网格回测、宏观信息与新闻视图。
- `app/pages/`：下一交易日情景、模型学习复盘、盘中路径、持仓规划、AI Agent 控制台，以及新增的多平台视频脚本与配图页。
- `app/forecast_pipeline.py`、`app/intraday_forecast.py`、`app/forecast_feedback.py`、`app/forecast_training_data.py`、`scripts/run_champion_challenger.py`、`scripts/run_intraday_learning_cycle.py` 等：继续使用原有预测、反馈、验证与盘中路径学习逻辑。
- 现有 `data/market_data.db`、`data/users.db`、持仓流水和 `reports/` 历史报告：保留在本地整合包里，方便接着开发与复盘。

## 统一后的入口

`dashboard.py` 是唯一推荐入口。`Start_QQQM_Assistant.bat` 和 `scripts/start_qqqm_assistant.ps1` 只负责启动这个入口，不会在启动时偷偷执行同步、训练或预测。所有 Streamlit 页面共用一次本地登录与一套导航。

## 新增 / 修正

- `app/pages/video_script_studio.py`：独立的公开行情刷新、官方新闻同步、中文口播稿和两张配图导出；原有 Alpaca 学习闭环放在高级操作区。
- `scripts/video_market_data.py`：无需独立密钥的公开行情接口，分开保存日线与标记为 `yahoo` 的十五分钟分时，不会把公开行情冒充 IEX。
- `scripts/run_unified_daily_workflow.py`：调度已有日常同步、预测反馈和盘中学习流程。它是编排器，不另写一套相互竞争的价格预测算法。
- 旧公众号生成脚本已移至 `legacy/` 归档，不再参与 GUI 导航或每日工作流。
- `review_qqqm_day.py`：针对已完成交易日复盘原始预测，并比较基础日线特征与增加 K 线特征的候选模型。
- `DEPLOYMENT.md`：说明网页公开分享和微信小程序的路线及部署前安全要求。
- `requirements.txt`：合并成单一依赖清单；`.env.example` 只含占位符，不包含真实 Key。

## 旧文件处理原则

旧的收盘点预测脚本、旧评估脚本和旧安装修复脚本移入 `legacy/`，保留作追溯用，但不再作为推荐入口。旧的市场 / 持仓 Dashboard 不删除，仍从统一导航进入。移除的是重复嵌套的临时整合文件夹、Python 缓存和重复依赖文件，而不是交易或模型历史记录。

## 测试范围与限制

- 整个 Python 项目 `compileall` 通过。
- 自动化单元测试最初 120 项通过；本次状态诊断新增 4 项测试后，全套 136 项通过。
- 用现有的已保存预测快照对文章生成器做了本地烟雾测试，HTML / Markdown / PNG 可生成。
- 市场数据库和账号数据库通过 SQLite `PRAGMA integrity_check`。
- 本次沙箱没有实际启动浏览器访问 Streamlit，因此最终 GUI 外观与用户 Windows 本地的 API 权限，仍需在本地运行验证。


## 2026-10-21 视频工作流调整
- 统一导航已将“公众号文章生成与导出”替换为“多平台视频脚本与配图”。
- 新增 `scripts/generate_video_script.py`，生成纯中文口播稿、中文发布文案、真实完整分时复盘图与下一交易日情景图，并打包 ZIP。
- 如果最新日线已更新但本地完整分时仍停留在更早交易日，视频流程会停止使用过期曲线。第一张复盘图只叠加同日真实留档预测，或用户已发布预测图的透明还原，不会事后回算一条线冒充事前预测。
- 每日统一工作流不再尝试导出公众号文章。
- 若 Alpaca 401 导致盘中数据无法更新，GUI 显示鉴权错误和修复步骤；旧预测被明确标为过期，不伪装成下一交易日预测。


## 2026-10-21 复盘图修正补充
- 用户提供的2026年10月20日已发布预测图已归档至 `reports/intraday/archive/QQQM_2026-10-20_已发布预测图.png`。
- 复盘数据文件 `QQQM_intraday_forecast_2026-10-20_image_restored.json` 标明 `source_kind=published_chart_digitized`；蓝线必须来自真实十五分钟行情。橙线中间点按图片坐标估读及插值，需在图注和文案中说明近似性。
- 若没有同日预测原件，不生成事后回算对比线。


## 数据健康与模型流转说明（本次修订）
- 新增 `app/services/system_health.py` 和“数据更新与系统诊断”页面，明确区分市场数据日期与抓取时间、提供凭据存在性检查（绝不显示密钥内容）和最近流程状态。
- 首页增加只读数据状态区；没有新鲜预测时不再 `st.stop()`，持仓和其余研究功能继续可用。
- 视频脚本页面明确说明视频生成不训练模型；新增日线反馈/重新拟合、本地滚动评估和 Alpaca 盘中闭环的分离入口。
- 缺少 Alpaca Key/Secret 时禁用盘中闭环按钮，说明错误发生在数据同步阶段，而不是模型准确性评估阶段。
- 完整十五分钟交易日现在至少要求 24 个点，并覆盖 09:30 至 15:45；这比此前 18 个点的门槛更严格。


## 2026-10-22 公开行情盘中学习流程修正
- 新增 `scripts/run_public_intraday_learning_cycle.py`：不依赖 Alpaca，先刷新公开行情与新闻，再以同一分时来源结算历史前瞻预测、运行盘中岭回归候选和滚动回测，写入 `reports/public_intraday_learning_cycle/latest_status.json`。
- `app/intraday_forecast.py` 支持指定目标日期，以美东交易时段确定下一次尚未开始的完整交易日，避免美股盘前/盘中把目标日错标。
- `scripts/generate_video_script.py` 逐来源检查数据日期与来源一致性，不拼接供应商曲线；只有同日、同源、事前保存的 Ridge 候选才标记为机器学习主路径，否则标为历史统计情景。输出 JSON 包含模型来源、基准日期、目标日期和模型家族。
- GUI 主按钮现为“更新行情 → 复盘历史预测 → 训练盘中候选 → 生成曲线和视频包”，并展示样本数、路径 MAE、相对历史统计基准改善与预测来源。流程失败时仍可尝试生成缓存内容，但会明确提示并不冒充机器学习预测。
- 新增美东时间目标日边界测试、跨来源隔离测试和指定目标日集成测试。
