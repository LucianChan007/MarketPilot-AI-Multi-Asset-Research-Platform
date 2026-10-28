# MarketPilot AI V2.2.1 集成修复说明（2026-10-28）

## 发现的根因

1. 默认「研究驾驶舱」仍用旧版 QQQM 页面逻辑，只读取 `data/market_data.db` 的 `market_daily` 表。
2. 多资产引擎单独保存到 `data/multi_asset_research.db`，两个数据库没有在首页正确衔接。
3. ZIP 为保护个人数据而排除 SQLite 数据库，但没有在首次启动时将随包的 `data/QQQM_tiingo_history.csv` 初始化到旧版数据库，所以干净安装可能显示「本地最新行情未知」。
4. 首页的「立即同步并更新预测」原本调用旧版 `app.daily_automation`，并非新版四资产的行情同步工作流。

## 修复内容

- 新增 `app/services/local_data_bootstrap.py`：首次启动时本地初始化 SQLite 并导入随包 QQQM 历史 CSV；不联网、不覆盖已存在数据，并明确标记为 `bundled_tiingo_history_csv`。
- 在统一入口登录成功后执行本地数据引导，且首页独立运行时也可自检，因此无论从默认首页启动还是直接测试首页，都不会依赖预先存在的私有数据库文件。
- 重做默认研究驾驶舱：四资产平行展示；无本地行情时显示「等待首次同步」，有历史缓存时显示数据日期和来源；不把历史缓存说成实时行情。
- 首页「仅更新行情」现在调用 `app.multi_asset_engine.sync_market_data`，完整流程调用 `run_cycle`；每个标的的失败原因、日线/分时日期和预测准备度可展开查看。
- QQQM 新版多资产日线同步成功后，以有来源优先级的方式镜像进旧版日线数据库，保持原有 QQQM 研究工具兼容。
- 原有 QQQM 完整日常工作流保留在首页折叠区，个人账本、旧版预测/复盘、宏观和内容工具仍在原目录与导航中。
- 新增 5 项 `local_data_bootstrap` 测试与 1 项真实 Streamlit AppTest 页面冒烟测试。

## 测试结果与边界

- 完整 unittest：175 项，174 项通过、1 项 Streamlit AppTest 因构建环境未安装 Streamlit 且无可用网络下载安装而跳过。
- `py_compile` 与 `compileall`：通过。
- 本地初始化使用真实随包 CSV 的隔离临时数据库验证：导入 1505 条 QQQM 日线，最新日期为 2026-10-09、收盘参考价为 309.39；这是原始历史缓存日期，不声称是当前实时行情。
- 默认页面通过了无联网 Streamlit 模拟执行冒烟检查，但这不等同于真实浏览器 UI 验收。安装 `requirements.txt` 后请运行 `python -m unittest tests.test_streamlit_page_smoke -v` 执行真实 Streamlit AppTest。
- 不保证外部行情 API 一定能访问。若 Yahoo/yfinance、Stooq 或可选 EODHD 都被网络、套餐或限流拒绝，页面会保留历史数据并显示失败详情，而不是谎报成功。
