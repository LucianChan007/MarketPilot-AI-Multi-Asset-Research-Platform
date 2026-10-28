# MarketPilot AI · 美股多资产研究台

MarketPilot AI 是一个面向个人研究与学习的本地美股研究工作台。V2.2.1 集成修复版将 QQQM、AAPL、MSFT、NVDA 作为四个平行研究标的，并保留添加更多股票/ETF 的能力。各标的分别保存行情、预测快照、回测结果和预测复盘，不把不同标的的数据混在一起。

> 本项目仅用于研究与学习，不是自动交易系统，不会自动下单，也不构成投资建议。行情供应商可能限流、延迟或调整接口；必须以页面显示的交易日期、来源和状态为准。

## Windows 快速启动

1. 将 ZIP 解压至独立的英文路径，例如 `E:\Projects\MarketPilot_AI`。不要覆盖旧项目，避免覆盖已有数据库、交易记录或报告。
2. 如需保留旧项目的数据源配置，手动复制旧项目根目录的 `.env` 到新目录根目录。不要复制旧的 `.venv`，也不要把 API Key 发到聊天或提交到 GitHub。没有配置文件时，可复制 `.env.example` 为 `.env`。
3. 双击 `Start_MarketPilot_AI.bat`。第一次启动会检查/创建 Python 3.11 虚拟环境并安装 `requirements.txt`，需要联网。启动只打开网页，不会自动同步、训练或预测。
4. 浏览器访问终端显示的网址，通常是 `http://localhost:8501`。登录后默认进入四资产总览。首次启动会在本地把随包的 QQQM 历史 CSV 导入缺失的旧版行情数据库，不会因此发起网络请求。
5. 第一次先点击首页的「仅更新行情」，检查每只资产的最新日线日期、分时日期、数据源、接口尝试结果和错误说明。行情成功更新之后，再按需要运行「完整多资产研究」。

如果启动失败，可在项目目录运行：

```powershell
.\Prepare_MarketPilot_Environment.bat
```

或先阅读 `V2_使用手册_中文.md` 的故障排查部分。请使用项目虚拟环境，不要误用 MSYS2 或其他 Python 环境。

## V2.2.1 集成修复（本轮）

- 修复首页与多资产引擎使用两套数据库、首页按钮仍调用旧版 QQQM 工作流的问题。
- 全新安装时从随包 `data/QQQM_tiingo_history.csv` 初始化旧版 QQQM 历史库，确保老页面有可读历史数据；该导入明确标记为历史缓存而非实时行情，并且不覆盖已有数据。
- 默认首页改为 QQQM、AAPL、MSFT、NVDA 四资产平行总览；「仅更新行情」调用多资产同步引擎。
- 新版 QQQM 日线成功同步后，会将已保存的日线桥接至旧版数据库，旧版账本、预测和分析页面仍可继续读取。
- 新增本地数据初始化测试与 Streamlit 页面冒烟测试；请在已安装依赖的本机执行页面冒烟测试，检查实际渲染。

## V2.2 原有重点改进

- 独立「仅更新行情」流程：只同步行情、检查日期并记录错误，不跑新闻分析或预测。
- Yahoo Finance 直接图表接口失败时，额外尝试 `yfinance` 客户端路径。两者仍是同一 Yahoo 数据来源，不代表两个独立供应商。
- 如果可用的预测同源行情仍过旧，会尝试 Stooq 的日线展示兜底。Stooq 数据不与 Yahoo/EODHD 的分时拼接，也不会作为盘中预测的数据来源。
- 页面分开显示最新可用日线、预测同源日线、完整分时日期和最近抓取时间；旧预测快照不会被标记为新预测。
- 默认四资产平行展示：QQQM、AAPL、MSFT、NVDA。SPY 暂时保留为未启用资产，可按需启用。
- 保留手动触发数据同步、完整研究、模型评估和公众号/视频内容生成的工作方式；打开网页本身不会启动昂贵任务。

## 核心目录

- `dashboard.py`：统一 Streamlit 入口。
- `app/multi_asset_engine.py`：多资产行情、数据库、模型、预测、评估和复盘。
- `app/pages/multi_asset_research.py`：多资产研究页面。
- `config/instruments.json`：启用/添加研究标的。
- `data/multi_asset_research.db`：V2 多资产独立数据库，首次使用时创建。
- `reports/multi_asset/`：状态、预测快照、曲线、评估和新闻分析结果。
- `V2_使用手册_中文.md`：安装、使用、添加股票、行情故障排查和限制说明。
- `V2_测试报告_20261028.md`：当前已执行的测试范围和未验证事项。

## 开发与测试

在已安装项目依赖的环境中执行：

```powershell
.\.venv\Scripts\python.exe -m py_compile dashboard.py app\multi_asset_engine.py app\pages\multi_asset_research.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本轮隔离环境回归测试为 175 项，其中 174 项通过、1 项 Streamlit AppTest 因构建环境没有安装 Streamlit 且无法联网安装而跳过。已通过核心文件语法检查、数据引导测试和无联网模拟页面执行；真实 Streamlit 渲染、用户网络和行情供应商连通性仍须在已安装依赖的本机执行 `python -m unittest tests.test_streamlit_page_smoke -v` 验收。
