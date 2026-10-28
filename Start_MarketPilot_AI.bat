@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title MarketPilot AI - US Multi-Asset Research

echo ============================================================
echo   MarketPilot AI - 美股多资产研究台
echo   先登录本地账号，默认进入四资产总览
echo ============================================================
echo.

call "Prepare_MarketPilot_Environment.bat"
if errorlevel 1 (
    echo [错误] 环境准备失败，请阅读上方提示。
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m py_compile dashboard.py app\pages\research_home.py app\services\local_data_bootstrap.py app\multi_asset_engine.py app\pages\multi_asset_research.py
if errorlevel 1 (
    echo [错误] 核心文件语法检查失败。
    pause
    exit /b 1
)

.venv\Scripts\python.exe -c "from app.services.local_data_bootstrap import ensure_bundled_qqqm_history; r=ensure_bundled_qqqm_history(); print('[MarketPilot AI] 本地历史数据初始化：', r.get('reason'), '行数=', r.get('rows_added', 0), '最新日=', r.get('latest_date') or '未知')"
if errorlevel 1 (
    echo [警告] 本地历史数据初始化遇到问题，网页仍会启动，请检查首页提示。
)

echo [提示] 正在启动网页。启动时只初始化本地历史缓存，不联网同步行情、训练或生成预测。
.venv\Scripts\python.exe -m streamlit run dashboard.py
if errorlevel 1 (
    echo [错误] 网页启动失败，请检查 Python 环境和终端报错。
    pause
)
endlocal
