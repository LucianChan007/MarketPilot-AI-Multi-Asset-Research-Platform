@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

call "%~dp0Prepare_QQQM_Environment.bat"
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo   QQQM 智能投资研究助手
echo   正在同步行情、新闻/宏观信息，并复盘/更新预测……
echo ============================================================
echo [说明] 若外部 API 不可用，程序会显示同步失败并保留最后可验证的数据。
echo.
"%~dp0.venv\Scripts\python.exe" -m app.daily_automation
if errorlevel 1 (
  echo.
  echo [提示] 本次更新存在失败步骤。网页仍会启动，请查看 reports\automation\latest_status.json 和终端日志。
)
echo.
echo [QQQM] 正在启动网页：http://localhost:8501
echo 请保持此终端窗口开启；按 Ctrl+C 可停止网页服务。
echo.
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0dashboard.py" --server.address 127.0.0.1 --server.port 8501
set "APP_EXIT=%ERRORLEVEL%"
echo.
echo [QQQM] 网页服务已退出，退出码 %APP_EXIT%。
pause
exit /b %APP_EXIT%
