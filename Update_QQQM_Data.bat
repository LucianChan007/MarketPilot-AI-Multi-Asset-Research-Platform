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
echo [QQQM] 正在同步行情、新闻/宏观数据并更新预测反馈……
"%~dp0.venv\Scripts\python.exe" -m app.daily_automation
set "UPDATE_EXIT=%ERRORLEVEL%"
if not "%UPDATE_EXIT%"=="0" (
  echo.
  echo [提示] 有同步或分析步骤失败。查看 reports\automation\latest_status.json 及最新日志。
) else (
  echo.
  echo [完成] 工作流结束。请刷新网页查看实际最新行情日期与数据来源。
)
echo.
pause
exit /b %UPDATE_EXIT%
