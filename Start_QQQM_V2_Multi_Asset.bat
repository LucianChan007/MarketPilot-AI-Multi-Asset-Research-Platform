@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo ============================================================
echo   QQQM 投资研究助手 V2：多股票研究与预测复盘
echo   启动网页，不会在打开时自动同步行情或训练模型
echo ============================================================
echo.
call "%~dp0Prepare_QQQM_Environment.bat"
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)
echo.
echo [V2] 正在启动网页：http://localhost:8501
echo 请保持此窗口开启。按 Ctrl+C 停止网页服务。
echo.
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0dashboard.py" --server.address 127.0.0.1 --server.port 8501
set "APP_EXIT=%ERRORLEVEL%"
echo.
echo [V2] 网页服务已退出，退出码 %APP_EXIT%。
pause
exit /b %APP_EXIT%
