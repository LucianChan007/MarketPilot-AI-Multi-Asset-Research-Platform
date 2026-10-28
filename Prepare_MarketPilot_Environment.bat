@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo [MarketPilot AI] 检查 Windows Python 环境……
if not exist "%VENV_PY%" goto create_venv
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.platform == 'win32' and sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [MarketPilot AI] 现有 .venv 不是可用的 Windows Python 3.11+ 环境，准备重建。
  rmdir /s /q "%~dp0.venv" >nul 2>nul
  goto create_venv
)
goto install_deps

:create_venv
where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 --version >nul 2>nul
  if not errorlevel 1 (
    echo [MarketPilot AI] 正在通过 Windows Python Launcher 创建项目专用环境……
    py -3.11 -m venv "%~dp0.venv"
    if not errorlevel 1 if exist "%VENV_PY%" goto install_deps
  )
)
rem Fallback for Python.org installations when the Python Launcher is not on PATH.
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  echo [MarketPilot AI] 正在使用当前用户的 Python 3.11 创建项目专用环境……
  "%LocalAppData%\Programs\Python\Python311\python.exe" -m venv "%~dp0.venv"
  if not errorlevel 1 if exist "%VENV_PY%" goto install_deps
)
if exist "%ProgramFiles%\Python311\python.exe" (
  echo [MarketPilot AI] 正在使用系统 Python 3.11 创建项目专用环境……
  "%ProgramFiles%\Python311\python.exe" -m venv "%~dp0.venv"
  if not errorlevel 1 if exist "%VENV_PY%" goto install_deps
)
if exist "%ProgramFiles(x86)%\Python311\python.exe" (
  echo [MarketPilot AI] 正在使用系统 Python 3.11 创建项目专用环境……
  "%ProgramFiles(x86)%\Python311\python.exe" -m venv "%~dp0.venv"
  if not errorlevel 1 if exist "%VENV_PY%" goto install_deps
)
goto python_failed

:install_deps
if exist "%~dp0.venv\.qqqm_requirements_installed" goto ready
echo [MarketPilot AI] 首次安装项目依赖，网络较慢时请耐心等待……
"%VENV_PY%" -m ensurepip --upgrade >nul 2>nul
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto dependency_failed
>"%~dp0.venv\.qqqm_requirements_installed" echo installed

:ready
echo [MarketPilot AI] Python 环境与依赖检查通过。
exit /b 0

:python_failed
echo.
echo [错误] 没能找到可用的 Windows Python 3.11。
echo 请安装 64 位 Python 3.11，并勾选 Python Launcher 与 Add Python to PATH，然后重新运行此文件。
echo 安装地址：https://www.python.org/downloads/windows/
exit /b 1

:dependency_failed
echo.
echo [错误] 依赖安装失败，通常是网络连接或软件源访问问题。
echo 可以先检查网络，然后重新运行此文件；不要改用 C:\msys64\mingw64\bin\python.exe。
exit /b 1
