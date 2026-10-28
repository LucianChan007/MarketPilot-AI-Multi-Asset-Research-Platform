$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Dashboard = Join-Path $Root 'dashboard.py'

if (-not (Test-Path $Python)) {
    throw "项目虚拟环境未找到：$Python`n请先按 README.md 创建 .venv 并安装 requirements.txt。"
}
if (-not (Test-Path $Dashboard)) {
    throw "统一 Dashboard 入口未找到：$Dashboard"
}

Write-Host ''
Write-Host 'QQQM Investment Assistant · 统一工作台' -ForegroundColor Cyan
Write-Host '启动前更新行情、新闻/宏观数据并复盘预测；数据变化时生成 AI 报告。' -ForegroundColor Gray
Write-Host '如果更新失败，程序会保留旧数据并记录错误，不会伪装成实时行情。' -ForegroundColor Gray
Write-Host '启动地址：http://localhost:8501' -ForegroundColor Green
Write-Host ''
Set-Location $Root
Write-Host '正在检查并更新市场数据，请稍候……' -ForegroundColor Cyan
& $Python -m app.daily_automation
$SyncExitCode = $LASTEXITCODE
if ($SyncExitCode -ne 0) {
    Write-Warning '本次数据更新未全部成功。仍将启动网页；请查看 reports\automation\latest_status.json 及上方错误，网页会显示数据库中实际最新日期。'
}
Write-Host ''
Write-Host '正在启动网页……' -ForegroundColor Green
& $Python -m streamlit run $Dashboard
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    Write-Warning "Streamlit 已退出，退出码：$ExitCode。请查看上方终端错误。"
}
exit $ExitCode
