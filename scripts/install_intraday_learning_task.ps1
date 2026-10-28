$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    throw "找不到项目虚拟环境 Python：$PythonPath"
}

$TaskName = "QQQM Investment Assistant - Intraday Learning Cycle"
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "-m scripts.run_intraday_learning_cycle" `
    -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At "10:00AM"
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 45) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Sync QQQM 15Min IEX bars, evaluate archived forecasts, generate next-day path charts, and build an auditable prospective dataset." `
    -Force | Out-Null

Write-Host "已创建计划任务：$TaskName"
Write-Host "每天上午 10:00 按当前 Windows 本地时间运行。"
Write-Host "首次建议先手动运行：& `"$PythonPath`" -m scripts.run_intraday_learning_cycle"
