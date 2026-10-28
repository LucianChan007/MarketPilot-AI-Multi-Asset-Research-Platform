$ErrorActionPreference = 'Stop'

# Register the QQQM intraday learning cycle as a separate Windows scheduled task.
# This script intentionally uses ASCII-only text for Windows PowerShell 5.1 compatibility.

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$CycleScript = Join-Path $ProjectRoot 'scripts\run_intraday_learning_cycle.py'
$TaskName = 'QQQM Investment Assistant - Intraday Learning Cycle'

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Project virtual-environment Python was not found: $PythonPath"
}

if (-not (Test-Path -LiteralPath $CycleScript -PathType Leaf)) {
    throw "Intraday learning cycle script was not found: $CycleScript"
}

$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$RunAt = (Get-Date).Date.AddHours(8).AddMinutes(50)

$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument '-m scripts.run_intraday_learning_cycle' `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At $RunAt

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::FromHours(2))

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description 'Runs the QQQM intraday forecast and learning cycle after daily data sync.' `
    -Force | Out-Null

$RegisteredTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction Stop

Write-Host ''
Write-Host 'Scheduled task registered successfully.' -ForegroundColor Green
Write-Host ("Task name: {0}" -f $RegisteredTask.TaskName)
Write-Host ("Project root: {0}" -f $ProjectRoot)
Write-Host ("Program: {0}" -f $PythonPath)
Write-Host 'Arguments: -m scripts.run_intraday_learning_cycle'
Write-Host 'Schedule: Daily at 08:50 (Windows local time)'
Write-Host ("Next run: {0}" -f $TaskInfo.NextRunTime)
Write-Host ("Last result: {0}" -f $TaskInfo.LastTaskResult)
Write-Host ''
Write-Host 'Note: the task runs under the current Windows user and requires the user session to be available.'
