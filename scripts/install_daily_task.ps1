$ErrorActionPreference = 'Stop'

# Install or update the QQQM daily automation task for the current Windows user.
# Schedule time follows the computer's current Windows time zone.

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$modulePath = Join-Path $projectRoot 'app\daily_automation.py'
$taskName = 'QQQM Investment Assistant - Daily Data Sync'
$automationLogDir = Join-Path $projectRoot 'reports\automation'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project virtual-environment Python was not found: $pythonPath"
}

if (-not (Test-Path -LiteralPath $modulePath -PathType Leaf)) {
    throw "Automation module was not found: $modulePath"
}

New-Item -ItemType Directory -Force -Path $automationLogDir | Out-Null

# Preflight: validate imports and command construction without calling data sources or AI.
Push-Location $projectRoot
try {
    & $pythonPath -m app.daily_automation --dry-run
    if ($LASTEXITCODE -ne 0) {
        throw "Automation dry-run failed with exit code $LASTEXITCODE. Task was not registered."
    }
}
finally {
    Pop-Location
}

# 08:30 local time, Tuesday through Saturday. On a UTC+8 computer this normally
# follows the previous US trading day's close, while allowing time for daily bars to publish.
$runAt = (Get-Date).Date.AddHours(8).AddMinutes(30)
$days = @('Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $runAt

# Use the project's virtual-environment Python directly; no cmd.exe or .bat file is needed.
$action = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument '-m app.daily_automation --period 1y' `
    -WorkingDirectory $projectRoot

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'Sync market daily bars, Fed/BLS news, FRED macro data, and conditionally generate AI reports.' `
    -Force | Out-Null

Write-Host "Task installed: $taskName" -ForegroundColor Green
Write-Host 'Schedule: Tuesday-Saturday at 08:30 in the Windows time zone.'
Write-Host "Project: $projectRoot"
Write-Host "Logs: $automationLogDir"
Write-Host 'The task runs as the current user when that user is logged in.'
Write-Host 'It uses --period 1y; AI reports run only when the automation detects relevant data changes.'
