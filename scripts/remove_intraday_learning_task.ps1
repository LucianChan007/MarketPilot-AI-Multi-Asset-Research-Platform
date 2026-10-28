$ErrorActionPreference = "Stop"
$TaskName = "QQQM Investment Assistant - Intraday Learning Cycle"
$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $Task) {
    Write-Host "任务不存在：$TaskName"
    exit 0
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "已删除任务：$TaskName。原有 Daily Data Sync 未更改。" -ForegroundColor Green
