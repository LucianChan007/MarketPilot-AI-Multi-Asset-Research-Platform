param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$PatchRoot = $PSScriptRoot
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

Write-Host "升级包目录：$PatchRoot"
Write-Host "目标项目目录：$ProjectRoot"

# Guard against accidentally installing into the extracted package directory.
if (-not (Test-Path (Join-Path $ProjectRoot 'app\intraday_forecast.py'))) {
    throw "目标目录下没有 app\intraday_forecast.py：$ProjectRoot。请从项目根目录运行，并通过 -ProjectRoot 指定目录。"
}
if (-not (Test-Path (Join-Path $ProjectRoot 'tests'))) {
    throw "目标目录下没有 tests 文件夹：$ProjectRoot。为避免误装入临时解压目录，安装已停止。"
}
if ((Resolve-Path $ProjectRoot).Path -eq (Resolve-Path $PatchRoot).Path) {
    throw '目标项目目录与升级包目录相同，拒绝安装。'
}

$relativeFiles = @(
    'app\intraday_forecast.py',
    'app\services\market_calendar.py',
    'tests\test_intraday_forecast.py',
    'tests\test_market_calendar.py'
)
$BackupTag = Get-Date -Format 'yyyyMMdd_HHmmss'

foreach ($relative in $relativeFiles) {
    $source = Join-Path $PatchRoot $relative
    $destination = Join-Path $ProjectRoot $relative
    if (-not (Test-Path $source)) {
        throw "升级包缺少文件：$source"
    }
    $destinationDir = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
    if (Test-Path $destination) {
        $backup = "$destination.bak_datefix_v2_$BackupTag"
        Copy-Item $destination $backup -Force
        Write-Host "已备份：$backup"
    }
    Copy-Item $source $destination -Force
    Write-Host "已安装：$relative"
}

# Remove only bytecode caches for the two replaced Python modules, so the
# interpreter cannot reuse bytecode compiled from the previous source version.
$pycacheTargets = @(
    (Join-Path $ProjectRoot 'app\__pycache__\intraday_forecast.*.pyc'),
    (Join-Path $ProjectRoot 'app\services\__pycache__\market_calendar.*.pyc')
)
foreach ($pattern in $pycacheTargets) {
    Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Remove-Item -Force
}

$installedForecast = Get-Content (Join-Path $ProjectRoot 'app\intraday_forecast.py') -Raw
if ($installedForecast -notmatch 'target_day\s*=\s*_parse_date_value\(forecast\.get\("forecast_date_weekday_estimate"\)\)') {
    throw '安装后校验失败：目标 app\intraday_forecast.py 仍未包含修复后的日期解析代码。'
}
if ($installedForecast -match 'target_day\s*=\s*date\.fromisoformat\(str\(forecast\.get\("forecast_date_weekday_estimate"\)\)\)') {
    throw '安装后校验失败：旧版 date.fromisoformat 调用仍然存在。'
}

Write-Host ''
Write-Host '日期解析修复 v2 已安装并验证目标文件。' -ForegroundColor Green
Write-Host "目标文件：$(Join-Path $ProjectRoot 'app\intraday_forecast.py')"
Write-Host '数据库、API 配置、报告和其他业务代码未修改。'
