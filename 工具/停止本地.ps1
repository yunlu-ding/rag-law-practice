<#
.SYNOPSIS
    停止本地服务。

.DESCRIPTION
    默认只停后端（这样数据库里的东西都还在，下次启动更快）。
    加 -All 连 PostgreSQL 一起停——它占内存，不用的时候可以关掉。

.EXAMPLE
    .\停止本地.ps1
    .\停止本地.ps1 -All

.NOTE
    ⚠️ 本文件必须保存为「UTF-8 带 BOM」，
    否则 PowerShell 5.1 会把中文按 GBK 解读成乱码并报语法错误。
#>

param(
    [switch]$All
)

$ErrorActionPreference = 'Continue'

$base    = 'D:\丁云璐\个人\Agent 产品项目'
$pgbin   = 'D:\anaconda\envs\pg\Library\bin'
$pgdata  = Join-Path $base 'pgdata'
$logDir  = Join-Path $base 'vibe-rag\logs'
$pidFile = Join-Path $logDir 'backend.pid'
$port    = 8000

Write-Host ''
Write-Host '===== 停止 =====' -ForegroundColor Cyan

# ---------- 后端 ----------
$stopped = $false
if (Test-Path -LiteralPath $pidFile) {
    $backendPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($backendPid -and (Get-Process -Id $backendPid -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $backendPid -Force -ErrorAction SilentlyContinue
        Write-Host "  后端已停止（PID $backendPid）" -ForegroundColor Green
        $stopped = $true
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    # 兜底：按端口找进程。pid 文件可能丢了（比如手动关过窗口）。
    $conn = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        foreach ($item in $conn) {
            Stop-Process -Id $item.OwningProcess -Force -ErrorAction SilentlyContinue
        }
        Write-Host "  后端已停止（按端口 $port 找到并结束）" -ForegroundColor Green
    } else {
        Write-Host '  后端本来就没在运行' -ForegroundColor DarkGray
    }
}

# ---------- PostgreSQL ----------
if ($All) {
    & "$pgbin\pg_ctl.exe" -D $pgdata stop 2>&1 | Out-Null
    Start-Sleep -Milliseconds 800
    $ready = & "$pgbin\pg_isready.exe" -h 127.0.0.1 -p 5432 2>&1
    if ($ready -match 'accepting connections') {
        Write-Host '  PostgreSQL 还在运行（可能有其他程序连着它）' -ForegroundColor Yellow
    } else {
        Write-Host '  PostgreSQL 已停止' -ForegroundColor Green
    }
} else {
    Write-Host '  PostgreSQL 保持运行（加 -All 可以一起停）' -ForegroundColor DarkGray
}

Write-Host ''
