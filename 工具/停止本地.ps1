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
if (Test-Path -LiteralPath $pidFile) {
    $backendPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($backendPid -and (Get-Process -Id $backendPid -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $backendPid -Force -ErrorAction SilentlyContinue
        Write-Host "  后端已停止（PID $backendPid）" -ForegroundColor Green
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

# ⚠️ 按端口再查一遍是**无条件**的，不能只在"pid 文件那条路没成功"时才做。
#
# 原来这里写的是 `if (-not $stopped)`，于是一个真实的坑出现了：
#   · pid 文件里记的是 6632，杀掉它——"成功"；
#   · 但真正监听 8000 的是另一个进程（13720，和 6632 同一秒启动的另一个 python），
#     它活得好好的，端口一直没释放；
#   · 因为 `$stopped` 已经是 $true，这段清理**被跳过**，
#     `启动本地.ps1` 下次看到端口被占用又会"跳过启动"。
#   · 结果：**停止说成功了、启动说成功了、端口上跑的还是 13 天前那个进程。**
#     这一条比 pid 文件不准更隐蔽——两道脚本都报告成功。
#
# 所以顺序反过来：先按 pid 文件杀，再**无条件**按端口兜底。
# 后者才是"端口有没有真的空出来"的判据。
Start-Sleep -Milliseconds 500
$conn = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($conn) {
    foreach ($item in $conn) {
        # 排除掉自己（万一脚本正好跑在这个端口上，虽然不可能）
        if ($item.OwningProcess -ne $PID) {
            Stop-Process -Id $item.OwningProcess -Force -ErrorAction SilentlyContinue
            Write-Host "  端口 $port 仍被 PID $($item.OwningProcess) 占用，已结束（pid 文件里记的不是它）" -ForegroundColor Yellow
        }
    }
    Start-Sleep -Milliseconds 500
}

# 复查一次。**"停止"这件事必须以端口空出来为准，不能以"我发过停止命令"为准。**
$still = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($still) {
    Write-Host "  ⚠️ 端口 $port 仍被占用，停止没成功。" -ForegroundColor Red
    Write-Host "     占用它的 PID：$($still.OwningProcess -join ', ')" -ForegroundColor Red
    Write-Host '     可以手工结束：Stop-Process -Id <PID> -Force' -ForegroundColor Yellow
} else {
    Write-Host '  后端已停止（端口已释放）' -ForegroundColor Green
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
