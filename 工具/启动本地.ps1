<#
.SYNOPSIS
    一键启动金融监管法规知识库（本地）。

.DESCRIPTION
    做三件事：
      1. 启动 PostgreSQL（它不会开机自启，每次都要先拉起来）
      2. 启动后端服务（前端页面由它同源托管，所以只需要一个地址）
      3. 等它就绪，打印地址并打开浏览器

.EXAMPLE
    .\启动本地.ps1

.NOTE
    如果提示"禁止运行脚本"，改用：
      powershell -ExecutionPolicy Bypass -File .\启动本地.ps1

    ⚠️ 本文件必须保存为「UTF-8 带 BOM」。
    PowerShell 5.1 默认按 GBK 读取没有 BOM 的 UTF-8 文件，
    中文会变成乱码并导致语法错误——报错信息看起来像脚本写错了，
    实际只是编码问题。用记事本或 VS Code 改完脚本后，
    请确认编码仍然是「UTF-8 with BOM」。
#>

$ErrorActionPreference = 'Continue'

# Windows 下 Python 默认按 GBK 输出，遇到中文或特殊字符会抛 UnicodeEncodeError。
# 这个异常发生在"打印日志"这一步，却会中断真正的业务流程——
# 一个与业务无关的编码问题，表现起来像是功能坏了。所以强制 UTF-8。
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# ---------- 路径 ----------
$base    = 'D:\丁云璐\个人\Agent 产品项目'
$pgbin   = 'D:\anaconda\envs\pg\Library\bin'
$pgdata  = Join-Path $base 'pgdata'
$python  = Join-Path $base '.venv\Scripts\python.exe'
$backend = Join-Path $base 'vibe-rag\backend'
$logDir  = Join-Path $base 'vibe-rag\logs'
$pidFile = Join-Path $logDir 'backend.pid'
$port    = 8000
$url     = "http://127.0.0.1:$port"

if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

Write-Host ''
Write-Host '===== 金融监管法规知识库 · 本地启动 =====' -ForegroundColor Cyan

# ---------- 0. 外网连通性 ----------
#
# 为什么把这一步放在最前面：
# 向量化和生成都要调百炼。如果这条链路不通，服务照样能起来、页面照样能打开，
# 但你一问问题就会失败——**问题在使用时才暴露，而根因在启动前就已经存在了。**
# 提前 4 秒检查一次，能把"用了才发现"变成"启动就知道"。
Write-Host ''
Write-Host '[0/3] 检查外网（百炼 + 向量库）...' -ForegroundColor Yellow

$netClient = New-Object System.Net.Sockets.TcpClient
$canReachDashScope = $false
try {
    $connectTask = $netClient.ConnectAsync('dashscope.aliyuncs.com', 443)
    $canReachDashScope = $connectTask.Wait(4000) -and $netClient.Connected
} catch {
    $canReachDashScope = $false
} finally {
    $netClient.Close()
}

if ($canReachDashScope) {
    Write-Host '      可以连到百炼' -ForegroundColor Green
} else {
    Write-Host '      ⚠️ 连不上 dashscope.aliyuncs.com:443' -ForegroundColor Red
    Write-Host '         服务仍然会启动、页面也能打开，但：' -ForegroundColor Red
    Write-Host '           · 向量化会失败 → 检索不到任何内容' -ForegroundColor Red
    Write-Host '           · 问答会提示"检索服务暂时不可用"' -ForegroundColor Red
    Write-Host '         这不是系统坏了，是这台机器连不上百炼。' -ForegroundColor Yellow
    Write-Host '         先确认网络或代理设置，再往下走。' -ForegroundColor Yellow
}

# ---------- 0.2 向量库 ----------
#
# 这一段是踩过坑之后补的：Zilliz 的免费实例**长时间不访问会被挂起**，
# 挂起时它的域名直接不再解析。表现是"服务能起来、页面能打开、
# 但一检索就报错"，而错误信息长得像代码写错了——
# 排查这种问题要花很久，而它其实只要去控制台点一下"恢复"。
#
# 所以启动时就探一次，把"用的时候才发现"变成"启动就知道"。
$milvusHost = $null
$envFile = Join-Path $backend '.env'
if (Test-Path -LiteralPath $envFile) {
    $milvusLine = Select-String -Path $envFile -Pattern '^MILVUS_URI=' | Select-Object -First 1
    if ($milvusLine) {
        $milvusUri = ($milvusLine.Line -split '=', 2)[1].Trim()
        if ($milvusUri -match '^https://([^/]+)') { $milvusHost = $Matches[1] }
    }
}

if (-not $milvusHost) {
    Write-Host '      向量库配置成的是本地地址，跳过外网检查' -ForegroundColor DarkGray
} else {
    $netClient2 = New-Object System.Net.Sockets.TcpClient
    $canReachVector = $false
    try {
        $task2 = $netClient2.ConnectAsync($milvusHost, 443)
        $canReachVector = $task2.Wait(5000) -and $netClient2.Connected
    } catch {
        $canReachVector = $false
    } finally {
        $netClient2.Close()
    }

    if ($canReachVector) {
        Write-Host '      可以连到向量库' -ForegroundColor Green
    } else {
        Write-Host "      ⚠️ 连不上向量库 $milvusHost`:443" -ForegroundColor Red
        Write-Host '         检索会失败，但服务本身能启动、页面也能打开。' -ForegroundColor Red
        Write-Host '         最常见的原因是集群被挂起（免费实例长时间不访问会挂起）。' -ForegroundColor Yellow
        Write-Host '         先确认网络，再去向量库控制台看实例状态；' -ForegroundColor Yellow
        Write-Host '         需要分辨是哪一层出了问题就跑：python 工具\诊断向量库.py' -ForegroundColor Yellow
    }
}

# ---------- 1. PostgreSQL ----------
Write-Host ''
Write-Host '[1/3] 检查 PostgreSQL ...' -ForegroundColor Yellow

$ready = & "$pgbin\pg_isready.exe" -h 127.0.0.1 -p 5432 2>&1
if ($ready -match 'accepting connections') {
    Write-Host '      已经在运行，跳过' -ForegroundColor Green
} else {
    Write-Host '      正在启动 ...'
    & "$pgbin\pg_ctl.exe" -D $pgdata -o '-p 5432' -l (Join-Path $pgdata 'server.log') start 2>&1 | Out-Null

    $started = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        $ready = & "$pgbin\pg_isready.exe" -h 127.0.0.1 -p 5432 2>&1
        if ($ready -match 'accepting connections') { $started = $true; break }
    }
    if ($started) {
        Write-Host '      已启动' -ForegroundColor Green
    } else {
        Write-Host '      启动失败。可以看 pgdata\server.log 排查。' -ForegroundColor Red
        Write-Host '      提示：数据库没起来的话，文档和问答都会报错。' -ForegroundColor Red
    }
}

# ---------- 2. 后端 ----------
Write-Host ''
Write-Host '[2/3] 启动后端服务 ...' -ForegroundColor Yellow

$listening = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($listening) {
    Write-Host "      端口 $port 已被占用，认为服务已在运行，跳过启动" -ForegroundColor Green
} else {
    $outLog = Join-Path $logDir 'backend_out.log'
    $errLog = Join-Path $logDir 'backend_err.log'
    $launched = $false

    try {
        $proc = Start-Process -FilePath $python `
            -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$port") `
            -WorkingDirectory $backend `
            -WindowStyle Hidden `
            -PassThru `
            -RedirectStandardOutput $outLog `
            -RedirectStandardError $errLog
        $proc.Id | Set-Content -Path $pidFile -Encoding UTF8
        Write-Host "      已启动，PID = $($proc.Id)" -ForegroundColor Green
        $launched = $true
    } catch {
        # 某些受限环境里 Start-Process 会因为环境变量大小写冲突而失败
        # （报错形如「字典中的关键字 Path 所添加的关键字 PATH」）。
        # 这里用 WMI 直接创建进程作为兜底，它不经过 PowerShell 的环境块处理。
        Write-Host '      Start-Process 不可用，改用 WMI 启动 ...' -ForegroundColor Yellow
        $cmdLine = "`"$python`" -m uvicorn app.main:app --host 127.0.0.1 --port $port"
        $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
            CommandLine    = $cmdLine
            CurrentDirectory = $backend
        }
        if ($result.ProcessId) {
            $result.ProcessId | Set-Content -Path $pidFile -Encoding UTF8
            Write-Host "      已启动，PID = $($result.ProcessId)" -ForegroundColor Green
            $launched = $true
        } else {
            Write-Host "      启动失败：$($result.ReturnValue)" -ForegroundColor Red
        }
    }
}

# ---------- 3. 等待就绪 ----------
Write-Host ''
Write-Host '[3/3] 等待服务就绪 ...' -ForegroundColor Yellow

$healthy = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $response = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) { $healthy = $true; break }
    } catch {
        # 还没起来，继续等
    }
}

Write-Host ''
if ($healthy) {
    Write-Host '===== 启动完成 =====' -ForegroundColor Green
    Write-Host ''
    Write-Host "  应用界面 : $url" -ForegroundColor Cyan
    Write-Host "  接口文档 : $url/api/v1/docs" -ForegroundColor Cyan
    Write-Host ''

    # 顺手报一下知识库状态，省得打开页面才发现是空的
    try {
        $docs = Invoke-RestMethod -Uri "$url/api/v1/documents" -TimeoutSec 5
        $usage = Invoke-RestMethod -Uri "$url/api/v1/usage" -TimeoutSec 5
        Write-Host "  知识库   : $($docs.total) 份文档，$((($docs.items | Measure-Object -Property chunk_count -Sum).Sum)) 个切片"
        Write-Host "  今日用量 : 问答 $($usage.qa_count) 次 ｜ 限额$($(if ($usage.limits_enabled) { '已开启' } else { '已关闭' }))"
    } catch {
        Write-Host '  （读取知识库状态失败，不影响使用）' -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host '  正在打开浏览器 ...' -ForegroundColor Green
    Start-Process $url
} else {
    Write-Host '===== 服务没能就绪 =====' -ForegroundColor Red
    Write-Host ''
    Write-Host "  看一下错误日志：$errLog" -ForegroundColor Yellow
    Get-Content (Join-Path $logDir 'backend_err.log') -Tail 20 -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host '停止服务：.\停止本地.ps1' -ForegroundColor DarkGray
