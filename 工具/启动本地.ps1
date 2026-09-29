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
$staleService = $false
if ($listening) {
    # ⚠️ 端口被占用**不等于**"我们的服务已经在跑"，更不等于"跑的是当前这份代码"。
    #
    # 这里踩过一个真实的坑，而且它一次比一次恶劣：
    #   1. uvicorn 默认**不热加载**（没开 --reload），服务一旦起来就锁定了
    #      那一刻的代码和配置；
    #   2. 原来这一支只判断"端口有没有被占用"，占用就跳过启动并打印"启动完成"；
    #   3. 于是一个 **13 天前启动的旧进程**一直占着 8000：它是**场景迁移之前**
    #      起来的，数据库和向量集合都指向迁移前的库，
    #      结果两条检索路都返回空、问答一律"无法判断"；
    #   4. 而用户每次"重启"都被这一支挡住，看到的永远是"启动完成"——
    #      **重启这条路被脚本自己堵死了。**
    #
    # 所以现在要多问一句：**这个服务是什么时候起来的？**
    # 拿它的启动时间和磁盘上代码、配置的最后修改时间比——
    # 比磁盘旧，就说明它跑的不是你手上这份代码。
    $health = $null
    try { $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 5 } catch { }

    $watch = @()
    $watch += Get-ChildItem -Path (Join-Path $backend 'app') -Recurse -File -Filter *.py -ErrorAction SilentlyContinue
    $envPath = Join-Path $backend '.env'
    if (Test-Path -LiteralPath $envPath) { $watch += Get-Item -LiteralPath $envPath }
    $newest = $watch | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if ($health -and $health.started_at) {
        $startedAt = [datetime]::Parse($health.started_at)
        $startedLocal = $startedAt.ToString('yyyy-MM-dd HH:mm:ss')
        if ($newest -and $startedAt -lt $newest.LastWriteTime) {
            $staleService = $true
            Write-Host "      ⚠️ 端口 $port 上的服务是**旧进程**，不是当前代码" -ForegroundColor Red
            Write-Host "         它在 $startedLocal 启动（PID $($health.pid)），" -ForegroundColor Red
            Write-Host "         而 $($newest.Name) 在那之后被改过（$($newest.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))）。" -ForegroundColor Red
            Write-Host '         它跑的是启动那一刻的代码和配置，你看到的答案可能来自旧版本。' -ForegroundColor Yellow
            Write-Host '         处理：先执行 .\停止本地.ps1，再重新执行本脚本。' -ForegroundColor Yellow
        } else {
            Write-Host "      端口 $port 已被本项目服务占用（$startedLocal 启动，PID $($health.pid)），跳过启动" -ForegroundColor Green
        }
    } else {
        # 占用端口的东西答不上来 /health —— 大概率不是本项目的服务。
        $staleService = $true
        Write-Host "      ⚠️ 端口 $port 被占用，但占用者不响应本项目的 /health" -ForegroundColor Red
        Write-Host '         它可能不是本项目的服务（也可能是很旧的版本）。' -ForegroundColor Yellow
        Write-Host '         处理：先执行 .\停止本地.ps1，再重新执行本脚本。' -ForegroundColor Yellow
    }
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
if ($healthy -and $staleService) {
    # 服务确实在响应，但它是旧的 —— 这时**不能**打印"启动完成"。
    # 用户看到"启动完成"就会以为没问题，然后继续对着旧版本提问题，
    # 而这正是这个坑最难查的地方：**一切显示正常。**
    Write-Host '===== 端口上的服务是旧版本，本次没有重启 =====' -ForegroundColor Red
    Write-Host ''
    Write-Host '  它照常能用，但用的是**启动那一刻**的代码和配置。' -ForegroundColor Yellow
    Write-Host '  重新加载当前代码需要：先 .\停止本地.ps1，再执行本脚本。' -ForegroundColor Yellow
    Write-Host ''
    Write-Host "  应用界面 : $url （当前跑的是旧版本）" -ForegroundColor DarkGray
} elseif ($healthy) {
    # ⚠️ 把**真正在跑的那个进程**的 PID 记进 pid 文件，而不是 Start-Process 返回的那个。
    #
    # 实测：`Start-Process python.exe -m uvicorn ...` 返回的是一个**启动器** PID，
    # 真正的 uvicorn 是它的子进程，两者 PID 不同（实测 13616 → 25332）。
    # pid 文件里记的是前者，于是"停止"只杀掉了启动器，真正监听端口的那个还活着——
    # 启动脚本下次看到端口被占用又跳过启动。**停止说成功、启动说成功、
    # 端口上跑的却是十几天前的旧进程**，就是这么来的。
    #
    # 现在从 /health 取真实 PID（那是服务自己报的，不可能错）再写文件。
    try {
        $healthNow = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 5
        if ($healthNow.pid) {
            $healthNow.pid | Set-Content -Path $pidFile -Encoding UTF8
        }
    } catch {
        # 取不到就保留原来的记录，不影响使用
    }

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

    # ---------- 冒烟检索 ----------
    #
    # 为什么要做这一步：上面那三项检查（外网、向量库、数据库）查的都是**能不能连上**，
    # 连得上不等于检索得出东西。踩过一次真实的坑——
    # 服务起来了、页面能开、一切"就绪"，然后问一个库里明明有的问题，
    # 回答是"无法判断"。日志里那一次的记录是：
    #
    #     向量=0 关键词=0 融合=0 返回=0   error=None
    #
    # 两条检索路都返回了空，而且都不报错。这种情况必须用**一次真实检索**才能发现，
    # 连通性检查永远看不出来。
    #
    # 探测用的是**库里一定有依据**的问题，而且是走条款直查的那一类，
    # 所以判据很硬：向量和关键词都为 0，就一定是链路有问题。
    Write-Host ''
    Write-Host '  冒烟检索（确认检索链路真的能召回）...' -ForegroundColor Yellow
    try {
        # ⚠️ body 必须先转成 **UTF-8 字节**再发。
        # 直接传字符串的话，PowerShell 会按控制台的默认编码（这里是 GBK）发送，
        # 中文查询在服务端就变成了乱码——冒烟检索会"失败"，而失败原因是发送端的编码，
        # 不是检索链路。日志里能看到那一条：query='????????????????????????'。
        $probeBody = [System.Text.Encoding]::UTF8.GetBytes(
            '{"query":"证券期货投资者适当性管理办法第二十九条怎么规定的","top_k":3}'
        )
        $probe = Invoke-RestMethod -Uri "$url/api/v1/retrieval/search" -Method Post `
            -ContentType 'application/json; charset=utf-8' `
            -Body $probeBody `
            -TimeoutSec 60
        if ($probe.vector_hit_count -gt 0 -and $probe.bm25_hit_count -gt 0) {
            Write-Host "      正常：向量召回 $($probe.vector_hit_count) 条 ｜ 关键词召回 $($probe.bm25_hit_count) 条 ｜ 条款直查 $($probe.exact_hit_count) 条" -ForegroundColor Green
        } else {
            Write-Host '      ⚠️ 检索链路异常：库里有数据，但两条路都没召回到东西。' -ForegroundColor Red
            Write-Host "         向量 $($probe.vector_hit_count) 条 ｜ 关键词 $($probe.bm25_hit_count) 条" -ForegroundColor Red
            if ($probe.error) { Write-Host "         原因：$($probe.error)" -ForegroundColor Red }
            Write-Host '         这时候问问题会得到"无法判断"，但**不是知识库里没有**。' -ForegroundColor Yellow
            Write-Host '         查一查向量库状态：python 工具\查看集合.py ；对账：python 工具\入库后校验.py' -ForegroundColor Yellow
        }
    } catch {
        Write-Host "      ⚠️ 冒烟检索没跑成：$($_.Exception.Message)" -ForegroundColor Red
        Write-Host '         不影响启动，但第一次提问可能会失败。' -ForegroundColor Yellow
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
