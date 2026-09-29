from __future__ import annotations

import uvicorn

"""金融监管法规知识库 · 后端启动入口。

启动方式：
    cd vibe-rag/backend
    python run.py

默认地址：
    应用界面    http://127.0.0.1:8000
    接口文档    http://127.0.0.1:8000/api/v1/docs

Windows 提示：
    如果终端出现 UnicodeEncodeError，先设置两个环境变量再启动：
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'
    这是 Windows 下 Python 默认按 GBK 输出导致的，与业务逻辑无关。
"""

if __name__ == '__main__':
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
