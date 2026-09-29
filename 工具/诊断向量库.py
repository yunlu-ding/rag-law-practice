"""诊断向量库连不上到底是哪一层的问题。

分三层看，因为它们对应的处理办法完全不同：

    1. DNS        —— 域名解析不了 = 集群被挂起或删除（去控制台恢复）
    2. TCP/TLS    —— 连不上端口 = 网络/防火墙问题
    3. 鉴权       —— 端口通但报 unauthorized = **token 失效或写错了**

第三层最容易被误判成"服务挂了"，因为它抛出来的错误信息
（"illegal connection params or server unavailable"）把两种情况说在了一起。
这个脚本把三层拆开验证，直接指出是哪一层。

⚠️ 本脚本**只打印状态码，不打印 token 内容**。
"""

from __future__ import annotations

import json
import socket
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

# Windows 控制台默认按 GBK 解码，而这个脚本会打印 ✅/⚠️ 这类符号——
# 不加保护的话，它会**直接崩在打印那一步**，而崩溃点常常在干完活之后
# （评测跑完了、钱花完了，明细一条都没落盘）。详见 工具/修控制台编码.py。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass

from app.config import get_settings  # noqa: E402


def check_dns(host: str) -> bool:
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, 443)}
    except socket.gaierror as exc:
        print(f'❌ DNS：解析失败（{exc}）')
        print('   集群可能被挂起或已删除。去 Zilliz 控制台确认实例状态。')
        return False
    print(f'✅ DNS：解析成功 → {", ".join(sorted(addresses))}')
    return True


def check_tls(host: str) -> bool:
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                print(f'✅ TLS：握手成功（{tls.version()}）')
        return True
    except Exception as exc:  # noqa: BLE001
        print(f'❌ TLS：握手失败（{type(exc).__name__}: {exc}）')
        return False


def check_auth(host: str, token: str | None) -> None:
    url = f'https://{host}/v2/vectordb/collections/list'
    body = json.dumps({}).encode('utf-8')

    for label, header in (
        ('不带凭据', {}),
        ('带配置里的 token', {'Authorization': f'Bearer {token}'} if token else None),
    ):
        if header is None:
            print('⚠️  鉴权：配置里没有 MILVUS_TOKEN，跳过')
            return
        request = urllib.request.Request(
            url, data=body, method='POST',
            headers={'Content-Type': 'application/json', **header},
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode('utf-8') or '{}')
                names = [item.get('name') for item in payload.get('data', [])]
                print(f'✅ 鉴权（{label}）：HTTP {response.status}')
                print(f'   现有集合：{names if names else "（空）"}')
        except urllib.error.HTTPError as exc:
            print(f'❌ 鉴权（{label}）：HTTP {exc.code} {exc.reason}')
            if exc.code == 401:
                print('   token 无效或已过期。去 Zilliz 控制台重新生成 API Key，')
                print('   然后更新 backend/.env 里的 MILVUS_TOKEN。')
        except Exception as exc:  # noqa: BLE001
            print(f'❌ 鉴权（{label}）：{type(exc).__name__}: {exc}')


def main() -> int:
    settings = get_settings()
    uri = settings.milvus_uri or settings.resolved_milvus_uri
    print(f'向量库地址：{uri}')
    print()

    if not uri.startswith('https://'):
        print('不是 https 托管地址，跳过 DNS/TLS 检查（应该是本地 Milvus）。')
        return 0

    host = uri.split('://', 1)[1].split('/', 1)[0]
    if not check_dns(host):
        return 1
    if not check_tls(host):
        return 1
    check_auth(host, settings.milvus_token)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
