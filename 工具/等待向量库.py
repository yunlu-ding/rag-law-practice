"""轮询等待向量库恢复可用。

为什么需要这个脚本，而不是"再手动试一次"：

Zilliz 的 serverless 实例从挂起状态恢复时，**DNS 记录不是一次性生效的**。
实测表现是：这一分钟能解析、下一分钟又解析不了，来回跳。
手工重试的问题是不知道该等多久、也判断不了"到底是在恢复还是彻底没了"。

这个脚本把"等"变成一件有结论的事：
    连着 N 次解析成功、并且真的连上了 —— 才算恢复。
    （只看到一次解析成功就宣布恢复，会被刚才那种跳变骗到。）

用法：
    python 工具/等待向量库.py                # 最多等 5 分钟
    python 工具/等待向量库.py --minutes 10
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
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

# 连续成功多少次才算稳定。
# 取 3 是因为实测的抖动周期在 10 秒左右——单次成功完全可能是撞上的。
STABLE_STREAK = 3
INTERVAL_SECONDS = 10


def probe(host: str) -> tuple[bool, str]:
    """探一次：先看解析，再真的连一下（解析通不等于服务通）。"""

    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror:
        return False, '域名解析失败'

    try:
        from pymilvus import MilvusClient

        settings = get_settings()
        client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
        collections = client.list_collections()
        return True, f'已连接，现有集合 {len(collections)} 个：{collections}'
    except Exception as exc:  # noqa: BLE001
        return False, f'解析通了但连接失败：{type(exc).__name__}'


def main() -> int:
    parser = argparse.ArgumentParser(description='等待向量库恢复')
    parser.add_argument('--minutes', type=float, default=5.0, help='最长等待分钟数')
    args = parser.parse_args()

    settings = get_settings()
    uri = settings.milvus_uri or ''
    if not uri.startswith('https://'):
        print(f'不是托管地址（{uri}），不需要等待。')
        return 0

    host = uri.split('://', 1)[1].split('/', 1)[0]
    deadline = time.monotonic() + args.minutes * 60
    streak = 0
    attempt = 0

    print(f'地址：{host}')
    print(f'最长等待 {args.minutes:g} 分钟，每 {INTERVAL_SECONDS} 秒探一次，')
    print(f'连续成功 {STABLE_STREAK} 次才算恢复。')
    print()

    while time.monotonic() < deadline:
        attempt += 1
        ok, message = probe(host)
        stamp = time.strftime('%H:%M:%S')
        if ok:
            streak += 1
            print(f'[{stamp}] {streak}/{STABLE_STREAK} ✅ {message}')
            if streak >= STABLE_STREAK:
                print()
                print('向量库已恢复，可以继续入库了。')
                return 0
        else:
            streak = 0
            print(f'[{stamp}] 第 {attempt} 次 ❌ {message}')
        time.sleep(INTERVAL_SECONDS)

    print()
    print(f'等了 {args.minutes:g} 分钟仍未稳定恢复。')
    print('建议去 Zilliz 控制台确认实例状态，并检查 backend/.env 里的 MILVUS_TOKEN。')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
