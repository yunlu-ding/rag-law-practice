"""条界校验：检验"一条一片"到底做没做到。

为什么不靠"看起来对不对"来判断：切片抽出来看一眼，绝大多数都是像样的，
坏的那几片混在几百片里根本不会被注意到。所以要换成可证伪的检查：

  1. **条号连续性**：第 1 条到第 N 条之间有没有缺口？
     缺口 = 某条真条文被误判成引用，会被并进上一条的切片——
     表现为"问第 30 条，引用却指向第 29 条"，而且不报错。
  2. **重号**：同一个条号出现两次 = 有引用没被过滤干净，多出一片碎片。
  3. **被拒清单**：过滤刷掉了哪些候选？逐条看语境，确认拒得对。
     这一步是给"过滤器"本身做的评测——过滤器也会错。

用法：
    python 工具/校验条界.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))
logging.getLogger('pypdf').setLevel(logging.ERROR)

from app.rag.chunker import build_chunks  # noqa: E402
from app.rag.loader import load_document  # noqa: E402
from app.rag.splitters.legal import (  # noqa: E402
    audit_article_boundaries,
    chinese_number_to_int,
    starts_with_article,
)
from app.rag.metadata import LEGAL_LEVEL_BY_FOLDER  # noqa: E402

# 要校验的文件。扫**层级目录**，与 工具/入库.py 保持一致：
# `语料/_原始件/` 放的是原始下载件和已被替换的旧版本，不是语料，不参与校验。
TARGETS: list[str] = []
for folder in sorted((ROOT / '语料').iterdir()):
    if not folder.is_dir() or folder.name not in LEGAL_LEVEL_BY_FOLDER:
        continue
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in ('.pdf', '.docx', '.html', '.txt', '.md'):
            TARGETS.append(f'{folder.name}/{path.name}')

for relative in TARGETS:
    path = ROOT / '语料' / relative
    loaded = load_document(path)
    text = loaded.full_text
    audit = audit_article_boundaries(text)
    accepted = audit['accepted']
    rejected = audit['rejected']

    numbers = [chinese_number_to_int(token) for _, token in accepted]
    numbers = [n for n in numbers if n is not None]
    found = set(numbers)
    missing = [n for n in range(1, max(numbers or [0]) + 1) if n not in found]

    if not numbers:
        continue  # 没有条结构的文件（指南类），跳过

    print('=' * 78)
    print(f'{path.name}')
    print(f'  识别条数：{len(numbers)}  最大条号：{max(numbers) if numbers else 0}')
    print(f'  重号：{len(numbers) - len(found)} 个')
    print(f'  条界缺口：{missing if missing else "无"}')
    print(f'  判为引用而过滤掉：{len(rejected)} 个')
    for position, token, reason in rejected[:6]:
        context = text[max(0, position - 18) : position + 22].replace('\n', '⏎')
        print(f'    ✗ {token:<10} {reason}   语境…{context}')
    if len(rejected) > 6:
        print(f'    …… 其余 {len(rejected) - 6} 个略')

    # ---- 切片层面的条号覆盖 ----
    #
    # 上面看的是"条界找得准不准"，这里看的是"每一片有没有拿到条号身份"。
    # 后者是条款级定位能成立的前提：一片没有条号，就等于它在这套定位体系里不存在。
    #
    # 三种片要分开数，因为它们的问题不一样：
    #   条文头 —— 正文以条号开头，是"这一条"的入口；
    #   续片   —— 有父条号但正文不以它开头，通常是跨页被切开的；
    #   无条号 —— 前言，或者根本不是法规文本。
    chunking = build_chunks(loaded)
    records = chunking.records
    heads = continuations = orphans = 0
    for record in records:
        if record.article_number and starts_with_article(record.content):
            heads += 1
        elif record.article_number:
            continuations += 1
        else:
            orphans += 1

    print(
        f'  切片 {len(records)} 片：条文头 {heads} ｜ 续片 {continuations} ｜ '
        f'无条号 {orphans}'
    )
