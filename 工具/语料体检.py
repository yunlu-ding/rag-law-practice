"""语料体检：把 语料/ 目录下的每份文件跑一遍解析 + 切分，产出真实数字。

为什么要有这个工具：

入库之前必须回答三个问题——
  1. 每份文件到底抽出了多少字？（抽不出来 = 这份文件等于没入库）
  2. 抽出来的字对不对？（HTML 抽成导航、PDF 抽成乱码，都是"看起来成功")
  3. 切出来的片是谁？（一条法规被切成几片、有没有跨条合并）

这三个问题只靠"上传成功了"是回答不了的。所以做成一个可重复运行的脚本，
每次换语料、改切分策略都跑一遍，拿数字说话。

用法：
    python 工具/语料体检.py
    python 工具/语料体检.py --output 11-语料体检报告.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / 'backend'
sys.path.insert(0, str(BACKEND))

import logging  # noqa: E402

# pypdf 遇到缺 ToUnicode 的 PDF 会逐字体刷警告，一次几千行。
# 这类文件的问题在体检单里由"汉字占比"一栏暴露，不需要刷屏。
logging.getLogger('pypdf').setLevel(logging.ERROR)

from app.rag.chunker import build_chunks  # noqa: E402
from app.rag.loader import load_document  # noqa: E402
from app.rag.splitters.legal import (  # noqa: E402
    audit_article_boundaries,
    chinese_number_to_int,
)
from app.rag.metadata import LEGAL_LEVEL_BY_FOLDER, extract_metadata  # noqa: E402
from app.rag.metadata_overrides import apply_overrides  # noqa: E402

CORPUS = ROOT / '语料'


def iter_corpus_files() -> list[tuple[str, Path]]:
    """按"层级目录"遍历语料，返回 (层级中文名, 文件路径)。

    层级信息来自目录名——这是人工确认过的，比从正文里猜准得多。
    """

    files: list[tuple[str, Path]] = []
    for folder in sorted(CORPUS.iterdir()):
        if not folder.is_dir():
            continue
        # 与 工具/入库.py 保持一致：只体检层级目录。
        if folder.name not in LEGAL_LEVEL_BY_FOLDER:
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.startswith('~$'):
                files.append((folder.name, path))
    return files


def inspect(path: Path, *, legal_level: str | None) -> dict:
    """体检单份文件：解析 → 切分 → 抽元数据 → 算指标。

    四步放在一起，是因为它们共用同一次解析结果。
    拆开写就得把同一份文件解析好几遍——13MB 的 PDF 解析一遍要十几秒。
    """

    loaded = load_document(path)
    chunking = build_chunks(loaded)
    stats = chunking.stats

    # 用**严格**条文数，不用宽松计数：引用（"依据本办法第二十九条"）不算条文。
    # 否则"片/条"这个比值会被虚高的条数压小，看不出真实的切分粒度。
    audit = audit_article_boundaries(loaded.full_text)
    article_total = len(audit['accepted'])
    rejected_total = len(audit['rejected'])

    # 内部一致性：条文编号应该是连续的。缺号说明有一条真条文被过滤错了，
    # 后果是它和上一条被并进同一片——这件事在成品里看不出来，只能在这里查。
    numbers = [
        n
        for n in (chinese_number_to_int(token) for _, token in audit['accepted'])
        if n is not None
    ]
    gaps = [n for n in range(1, max(numbers or [0]) + 1) if n not in set(numbers)]
    duplicates = len(numbers) - len(set(numbers))

    # 汉字占比：判断"这份文件的文本层是不是坏的"最直接的一个数字。
    # 一份中文法规的汉字占比应该在 70% 以上；掉到 10% 以下基本可以断定
    # 抽出来的是字形索引或乱码，而不是文字。
    visible = [ch for ch in loaded.full_text if ch.isprintable() and not ch.isspace()]
    han_ratio = sum(1 for ch in visible if '\u4e00' <= ch <= '\u9fff') / max(len(visible), 1)

    try:
        metadata = extract_metadata(
            filename=path.name,
            text=loaded.full_text,
            legal_level=legal_level,
        )
        # 和入库流程保持一致：体检报告里显示的应该是**最终会入库的值**，
        # 否则报告和库里的数据会不一样，人会以为入库出错了。
        metadata = apply_overrides(metadata, path.name)
    except Exception as exc:  # noqa: BLE001
        metadata = {'error': f'{type(exc).__name__}: {exc}'}

    # "一条一片"的达成率：切片数与条数接近，说明基本做到一条一片；
    # 明显小于条数，说明发生了跨条合并。
    per_article = round(stats['chunk_count'] / article_total, 2) if article_total else None

    return {
        'filename': path.name,
        'file_type': path.suffix.lower().lstrip('.'),
        'legal_level': legal_level,
        'size_kb': round(path.stat().st_size / 1024, 1),
        'parser': loaded.parser_name,
        'sections': len(loaded.sections),
        'chars': loaded.char_count,
        'han_ratio': han_ratio,
        'articles': article_total,
        'article_rejected': rejected_total,
        'article_gaps': gaps,
        'article_duplicates': duplicates,
        'chunks': stats['chunk_count'],
        'per_article': per_article,
        'avg_length': stats['avg_length'],
        'max_length': stats['max_length'],
        'min_length': stats['min_length'],
        'mid_word_ratio': stats['mid_word_ratio'],
        'splitter_usage': stats['splitter_usage'],
        'warnings': loaded.warnings,
        'chunking_warnings': chunking.stats.get('warnings', []),
        'metadata': metadata,
    }


def build_report(rows: list[dict], failures: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    add = lines.append

    total_chunks = sum(row['chunks'] for row in rows)
    total_chars = sum(row['chars'] for row in rows)
    total_articles = sum(row['articles'] for row in rows)
    splitter_counter: Counter[str] = Counter()
    for row in rows:
        splitter_counter.update(row['splitter_usage'])

    add('# 语料体检报告（法规版）')
    add('')
    add('> 本报告由 `工具/语料体检.py` 自动生成，所有数字均为真实运行结果，未做任何人工修饰。')
    add('')
    add('## 一、总览')
    add('')
    add(f'- 语料文件：**{len(rows)} 份**（另有 {len(failures)} 份解析失败）')
    add(f'- 正文总量：**{total_chars:,} 字**')
    add(f'- 识别条文：**{total_articles:,} 条**（剔除跨条引用后的真条文起点数）')
    add(f'- 切片总数：**{total_chunks:,} 片**')
    distribution = '、'.join(f'{name} {count} 片' for name, count in splitter_counter.most_common())
    add(f'- 切片策略分布：{distribution or "—"}')
    add('')

    if failures:
        add('### 解析失败')
        add('')
        add('| 文件 | 错误 |')
        add('|---|---|')
        for name, error in failures:
            add(f'| {name} | {error} |')
        add('')

    add('## 二、逐份明细')
    add('')
    add('| # | 文件 | 类型 | 正文(字) | 汉字占比 | 条数 | 切片 | 片/条 | 均长 | 残句率 | 条界缺口 | 重号 |')
    add('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for index, row in enumerate(rows, start=1):
        per_article = row['per_article'] if row['per_article'] is not None else '—'
        gaps = row['article_gaps']
        gap_text = '无' if not gaps else (f'{len(gaps)} 处' if len(gaps) > 4 else '、'.join(map(str, gaps)))
        add(
            f"| {index} | {row['filename']} | {row['file_type']} | {row['chars']:,} "
            f"| {row['han_ratio']:.1%} | {row['articles']} | {row['chunks']} | {per_article} "
            f"| {row['avg_length']} | {row['mid_word_ratio']:.1%} "
            f"| {gap_text} | {row['article_duplicates']} |"
        )
    add('')

    add('### 条文切分质量')
    add('')
    add('"条"是法规问答的引用单位，所以这里单独看三个数字：')
    add('')
    add('- **条数 / 切片数**：两者接近说明"一条一片"基本达成；切片数明显更大，说明有条被切碎了。')
    add('- **条界缺口**：条号本该连续。出现缺口 = 某条真条文被误判成引用，')
    add('  会被并进上一条的切片里，表现为"引用指向隔壁那条"。')
    add('- **重号**：同一编号出现两次 = 有引用没被过滤掉，会多出一片不完整的碎片。')
    add('')
    add('| 文件 | 条数 | 切片 | 片/条 | 被过滤的引用 | 条界缺口 | 重号 |')
    add('|---|---|---|---|---|---|---|')
    for row in rows:
        if not row['articles']:
            continue
        per_article = row['per_article']
        gaps = row['article_gaps']
        add(
            f"| {row['filename']} | {row['articles']} | {row['chunks']} | {per_article} "
            f"| {row['article_rejected']} | {'无' if not gaps else len(gaps)} "
            f"| {row['article_duplicates']} |"
        )
    add('')

    # 文本层坏掉的文件必须单独点名。它们混在中间时和正常文件长得一模一样，
    # 只有把汉字占比拎出来看，才能发现"这份文件其实是 1600 片乱码"。
    broken = [row for row in rows if row['han_ratio'] < 0.5]
    add('## 三、文本层可用性')
    add('')
    if not broken:
        add('全部文件的汉字占比均在 50% 以上，文本层正常。')
    else:
        add('以下文件的汉字占比异常偏低，**抽出来的不是中文文字，入库等于往库里灌乱码**：')
        add('')
        add('| 文件 | 正文(字) | 汉字占比 | 建议 |')
        add('|---|---|---|---|')
        for row in broken:
            add(
                f"| {row['filename']} | {row['chars']:,} | {row['han_ratio']:.1%} "
                f"| 换一份带文本层的版本，或提供网页/Word 版 |"
            )
        add('')

    warned = [row for row in rows if row['warnings'] or row['chunking_warnings']]
    add('## 四、解析质检提示')
    add('')
    if not warned:
        add('无。所有文件解析均未触发告警。')
    else:
        for row in warned:
            add(f"### {row['filename']}")
            add('')
            for warning in list(row['warnings']) + list(row['chunking_warnings']):
                add(f'- {warning}')
            add('')

    add('')
    add('## 五、元数据抽取结果')
    add('')
    add('> 每个字段都记了它是从哪儿抽出来的（evidence）。')
    add('> **抽不到 ≠ 没有**——抽不到的字段需要人工补，不能当作"这份文件没有这个信息"。')
    add('')
    add('| 文件 | 层级 | 标题 | 发布机关 | 文号 | 公布 | 施行 | 效力 | 适用范围 |')
    add('|---|---|---|---|---|---|---|---|---|')
    for row in rows:
        meta = row['metadata']
        if 'error' in meta:
            add(f"| {row['filename']} | {row['legal_level']} | — | — | — | — | — | — | 抽取异常：{meta['error']} |")
            continue
        scope = '、'.join(meta['scope']) if meta['scope'] else '—'
        add(
            f"| {row['filename']} | {row['legal_level']} "
            f"| {meta['title']} | {meta['regulator'] or '—'} | {meta['doc_number'] or '—'} "
            f"| {meta['issued_date'] or '—'} | {meta['effective_date'] or '—'} "
            f"| {meta['validity']} | {scope} |"
        )
    add('')

    add('### 待人工复核的字段')
    add('')
    add('| 文件 | 字段 | 抽取说明 |')
    add('|---|---|---|')
    for row in rows:
        meta = row['metadata']
        if 'error' in meta:
            continue
        for field in ('regulator', 'doc_number', 'issued_date', 'effective_date', 'scope'):
            if meta.get(field):
                continue
            add(f"| {row['filename']} | `{field}` | {meta['evidence'].get(field, '—')} |")
    add('')

    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description='语料体检')
    parser.add_argument('--output', default=str(ROOT / '11-语料体检报告.md'))
    args = parser.parse_args()

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for folder_name, path in iter_corpus_files():
        level = LEGAL_LEVEL_BY_FOLDER.get(folder_name, '?')
        try:
            row = inspect(path, legal_level=level)
        except Exception as exc:  # noqa: BLE001
            failures.append((f'{folder_name}/{path.name}', f'{type(exc).__name__}: {exc}'))
            print(f'[失败] {folder_name}/{path.name} -> {type(exc).__name__}: {exc}')
            continue
        row['folder'] = folder_name
        rows.append(row)
        print(
            f"[{level:6}] {path.name[:34]:36} "
            f"{row['chars']:>7,}字 {row['articles']:>3}条 "
            f"{row['chunks']:>4}片 均长{row['avg_length']:>4} "
            f"汉字{row['han_ratio']:>5.1%} 残句率{row['mid_word_ratio']:.0%}"
        )

    report = build_report(rows, failures)
    output = Path(args.output)
    output.write_text(report, encoding='utf-8')

    print()
    print(f'共 {len(rows)} 份，{sum(r["chars"] for r in rows):,} 字，'
          f'{sum(r["articles"] for r in rows):,} 条，{sum(r["chunks"] for r in rows):,} 片')
    print(f'报告：{output}')
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
