"""给工具脚本补上"控制台按 UTF-8 输出"的保护。**幂等，可重复运行。**

## 为什么需要它

Windows 控制台默认按 GBK 解码，而这些工具会打印 ✅ / ⚠️ / ❌。
后果不是"显示成乱码"，而是**脚本直接崩在打印那一步**——
而崩溃点往往在**干完活之后**：

  · 跑评测时：120 题全跑完、钱花完了，明细一条都没落盘；
  · 跑入库时：文件已经写进库了，最后那行汇总打印不出来，看起来像失败。

这个坑在搭建过程中踩了四次（评测、诊断、编译词条、升级表结构各一次），
每次都是**单独修那一个文件**。第五次的时候停下来了：与其一个一个踩，
不如扫一遍，把有同样问题的都补上。

## 它做什么

在 `sys.path.insert(...)` 那一行后面插入：

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

插在这个位置是因为这些脚本都有那一行（都要把自己或 backend 加进 sys.path），
而且它一定在 `import sys` 之后——不用做语法分析，也不需要猜 import 结构。

## 幂等性

已经有 `reconfigure` 的文件会被跳过，所以重复运行不会重复插入。
插完会用 `py_compile` 复查一遍，语法坏了会明确报出来。

用法：
    python 工具/修控制台编码.py            # 只看要改哪些（默认演练）
    python 工具/修控制台编码.py --apply    # 真正写入
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 会出现在打印里的那些符号。有它们，又没有编码保护，就会崩。
EMOJI = re.compile(r'[\u2705\u274c\u26a0\u2753\U0001f6aa\U0001f64b]')

GUARD = '''
# Windows 控制台默认按 GBK 解码，而这个脚本会打印 ✅/⚠️ 这类符号——
# 不加保护的话，它会**直接崩在打印那一步**，而崩溃点常常在干完活之后
# （评测跑完了、钱花完了，明细一条都没落盘）。详见 工具/修控制台编码.py。
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:  # noqa: BLE001
    pass
'''

ANCHOR = re.compile(r'^sys\.path\.insert\(.*\)\s*$', re.MULTILINE)


def patch(path: Path) -> str:
    text = path.read_text(encoding='utf-8')
    if 'reconfigure' in text:
        return '跳过（已有保护）'
    if not EMOJI.search(text):
        return '跳过（不打印符号）'
    match = ANCHOR.search(text)
    if match is None:
        return '跳过（找不到 sys.path.insert 这一行，需要人工看）'
    if not re.search(r'^import sys\b', text, re.MULTILINE):
        return '跳过（没有 import sys，需要人工看）'

    updated = text[: match.end()] + GUARD + text[match.end() :]
    path.write_text(updated, encoding='utf-8')
    return '已补上'


def main() -> int:
    parser = argparse.ArgumentParser(description='给工具脚本补控制台编码保护')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    targets = sorted(
        [*Path(ROOT / '工具').glob('*.py'), *Path(ROOT / '评测').glob('*.py')]
    )
    changed = 0
    for path in targets:
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding='utf-8')
        if 'reconfigure' in text or not EMOJI.search(text):
            continue
        print(f'  {path.name}')
        changed += 1
        if args.apply:
            print(f'    → {patch(path)}')

    print()
    print(f'{changed} 个文件需要改')
    if not args.apply:
        print('演练模式：没有写入。确认后加 --apply。')
        return 0

    # 改完复查一遍语法：批量改文件最怕的就是改坏一个而没人发现。
    import py_compile

    broken = []
    for path in targets:
        try:
            py_compile.compile(str(path), doraise=True, quiet=2)
        except py_compile.PyCompileError as exc:
            broken.append(f'{path.name}: {exc}')
    if broken:
        print()
        print('⚠️ 语法检查没过：')
        for item in broken:
            print('  ', item)
        return 1
    print('语法检查通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
