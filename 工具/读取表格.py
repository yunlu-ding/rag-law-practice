"""读 .xlsx（并可选导出成 CSV）。

为什么自己写而不装 openpyxl：

xlsx 的本质是一个 zip 包，里面装着几个 XML（共享字符串表 + 每张表的单元格）。
读它需要的代码不到一百行，而且**只用标准库**。
评测集这种"人用 Excel 编辑、机器读"的文件，加一个只在读表时用到的第三方依赖，
换来的便利抵不上"环境里少一个依赖"的价值——何况这个依赖装不上时，
表现是"评测跑不起来"，跟评测本身毫无关系。

用法：
    python 工具/读取表格.py 评测/法规评测集.xlsx                  # 打印前若干行
    python 工具/读取表格.py 评测/法规评测集.xlsx --csv 输出.csv    # 导出 CSV 给评测脚本用
    python 工具/读取表格.py 评测/法规评测集.xlsx --column 备注     # 只看某一列
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
COLUMN_LETTERS = re.compile(r'^([A-Z]+)')


def _column_index(letters: str) -> int:
    """把 "A" / "AB" 这样的列号转成 0 基下标。"""

    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord('A') + 1)
    return index - 1


def read_xlsx(path: Path) -> list[list[str]]:
    """返回按行组织的二维字符串表。空单元格用空串补齐。"""

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

        shared: list[str] = []
        if 'xl/sharedStrings.xml' in names:
            root = ElementTree.fromstring(archive.read('xl/sharedStrings.xml'))
            for item in root.findall(f'{NS}si'):
                # 一个 <si> 里可能有多个 <t>（富文本被拆成了几段），要拼起来。
                # 只取第一个 <t> 会把加粗、换色的那部分文字整段丢掉。
                shared.append(''.join(node.text or '' for node in item.iter(f'{NS}t')))

        sheet_files = sorted(
            name
            for name in names
            if name.startswith('xl/worksheets/sheet') and name.endswith('.xml')
        )
        if not sheet_files:
            raise ValueError(f'{path.name} 里没有工作表')

        root = ElementTree.fromstring(archive.read(sheet_files[0]))

    table: list[list[str]] = []
    for xml_row in root.iter(f'{NS}row'):
        cells: dict[int, str] = {}
        for cell in xml_row.findall(f'{NS}c'):
            reference = cell.get('r') or ''
            match = COLUMN_LETTERS.match(reference)
            if not match:
                continue
            index = _column_index(match.group(1))

            kind = cell.get('t')
            value_node = cell.find(f'{NS}v')
            if kind == 'inlineStr':
                inline = cell.find(f'{NS}is')
                value = ''.join(node.text or '' for node in inline.iter(f'{NS}t')) if inline is not None else ''
            elif kind == 's' and value_node is not None:
                # 共享字符串：单元格里存的是下标，真正的文字在共享表里。
                value = shared[int(value_node.text or '0')]
            elif value_node is not None:
                value = value_node.text or ''
            else:
                value = ''
            cells[index] = value.replace('\r\n', '\n').strip()

        width = max(cells) + 1 if cells else 0
        table.append([cells.get(index, '') for index in range(width)])
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description='读 xlsx')
    parser.add_argument('path')
    parser.add_argument('--csv', default=None, help='导出成 CSV')
    parser.add_argument('--column', default=None, help='只显示这一列（按表头名匹配）')
    parser.add_argument('--rows', type=int, default=5, help='打印前 N 行（默认 5）')
    args = parser.parse_args()

    table = read_xlsx(Path(args.path))
    if not table:
        print('表是空的')
        return 1

    width = max(len(row) for row in table)
    header = table[0] + [''] * (width - len(table[0]))
    print(f'工作表：{len(table) - 1} 行 × {width} 列')
    print(f'表头：{header}')
    print()

    if args.column:
        if args.column not in header:
            print(f'没有这一列：{args.column}（可选：{header}）')
            return 1
        index = header.index(args.column)
        for row in table[1:]:
            value = row[index] if index < len(row) else ''
            if value:
                print(f'  {row[0]:<8} {value}')
        return 0

    for row in table[: args.rows]:
        print('  | ' + ' | '.join(cell[:26] for cell in row))

    if args.csv:
        output = Path(args.csv)
        with output.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            for row in table:
                writer.writerow(row + [''] * (width - len(row)))
        print()
        print(f'已导出 {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
