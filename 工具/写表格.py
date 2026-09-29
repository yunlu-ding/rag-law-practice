"""写 .xlsx（只用标准库，不依赖 openpyxl）。

和 `读取表格.py` 配套：评测集由人来编辑，而 Excel/WPS 用起来比 CSV 顺手
（不会因为单元格里有个逗号就串列），所以**xlsx 是唯一源**，
脚本读写它。既然要读，就得也能写——不能只让机器读、不让人改。

xlsx 的最小合法结构就是五个文件：

    [Content_Types].xml              声明各部分是什么
    _rels/.rels                      包级关系
    xl/workbook.xml                  工作簿（有哪几张表）
    xl/_rels/workbook.xml.rels       工作簿到表的引用
    xl/worksheets/sheet1.xml         表本身

单元格文字用 `inlineStr` 直接内联，**不建共享字符串表**。
共享字符串是给"同一个字符串重复上千次"省空间用的，我们一张表才一百多行，
省不了多少，却要多维护一张索引表——而且改一个字就得同步两处。

⚠️ 这个写入器**只负责生成新文件**，不做"在原表上改几个单元格"。
后者要处理共享字符串索引、样式、合并单元格，很容易把原文件的其它东西弄坏。
我们的用法是"读出来 → 在内存里改 → 整体重写"，简单且不会留下半截状态。

用法：
    python 工具/写表格.py 输出.xlsx 输入.csv
"""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""

WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="{title}" sheetId="1" r:id="rId1"/></sheets>
</workbook>
"""

WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""


def column_name(index: int) -> str:
    """0 → A，25 → Z，26 → AA。"""

    name = ''
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord('A') + remainder) + name
    return name


def _cell(reference: str, value: str) -> str:
    if value == '':
        return f'<c r="{reference}"/>'
    # xml:space="preserve" 是必须的：不加的话 Excel 会把单元格首尾的空格吃掉，
    # 而我们的备注列经常以空格结尾。
    return (
        f'<c r="{reference}" t="inlineStr">'
        f'<is><t xml:space="preserve">{escape(value)}</t></is></c>'
    )


def write_xlsx(path: Path, rows: list[list[str]], *, title: str = 'Sheet1') -> None:
    """把二维表写成 xlsx。第一行是表头。"""

    body: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = ''.join(
            _cell(f'{column_name(column_index)}{row_index}', str(value))
            for column_index, value in enumerate(row)
        )
        body.append(f'<row r="{row_index}">{cells}</row>')

    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(body)}</sheetData>'
        '</worksheet>'
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', CONTENT_TYPES)
        archive.writestr('_rels/.rels', ROOT_RELS)
        archive.writestr('xl/workbook.xml', WORKBOOK.format(title=escape(title)))
        archive.writestr('xl/_rels/workbook.xml.rels', WORKBOOK_RELS)
        archive.writestr('xl/worksheets/sheet1.xml', sheet)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    target = Path(sys.argv[1])
    source = Path(sys.argv[2])

    with source.open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.reader(handle)]
    write_xlsx(target, rows, title=target.stem)
    print(f'已写出 {target}（{len(rows)} 行 × {max(len(r) for r in rows)} 列）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
