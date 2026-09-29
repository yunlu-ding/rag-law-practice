"""切分策略。

为什么要有"策略"这个概念，而不是写一个切分函数：
不同文档的语义边界藏在完全不同的地方——

- **法规**文本，原子单位是"条"（第N条），一条就该是一片；
- 行业标准、技术规范这类规则型文本，边界是作者写好的条款编号（Standard III(B)）；
- 教材、讲义这类**概念型**文本，边界是标题和段落；
- 没有结构的纯文本，只能靠长度兜底。

用一套通用切分器通吃，结果就是每一类都丢一部分。
所以这里做成注册表，按文档自身的特点选策略。

顺序也有讲究：**先认法规，再认通用结构，最后才是长度。**
判断条件是"越来越宽松"的，所以必须从最严格的那个开始问。
"""

from app.rag.splitters.base import SplitChunk, find_best_split_point, starts_mid_word
from app.rag.splitters.legal import (
    article_boundaries,
    count_article_markers,
    looks_legal,
    split_legal_text,
)
from app.rag.splitters.semantic import looks_structured, split_semantic_text
from app.rag.splitters.unstructured import split_unstructured_text

# 注册表：新增策略只要在这里加一行，上传接口和前端下拉框会自动带上它。
SPLITTER_REGISTRY = {
    'legal': split_legal_text,
    'semantic': split_semantic_text,
    'unstructured': split_unstructured_text,
}

# 给用户看的说明。前端下拉框直接展示它。
SPLITTER_DESCRIPTIONS = {
    'auto': '自动判断：法规按条切，其它按结构切，没有结构才按长度切',
    'legal': '法条感知：一条一片，绝不跨条合并，保证引用能精确到"第几条"',
    'semantic': '结构感知：优先按条款编号、标题等作者划好的边界切，超长才降级到句子边界',
    'unstructured': '按长度切：每片固定长度并保留重叠，是最朴素的兜底方案',
}


def choose_splitter(text: str) -> str:
    """自动判断该用哪种策略。

    三级判断，从最严格到最宽松：
      ① 像法规（有足够多"第N条"）→ legal
      ② 有结构标记（标题、编号）→ semantic
      ③ 都没有 → unstructured

    为什么法规要单独一档，而不是并进 ②：
    因为 ② 允许"不超长就合并"，而这个口子在法规上会造成跨条合并——
    一旦两条法规共享一片，引用就无法精确到某一条。
    """

    if looks_legal(text):
        return 'legal'
    return 'semantic' if looks_structured(text) else 'unstructured'


__all__ = [
    'SPLITTER_REGISTRY',
    'SPLITTER_DESCRIPTIONS',
    'SplitChunk',
    'article_boundaries',
    'choose_splitter',
    'count_article_markers',
    'find_best_split_point',
    'looks_legal',
    'looks_structured',
    'split_legal_text',
    'split_semantic_text',
    'split_unstructured_text',
    'starts_mid_word',
]
