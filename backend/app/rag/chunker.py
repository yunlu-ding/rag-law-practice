from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.rag.loader import LoadedDocument, LoadedSection
from app.rag.splitters import (
    SPLITTER_REGISTRY,
    choose_splitter,
    starts_mid_word,
)
from app.utils.text import estimate_token_count

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """一条待入库的切片。"""

    chunk_index: int
    content: str
    splitter_name: str
    content_type: str
    section_index: int | None = None
    section_type: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None

    @property
    def token_count(self) -> int:
        return estimate_token_count(self.content)


@dataclass
class ChunkingResult:
    """切分结果 + 质量指标。"""

    records: list[ChunkRecord] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def build_chunks(
    loaded: LoadedDocument,
    *,
    preferred_splitter: str | None = None,
) -> ChunkingResult:
    """把解析结果切成入库切片。

    两步走：
    1. 逐个 section 决定用哪种切分策略（用户指定 / 自动判断）；
    2. 切完之后统计质量指标。

    第 2 步和切分本身一样重要——**没有指标，就没法回答"这样切到底好不好"**。
    所以这里顺手把三个关键数字算出来：切片数量、平均长度、
    以及"从单词中间开始"的比例（残句率）。
    最后一个指标最硬：它直接决定模型读到的是完整句子还是半句话。
    """

    records: list[ChunkRecord] = []
    splitter_usage: dict[str, int] = {}
    global_index = 0

    # 结构判断要做在**文档级**，不是 section 级。
    #
    # 这不是拍脑袋定的，而是被实验打回来一次的结果：
    # 最初按 section 判断（PDF 的 section 是"一页"），
    # 结果单页很少包含 3 个以上标题，于是 214 页的手册里绝大多数页
    # 被判成"无结构"，整体退化成按长度切——残句率 50.5%，
    # 比全程用结构感知切分（0%）还差。
    #
    # 所以改成：先看整篇文档有没有结构，用它作为这一批 section 的默认策略；
    # 再对每个 section 做一次"升级"判断——某一段自己结构明显，就用结构感知。
    document_splitter = _resolve_document_splitter(loaded, preferred_splitter)

    for section in loaded.sections:
        splitter_name = _resolve_section_splitter(section, preferred_splitter, document_splitter)
        splitter = SPLITTER_REGISTRY[splitter_name]
        pieces = splitter(section.text)
        splitter_usage[splitter_name] = splitter_usage.get(splitter_name, 0) + len(pieces)

        for piece in pieces:
            records.append(
                ChunkRecord(
                    chunk_index=global_index,
                    content=piece.content,
                    splitter_name=splitter_name,
                    content_type='text',
                    section_index=_as_int(section.metadata.get('section_index')),
                    section_type=_as_str(section.metadata.get('section_type')),
                    section_title=_as_str(section.metadata.get('section_title')),
                    page_number=_as_int(section.metadata.get('page_number')),
                    start_offset=piece.start_offset,
                    end_offset=piece.end_offset,
                )
            )
            global_index += 1

    stats = _build_stats(loaded, records, splitter_usage)
    logger.info(
        '[CHUNKER] 切分完成: file=%s 切片数=%s 平均长度=%s 最长=%s 残句率=%s 策略分布=%s',
        loaded.filename,
        stats.get('chunk_count'),
        stats.get('avg_length'),
        stats.get('max_length'),
        stats.get('mid_word_ratio'),
        splitter_usage,
    )
    return ChunkingResult(records=records, stats=stats)


def _resolve_document_splitter(loaded: LoadedDocument, preferred: str | None) -> str:
    """决定整篇文档的默认切分策略。

    优先级：用户指定 > 按整篇文档自动判断。
    """

    if preferred and preferred in SPLITTER_REGISTRY:
        return preferred
    return choose_splitter(loaded.full_text)


def _resolve_section_splitter(
    section: LoadedSection,
    preferred: str | None,
    document_splitter: str,
) -> str:
    """决定某一段实际用哪种策略。

    在文档级默认值的基础上做一次"升级"：
    **这一段自己结构明显，就单独用结构感知切分。**

    实际意义在于：一份以叙述为主的材料里，某个章节可能是条款列表，
    这时候按文档级默认值走会把它切成碎片，而它本来有很好的天然边界。
    """

    if preferred and preferred in SPLITTER_REGISTRY:
        return preferred

    # 法规文档**不允许被降级**。
    #
    # 这条判据是为了修一个实测出来的漏切。原来的逻辑是"这一段自己结构明显，
    # 就升级成结构感知"——但它反过来也会生效：法规文档的 section 是"一页"，
    # 一页里往往只有一两条法规，达不到"像法规"的门槛（5 个条界），
    # 于是这一页被**降级**成结构感知切分，页内的条界就被丢掉了。
    #
    # 实测表现：《证券期货投资者适当性管理办法》的第二十九条、
    # 第三条、第四条等十几条都没有成为任何切片的开头，
    # 而是被并进了上一页末尾那一片里——引用它们时会指到隔壁那一条。
    #
    # 所以：文档级判定是 legal，就整篇按 legal 走。
    # "这一页有没有条界"由 split_legal_text 自己处理（它没有条界时会退回结构感知），
    # 不需要外面再替它判断一次。
    if document_splitter == 'legal':
        return 'legal'

    if document_splitter != 'semantic' and choose_splitter(section.text) == 'semantic':
        return 'semantic'
    return document_splitter


def _build_stats(
    loaded: LoadedDocument,
    records: list[ChunkRecord],
    splitter_usage: dict[str, int],
) -> dict:
    """算切分质量指标。"""

    if not records:
        return {
            'chunk_count': 0,
            'avg_length': 0,
            'max_length': 0,
            'min_length': 0,
            'mid_word_start_count': 0,
            'mid_word_ratio': 0.0,
            'splitter_usage': {},
        }

    lengths = [len(record.content) for record in records]
    mid_word_count = 0

    for record in records:
        section_text = _section_text(loaded, record.section_index)
        if section_text and record.start_offset is not None:
            if starts_mid_word(section_text, record.start_offset):
                mid_word_count += 1

    return {
        'chunk_count': len(records),
        'avg_length': round(sum(lengths) / len(lengths)),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'mid_word_start_count': mid_word_count,
        'mid_word_ratio': round(mid_word_count / len(records), 4),
        'splitter_usage': splitter_usage,
    }


def _section_text(loaded: LoadedDocument, section_index: int | None) -> str:
    """按下标取回所属 section 的原文。

    残句率必须拿 section 原文来算，不能用切片自身的文本——
    切片自己当然看不出"开头被切断了"，必须和原始上下文对照。
    """

    if section_index is None or section_index < 0 or section_index >= len(loaded.sections):
        return ''
    return loaded.sections[section_index].text


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _as_str(value: object) -> str | None:
    return str(value) if value is not None else None
