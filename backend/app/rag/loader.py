from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.utils.text import clean_text, count_long_lines, unwrap_pdf_lines

logger = logging.getLogger(__name__)

_MARKDOWN_HEADING = re.compile(r'^(#{1,6})\s+(.*)$')


@dataclass
class LoadedSection:
    """文档里的一个结构单元。

    为什么解析阶段就要按"结构单元"切，而不是直接吐一大段纯文本：
    后面切分时，这些结构边界（标题、页码）就是最准的切分点。
    如果解析阶段把结构信息丢掉，切分阶段就只能按长度猜，
    而按长度猜会把一句话、一条规则拦腰截断。
    """

    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """解析结果。"""

    filename: str
    file_type: str
    parser_name: str
    sections: list[LoadedSection]
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return '\n\n'.join(section.text for section in self.sections if section.text)

    @property
    def char_count(self) -> int:
        return len(self.full_text)


def _detect_encoding(raw: bytes) -> str:
    """探测文本编码。

    中文材料经常是 GBK 而不是 UTF-8，硬按 UTF-8 解会得到一堆乱码，
    而乱码一旦入库，检索就会彻底失效且很难查原因。
    """

    try:
        import chardet

        detected = chardet.detect(raw[:20000]) or {}
        return detected.get('encoding') or 'utf-8'
    except Exception:  # noqa: BLE001
        return 'utf-8'


def _load_text_file(file_path: Path) -> LoadedDocument:
    raw = file_path.read_bytes()
    encoding = _detect_encoding(raw)
    text = clean_text(raw.decode(encoding, errors='replace'))
    return LoadedDocument(
        filename=file_path.name,
        file_type='txt',
        parser_name='plain_text_loader',
        sections=[LoadedSection(text=text, metadata={'section_type': 'full_text', 'section_index': 0})],
    )


def _split_markdown_sections(text: str) -> list[LoadedSection]:
    """按 Markdown 标题切段落。

    标题本身就是作者划下的语义边界，用它切最准。
    没有标题的文档会退化成"整篇一段"，后面的切分策略再兜底。
    """

    sections: list[LoadedSection] = []
    current_title = '（正文开始）'
    current_lines: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index
        content = clean_text('\n'.join(current_lines))
        if not content:
            return
        sections.append(
            LoadedSection(
                text=content,
                metadata={
                    'section_type': 'markdown_heading',
                    'section_title': current_title,
                    'section_index': section_index,
                },
            )
        )
        section_index += 1

    for line in text.splitlines():
        match = _MARKDOWN_HEADING.match(line.strip())
        if match:
            flush()
            current_title = match.group(2).strip() or '未命名标题'
            current_lines = [line]
            continue
        current_lines.append(line)

    flush()
    return sections


def _load_markdown_file(file_path: Path) -> LoadedDocument:
    raw = file_path.read_bytes()
    encoding = _detect_encoding(raw)
    text = clean_text(raw.decode(encoding, errors='replace'))
    return LoadedDocument(
        filename=file_path.name,
        file_type='md',
        parser_name='markdown_loader',
        sections=_split_markdown_sections(text),
    )


def _load_pdf_file(file_path: Path) -> LoadedDocument:
    """按页解析 PDF。

    这一步同时承担**质检**职责：抽出多少字、有多少空白页、
    有多少疑似串行的超长行，都会记进 warnings。
    原因是"这份 PDF 其实没解析出东西"是一件必须让用户看见的事——
    否则用户会以为文件进库了，然后奇怪为什么搜不到。
    """

    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    sections: list[LoadedSection] = []
    warnings: list[str] = []
    empty_pages = 0
    long_line_total = 0

    for page_index, page in enumerate(reader.pages):
        # 先 clean_text 再去断行：清洗会把中文字之间的空格去掉，
        # 让"还原断行"的判断更准（否则"证券期货 "和"投资者"会被当成
        # 两侧不同的字，接起来时多出一个空格）。
        page_text = unwrap_pdf_lines(clean_text(page.extract_text() or ''))
        long_line_total += count_long_lines(page_text)
        if not page_text:
            empty_pages += 1
        sections.append(
            LoadedSection(
                text=page_text,
                metadata={
                    'section_type': 'pdf_page',
                    'page_number': page_index + 1,
                    'section_index': page_index,
                },
            )
        )

    page_count = len(reader.pages)
    if page_count and empty_pages / page_count >= 0.5:
        warnings.append(
            f'{empty_pages}/{page_count} 页没有抽出文本，这份 PDF 可能是扫描件。'
            f'扫描件需要 OCR，当前版本还没有接入。'
        )
    elif empty_pages:
        warnings.append(f'{empty_pages}/{page_count} 页没有抽出文本，可能是章节分隔页或含大量图片。')

    if long_line_total > page_count:
        warnings.append(
            f'检测到 {long_line_total} 行超长文本，可能是双栏排版被串成了单栏，'
            f'解析质量存疑，建议抽查。'
        )

    return LoadedDocument(
        filename=file_path.name,
        file_type='pdf',
        parser_name='pypdf_loader',
        sections=sections,
        page_count=page_count,
        warnings=warnings,
    )


def _load_docx_file(file_path: Path) -> LoadedDocument:
    from docx import Document as DocxDocument

    document = DocxDocument(str(file_path))
    sections: list[LoadedSection] = []
    warnings: list[str] = []
    current_title = '（正文开始）'
    current_lines: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index
        content = clean_text('\n'.join(current_lines))
        if not content:
            return
        sections.append(
            LoadedSection(
                text=content,
                metadata={
                    'section_type': 'docx_heading_block',
                    'section_title': current_title,
                    'section_index': section_index,
                },
            )
        )
        section_index += 1

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = getattr(paragraph.style, 'name', '') or ''
        if style_name.startswith('Heading'):
            flush()
            current_title = text
            current_lines = [text]
            continue
        current_lines.append(text)

    flush()

    table_count = len(document.tables)
    if table_count:
        # 这一条属于"缺口可见"：现在还不能把表格切好入库，
        # 那就明确标注出来，而不是让它静默消失。
        warnings.append(
            f'文档里有 {table_count} 个表格，当前版本只读取了正文，表格内容尚未入库。'
        )

    return LoadedDocument(
        filename=file_path.name,
        file_type='docx',
        parser_name='docx_loader',
        sections=sections,
        warnings=warnings,
    )


def build_loaded_document_from_text(filename: str, text: str) -> LoadedDocument:
    """把一段纯文本包装成解析结果，用于快速验证链路。"""

    cleaned = clean_text(text)
    suffix = Path(filename).suffix.lower().lstrip('.') or 'txt'
    sections = (
        _split_markdown_sections(cleaned)
        if suffix == 'md'
        else [LoadedSection(text=cleaned, metadata={'section_type': 'inline_text', 'section_index': 0})]
    )
    return LoadedDocument(
        filename=filename,
        file_type=suffix,
        parser_name='inline_text_loader',
        sections=sections,
    )


# ---------------------------------------------------------------------------
# HTML
#
# 为什么要专门写一个 HTML 解析器，而不是"用正则把标签去掉"：
#
# 监管规则在网上的正式发布页，绝大多数是"整页另存"下来的。这类文件里
# 真正的法条通常只占 5%~10%，其余是导航栏、面包屑、页脚、内联 <style>、
# 埋点和统计脚本。正则去标签会把这一整锅都留下来——
# 结果是：语料里混进了几百字 CSS，检索时它和法条一样有机会被召回。
#
# 实测两份 HTML 语料：整页 147,115 / 99,767 字，真正的正文只有
# 8,981 / 7,789 字。**噪声是正文的 13 倍以上。**
#
# 所以这里的做法是"先找正文容器，再从容器里取段落"，而不是"整页去标签"。
# ---------------------------------------------------------------------------

# 这些标签里的内容一定不是正文。
#
# ⚠️ 注意这里**没有 head**，这是一个踩过的坑。
#
# 最初把 head 整块丢掉，理由是"head 里只有标题、meta 和样式"。
# 这对结构规范的页面成立，但政府网站的页面常常不规范——
# 实测某份国务院公报的页面里，真正的正文 `<div class="pages_content">`
# 因为前面有未闭合的标签，被 HTML 解析器归到了 `head` 里面。
# 结果：一份 18 万字的规章正文，只抽出了 184 字（导航条的文字）。
#
# 所以改成"丢掉 head 里那些确定不是正文的标签"（title/meta/link），
# 而不是丢掉 head 本身。这样无论解析器把正文放在哪个容器里，都找得到。
_HTML_DROP_TAGS = (
    'script', 'style', 'noscript', 'iframe', 'svg',
    'title', 'meta', 'link', 'base', 'template',
)

# 块级标签：遇到它们就断句。段落边界是法条边界的基础，
# 丢了这个信息，后面就只能靠正则猜——能猜对，但不该退到那一步。
_HTML_BLOCK_TAGS = {
    'p', 'div', 'li', 'tr', 'td', 'th', 'br', 'section', 'article',
    'blockquote', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dd', 'dt',
}

# 候选正文容器的 id/class 特征。命中这些词的节点才参与打分。
_HTML_CONTENT_HINT = re.compile(r'(^|[\s_-])(content|article|main|text|zoom|body)([\s_-]|$)', re.I)

# 章节标题。用它把正文切成 section，让"第几章"成为可展示的定位信息。
_CHINESE_CHAPTER = re.compile(r'^第([一二三四五六七八九十百零〇\d]+)章\s*(.*)$')


def _html_paragraphs(node) -> list[str]:
    """把一个 DOM 节点拍平成"段落列表"。

    关键点是**在块级标签的边界断行**。直接调 text_content() 会把
    `<p>第一条 ...</p><p>第二条 ...</p>` 粘成一行，
    于是"第一条"和"第二条"的边界就消失了——而这个边界正是法条级检索的依据。
    """

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        text = re.sub(r'[\s\u3000]+', ' ', ''.join(buffer)).strip()
        if text:
            paragraphs.append(text)
        buffer.clear()

    def walk(element) -> None:
        if element.text:
            buffer.append(element.text)
        for child in element:
            tag = child.tag if isinstance(child.tag, str) else ''
            if tag in _HTML_BLOCK_TAGS:
                flush()
                walk(child)
                flush()
            else:
                walk(child)
            if child.tail:
                buffer.append(child.tail)

    walk(node)
    flush()
    return paragraphs


def _pick_html_container(tree) -> tuple[object, str, int]:
    """挑出正文容器。

    打分口径：**条款标记数优先，长度次之。**

    为什么不是"谁最长就选谁"：整页 <body> 一定是最长的，
    但它也一定包含全部噪声。而条款标记（第N条 / 第N章）几乎只出现在正文里，
    用它当主轴，能直接把导航和页脚筛掉。

    返回 (容器, 说明, 整页文本长度)，说明会写进解析质检报告——
    将来某份文件解析歪了，看这一条就能知道当初选的是哪个容器。
    """

    # 遍历**整个文档**，而不是遍历 `root.iter()`。
    #
    # 这里有三个叠在一起的坑，每一个都能让正文一个字都抽不到：
    #
    # 1. 不规范页面的正文可能被解析器归到 `head` 里，只搜 `body` 会漏掉。
    # 2. lxml 的 `fromstring` 返回的**不一定包含全部内容**：文档开头
    #    如果有解析器不认识的东西，它可能把后面的内容放到 `<html>` **外面**，
    #    成为同级节点。
    # 3. 于是出现了看起来自相矛盾的现象——
    #    `tree.xpath('//*[contains(@class,"pages_content")]')` 能找到 42,694 字的正文，
    #    而 `tree.text_content()` 只有 3,482 字。
    #    原因：**xpath 里的 `//` 是绝对路径**（从文档根算起），
    #    而 `text_content()` / `iter()` 只走自己的子树。
    #
    # 实测后果：这份国务院公报的正文一个字都没抽到，只剩导航条。
    # 修法就是统一用绝对 xpath 遍历。
    root = tree.getroottree().getroot()
    elements = root.xpath('//*')
    full_text_length = sum(
        len(' '.join(node.text_content().split())) for node in root.xpath('/*')
    )

    from app.rag.splitters import count_article_markers

    best_node = None
    best_score = -1
    best_signature = ''

    for node in elements:
        tag = node.tag if isinstance(node.tag, str) else ''
        if tag not in ('div', 'article', 'section', 'td', 'main'):
            continue
        signature = f"{node.get('id') or ''} {node.get('class') or ''}".strip()
        if not _HTML_CONTENT_HINT.search(signature):
            continue

        text = ' '.join(node.text_content().split())
        if len(text) < 200:
            continue

        score = count_article_markers(text) * 1000 + len(text)
        if score > best_score:
            best_score, best_node, best_signature = score, node, signature

    if best_node is None:
        # 没有任何"看起来像正文"的容器。这时退回整页，但必须把这件事说出来——
        # 静默退回整页，等于让噪声混进语料而没人知道。
        return root, '（未找到正文容器，退回整页）', full_text_length

    return best_node, f'{best_node.tag}[{best_signature}]', full_text_length


def _split_html_legal_sections(
    paragraphs: list[str],
    *,
    document_title: str,
) -> list[LoadedSection]:
    """按"章"把段落组装成 section。

    只在**识别到章节标记**时才分节。识别不到就整篇作为一个 section，
    交给切分阶段处理——不假装自己找到了结构。
    """

    chapter_count = sum(1 for line in paragraphs if _CHINESE_CHAPTER.match(line))
    if chapter_count < 2:
        text = clean_text('\n'.join(paragraphs))
        return [
            LoadedSection(
                text=text,
                metadata={
                    'section_type': 'html_body',
                    'section_title': document_title,
                    'section_index': 0,
                },
            )
        ]

    sections: list[LoadedSection] = []
    current_title = f'{document_title}（前言）'
    current_lines: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index
        text = clean_text('\n'.join(current_lines))
        if not text:
            return
        sections.append(
            LoadedSection(
                text=text,
                metadata={
                    'section_type': 'html_chapter',
                    'section_title': current_title,
                    'section_index': section_index,
                },
            )
        )
        section_index += 1

    for line in paragraphs:
        match = _CHINESE_CHAPTER.match(line)
        if match:
            flush()
            suffix = match.group(2).strip()
            current_title = f'第{match.group(1)}章 {suffix}'.strip()
            current_lines = [line]
            continue
        current_lines.append(line)

    flush()
    return sections


def _load_html_file(file_path: Path) -> LoadedDocument:
    """解析"整页另存"的 HTML 监管规则页。"""

    from lxml import html as lxml_html

    raw = file_path.read_bytes()
    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors='replace')

    warnings: list[str] = []
    try:
        tree = lxml_html.fromstring(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f'HTML 解析失败：{exc}') from exc

    # 取文档根节点：fromstring 返回的不一定是 <html>，见 _pick_html_container 的说明。
    root = tree.getroottree().getroot()

    # 标题要在丢标签**之前**取——title 现在也在丢弃名单里。
    title_node = root.find('.//title')
    document_title = (title_node.text or '').strip() if title_node is not None else ''
    if not document_title:
        document_title = file_path.stem

    dropped = 0
    for tag in _HTML_DROP_TAGS:
        for node in root.xpath(f'//{tag}'):
            parent = node.getparent()
            if parent is not None:
                dropped += 1
                parent.remove(node)

    container, description, full_length = _pick_html_container(root)
    paragraphs = _html_paragraphs(container)
    body_text = clean_text('\n'.join(paragraphs))

    if not body_text:
        warnings.append('正文容器里没有抽出任何文本，这份 HTML 可能只有脚本渲染的内容。')
    else:
        noise_ratio = 1 - len(body_text) / full_length if full_length else 0.0
        if noise_ratio > 0.5:
            warnings.append(
                f'整页 {full_length} 字里只取出 {len(body_text)} 字正文，'
                f'丢弃了 {noise_ratio:.0%} 的页面噪声（导航/样式/脚本）。'
                f'这是"整页另存"的正常现象，但建议抽查正文首尾是否完整。'
            )
            # 这一条不是"出错了"，而是"我们做了什么"。放进 warnings 是为了留痕。

    sections = _split_html_legal_sections(paragraphs, document_title=document_title)
    chapter_count = sum(
        1 for section in sections if section.metadata.get('section_type') == 'html_chapter'
    )
    if body_text and chapter_count < 2:
        warnings.append('未识别到章节标记，整个正文按一段处理（后续按条款/长度切分兜底）。')

    logger.info(
        '[LOADER] HTML 正文定位: file=%s 容器=%s 整页=%s字 正文=%s字 段落=%s 章节=%s',
        file_path.name,
        description,
        full_length,
        len(body_text),
        len(paragraphs),
        chapter_count,
    )

    return LoadedDocument(
        filename=file_path.name,
        file_type='html',
        parser_name='lxml_html_loader',
        sections=sections,
        warnings=warnings,
    )


LOADER_REGISTRY = {
    'txt': _load_text_file,
    'md': _load_markdown_file,
    'pdf': _load_pdf_file,
    'docx': _load_docx_file,
    'html': _load_html_file,
    'htm': _load_html_file,
}


def load_document(file_path: str | Path) -> LoadedDocument:
    """按文件类型分派解析器。

    为什么分派而不是用一个通用解析器：
    txt / md / pdf / docx / html 的结构信息藏在完全不同的地方
    （纯文本没有、Markdown 在标题、PDF 在页、Word 在样式、
      HTML 在正文容器的段落里），
    用一个通用解析器相当于放弃所有结构，然后靠长度硬切——
    而长度硬切正是"把一条规则切成两半"的根源。
    """

    path = Path(file_path)
    file_type = path.suffix.lower().lstrip('.')
    loader = LOADER_REGISTRY.get(file_type)
    if loader is None:
        raise ValueError(
            f'暂不支持的文件类型：.{file_type}（当前支持 txt / md / pdf / docx / html）'
        )

    logger.info('[LOADER] 解析开始: file=%s type=%s', path.name, file_type)
    loaded = loader(path)
    logger.info(
        '[LOADER] 解析完成: file=%s parser=%s sections=%s pages=%s chars=%s warnings=%s',
        path.name,
        loaded.parser_name,
        len(loaded.sections),
        loaded.page_count,
        loaded.char_count,
        loaded.warnings,
    )
    return loaded
