"""沿用 ParsedText→Text 的解析分层，额外把表格文字接入文本证据。"""

import asyncio
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from paperqa.types import ChunkMetadata, Doc, ParsedMetadata, ParsedText, Text
from paperqa.utils import ImpossibleParsingError, clean_invalid_unicode, stable_id


@runtime_checkable
class PDFParser(Protocol):
    def __call__(self, path: str | Path, **kwargs) -> ParsedText: ...


@runtime_checkable
class AsyncPDFParser(Protocol):
    async def __call__(self, path: str | Path, **kwargs) -> ParsedText: ...


PDFParserFn = PDFParser | AsyncPDFParser


def resolve_page_range(page_range, page_count: int) -> range:
    """外部 1 起始、闭区间，转换为解析库的 0 起始下标。"""
    if page_range is None:
        return range(page_count)
    start, end = (page_range, page_range) if isinstance(page_range, int) else page_range
    if not 1 <= start <= end <= page_count:
        raise ValueError("PDF 页码范围无效")
    return range(start - 1, end)


def _make_chunk(doc: Doc, content: str, pages: list[int], order: int, kind="body") -> Text:
    key = stable_id(f"{doc.dockey}:{order}:{kind}:{content}")
    label = f"PDF页{','.join(map(str, pages))}" if pages else "文本"
    return Text(
        chunk_id=key,
        text=content,
        name=f"{doc.docname} {label} #{order}",
        doc=doc,
        order=order,
        pages=pages,
        kind=kind,
    )


def _split(text: str, size: int, overlap: int):
    if overlap >= size:
        raise ValueError("overlap 必须小于 chunk_chars")
    start = 0
    while start < len(text):
        yield text[start : start + size]
        if start + size >= len(text):
            break
        start += size - overlap


def chunk_text(parsed: ParsedText, doc: Doc, chunk_chars=1800, overlap=180) -> list[Text]:
    return [
        _make_chunk(doc, s, [], i)
        for i, s in enumerate(_split(parsed.reduce_content(), chunk_chars, overlap))
        if s.strip()
    ]


def chunk_pdf(
    parsed: ParsedText, doc: Doc, chunk_chars=1800, overlap=180, table_context_chars=2000,
    layout_tables=None,
) -> list[Text]:
    if not isinstance(parsed.content, dict):
        raise ImpossibleParsingError("PDF 解析结果必须按页组织")
    result = []
    for page_num, value in sorted(parsed.content.items(), key=lambda x: int(x[0])):
        body, media = (value, []) if isinstance(value, str) else value
        body = clean_invalid_unicode(body)
        page = int(page_num)
        table_texts = [m for m in media if m.info.get("type") == "table" and m.text]
        uncertain = (
            bool(re.search(r"(?m)^\s*(?:Table\s+\d+|表\s*\d+)\s*[:：.]", body))
            and not table_texts
        )
        if uncertain:
            doc.parse_warnings.append(f"PDF第{page}页含表题，但没有恢复结构化列关系。")
        for part in _split(body, chunk_chars, overlap):
            if part.strip():
                chunk = _make_chunk(doc, part, [page], len(result))
                chunk.layout_uncertain = uncertain
                result.append(chunk)
        for m in media:
            if m.info.get("type") != "table" or not m.text:
                continue
            # 同页首尾上下文帮助定位表题/表注；不猜多级列关系，也不拆开整表。
            half = table_context_chars // 2
            context = (
                body
                if len(body) <= table_context_chars
                else (body[:half] + "\n[同页正文中间已省略]\n" + body[-half:] if half else "")
            )
            text = f"[解析器导出的表格，列关系仍需核对]\n{m.text}\n[同页正文]\n{context}"
            result.append(_make_chunk(doc, text, [page], len(result), "table"))
        # 布局表格独立补充为片段，原正文的“不确定”标记继续保留。
        for region in (layout_tables or {}).get(page, []):
            chunk = _make_chunk(doc, region["text"], [page], len(result), "table")
            chunk.layout_uncertain = region.get("layout_uncertain", False)
            result.append(chunk)
    if not result:
        raise ImpossibleParsingError("未提取到可用文字；首版不提供 OCR")
    parsed.metadata.chunk_metadata = ChunkMetadata(size=chunk_chars, overlap=overlap)
    return result


def parse_text(path: str | Path) -> ParsedText:
    text = Path(path).read_text(encoding="utf-8-sig")
    return ParsedText(
        content=text,
        metadata=ParsedMetadata(
            parsing_libraries=["utf-8"], total_parsed_text_length=len(text)
        ),
    )


async def read_doc(path: str | Path, doc: Doc, settings) -> list[Text]:
    path = Path(path)
    p = settings.parsing
    if path.suffix.lower() == ".txt":
        parsed = await asyncio.to_thread(parse_text, path)
        texts = chunk_text(parsed, doc, p.chunk_chars, p.overlap)
    elif path.suffix.lower() == ".pdf":
        # 延迟导入：插件本身会反向导入 paperqa.types/readers/utils。
        from paperqa_pymupdf import parse_pdf_to_pages

        parsed = await asyncio.to_thread(
            parse_pdf_to_pages, path, parse_media=True, full_page=False
        )
        layout_tables = None
        if p.table_layout_fallback:
            from paperqa.table_layout import read_ruled_tables

            layout_tables = await asyncio.to_thread(read_ruled_tables, path)
        texts = chunk_pdf(
            parsed, doc, p.chunk_chars, p.overlap, p.table_context_chars, layout_tables
        )
        doc.parse_warnings.append("表格文字来自自动解析；跨页/多级表头未自动证明正确。")
    else:
        raise ImpossibleParsingError("第一版仅接收 PDF 和 UTF-8 TXT")
    if not texts:
        raise ImpossibleParsingError("文件没有可用文字")
    return texts
