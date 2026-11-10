from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from pypdf import PdfReader

# DOCX files do not store layout pages reliably unless rendered by Word.
# Use conservative heuristics to avoid severe undercounting.
_DOCX_PSEUDO_PAGE_MAX_CHARS = 2200
_DOCX_PSEUDO_PAGE_MAX_WORDS = 420
_DOCX_PSEUDO_PAGE_MAX_BLOCKS = 18
_DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def read_document(path: Path) -> list[PageText]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".txt":
        return _read_txt(path)
    if suffix == ".docx":
        return _read_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .docx")


def _read_pdf(path: Path) -> list[PageText]:
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page_number=index, text=text.strip()))
    if not pages:
        raise ValueError(f"No pages found in PDF: {path}")
    return pages


def _read_txt(path: Path) -> list[PageText]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"Text file is empty: {path}")
    return [PageText(page_number=1, text=stripped)]


def _read_docx(path: Path) -> list[PageText]:
    doc = Document(str(path))
    doc_blocks = _iter_docx_text_blocks(doc)
    xml_blocks = _iter_docx_text_blocks_from_xml(path)
    blocks_source = _select_richer_docx_blocks(doc_blocks, xml_blocks)
    blocks: list[tuple[str, bool]] = []
    for block_text, has_page_break in blocks_source:
        if block_text:
            blocks.append((block_text, has_page_break))
        elif has_page_break and blocks:
            last_text, _ = blocks[-1]
            blocks[-1] = (last_text, True)

    if not blocks:
        raise ValueError(f"DOCX file is empty: {path}")

    reported_page_count = _read_docx_reported_page_count(path)
    estimated_total_chars = sum(len(block_text) + 2 for block_text, _ in blocks)
    estimated_total_words = sum(len(block_text.split()) for block_text, _ in blocks)
    estimated_by_chars = ceil(estimated_total_chars / _DOCX_PSEUDO_PAGE_MAX_CHARS)
    estimated_by_words = ceil(estimated_total_words / _DOCX_PSEUDO_PAGE_MAX_WORDS)
    estimated_by_blocks = ceil(len(blocks) / _DOCX_PSEUDO_PAGE_MAX_BLOCKS)
    estimated_page_count = max(1, estimated_by_chars, estimated_by_words, estimated_by_blocks)
    explicit_page_count = 1 + sum(1 for _, hard_break_after in blocks if hard_break_after)
    if reported_page_count is not None and reported_page_count > 1:
        # Prefer Word-reported page count when present.
        target_page_count = max(reported_page_count, explicit_page_count)
    else:
        target_page_count = max(estimated_page_count, explicit_page_count)

    pages = _paginate_docx_blocks(blocks, target_page_count)
    if not pages:
        raise ValueError(f"DOCX file is empty: {path}")
    return pages


def _paginate_docx_blocks(blocks: list[tuple[str, bool]], target_page_count: int) -> list[PageText]:
    if not blocks:
        return []
    target_page_count = max(1, target_page_count)
    total_chars = sum(len(block_text) + 2 for block_text, _ in blocks)
    target_chars = max(900, round(total_chars / target_page_count))

    page_blocks: list[list[str]] = []
    current_blocks: list[str] = []
    current_chars = 0

    for block_text, hard_break_after in blocks:
        projected = current_chars + len(block_text) + 2
        remaining_pages = target_page_count - len(page_blocks)
        if (
            current_blocks
            and projected > target_chars
            and remaining_pages > 1
        ):
            page_blocks.append(current_blocks)
            current_blocks = []
            current_chars = 0
        current_blocks.append(block_text)
        current_chars += len(block_text) + 2
        if hard_break_after and current_blocks:
            page_blocks.append(current_blocks)
            current_blocks = []
            current_chars = 0

    if current_blocks:
        page_blocks.append(current_blocks)

    page_blocks = _rebalance_page_blocks(page_blocks, target_page_count)
    return [
        PageText(page_number=index, text="\n\n".join(blocks_for_page).strip())
        for index, blocks_for_page in enumerate(page_blocks, start=1)
        if any(blocks_for_page)
    ]


def _rebalance_page_blocks(page_blocks: list[list[str]], target_page_count: int) -> list[list[str]]:
    blocks = [list(page) for page in page_blocks if page]
    if not blocks:
        return []

    while len(blocks) > target_page_count and len(blocks) > 1:
        merge_index = min(
            range(1, len(blocks)),
            key=lambda idx: sum(len(item) for item in blocks[idx]),
        )
        blocks[merge_index - 1].extend(blocks[merge_index])
        del blocks[merge_index]

    while len(blocks) < target_page_count:
        split_index = max(
            range(len(blocks)),
            key=lambda idx: (len(blocks[idx]), sum(len(item) for item in blocks[idx])),
        )
        if len(blocks[split_index]) < 2:
            break
        midpoint = max(1, len(blocks[split_index]) // 2)
        left = blocks[split_index][:midpoint]
        right = blocks[split_index][midpoint:]
        blocks[split_index : split_index + 1] = [left, right]

    return blocks


def _read_docx_reported_page_count(path: Path) -> int | None:
    try:
        with ZipFile(path) as archive:
            xml_bytes = archive.read("docProps/app.xml")
    except (KeyError, FileNotFoundError, OSError):
        return None

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    namespace = {"ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"}
    pages_node = root.find("ep:Pages", namespace)
    if pages_node is None or not (pages_text := (pages_node.text or "").strip()).isdigit():
        return None
    page_count = int(pages_text)
    return page_count if page_count > 0 else None


def _iter_docx_text_blocks(doc: DocxDocument) -> list[tuple[str, bool]]:
    blocks: list[tuple[str, bool]] = []
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, doc)
            text = paragraph.text.strip()
            blocks.append((text, _paragraph_has_page_break(paragraph)))
            continue
        if isinstance(child, CT_Tbl):
            table = Table(child, doc)
            table_rows: list[str] = []
            for row in table.rows:
                cells = [" ".join(cell.text.split()) for cell in row.cells if cell.text and cell.text.strip()]
                if cells:
                    table_rows.append(" | ".join(cells))
            table_text = "\n".join(table_rows).strip()
            blocks.append((table_text, False))
    return blocks


def _iter_docx_text_blocks_from_xml(path: Path) -> list[tuple[str, bool]]:
    try:
        with ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (KeyError, FileNotFoundError, OSError):
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    blocks: list[tuple[str, bool]] = []
    body = root.find("w:body", _DOCX_NS)
    if body is None:
        return blocks

    for paragraph in body.iterfind(".//w:p", _DOCX_NS):
        text_parts = []
        for text_node in paragraph.findall(".//w:t", _DOCX_NS):
            if text_node.text:
                text_parts.append(text_node.text)
        text = "".join(text_parts).strip()
        has_page_break = False
        for br in paragraph.findall(".//w:br", _DOCX_NS):
            if br.attrib.get(f"{{{_DOCX_NS['w']}}}type") == "page":
                has_page_break = True
                break
        if not has_page_break and paragraph.find(".//w:lastRenderedPageBreak", _DOCX_NS) is not None:
            has_page_break = True
        blocks.append((text, has_page_break))
    return blocks


def _select_richer_docx_blocks(
    primary_blocks: list[tuple[str, bool]],
    alternate_blocks: list[tuple[str, bool]],
) -> list[tuple[str, bool]]:
    if not primary_blocks:
        return alternate_blocks
    if not alternate_blocks:
        return primary_blocks

    primary_chars = sum(len(text) for text, _ in primary_blocks)
    alternate_chars = sum(len(text) for text, _ in alternate_blocks)
    primary_nonempty = sum(1 for text, _ in primary_blocks if text)
    alternate_nonempty = sum(1 for text, _ in alternate_blocks if text)
    if alternate_chars > primary_chars * 1.15 or alternate_nonempty > primary_nonempty * 1.2:
        return alternate_blocks
    return primary_blocks


def _paragraph_has_page_break(paragraph: Paragraph) -> bool:
    for run in paragraph.runs:
        if _run_has_page_break(run):
            return True
    return False


def _run_has_page_break(run: Run) -> bool:
    # Covers explicit page breaks and rendered page break markers.
    if "\f" in run.text:
        return True
    run_xml = run._element.xml  # noqa: SLF001 - required for break-marker detection
    return (
        'w:type="page"' in run_xml
        or "w:lastRenderedPageBreak" in run_xml
        or "lastRenderedPageBreak" in run_xml
    )
