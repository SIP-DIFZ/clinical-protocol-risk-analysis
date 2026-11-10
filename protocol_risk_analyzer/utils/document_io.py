from pathlib import Path

from protocol_risk_analyzer.chunking import DocumentChunk, chunk_pages
from protocol_risk_analyzer.io import PageText, read_document


def read_document_pages(input_path: Path) -> list[PageText]:
    return read_document(input_path)


def chunk_document_pages(
    pages: list[PageText],
    *,
    max_chars: int = 14_000,
    overlap_chars: int = 1_200,
) -> list[DocumentChunk]:
    return chunk_pages(pages, max_chars=max_chars, overlap_chars=overlap_chars)


__all__ = ["DocumentChunk", "PageText", "chunk_document_pages", "read_document_pages"]
