from __future__ import annotations

from dataclasses import dataclass

from protocol_risk_analyzer.io import PageText


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: int
    start_page: int
    end_page: int
    text: str


def chunk_pages(
    pages: list[PageText],
    max_chars: int = 14_000,
    overlap_chars: int = 1_200,
) -> list[DocumentChunk]:
    if not pages:
        raise ValueError("Cannot chunk empty pages list.")
    if max_chars <= 500:
        raise ValueError("max_chars must be > 500.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and less than max_chars.")

    chunks: list[DocumentChunk] = []
    chunk_id = 1
    start_idx = 0

    while start_idx < len(pages):
        current = []
        current_chars = 0
        end_idx = start_idx
        while end_idx < len(pages):
            page = pages[end_idx]
            block = f"\n\n[PAGE {page.page_number}]\n{page.text}"
            next_size = current_chars + len(block)
            if current and next_size > max_chars:
                break
            current.append(block)
            current_chars = next_size
            end_idx += 1

        start_page = pages[start_idx].page_number
        end_page = pages[end_idx - 1].page_number
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                start_page=start_page,
                end_page=end_page,
                text="".join(current).strip(),
            )
        )
        chunk_id += 1

        if end_idx >= len(pages):
            break

        restart_idx = _find_overlap_restart(pages, end_idx, overlap_chars)
        if restart_idx <= start_idx:
            restart_idx = start_idx + 1
        start_idx = restart_idx

    return chunks


def _find_overlap_restart(pages: list[PageText], end_idx: int, overlap_chars: int) -> int:
    if overlap_chars == 0:
        return end_idx

    total = 0
    idx = end_idx - 1
    while idx > 0:
        total += len(pages[idx].text)
        if total >= overlap_chars:
            return idx
        idx -= 1
    return 0
