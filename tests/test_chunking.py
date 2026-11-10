from protocol_risk_analyzer.chunking import chunk_pages
from protocol_risk_analyzer.io import PageText


def test_chunk_pages_creates_multiple_chunks_with_overlap() -> None:
    pages = [
        PageText(page_number=1, text="A" * 1200),
        PageText(page_number=2, text="B" * 1200),
        PageText(page_number=3, text="C" * 1200),
        PageText(page_number=4, text="D" * 1200),
    ]

    chunks = chunk_pages(pages, max_chars=3000, overlap_chars=600)

    assert len(chunks) >= 2
    assert chunks[0].start_page == 1
    assert chunks[0].end_page >= 2
    assert chunks[-1].end_page == 4


def test_chunk_pages_validates_inputs() -> None:
    try:
        chunk_pages([], max_chars=3000, overlap_chars=600)
        assert False, "Expected ValueError for empty pages"
    except ValueError as exc:
        assert "empty pages" in str(exc).lower()
