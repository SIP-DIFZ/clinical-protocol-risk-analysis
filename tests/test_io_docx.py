from docx import Document

from protocol_risk_analyzer.io import read_document


def test_read_document_supports_docx(tmp_path) -> None:
    path = tmp_path / "protocol.docx"
    doc = Document()
    doc.add_paragraph("Section 1")
    doc.add_paragraph("Eligibility criteria are strict.")
    doc.save(path)

    pages = read_document(path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Section 1" in pages[0].text
    assert "Eligibility criteria are strict." in pages[0].text


def test_read_document_docx_splits_on_page_break(tmp_path) -> None:
    path = tmp_path / "paged.docx"
    doc = Document()
    doc.add_paragraph("Page one content.")
    doc.add_page_break()
    doc.add_paragraph("Page two content.")
    doc.save(path)

    pages = read_document(path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert "Page one content." in pages[0].text
    assert "Page two content." in pages[1].text


def test_read_document_docx_uses_conservative_fallback_page_sizing(tmp_path) -> None:
    path = tmp_path / "long.docx"
    doc = Document()
    doc.add_paragraph("A" * 2000)
    doc.add_paragraph("B" * 1200)
    doc.save(path)

    pages = read_document(path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_read_document_docx_uses_block_count_fallback(tmp_path) -> None:
    path = tmp_path / "many-blocks.docx"
    doc = Document()
    for i in range(40):
        doc.add_paragraph(f"Short block {i+1}")
    doc.save(path)

    pages = read_document(path)

    assert len(pages) >= 2


def test_read_document_docx_includes_table_text(tmp_path) -> None:
    path = tmp_path / "table.docx"
    doc = Document()
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Column A"
    table.cell(0, 1).text = "Column B"
    table.cell(1, 0).text = "Eligibility"
    table.cell(1, 1).text = "Strict criteria"
    doc.save(path)

    pages = read_document(path)

    assert len(pages) >= 1
    assert "Eligibility" in pages[0].text
    assert "Strict criteria" in pages[0].text


def test_read_document_raises_for_empty_docx(tmp_path) -> None:
    path = tmp_path / "empty.docx"
    Document().save(path)

    try:
        read_document(path)
        assert False, "Expected ValueError for empty DOCX"
    except ValueError as exc:
        assert "DOCX file is empty" in str(exc)
