from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable
from uuid import uuid4

from pypdf import PdfReader

from protocol_risk_analyzer.io import read_document

_DOCX_CONVERSION_LOCK = Lock()


def convert_docx_to_pdf_for_analysis(docx_path: Path, pdf_path: Path) -> Path:
    if docx_path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx input, got: {docx_path.suffix}")

    try:
        from docx2pdf import convert  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DOCX analysis requires DOCX->PDF conversion but docx2pdf is not installed. "
            "Install dependencies from requirements.txt and ensure Microsoft Word is available."
        ) from exc

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with _DOCX_CONVERSION_LOCK:
        if pdf_path.exists():
            pdf_path.unlink(missing_ok=True)
        try:
            _run_docx2pdf_conversion(convert, docx_path, pdf_path)
        except Exception as exc:  # pragma: no cover - depends on local Office setup
            repaired_docx_path = pdf_path.parent / f"{docx_path.stem}__repaired_for_conversion_{uuid4().hex[:8]}.docx"
            try:
                _repair_docx_for_conversion(docx_path, repaired_docx_path)
                _run_docx2pdf_conversion(convert, repaired_docx_path, pdf_path)
            except Exception as retry_exc:  # pragma: no cover - depends on local Office setup
                raise RuntimeError(
                    "Failed to convert DOCX to PDF for analysis. Ensure Microsoft Word is installed "
                    "and the document can be opened normally. "
                    f"First attempt error: {_format_exception(exc)}. "
                    f"Retry with repaired DOCX error: {_format_exception(retry_exc)}."
                ) from retry_exc
            finally:
                if repaired_docx_path.exists():
                    repaired_docx_path.unlink(missing_ok=True)

    if not pdf_path.exists():
        raise RuntimeError("DOCX conversion completed without producing a PDF file.")

    _validate_conversion_content(docx_path=docx_path, pdf_path=pdf_path)
    return pdf_path


def _run_docx2pdf_conversion(
    convert_func: Callable[[str, str], None],
    source_docx_path: Path,
    output_pdf_path: Path,
) -> None:
    pythoncom = None
    initialized_com = False
    try:
        import pythoncom  # type: ignore

        pythoncom.CoInitialize()
        initialized_com = True
    except ImportError:
        # Non-Windows environments or setups without pywin32 may not expose pythoncom.
        pythoncom = None

    try:
        convert_func(str(source_docx_path), str(output_pdf_path))
    finally:
        if initialized_com and pythoncom is not None:
            pythoncom.CoUninitialize()


def _repair_docx_for_conversion(docx_path: Path, repaired_docx_path: Path) -> None:
    from docx import Document

    repaired_docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document(str(docx_path))
    doc.save(str(repaired_docx_path))


def _format_exception(exc: Exception) -> str:
    details = str(exc).strip()
    if details:
        return f"{type(exc).__name__}: {details}"
    return type(exc).__name__


def _validate_conversion_content(docx_path: Path, pdf_path: Path) -> None:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover - malformed PDF from external converter
        raise RuntimeError("Converted PDF could not be opened for analysis validation.") from exc

    if not reader.pages:
        raise RuntimeError("Converted PDF has no pages; cannot analyze this document.")

    pdf_pages = read_document(pdf_path)
    docx_pages = read_document(docx_path)
    pdf_chars = sum(len(page.text) for page in pdf_pages)
    docx_chars = sum(len(page.text) for page in docx_pages)

    # A large drop usually indicates conversion failure or inaccessible content.
    if docx_chars > 0 and pdf_chars < max(500, int(docx_chars * 0.5)):
        raise RuntimeError(
            "DOCX->PDF conversion appears incomplete (too little extracted text in PDF). "
            "Please check the source file and conversion environment."
        )
