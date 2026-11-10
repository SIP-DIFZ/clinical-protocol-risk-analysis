from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import protocol_risk_analyzer.docx_conversion as docx_conversion


def test_convert_docx_to_pdf_retries_with_repaired_docx(monkeypatch, tmp_path: Path) -> None:
    source_docx = tmp_path / "protocol.docx"
    source_docx.write_bytes(b"placeholder")
    output_pdf = tmp_path / "analysis.pdf"
    repaired_docx_paths: list[Path] = []

    calls: list[tuple[str, str]] = []

    def fake_convert(src: str, dst: str) -> None:
        calls.append((src, dst))
        if len(calls) == 1:
            raise RuntimeError("first conversion failed")
        repaired_docx_paths.append(Path(src))
        Path(dst).write_bytes(b"%PDF-1.4\n%%EOF")

    monkeypatch.setitem(sys.modules, "docx2pdf", SimpleNamespace(convert=fake_convert))
    monkeypatch.setattr(
        docx_conversion,
        "_repair_docx_for_conversion",
        lambda _src, dst: Path(dst).write_bytes(b"repaired"),
    )
    monkeypatch.setattr(docx_conversion, "_validate_conversion_content", lambda **_kwargs: None)

    result = docx_conversion.convert_docx_to_pdf_for_analysis(source_docx, output_pdf)

    assert result == output_pdf
    assert output_pdf.exists()
    assert len(calls) == 2
    assert calls[0] == (str(source_docx), str(output_pdf))
    assert calls[1][1] == str(output_pdf)
    assert calls[1][0].endswith(".docx")
    assert "__repaired_for_conversion_" in calls[1][0]
    assert repaired_docx_paths
    assert not repaired_docx_paths[0].exists()


def test_convert_docx_to_pdf_raises_if_retry_fails(monkeypatch, tmp_path: Path) -> None:
    source_docx = tmp_path / "protocol.docx"
    source_docx.write_bytes(b"placeholder")
    output_pdf = tmp_path / "analysis.pdf"

    call_count = {"count": 0}

    def always_fail_convert(_src: str, _dst: str) -> None:
        call_count["count"] += 1
        raise RuntimeError(f"conversion failed {call_count['count']}")

    monkeypatch.setitem(sys.modules, "docx2pdf", SimpleNamespace(convert=always_fail_convert))
    monkeypatch.setattr(
        docx_conversion,
        "_repair_docx_for_conversion",
        lambda _src, dst: Path(dst).write_bytes(b"repaired"),
    )
    monkeypatch.setattr(docx_conversion, "_validate_conversion_content", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="Failed to convert DOCX to PDF for analysis") as exc_info:
        docx_conversion.convert_docx_to_pdf_for_analysis(source_docx, output_pdf)
    error_text = str(exc_info.value)
    assert "First attempt error: RuntimeError: conversion failed 1." in error_text
    assert "Retry with repaired DOCX error: RuntimeError: conversion failed 2." in error_text
    repaired_candidates = list(tmp_path.glob("protocol__repaired_for_conversion_*.docx"))
    assert repaired_candidates == []


def test_convert_docx_to_pdf_removes_stale_pdf_before_conversion(monkeypatch, tmp_path: Path) -> None:
    source_docx = tmp_path / "protocol.docx"
    source_docx.write_bytes(b"placeholder")
    output_pdf = tmp_path / "analysis.pdf"
    output_pdf.write_bytes(b"stale")

    def fake_convert(_src: str, dst: str) -> None:
        assert output_pdf.exists() is False
        Path(dst).write_bytes(b"%PDF-1.4\n%%EOF")

    monkeypatch.setitem(sys.modules, "docx2pdf", SimpleNamespace(convert=fake_convert))
    monkeypatch.setattr(docx_conversion, "_validate_conversion_content", lambda **_kwargs: None)

    result = docx_conversion.convert_docx_to_pdf_for_analysis(source_docx, output_pdf)
    assert result == output_pdf
    assert output_pdf.read_bytes().startswith(b"%PDF")


def test_run_docx2pdf_conversion_initializes_and_uninitializes_com(monkeypatch, tmp_path: Path) -> None:
    source_docx = tmp_path / "protocol.docx"
    output_pdf = tmp_path / "analysis.pdf"
    source_docx.write_bytes(b"placeholder")

    com_calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "pythoncom",
        SimpleNamespace(
            CoInitialize=lambda: com_calls.append("init"),
            CoUninitialize=lambda: com_calls.append("uninit"),
        ),
    )

    convert_calls: list[tuple[str, str]] = []

    def fake_convert(src: str, dst: str) -> None:
        convert_calls.append((src, dst))

    docx_conversion._run_docx2pdf_conversion(fake_convert, source_docx, output_pdf)

    assert convert_calls == [(str(source_docx), str(output_pdf))]
    assert com_calls == ["init", "uninit"]
