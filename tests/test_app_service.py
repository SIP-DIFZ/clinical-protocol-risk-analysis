from pathlib import Path

import protocol_risk_analyzer.app_service as app_service
from protocol_risk_analyzer.app_service import _count_analyzed_pages, _resolve_analysis_input_path, build_revision_change_risks
from protocol_risk_analyzer.chunking import DocumentChunk
from protocol_risk_analyzer.models import RiskDomain


def test_count_analyzed_pages_counts_unique_page_coverage() -> None:
    chunks = [
        DocumentChunk(chunk_id=1, start_page=1, end_page=3, text="a"),
        DocumentChunk(chunk_id=2, start_page=3, end_page=5, text="b"),
    ]
    assert _count_analyzed_pages(chunks, total_pages=5) == 5


def test_count_analyzed_pages_caps_at_total_pages() -> None:
    chunks = [DocumentChunk(chunk_id=1, start_page=1, end_page=10, text="a")]
    assert _count_analyzed_pages(chunks, total_pages=6) == 6


def test_resolve_analysis_input_path_keeps_non_docx() -> None:
    source = Path("C:\\tmp\\protocol.pdf")
    resolved = _resolve_analysis_input_path(source, Path("C:\\tmp\\out"))
    assert resolved == source


def test_resolve_analysis_input_path_converts_docx(monkeypatch) -> None:
    source = Path("C:\\tmp\\protocol.docx")
    output_dir = Path("C:\\tmp\\out")
    expected = output_dir / "protocol__analysis_source.pdf"

    def fake_convert(docx_path: Path, pdf_path: Path) -> Path:
        assert docx_path == source
        assert pdf_path == expected
        return expected

    monkeypatch.setattr(app_service, "convert_docx_to_pdf_for_analysis", fake_convert)
    resolved = _resolve_analysis_input_path(source, output_dir)
    assert resolved == expected


def test_build_revision_change_risks_maps_approved_changes_to_risks() -> None:
    risks = build_revision_change_risks(
        [
            {
                "summary": "Add monthly ECG monitoring",
                "requested_change": "Add monthly ECG monitoring for participants receiving active treatment.",
                "rationale": "Supports ongoing cardiac safety oversight for higher-risk participants.",
                "affected_sections": ["Safety Monitoring", "Schedule of Assessments"],
                "focus": "health",
                "estimated_edits": 3,
            }
        ]
    )
    assert len(risks) == 1
    assert risks[0].title == "Add monthly ECG monitoring"
    assert risks[0].domain == RiskDomain.HEALTH
    assert risks[0].category == "Safety Monitoring"
    assert risks[0].mitigations == ["Add monthly ECG monitoring for participants receiving active treatment."]
