from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from protocol_risk_analyzer.models import AnalysisResult, RiskAssessment


def write_analysis_json(result: AnalysisResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def build_markdown_report(result: AnalysisResult) -> str:
    lines: list[str] = []
    lines.append(f"# Protocol Risk Report: {result.metadata.document_name}")
    lines.append("")
    lines.append(f"- Generated (UTC): {result.metadata.generated_at_utc}")
    lines.append(f"- Pages analyzed: {result.metadata.page_count}")
    lines.append(f"- Chunks analyzed: {result.metadata.chunk_count}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(result.executive_summary)
    lines.append("")

    if result.key_assumptions:
        lines.append("## Key Assumptions")
        for assumption in result.key_assumptions:
            lines.append(f"- {assumption}")
        lines.append("")

    lines.append("## Risk Register")
    for idx, risk in enumerate(result.risks, start=1):
        lines.extend(build_risk_markdown_lines(risk, idx))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_risk_markdown_lines(risk: RiskAssessment, index: int | None = None) -> list[str]:
    prefix = f"{index}. " if index is not None else ""
    lines = [
        f"### {prefix}{risk.title} ({risk.domain.value.title()}) | Score: {risk.risk_score}",
        "",
        "#### Risk Metrics",
        f"- **Category:** {risk.category}",
        f"- **Likelihood:** {risk.likelihood}/5",
        f"- **Impact:** {risk.impact}/5",
        f"- **Confidence:** {risk.confidence}",
        f"- **Score:** {risk.risk_score}",
        "",
        "#### Why This Risk Matters",
        risk.rationale,
    ]
    if risk.factor_triggers:
        lines.append("")
        lines.append("#### Triggered Factors")
        for factor in risk.factor_triggers:
            lines.append(f"- {factor}")
    if risk.mitigations:
        lines.append("")
        lines.append("#### Suggested Mitigations")
        for mitigation in risk.mitigations:
            lines.append(f"- {mitigation}")
    if risk.evidence:
        lines.append("")
        lines.append("#### Evidence")
        for ev in risk.evidence:
            excerpt = ev.excerpt.replace("\n", " ").strip()
            lines.append(f"- **p.{ev.page_start}-{ev.page_end}:** {excerpt}")
    return lines


def build_risk_markdown_section(risk: RiskAssessment, index: int | None = None) -> str:
    return "\n".join(build_risk_markdown_lines(risk, index))


def write_markdown_report(result: AnalysisResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.md"
    path.write_text(build_markdown_report(result), encoding="utf-8")
    return path


def write_pdf_report(result: AnalysisResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.pdf"
    content = build_markdown_report(result)
    _render_text_to_pdf(content, path)
    return path


def _render_text_to_pdf(text: str, output_path: Path) -> None:
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    line_height = 13
    font_name = "Helvetica"
    font_size = 10
    c.setFont(font_name, font_size)

    max_width = width - 2 * margin
    for raw_line in text.splitlines():
        wrapped_lines = _wrap_line(raw_line, font_name, font_size, max_width)
        for line in wrapped_lines:
            if y <= margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height
    c.save()


def _wrap_line(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    stripped = text.rstrip()
    if not stripped:
        return [""]

    words = stripped.split(" ")
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width = pdfmetrics.stringWidth(candidate, font_name, font_size)
        if candidate_width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
