from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from protocol_risk_analyzer.analysis import analyze_protocol
from protocol_risk_analyzer.chat_service import ChatResult, ProtocolChatService
from protocol_risk_analyzer.chunking import DocumentChunk, chunk_pages
from protocol_risk_analyzer.config import load_config
from protocol_risk_analyzer.docx_conversion import convert_docx_to_pdf_for_analysis
from protocol_risk_analyzer.editor import (
    ProtocolEditPatch,
    ProtocolParagraph,
    RevisedProtocolOutputPaths,
    build_protocol_paragraphs_for_editing,
    create_revised_protocol_outputs,
    generate_revised_protocol_patch_plan,
    write_revised_protocol_outputs_from_patches,
)
from protocol_risk_analyzer.io import PageText, read_document
from protocol_risk_analyzer.llm import RiskLlmClient, build_openai_client_with_resilience
from protocol_risk_analyzer.models import AnalysisResult, EvidenceSnippet, RiskAssessment, RiskDomain
from protocol_risk_analyzer.revision_assistant import (
    ProtocolProposalEditService,
    ProtocolRevisionPlannerService,
    RevisionPlannerResult,
)
from protocol_risk_analyzer.reporting import (
    write_analysis_json,
    write_markdown_report,
    write_pdf_report,
)


@dataclass(frozen=True)
class AnalysisStats:
    analyzed_pages: int
    total_pages: int
    chunk_count: int


@dataclass(frozen=True)
class OutputPaths:
    analysis_json_path: Path
    report_md_path: Path
    report_pdf_path: Path


@dataclass(frozen=True)
class RevisedProtocolPaths:
    mitigated_md_path: Path
    mitigated_pdf_path: Path
    mitigated_docx_path: Path
    patch_log_path: Path


def run_protocol_analysis(
    input_path: Path,
    output_dir: Path,
    model: str | None = None,
    chunk_chars: int = 14_000,
    overlap_chars: int = 1_200,
    progress_callback: Callable[..., None] | None = None,
    max_workers: int | None = None,
) -> tuple[str, AnalysisStats, OutputPaths]:
    cfg = load_config()
    chosen_model = model or cfg.default_model

    analysis_input_path = _resolve_analysis_input_path(input_path=input_path, output_dir=output_dir)
    pages = read_document(analysis_input_path)
    chunks = chunk_pages(pages, max_chars=chunk_chars, overlap_chars=overlap_chars)
    analyzed_pages = _count_analyzed_pages(chunks, total_pages=len(pages))

    llm_client = RiskLlmClient(
        client=build_openai_client_with_resilience(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            timeout_seconds=cfg.openai_timeout_seconds,
            max_retries=cfg.openai_max_retries,
        ),
        model=chosen_model,
    )
    result = analyze_protocol(
        source_path=input_path,
        chunks=chunks,
        llm=llm_client,
        progress_callback=progress_callback,
        max_workers=max_workers,
    )

    json_path = write_analysis_json(result, output_dir)
    md_path = write_markdown_report(result, output_dir)
    pdf_path = write_pdf_report(result, output_dir)

    return (
        chosen_model,
        AnalysisStats(
            analyzed_pages=analyzed_pages,
            total_pages=len(pages),
            chunk_count=len(chunks),
        ),
        OutputPaths(
            analysis_json_path=json_path,
            report_md_path=md_path,
            report_pdf_path=pdf_path,
        ),
    )


def run_protocol_editor(
    input_path: Path,
    output_dir: Path,
    selected_risks: list[RiskAssessment],
    model: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[str, RevisedProtocolPaths]:
    cfg = load_config()
    chosen_model = model or cfg.default_model
    pages = read_document(input_path)
    llm_client = RiskLlmClient(
        client=build_openai_client_with_resilience(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            timeout_seconds=cfg.openai_timeout_seconds,
            max_retries=cfg.openai_max_retries,
        ),
        model=chosen_model,
    )
    outputs: RevisedProtocolOutputPaths = create_revised_protocol_outputs(
        pages=pages,
        selected_risks=selected_risks,
        llm=llm_client,
        output_dir=output_dir,
        source_path=input_path,
        progress_callback=progress_callback,
    )
    return (
        chosen_model,
        RevisedProtocolPaths(
            mitigated_md_path=outputs.protocol_md_path,
            mitigated_pdf_path=outputs.protocol_pdf_path,
            mitigated_docx_path=outputs.protocol_docx_path,
            patch_log_path=outputs.patch_log_path,
        ),
    )


def run_protocol_editor_proposals(
    input_path: Path,
    selected_risks: list[RiskAssessment],
    model: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    proposal_callback: Callable[[list[ProtocolParagraph], list[ProtocolEditPatch]], None] | None = None,
) -> tuple[str, list[ProtocolParagraph], list[ProtocolEditPatch], list[str]]:
    cfg = load_config()
    chosen_model = model or cfg.default_model
    pages = read_document(input_path)
    llm_client = RiskLlmClient(
        client=build_openai_client_with_resilience(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            timeout_seconds=cfg.openai_timeout_seconds,
            max_retries=cfg.openai_max_retries,
        ),
        model=chosen_model,
    )
    paragraphs, patches, validation_notes = generate_revised_protocol_patch_plan(
        pages=pages,
        selected_risks=selected_risks,
        llm=llm_client,
        source_path=input_path,
        progress_callback=progress_callback,
        proposal_callback=proposal_callback,
    )
    return chosen_model, paragraphs, patches, validation_notes


def run_protocol_editor_from_patches(
    input_path: Path,
    output_dir: Path,
    patches: list[ProtocolEditPatch],
    model: str | None = None,
) -> tuple[str, RevisedProtocolPaths]:
    cfg = load_config()
    chosen_model = model or cfg.default_model
    pages: list[PageText] = [] if input_path.suffix.lower() == ".docx" else read_document(input_path)
    paragraphs = build_protocol_paragraphs_for_editing(pages=pages, source_path=input_path)
    outputs: RevisedProtocolOutputPaths = write_revised_protocol_outputs_from_patches(
        paragraphs=paragraphs,
        patches=patches,
        output_dir=output_dir,
        source_path=input_path,
        validation_notes=[],
    )
    return (
        chosen_model,
        RevisedProtocolPaths(
            mitigated_md_path=outputs.protocol_md_path,
            mitigated_pdf_path=outputs.protocol_pdf_path,
            mitigated_docx_path=outputs.protocol_docx_path,
            patch_log_path=outputs.patch_log_path,
        ),
    )


def run_protocol_chat(
    input_path: Path,
    analysis_json_text: str,
    question: str,
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
    pages: list[PageText] | None = None,
    chunks: list[DocumentChunk] | None = None,
) -> ChatResult:
    cfg = load_config()
    chosen_model = model or cfg.default_model
    analysis = AnalysisResult.model_validate_json(analysis_json_text)
    chat_service = ProtocolChatService(
        client=build_openai_client_with_resilience(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            timeout_seconds=cfg.openai_timeout_seconds,
            max_retries=cfg.openai_max_retries,
        ),
        model=chosen_model,
    )
    return chat_service.answer_question(
        analysis=analysis,
        question=question,
        history=history or [],
        pages=pages,
        chunks=chunks,
        input_path=input_path,
    )


def run_protocol_revision_planner(
    analysis_json_text: str,
    question: str,
    history: list[dict[str, str]] | None = None,
    revision_session: dict[str, object] | None = None,
    model: str | None = None,
) -> RevisionPlannerResult:
    cfg = load_config()
    chosen_model = model or cfg.default_model
    analysis = AnalysisResult.model_validate_json(analysis_json_text)
    planner = ProtocolRevisionPlannerService(
        client=build_openai_client_with_resilience(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            timeout_seconds=cfg.openai_timeout_seconds,
            max_retries=cfg.openai_max_retries,
        ),
        model=chosen_model,
    )
    return planner.plan_turn(
        analysis=analysis,
        question=question,
        history=history or [],
        revision_session=revision_session or {},
    )


def run_protocol_proposal_edit(
    *,
    target_label: str,
    risk_title: str,
    original_text: str,
    current_text: str,
    current_reason: str,
    user_request: str,
    model: str | None = None,
) -> tuple[str, str]:
    cfg = load_config()
    chosen_model = model or cfg.default_model
    service = ProtocolProposalEditService(
        client=build_openai_client_with_resilience(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            timeout_seconds=cfg.openai_timeout_seconds,
            max_retries=cfg.openai_max_retries,
        ),
        model=chosen_model,
    )
    payload = service.revise_proposal(
        target_label=target_label,
        risk_title=risk_title,
        original_text=original_text,
        current_text=current_text,
        current_reason=current_reason,
        user_request=user_request,
    )
    return payload.updated_text, payload.reason


def run_protocol_revision_editor(
    input_path: Path,
    output_dir: Path,
    approved_changes: list[dict[str, object]],
    model: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[str, RevisedProtocolPaths]:
    if not approved_changes:
        raise ValueError("At least one approved revision change is required.")
    selected_risks = build_revision_change_risks(approved_changes)
    return run_protocol_editor(
        input_path=input_path,
        output_dir=output_dir,
        selected_risks=selected_risks,
        model=model,
        progress_callback=progress_callback,
    )


def _count_analyzed_pages(chunks: list[DocumentChunk], total_pages: int) -> int:
    covered_pages: set[int] = set()
    for chunk in chunks:
        covered_pages.update(range(chunk.start_page, chunk.end_page + 1))
    return min(len(covered_pages), total_pages)


def _resolve_analysis_input_path(input_path: Path, output_dir: Path) -> Path:
    if input_path.suffix.lower() != ".docx":
        return input_path
    converted_pdf_path = output_dir / f"{input_path.stem}__analysis_source.pdf"
    return convert_docx_to_pdf_for_analysis(input_path, converted_pdf_path)


def build_revision_change_risks(approved_changes: list[dict[str, object]]) -> list[RiskAssessment]:
    risks: list[RiskAssessment] = []
    for idx, change in enumerate(approved_changes, start=1):
        if not isinstance(change, dict):
            continue
        summary = str(change.get("summary", "")).strip() or f"Revision Change {idx}"
        requested_change = str(change.get("requested_change", "")).strip() or summary
        rationale = str(change.get("rationale", "")).strip() or "User-approved protocol revision request."
        affected_sections_raw = change.get("affected_sections", [])
        affected_sections = (
            [str(item).strip() for item in affected_sections_raw if str(item).strip()]
            if isinstance(affected_sections_raw, list)
            else []
        )
        focus = str(change.get("focus", "general")).strip().lower()
        domain = _focus_to_domain(focus)
        category = affected_sections[0] if affected_sections else "Protocol Revision"
        evidence_excerpt = (
            f"User-approved change request: {requested_change}. "
            + (f"Affected sections: {', '.join(affected_sections)}." if affected_sections else "")
        ).strip()
        risks.append(
            RiskAssessment(
                title=summary,
                domain=domain,
                category=category,
                likelihood=3,
                impact=4,
                confidence=0.85,
                rationale=rationale,
                mitigation_status="unmitigated",
                mitigation_status_reason="User-directed revision item pending implementation in protocol text.",
                mitigations=[requested_change],
                factor_triggers=affected_sections[:3] if affected_sections else ["User requested protocol revision"],
                evidence=[EvidenceSnippet(excerpt=evidence_excerpt, page_start=1, page_end=1)],
            )
        )
    if not risks:
        raise ValueError("No valid approved revision changes were available for protocol editing.")
    return risks


def _focus_to_domain(focus: str) -> RiskDomain:
    normalized = focus.strip().lower()
    mapping = {
        "health": RiskDomain.HEALTH,
        "financial": RiskDomain.FINANCIAL,
        "ethical": RiskDomain.ETHICAL,
        "eligibility": RiskDomain.ELIGIBILITY,
        "intervention": RiskDomain.INTERVENTION,
        "operational": RiskDomain.FINANCIAL,
    }
    return mapping.get(normalized, RiskDomain.HEALTH)
