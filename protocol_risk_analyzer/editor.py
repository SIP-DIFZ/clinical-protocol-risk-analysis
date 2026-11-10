from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from docx import Document
from docx.oxml import OxmlElement
from docx.shared import RGBColor
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from protocol_risk_analyzer.io import PageText, read_document
from protocol_risk_analyzer.llm import RiskLlmClient
from protocol_risk_analyzer.models import ProtocolEditPatch, RiskAssessment

_TOPIC_STOPWORDS = {
    "about",
    "after",
    "before",
    "between",
    "from",
    "into",
    "with",
    "without",
    "under",
    "over",
    "during",
    "within",
    "these",
    "those",
    "their",
    "there",
    "where",
    "which",
    "while",
    "will",
    "must",
    "should",
    "could",
    "would",
    "have",
    "has",
    "been",
    "being",
    "were",
    "this",
    "that",
    "they",
    "them",
    "then",
    "than",
    "also",
    "each",
    "such",
    "when",
    "what",
    "protocol",
    "study",
    "participant",
    "participants",
    "trial",
    "section",
    "paragraph",
    "include",
    "including",
    "includes",
    "required",
    "requirement",
    "requirements",
}


@dataclass(frozen=True)
class ProtocolParagraph:
    paragraph_id: str
    page_number: int
    text: str
    is_heading: bool
    is_editable: bool
    paragraph_number_on_page: int = 1


@dataclass(frozen=True)
class RevisedProtocolOutputPaths:
    protocol_md_path: Path
    protocol_pdf_path: Path
    protocol_docx_path: Path
    patch_log_path: Path


@dataclass(frozen=True)
class _RenderedParagraph:
    text: str
    edited: bool
    is_heading: bool


@dataclass(frozen=True)
class _RiskEditContext:
    risk: RiskAssessment
    anchor_paragraph_ids: tuple[str, ...]
    evidence_anchor_paragraph_ids: tuple[str, ...]
    candidate_paragraph_ids: tuple[str, ...]
    must_cover_mitigations: tuple[str, ...]


def format_protocol_paragraph_reference(paragraph: ProtocolParagraph | None, fallback: str = "Unknown paragraph") -> str:
    if paragraph is None:
        return fallback
    if paragraph.page_number > 0 and paragraph.paragraph_number_on_page > 0:
        return f"Page {paragraph.page_number} - Paragraph {paragraph.paragraph_number_on_page}"
    if paragraph.paragraph_id:
        return paragraph.paragraph_id
    return fallback


def create_revised_protocol_outputs(
    pages: list[PageText],
    selected_risks: list[RiskAssessment],
    llm: RiskLlmClient,
    output_dir: Path,
    source_path: Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> RevisedProtocolOutputPaths:
    paragraphs, patches, validation_notes = generate_revised_protocol_patch_plan(
        pages=pages,
        selected_risks=selected_risks,
        llm=llm,
        source_path=source_path,
        progress_callback=progress_callback,
    )
    return write_revised_protocol_outputs_from_patches(
        paragraphs=paragraphs,
        patches=patches,
        output_dir=output_dir,
        source_path=source_path,
        validation_notes=validation_notes,
        progress_callback=progress_callback,
    )


def generate_revised_protocol_patch_plan(
    pages: list[PageText],
    selected_risks: list[RiskAssessment],
    llm: RiskLlmClient,
    source_path: Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    proposal_callback: Callable[[list[ProtocolParagraph], list[ProtocolEditPatch]], None] | None = None,
) -> tuple[list[ProtocolParagraph], list[ProtocolEditPatch], list[str]]:
    if not selected_risks:
        raise ValueError("At least one risk must be selected for mitigation.")

    if progress_callback:
        progress_callback(1, 5, "Preparing protocol sections...")
    paragraphs = build_protocol_paragraphs_for_editing(pages, source_path=source_path)
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    risk_contexts = _build_risk_edit_contexts(paragraphs, selected_risks)
    risk_batches = _chunk_risks(selected_risks, batch_size=6)
    total_steps = 6 + len(risk_batches)
    patches: list[ProtocolEditPatch] = []
    for batch_idx, risk_batch in enumerate(risk_batches, start=1):
        if progress_callback:
            progress_callback(
                1 + batch_idx,
                total_steps,
                f"Editor Agent generating patches ({batch_idx}/{len(risk_batches)})...",
            )
        batch_contexts = [
            context
            for context in risk_contexts
            if _normalize_risk_key(context.risk.title)
            in {_normalize_risk_key(risk.title) for risk in risk_batch}
        ]
        batch_candidate_ids: list[str] = []
        seen_candidate_ids: set[str] = set()
        for context in batch_contexts:
            for paragraph_id in context.candidate_paragraph_ids:
                if paragraph_id not in seen_candidate_ids:
                    seen_candidate_ids.add(paragraph_id)
                    batch_candidate_ids.append(paragraph_id)
        batch_candidate_payload = [
            {
                "paragraph_id": paragraph_by_id[paragraph_id].paragraph_id,
                "page_number": paragraph_by_id[paragraph_id].page_number,
                "text": paragraph_by_id[paragraph_id].text,
            }
            for paragraph_id in batch_candidate_ids
            if paragraph_id in paragraph_by_id
        ]
        batch_patches = llm.editor_agent_generate_patches(
            protocol_paragraphs=batch_candidate_payload,
            selected_risks=_serialize_risk_contexts(batch_contexts),
        )
        patches.extend(batch_patches)
        if proposal_callback:
            provisional_validated, _ = _validate_generated_patches(paragraphs, selected_risks, patches)
            provisional_consolidated = _consolidate_patches_for_preview(paragraphs, provisional_validated)
            provisional_deduped = _dedupe_patches_by_target_paragraph(provisional_consolidated)
            proposal_callback(paragraphs, provisional_deduped)

    patches, validation_notes = _validate_generated_patches(paragraphs, selected_risks, patches)
    missing_risks = _missing_coverage_risks(paragraphs, selected_risks, patches)
    coverage_retry = 0
    while missing_risks and coverage_retry < 2:
        coverage_retry += 1
        if progress_callback:
            progress_callback(
                1 + len(risk_batches) + coverage_retry,
                total_steps,
                f"Improving mitigation coverage ({coverage_retry}/2)...",
            )
        retry_contexts = _build_risk_edit_contexts(paragraphs, missing_risks)
        retry_batches = _chunk_contexts(retry_contexts, batch_size=15)
        for retry_batch in retry_batches:
            retry_candidate_ids: list[str] = []
            seen_retry_candidate_ids: set[str] = set()
            for context in retry_batch:
                for paragraph_id in context.candidate_paragraph_ids:
                    if paragraph_id not in seen_retry_candidate_ids:
                        seen_retry_candidate_ids.add(paragraph_id)
                        retry_candidate_ids.append(paragraph_id)
            retry_candidate_payload = [
                {
                    "paragraph_id": paragraph_by_id[paragraph_id].paragraph_id,
                    "page_number": paragraph_by_id[paragraph_id].page_number,
                    "text": paragraph_by_id[paragraph_id].text,
                }
                for paragraph_id in retry_candidate_ids
                if paragraph_id in paragraph_by_id
            ]
            retry_patches = llm.editor_agent_generate_patches(
                protocol_paragraphs=retry_candidate_payload,
                selected_risks=_serialize_risk_contexts(retry_batch),
            )
            patches.extend(retry_patches)
        missing_risks = _missing_coverage_risks(paragraphs, selected_risks, patches)

    if missing_risks:
        fallback_patches = _build_fallback_patches_for_missing_risks(paragraphs, missing_risks)
        if fallback_patches:
            patches.extend(fallback_patches)
            validation_notes.append(
                f"Added {len(fallback_patches)} fallback patch(es) after validation to cover uncovered risks."
            )

    max_patches = max(30, min(len(selected_risks), 120))
    patches = _reduce_patch_set(
        paragraphs,
        patches,
        max_total=max_patches,
        required_risks=selected_risks,
    )
    patches = _consolidate_patches_for_application(paragraphs, patches, selected_risks, llm, validation_notes)
    patches = _filter_high_risk_expansion_patches(paragraphs, patches, required_risks=selected_risks)
    final_missing_risks = _missing_coverage_risks(paragraphs, selected_risks, patches)
    if final_missing_risks:
        fallback_patches = _build_fallback_patches_for_missing_risks(paragraphs, final_missing_risks)
        if fallback_patches:
            patches.extend(fallback_patches)
            validation_notes.append(
                f"Added {len(fallback_patches)} fallback patch(es) after final validation to preserve risk coverage."
            )
            patches = _reduce_patch_set(
                paragraphs,
                patches,
                max_total=max_patches,
                required_risks=selected_risks,
            )
    patches = _dedupe_patches_by_target_paragraph(patches)
    return paragraphs, patches, validation_notes


def build_protocol_paragraphs_for_editing(
    pages: list[PageText],
    source_path: Path | None = None,
) -> list[ProtocolParagraph]:
    if source_path is not None and source_path.suffix.lower() == ".docx":
        return _build_docx_protocol_paragraphs(source_path)
    return _build_protocol_paragraphs(pages)


def write_revised_protocol_outputs_from_patches(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
    output_dir: Path,
    source_path: Path | None = None,
    validation_notes: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> RevisedProtocolOutputPaths:
    total_steps = 2
    if progress_callback:
        progress_callback(1, total_steps, "Applying accepted edits to protocol text...")
    rendered = _apply_protocol_patches(paragraphs, patches)

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "mitigated_protocol.md"
    pdf_path = output_dir / "mitigated_protocol.pdf"
    docx_path = output_dir / "mitigated_protocol.docx"
    patch_log_path = output_dir / "mitigated_protocol_patches.txt"

    if progress_callback:
        progress_callback(2, total_steps, "Writing downloadable files...")
    md_path.write_text(_render_markdown_protocol(rendered), encoding="utf-8")
    _write_colored_pdf(rendered, pdf_path)
    if source_path is not None and source_path.suffix.lower() == ".docx":
        _write_preserved_docx(source_path, patches, docx_path)
    else:
        _write_colored_docx(rendered, docx_path)
    patch_log_path.write_text(
        _render_patch_log(paragraphs, patches, validation_notes=validation_notes),
        encoding="utf-8",
    )

    return RevisedProtocolOutputPaths(
        protocol_md_path=md_path,
        protocol_pdf_path=pdf_path,
        protocol_docx_path=docx_path,
        patch_log_path=patch_log_path,
    )


def _build_protocol_paragraphs(pages: list[PageText]) -> list[ProtocolParagraph]:
    paragraphs: list[ProtocolParagraph] = []
    counter = 1
    for page in pages:
        paragraph_number_on_page = 0
        blocks = re.split(r"\n\s*\n", page.text.strip())
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            text = "\n".join(lines).strip()
            if not text:
                continue
            paragraph_number_on_page += 1
            sanitized_text = _sanitize_xml_text(text)
            paragraphs.append(
                ProtocolParagraph(
                    paragraph_id=f"P{counter}",
                    page_number=page.page_number,
                    text=sanitized_text,
                    is_heading=_is_heading_like(sanitized_text),
                    is_editable=not _is_front_matter_like(
                        sanitized_text,
                        is_heading=_is_heading_like(sanitized_text),
                        page_number=page.page_number,
                    ),
                    paragraph_number_on_page=paragraph_number_on_page,
                )
            )
            counter += 1

    if not paragraphs:
        raise ValueError("Unable to build protocol paragraphs from the source document.")
    return paragraphs


def _build_docx_protocol_paragraphs(path: Path) -> list[ProtocolParagraph]:
    doc = Document(str(path))
    extracted_pages = read_document(path)
    page_blocks = [
        [block.strip() for block in page.text.split("\n\n") if block.strip()]
        for page in extracted_pages
    ]
    block_positions: list[tuple[int, int, str]] = []
    for page_index, page_block in enumerate(page_blocks):
        for block_index, block_text in enumerate(page_block):
            block_positions.append((page_index, block_index, _sanitize_xml_text(block_text)))
    block_positions_cursor = 0
    paragraphs: list[ProtocolParagraph] = []
    counter = 1
    body_started = False
    in_protected_admin_section = False
    page_paragraph_counters: dict[int, int] = {}
    for paragraph in doc.paragraphs:
        text = _sanitize_xml_text(paragraph.text.strip())
        if not text:
            continue
        page_number = 1
        matched_positions = None
        while block_positions_cursor < len(block_positions):
            candidate_page_index, candidate_block_index, candidate_text = block_positions[block_positions_cursor]
            if candidate_text == text:
                matched_positions = (candidate_page_index, candidate_block_index)
                page_number = extracted_pages[candidate_page_index].page_number
                block_positions_cursor += 1
                break
            block_positions_cursor += 1
        if matched_positions is None:
            page_number = extracted_pages[min(len(extracted_pages) - 1, max(0, len(paragraphs)))].page_number
        style_name = paragraph.style.name if paragraph.style is not None else ""
        is_heading = style_name.lower().startswith("heading") or _is_heading_like(text)
        is_body_heading = _is_body_section_heading(text, style_name)
        first_line = text.splitlines()[0].strip() if text.splitlines() else text.strip()
        is_admin_heading = is_heading and _is_protected_administrative_heading(first_line)
        if is_body_heading and not is_admin_heading:
            in_protected_admin_section = False
        if is_body_heading:
            body_started = True
        is_editable = not _is_front_matter_like(
            text,
            is_heading=is_heading,
            page_number=page_number,
            style_name=style_name,
            body_started=body_started,
        )
        if in_protected_admin_section:
            is_editable = False
        if is_admin_heading:
            in_protected_admin_section = True
        page_paragraph_number = page_paragraph_counters.get(page_number, 0) + 1
        page_paragraph_counters[page_number] = page_paragraph_number
        paragraphs.append(
            ProtocolParagraph(
                paragraph_id=f"P{counter}",
                page_number=page_number,
                text=text,
                is_heading=is_heading,
                is_editable=is_editable,
                paragraph_number_on_page=page_paragraph_number,
            )
        )
        counter += 1
    if not paragraphs:
        raise ValueError(f"DOCX file is empty: {path}")
    return paragraphs


def _build_risk_edit_contexts(
    paragraphs: list[ProtocolParagraph],
    selected_risks: list[RiskAssessment],
) -> list[_RiskEditContext]:
    editable_paragraphs = [paragraph for paragraph in paragraphs if paragraph.is_editable]
    if not editable_paragraphs:
        editable_paragraphs = paragraphs
    contexts: list[_RiskEditContext] = []
    for risk in selected_risks:
        evidence_pages: set[int] = set()
        for ev in risk.evidence:
            evidence_pages.update(range(ev.page_start, ev.page_end + 1))
        risk_tokens = {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", f"{risk.title} {risk.category}".lower())
            if len(token) > 3
        }
        evidence_anchor_ids = _find_evidence_anchor_paragraph_ids(editable_paragraphs, risk)
        anchor_ids = evidence_anchor_ids or _find_anchor_paragraph_ids(editable_paragraphs, risk, evidence_pages)
        candidate_ids = _expand_local_candidate_ids(editable_paragraphs, anchor_ids, evidence_pages, risk_tokens)
        must_cover_mitigations = tuple(
            dict.fromkeys(
                mitigation.strip()
                for mitigation in risk.mitigations
                if mitigation and mitigation.strip()
            )
        )
        contexts.append(
            _RiskEditContext(
                risk=risk,
                anchor_paragraph_ids=tuple(anchor_ids),
                evidence_anchor_paragraph_ids=tuple(evidence_anchor_ids),
                candidate_paragraph_ids=tuple(candidate_ids),
                must_cover_mitigations=must_cover_mitigations,
            )
        )
    return contexts


def _find_evidence_anchor_paragraph_ids(
    paragraphs: list[ProtocolParagraph],
    risk: RiskAssessment,
) -> list[str]:
    anchors: list[str] = []
    seen_ids: set[str] = set()
    for evidence in risk.evidence:
        evidence_key = _normalize_match_text(evidence.excerpt)
        if not evidence_key:
            continue
        ranked = sorted(
            (
                paragraph
                for paragraph in paragraphs
                if paragraph.page_number in range(evidence.page_start, evidence.page_end + 1)
            ),
            key=lambda paragraph: _evidence_match_score(paragraph, evidence_key),
            reverse=True,
        )
        for paragraph in ranked[:2]:
            if paragraph.paragraph_id in seen_ids:
                continue
            if _evidence_match_score(paragraph, evidence_key) < 10:
                continue
            anchors.append(paragraph.paragraph_id)
            seen_ids.add(paragraph.paragraph_id)
    return anchors


def _find_anchor_paragraph_ids(
    paragraphs: list[ProtocolParagraph],
    risk: RiskAssessment,
    evidence_pages: set[int],
) -> list[str]:
    anchors: list[str] = []
    risk_tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", f"{risk.title} {risk.category}".lower())
        if len(token) > 3
    }
    ranked = sorted(
        paragraphs,
        key=lambda paragraph: _paragraph_match_score(paragraph, evidence_pages, risk_tokens),
        reverse=True,
    )
    for paragraph in ranked[:2]:
        if _paragraph_match_score(paragraph, evidence_pages, risk_tokens) > 0 and _paragraph_contains_risk_tokens(
            paragraph,
            risk_tokens,
        ):
            anchors.append(paragraph.paragraph_id)
    return anchors


def _expand_local_candidate_ids(
    paragraphs: list[ProtocolParagraph],
    anchor_ids: list[str],
    evidence_pages: set[int],
    risk_tokens: set[str],
) -> list[str]:
    if not paragraphs:
        return []
    by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    ordered_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    anchor_set = set(anchor_ids)
    candidate_ids: list[str] = []
    seen_ids: set[str] = set()

    def add_candidate(paragraph_id: str) -> None:
        if paragraph_id in seen_ids or paragraph_id not in by_id:
            return
        seen_ids.add(paragraph_id)
        candidate_ids.append(paragraph_id)

    def is_directly_relevant(paragraph: ProtocolParagraph) -> bool:
        return paragraph.is_editable and _paragraph_contains_risk_tokens(paragraph, risk_tokens)

    for anchor_id in anchor_ids:
        add_candidate(anchor_id)
        anchor_index = ordered_ids.index(anchor_id)
        anchor_page = by_id[anchor_id].page_number
        for offset in (-1, 1):
            neighbor_index = anchor_index + offset
            if 0 <= neighbor_index < len(ordered_ids):
                neighbor = by_id[ordered_ids[neighbor_index]]
                if abs(neighbor.page_number - anchor_page) <= 1 and is_directly_relevant(neighbor):
                    add_candidate(neighbor.paragraph_id)

    if candidate_ids:
        return candidate_ids

    ranked = sorted(
        paragraphs,
        key=lambda paragraph: _paragraph_match_score(paragraph, evidence_pages, risk_tokens),
        reverse=True,
    )
    for paragraph in ranked[:4]:
        if _paragraph_match_score(paragraph, evidence_pages, risk_tokens) > 0 and is_directly_relevant(paragraph):
            add_candidate(paragraph.paragraph_id)
    if candidate_ids:
        return candidate_ids

    return []


def _apply_protocol_patches(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
) -> list[_RenderedParagraph]:
    if not patches:
        return [
            _RenderedParagraph(text=p.text, edited=False, is_heading=p.is_heading) for p in paragraphs
        ]

    rendered = [_RenderedParagraph(text=p.text, edited=False, is_heading=p.is_heading) for p in paragraphs]
    paragraph_index = {p.paragraph_id: idx for idx, p in enumerate(paragraphs)}
    indexed_patches: list[tuple[int, ProtocolEditPatch]] = []
    for patch in patches:
        if patch.target_paragraph_id not in paragraph_index:
            raise RuntimeError(f"Editor Agent produced unknown paragraph ID: {patch.target_paragraph_id}")
        indexed_patches.append((paragraph_index[patch.target_paragraph_id], patch))

    for idx, patch in sorted(indexed_patches, key=lambda item: item[0], reverse=True):
        sanitized_new_text = _sanitize_xml_text(patch.new_text.strip())

        if patch.action == "replace":
            rendered[idx] = _RenderedParagraph(
                text=sanitized_new_text,
                edited=True,
                is_heading=rendered[idx].is_heading,
            )
            continue

        insertion = _RenderedParagraph(
            text=sanitized_new_text,
            edited=True,
            is_heading=_is_heading_like(sanitized_new_text),
        )
        if patch.action == "insert_before":
            rendered.insert(idx, insertion)
            continue
        if patch.action == "insert_after":
            rendered.insert(idx + 1, insertion)
            continue
        raise RuntimeError(f"Unsupported edit action from Editor Agent: {patch.action}")
    return rendered


def _render_markdown_protocol(paragraphs: list[_RenderedParagraph]) -> str:
    lines = [
        "# Mitigated Protocol Draft",
        "",
        "Edited text is shown in red.",
        "",
    ]
    for paragraph in paragraphs:
        text = _sanitize_xml_text(paragraph.text.strip())
        if paragraph.is_heading and "\n" not in text and len(text) <= 120:
            lines.append(f"## {text}")
        elif paragraph.edited:
            lines.append(f"<span style=\"color:#B00020\">{text}</span>")
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_patch_log(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
    validation_notes: list[str] | None = None,
) -> str:
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    lines = [
        "# Editor Agent Patch Log",
        "",
        f"- Total patches: {len(patches)}",
        "",
    ]
    if validation_notes:
        lines.extend(["## Validation Notes", ""])
        for note in validation_notes:
            cleaned_note = _sanitize_xml_text(note.strip())
            if cleaned_note:
                lines.append(f"- {cleaned_note}")
        lines.append("")
    for idx, patch in enumerate(patches, start=1):
        patch_kind = "Fallback mitigation patch" if patch.reason.startswith("Fallback mitigation patch") else "Generated mitigation patch"
        lines.extend(
            [
                f"## Patch {idx}",
                f"- Risk: {_sanitize_xml_text(patch.risk_title)}",
                f"- Action: {patch.action}",
                f"- Target: {_sanitize_xml_text(format_protocol_paragraph_reference(paragraph_by_id.get(patch.target_paragraph_id), patch.target_paragraph_id))}",
                f"- Kind: {patch_kind}",
                f"- Reason: {_sanitize_xml_text(patch.reason)}",
                "",
                "### Original Text",
                _sanitize_xml_text(
                    paragraph_by_id.get(
                        patch.target_paragraph_id,
                        ProtocolParagraph(
                            patch.target_paragraph_id,
                            0,
                            "[Original text unavailable]",
                            False,
                            False,
                            0,
                        ),
                    ).text.strip()
                ),
                "",
                "### New Text",
                _sanitize_xml_text(patch.new_text.strip()),
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _write_colored_docx(paragraphs: list[_RenderedParagraph], output_path: Path) -> None:
    doc = Document()
    doc.add_heading("Mitigated Protocol Draft", level=1)
    note = doc.add_paragraph("Edited text is shown in red.")
    note.runs[0].italic = True
    for paragraph in paragraphs:
        p = doc.add_paragraph(style="Heading 2" if paragraph.is_heading else None)
        run = p.add_run(_sanitize_xml_text(paragraph.text.strip()))
        if paragraph.edited:
            run.font.color.rgb = RGBColor(176, 0, 32)
    doc.save(str(output_path))


def _write_preserved_docx(source_path: Path, patches: list[ProtocolEditPatch], output_path: Path) -> None:
    doc = Document(str(source_path))
    editable_paragraphs = [p for p in doc.paragraphs if p.text and p.text.strip()]
    paragraph_index = {f"P{idx}": idx - 1 for idx in range(1, len(editable_paragraphs) + 1)}

    indexed_patches: list[tuple[int, ProtocolEditPatch]] = []
    for patch in patches:
        idx = paragraph_index.get(patch.target_paragraph_id)
        if idx is None:
            continue
        indexed_patches.append((idx, patch))

    for idx, patch in sorted(indexed_patches, key=lambda item: item[0], reverse=True):
        target = editable_paragraphs[idx]
        sanitized_new_text = _sanitize_xml_text(patch.new_text.strip())
        if patch.action == "replace":
            _replace_paragraph_text(target, sanitized_new_text)
            continue

        if patch.action == "insert_before":
            inserted = target.insert_paragraph_before(sanitized_new_text)
            if target.style is not None:
                inserted.style = target.style
            _color_paragraph_red(inserted)
            continue

        if patch.action == "insert_after":
            inserted = _insert_paragraph_after(target, sanitized_new_text)
            if target.style is not None:
                inserted.style = target.style
            _color_paragraph_red(inserted)
            continue

        raise RuntimeError(f"Unsupported edit action from Editor Agent: {patch.action}")

    doc.save(str(output_path))


def _consolidate_patches_for_application(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
    selected_risks: list[RiskAssessment],
    llm: RiskLlmClient,
    validation_notes: list[str] | None = None,
) -> list[ProtocolEditPatch]:
    if len(patches) <= 1:
        return patches
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    risk_by_key = {_normalize_risk_key(risk.title): risk for risk in selected_risks}
    grouped: dict[tuple[str, str], list[ProtocolEditPatch]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for patch in patches:
        key = (patch.target_paragraph_id, patch.action)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(patch)

    consolidated: list[ProtocolEditPatch] = []
    for key in ordered_keys:
        target_paragraph_id, action = key
        group = grouped[key]
        if action != "replace" or len(group) == 1:
            consolidated.extend(group)
            continue
        original_text = paragraph_by_id.get(target_paragraph_id)
        related_risks = [
            risk_by_key[risk_key]
            for risk_key in [_normalize_risk_key(patch.risk_title) for patch in group]
            if risk_key in risk_by_key
        ]
        if original_text is None or not related_risks:
            merged_text = _merge_replace_patch_texts(
                original_text.text if original_text is not None else "",
                [patch.new_text for patch in group],
            )
            merged_reason = _summarize_patch_reasons(group)
        else:
            try:
                merged_text, merged_reason = llm.editor_agent_rewrite_overlapping_paragraph(
                    target_paragraph={
                        "paragraph_id": original_text.paragraph_id,
                        "page_number": original_text.page_number,
                        "text": original_text.text,
                    },
                    overlapping_risks=[risk.model_dump() for risk in related_risks],
                    candidate_rewrites=[
                        {
                            "risk_title": patch.risk_title,
                            "reason": patch.reason,
                            "new_text": patch.new_text,
                        }
                        for patch in group
                    ],
                )
            except (RuntimeError, ValueError) as exc:
                merged_text = _merge_replace_patch_texts(
                    original_text.text if original_text is not None else "",
                    [patch.new_text for patch in group],
                )
                merged_reason = _summarize_patch_reasons(group)
                if validation_notes is not None:
                    validation_notes.append(
                        f"Used deterministic merge for paragraph {target_paragraph_id} after editor rewrite failed: {exc}"
                    )
            else:
                candidate_patch = ProtocolEditPatch(
                    risk_title=_summarize_patch_risks(group),
                    action="replace",
                    target_paragraph_id=target_paragraph_id,
                    new_text=merged_text,
                    reason=merged_reason,
                )
                if not all(
                    _patch_effectively_mitigates_risk(candidate_patch, paragraph_by_id, risk)
                    for risk in related_risks
                ):
                    merged_text = _merge_replace_patch_texts(
                        original_text.text if original_text is not None else "",
                        [patch.new_text for patch in group],
                    )
                    merged_reason = _summarize_patch_reasons(group)
                    if validation_notes is not None:
                        validation_notes.append(
                            f"Used deterministic merge for paragraph {target_paragraph_id} after editor rewrite did not validate."
                        )
        consolidated.append(
            ProtocolEditPatch(
                risk_title=_summarize_patch_risks(group),
                action="replace",
                target_paragraph_id=target_paragraph_id,
                new_text=merged_text,
                reason=merged_reason,
            )
        )
    return consolidated


def _consolidate_patches_for_preview(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
) -> list[ProtocolEditPatch]:
    if len(patches) <= 1:
        return patches
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    grouped: dict[tuple[str, str], list[ProtocolEditPatch]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for patch in patches:
        key = (patch.target_paragraph_id, patch.action)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(patch)

    consolidated: list[ProtocolEditPatch] = []
    for target_paragraph_id, action in ordered_keys:
        group = grouped[(target_paragraph_id, action)]
        if action != "replace" or len(group) == 1:
            consolidated.extend(group)
            continue
        original_text = paragraph_by_id.get(target_paragraph_id)
        consolidated.append(
            ProtocolEditPatch(
                risk_title=_summarize_patch_risks(group),
                action="replace",
                target_paragraph_id=target_paragraph_id,
                new_text=_merge_replace_patch_texts(
                    original_text.text if original_text is not None else "",
                    [patch.new_text for patch in group],
                ),
                reason=_summarize_patch_reasons(group),
            )
        )
    return consolidated


def _validate_generated_patches(
    paragraphs: list[ProtocolParagraph],
    selected_risks: list[RiskAssessment],
    patches: list[ProtocolEditPatch],
) -> tuple[list[ProtocolEditPatch], list[str]]:
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    risk_by_key = {_normalize_risk_key(risk.title): risk for risk in selected_risks}
    validated: list[ProtocolEditPatch] = []
    notes: list[str] = []
    for patch in patches:
        paragraph = paragraph_by_id.get(patch.target_paragraph_id)
        if paragraph is None:
            _append_validation_note(
                notes,
                f"Discarded patch for {patch.target_paragraph_id} because the target paragraph was not found.",
            )
            continue
        if not paragraph.is_editable:
            _append_validation_note(
                notes,
                f"Discarded patch for {patch.target_paragraph_id} because the target paragraph is not editable.",
            )
            continue
        risk = risk_by_key.get(_normalize_risk_key(patch.risk_title))
        if risk is None:
            _append_validation_note(
                notes,
                f"Discarded patch for {patch.target_paragraph_id} because the risk title did not match a selected risk.",
            )
            continue
        new_text = patch.new_text.strip()
        original_text = paragraph.text.strip()
        if not new_text or _normalize_match_text(new_text) == _normalize_match_text(original_text):
            _append_validation_note(
                notes,
                f"Discarded patch for {patch.target_paragraph_id} because it did not materially change the protocol text.",
            )
            continue
        if patch.action == "replace":
            if not _preserves_paragraph_subject(original_text, new_text):
                _append_validation_note(
                    notes,
                    f"Discarded replace patch for {patch.target_paragraph_id} because it strayed from the original paragraph topic.",
                )
                continue
            if _is_length_and_topic_drift(original_text, new_text):
                _append_validation_note(
                    notes,
                    f"Discarded replace patch for {patch.target_paragraph_id} because the rewrite is significantly longer than the original and topic overlap is insufficient.",
                )
                continue
            if not _patch_effectively_mitigates_risk(patch, paragraph_by_id, risk):
                _append_validation_note(
                    notes,
                    f"Discarded replace patch for {patch.target_paragraph_id} because it did not validate as a mitigation.",
                )
                continue
        elif patch.action in {"insert_before", "insert_after"}:
            normalized_new_text = _normalize_match_text(new_text)
            if _is_high_risk_expansion(original_text, new_text) or not (
                _contains_control_language(normalized_new_text) or _mentions_risk_mitigations(risk, normalized_new_text)
            ):
                _append_validation_note(
                    notes,
                    f"Discarded insert patch for {patch.target_paragraph_id} because it did not add clear mitigation language.",
                )
                continue
            if not _preserves_paragraph_subject(original_text, new_text):
                _append_validation_note(
                    notes,
                    f"Discarded insert patch for {patch.target_paragraph_id} because it strayed from the original paragraph topic.",
                )
                continue
        else:
            _append_validation_note(
                notes,
                f"Discarded patch for {patch.target_paragraph_id} because the action was unsupported.",
            )
            continue
        validated.append(patch)
    return validated, notes


def _append_validation_note(notes: list[str], note: str, max_notes: int = 8) -> None:
    if len(notes) < max_notes:
        notes.append(note)
        return
    if len(notes) == max_notes:
        notes.append("Additional generated patches were discarded during validation.")


def _merge_replace_patch_texts(original_text: str, candidate_texts: list[str]) -> str:
    cleaned_candidates = [text.strip() for text in candidate_texts if text.strip()]
    if not cleaned_candidates:
        return original_text.strip()
    base_text = max(cleaned_candidates, key=len)
    merged_sentences = _split_sentences(base_text)
    seen = {_normalize_match_text(sentence) for sentence in merged_sentences}
    for candidate in cleaned_candidates:
        for sentence in _split_sentences(candidate):
            normalized = _normalize_match_text(sentence)
            if not normalized or normalized in seen:
                continue
            if any(SequenceMatcher(None, normalized, existing).ratio() >= 0.92 for existing in seen):
                continue
            merged_sentences.append(sentence)
            seen.add(normalized)
    merged_text = " ".join(merged_sentences).strip()
    if merged_text:
        return merged_text
    return base_text


def _preserves_paragraph_subject(original_text: str, new_text: str) -> bool:
    original_tokens = _topic_tokens(original_text)
    new_tokens = _topic_tokens(new_text)
    if not original_tokens or not new_tokens:
        return False
    shared_tokens = original_tokens.intersection(new_tokens)
    if len(original_tokens) <= 3:
        return bool(shared_tokens)
    shared_ratio = len(shared_tokens) / max(1, len(original_tokens))
    if len(original_tokens) <= 8:
        return len(shared_tokens) >= 2 and shared_ratio >= 0.4
    return len(shared_tokens) >= 3 and shared_ratio >= 0.35


def _is_length_and_topic_drift(original_text: str, new_text: str) -> bool:
    orig_words = len(original_text.split())
    new_words = len(new_text.split())
    if orig_words == 0:
        return False
    ratio = new_words / orig_words
    if ratio <= 2.0:
        return False
    # At 2x–3x length, require strong topic overlap to pass.
    original_tokens = _topic_tokens(original_text)
    new_tokens = _topic_tokens(new_text)
    if not original_tokens:
        return ratio > 3.0
    shared_tokens = original_tokens.intersection(new_tokens)
    shared_ratio = len(shared_tokens) / max(1, len(original_tokens))
    if ratio > 3.0:
        return True
    # Between 2x and 3x: reject if topic overlap drops below a higher threshold.
    return shared_ratio < 0.50


def _topic_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 4 and token not in _TOPIC_STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _summarize_patch_risks(patches: list[ProtocolEditPatch]) -> str:
    titles = list(dict.fromkeys(patch.risk_title.strip() for patch in patches if patch.risk_title.strip()))
    if not titles:
        return "Consolidated mitigation patch"
    if len(titles) <= 3:
        return " | ".join(titles)
    return f"{' | '.join(titles[:3])} | +{len(titles) - 3} more"


def _summarize_patch_reasons(patches: list[ProtocolEditPatch]) -> str:
    reasons = list(dict.fromkeys(patch.reason.strip() for patch in patches if patch.reason.strip()))
    if not reasons:
        return "Consolidated mitigation patch."
    if len(reasons) == 1:
        return reasons[0]
    combined = " ".join(reasons)
    return combined[:500].rstrip()


def _filter_high_risk_expansion_patches(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
    required_risks: list[RiskAssessment] | None = None,
) -> list[ProtocolEditPatch]:
    if not patches:
        return []
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    filtered: list[ProtocolEditPatch] = []
    for patch in patches:
        if patch.action != "replace":
            filtered.append(patch)
            continue
        paragraph = paragraph_by_id.get(patch.target_paragraph_id)
        if paragraph is None:
            continue
        if _is_high_risk_expansion(paragraph.text, patch.new_text):
            continue
        filtered.append(patch)

    if required_risks is None:
        return filtered
    missing = _missing_coverage_risks(paragraphs, required_risks, filtered)
    if not missing:
        return filtered
    # Re-add at most one strongest patch per missing risk if we over-filtered.
    by_risk_key: dict[str, list[ProtocolEditPatch]] = {}
    for patch in patches:
        by_risk_key.setdefault(_normalize_risk_key(patch.risk_title), []).append(patch)
    for risk in missing:
        key = _normalize_risk_key(risk.title)
        candidates = by_risk_key.get(key, [])
        if not candidates:
            continue
        preferred = sorted(
            candidates,
            key=lambda patch: (
                0 if patch.action == "replace" else 1,
                _paragraph_id_sort_key(patch.target_paragraph_id),
                len(patch.new_text),
            ),
        )[0]
        if preferred not in filtered:
            filtered.append(preferred)
    return filtered


def _is_high_risk_expansion(original_text: str, new_text: str) -> bool:
    original_norm = f" {_normalize_match_text(original_text)} "
    new_norm = f" {_normalize_match_text(new_text)} "
    if _has_explanatory_rationale_language(original_norm, new_norm):
        return True

    risk_markers = (
        " may ",
        " could ",
        " might ",
        " risk ",
        " severe ",
        " adverse ",
        " burden ",
        " dropout ",
        " uncertain ",
        " unknown ",
        " however ",
    )
    control_markers = (
        " will ",
        " must ",
        " criteria ",
        " threshold ",
        " monitoring ",
        " observation ",
        " emergency ",
        " required ",
        " escalation ",
        " protocol ",
        " documented ",
        " defined ",
    )
    risk_delta = sum(new_norm.count(marker) - original_norm.count(marker) for marker in risk_markers)
    control_delta = sum(new_norm.count(marker) - original_norm.count(marker) for marker in control_markers)
    if risk_delta > control_delta + 2:
        return True
    if (" may cause " in new_norm and " may cause " not in original_norm) or (
        " risk of " in new_norm and " risk of " not in original_norm
    ):
        return True
    return False


def _has_explanatory_rationale_language(original_norm: str, new_norm: str) -> bool:
    rationale_markers = (
        " because ",
        " in order to ",
        " to mitigate ",
        " to reduce the risk ",
        " to minimize the risk ",
        " to address this risk ",
        " to address the risk ",
        " to explain ",
        " this section will define ",
        " this paragraph will define ",
        " this change ",
        " this update ",
    )
    new_only_markers = [marker for marker in rationale_markers if marker in new_norm and marker not in original_norm]
    if new_only_markers:
        return True

    explanatory_starts = (
        "to mitigate ",
        "to reduce the risk ",
        "to minimize the risk ",
        "to address this risk ",
        "to address the risk ",
        "because ",
        "in order to ",
        "this section will define ",
        "this paragraph will define ",
    )
    stripped = new_norm.strip()
    return any(stripped.startswith(marker.strip()) for marker in explanatory_starts)


def _replace_paragraph_text(paragraph: Paragraph, text: str) -> None:
    paragraph.text = text
    _color_paragraph_red(paragraph)


def _insert_paragraph_after(paragraph: Paragraph, text: str) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)  # noqa: SLF001 - required for insertion at exact location
    new_para = Paragraph(new_p, paragraph._parent)  # noqa: SLF001 - paragraph parent is required by python-docx
    new_para.text = text
    return new_para


def _color_paragraph_red(paragraph: Paragraph) -> None:
    if not paragraph.runs:
        run = paragraph.add_run(paragraph.text)
        run.font.color.rgb = RGBColor(176, 0, 32)
        return
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor(176, 0, 32)


def _write_colored_pdf(paragraphs: list[_RenderedParagraph], output_path: Path) -> None:
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    line_height = 13
    font_name = "Helvetica"
    font_size = 10

    c.setFont(font_name, font_size)
    c.drawString(margin, y, "Mitigated Protocol Draft")
    y -= line_height * 2
    c.setFillColor(colors.black)
    c.drawString(margin, y, "Edited text is shown in red.")
    y -= line_height * 2

    max_width = width - 2 * margin
    for paragraph in paragraphs:
        c.setFillColor(colors.HexColor("#B00020") if paragraph.edited else colors.black)
        paragraph_text = _sanitize_xml_text(paragraph.text.strip())
        if paragraph.is_heading and "\n" not in paragraph_text and len(paragraph_text) <= 120:
            c.setFont("Helvetica-Bold", 11)
        else:
            c.setFont(font_name, font_size)
        wrapped_lines = _wrap_line(paragraph_text, font_name, font_size, max_width)
        for line in wrapped_lines:
            if y <= margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - margin
                c.setFillColor(colors.HexColor("#B00020") if paragraph.edited else colors.black)
            c.drawString(margin, y, line)
            y -= line_height
        y -= line_height
    c.save()


def _wrap_line(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    if not text:
        return [""]
    words = text.split(" ")
    current = words[0]
    lines: list[str] = []
    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _is_heading_like(text: str) -> bool:
    line = text.splitlines()[0].strip()
    if not line:
        return False
    if len(line) > 120:
        return False
    if re.match(r"^\d+(\.\d+)*\s+[A-Za-z].*$", line):
        return True
    if re.match(r"^[A-Z][A-Z0-9\s\-\(\)\/]{4,}$", line):
        return True
    return False


def _is_body_section_heading(text: str, style_name: str) -> bool:
    line = text.splitlines()[0].strip()
    if _is_protected_administrative_heading(line):
        return False
    style_lower = style_name.lower()
    if style_lower.startswith("heading") and len(line.split()) <= 12:
        return True
    return bool(re.match(r"^\d+(\.\d+)*[\.\)]?\s+[A-Za-z].*$", line))


def _is_front_matter_like(
    text: str,
    is_heading: bool,
    page_number: int,
    style_name: str = "",
    body_started: bool = True,
) -> bool:
    normalized = " ".join(text.lower().split())
    words = normalized.split()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text.strip()
    if _is_protected_administrative_heading(first_line):
        return True
    if not body_started and page_number <= 3:
        return True
    if is_heading and page_number <= 2 and len(words) <= 12:
        return True
    if len(words) <= 3 and page_number <= 2:
        return True
    if any(
        marker in normalized
        for marker in (
            "prepared by",
            "sponsored by",
            "version",
            "protocol number",
            "table of contents",
            "contents",
        )
    ):
        return True
    if style_name.lower().startswith("toc"):
        return True
    if re.match(r"^.+\.{3,}\s*\d+$", text.strip()):
        return True
    if _looks_like_name_list(normalized):
        return True
    if _looks_like_administrative_people_block(text, normalized):
        return True
    if (
        page_number <= 3
        and len(words) <= 25
        and any(ch in text for ch in (",", ";", "|", "@"))
    ):
        return True
    if style_name.lower() in {"title", "subtitle"}:
        return True
    return False


def _looks_like_name_list(normalized: str) -> bool:
    if len(normalized.split()) > 30:
        return False
    credentials = ("md", "phd", "mph", "ms", "rn", "dr", "prof", "m.d", "ph.d")
    credential_hits = sum(1 for token in credentials if f" {token}" in f" {normalized}")
    separators = normalized.count(";") + normalized.count(",")
    return credential_hits >= 2 or separators >= 4


def _looks_like_personnel_roster_entry(text: str, normalized: str) -> bool:
    if len(normalized.split()) > 20:
        return False
    if not any(char.isalpha() for char in text):
        return False

    name_match = re.match(
        r"^[A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?(?:,\s*[A-Za-z\.\- ]+)?$",
        text.strip(),
    )
    if not name_match:
        return False

    role_or_credential_markers = (
        "investigator",
        "co-investigator",
        "sub-investigator",
        "study coordinator",
        "site coordinator",
        "medical monitor",
        "project manager",
        "trial manager",
        "md",
        "phd",
        "mph",
        "rn",
        "msc",
    )
    return any(marker in normalized for marker in role_or_credential_markers) or ("," in text)


def _is_protected_administrative_heading(line: str) -> bool:
    normalized = " ".join(line.lower().split()).strip(" :-")
    if not normalized:
        return False
    normalized = re.sub(r"^\d+(\.\d+)*[\.\)]?\s+", "", normalized)
    heading_markers = (
        "investigator",
        "investigators",
        "principal investigator",
        "co-investigator",
        "sub-investigator",
        "study team",
        "study staff",
        "study personnel",
        "personnel",
        "clinical staff",
        "author",
        "authors",
        "authorship",
        "affiliations",
        "contributing investigators",
        "investigator list",
        "participating investigators",
        "contributors",
        "coordination team",
        "coordinating center",
        "steering committee",
        "data safety monitoring board",
        "acknowledgements",
        "acknowledgments",
        "sponsor contacts",
        "contact information",
        "contact details",
        "distribution list",
        "signature page",
        "signatures",
        "approvals",
        "revision history",
        "document history",
    )
    return any(normalized == marker or normalized.startswith(f"{marker} ") for marker in heading_markers)


def _looks_like_administrative_people_block(text: str, normalized: str) -> bool:
    words = normalized.split()
    if len(words) > 55:
        return False

    direct_admin_phrases = (
        "study contact information",
        "contact information",
        "contact details",
        "sponsor contact information",
        "investigator contact information",
        "study staff",
        "study personnel",
        "coordinating center",
        "steering committee",
        "investigator list",
        "participating investigators",
    )
    if any(normalized.startswith(phrase) for phrase in direct_admin_phrases):
        return True

    role_markers = (
        "principal investigator",
        "co-investigator",
        "sub-investigator",
        "study chair",
        "medical monitor",
        "sponsor contact",
        "contact person",
        "study coordinator",
        "site coordinator",
        "project manager",
        "trial manager",
    )
    contact_markers = (
        "email",
        "e-mail",
        "telephone",
        "phone",
        "fax",
        "address",
    )
    has_role = any(marker in normalized for marker in role_markers)
    has_contact = any(marker in normalized for marker in contact_markers) or ("@" in text)
    separators = text.count(";") + text.count(",")
    if _looks_like_personnel_roster_entry(text, normalized):
        return True
    return (has_role and (has_contact or separators >= 2)) or (has_contact and separators >= 3)


def _paragraph_match_score(
    paragraph: ProtocolParagraph,
    evidence_pages: set[int],
    risk_tokens: set[str],
) -> int:
    if not paragraph.is_editable:
        return -1000
    score = 0
    if evidence_pages and paragraph.page_number in evidence_pages:
        score += 10
    paragraph_text = paragraph.text.lower()
    token_hits = sum(1 for token in risk_tokens if token in paragraph_text)
    score += token_hits * 3
    if paragraph.is_heading:
        score -= 10
    if len(paragraph.text.split()) < 8:
        score -= 4
    return score


def _paragraph_has_page_break(paragraph: Paragraph) -> bool:
    return any(_run_has_page_break(run) for run in paragraph.runs)


def _run_has_page_break(run: Run) -> bool:
    if "\f" in run.text:
        return True
    run_xml = run._element.xml  # noqa: SLF001 - required for break-marker detection
    return (
        'w:type="page"' in run_xml
        or "w:lastRenderedPageBreak" in run_xml
        or "lastRenderedPageBreak" in run_xml
    )


def _reduce_patch_set(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
    max_total: int,
    required_risks: list[RiskAssessment] | None = None,
) -> list[ProtocolEditPatch]:
    paragraph_ids = {p.paragraph_id for p in paragraphs if p.is_editable}
    valid_patches = [patch for patch in patches if patch.target_paragraph_id in paragraph_ids]
    grouped: dict[str, list[ProtocolEditPatch]] = {}
    for patch in valid_patches:
        grouped.setdefault(patch.target_paragraph_id, []).append(patch)

    action_priority = {"replace": 0, "insert_after": 1, "insert_before": 2}
    reduced: list[ProtocolEditPatch] = []
    used_keys: set[tuple[str, str, str]] = set()
    required_by_key = (
        {_normalize_risk_key(risk.title): risk for risk in required_risks} if required_risks else {}
    )
    # First pass: guarantee one patch per selected risk when possible.
    for risk_key in required_by_key.keys():
        risk_candidates = [
            patch
            for patch in valid_patches
            if _normalize_risk_key(patch.risk_title) == risk_key
        ]
        if not risk_candidates:
            continue
        chosen = sorted(
            risk_candidates,
            key=lambda patch: (
                action_priority.get(patch.action, 9),
                _paragraph_id_sort_key(patch.target_paragraph_id),
                len(patch.new_text),
            ),
        )[0]
        dedupe_key = (
            _normalize_risk_key(chosen.risk_title),
            chosen.action,
            chosen.target_paragraph_id,
        )
        if dedupe_key not in used_keys:
            reduced.append(chosen)
            used_keys.add(dedupe_key)
        if len(reduced) >= max_total:
            return reduced[:max_total]

    # Second pass: add additional high-value patches.
    for target_id in sorted(grouped.keys(), key=_paragraph_id_sort_key):
        target_group = sorted(
            grouped[target_id],
            key=lambda patch: (action_priority.get(patch.action, 9), len(patch.new_text)),
        )
        # Keep limited patch density per paragraph to avoid destructive over-editing.
        for patch in target_group[:3]:
            dedupe_key = (
                _normalize_risk_key(patch.risk_title),
                patch.action,
                patch.target_paragraph_id,
            )
            if dedupe_key in used_keys:
                continue
            reduced.append(patch)
            used_keys.add(dedupe_key)
        if len(reduced) >= max_total:
            break
    return reduced[:max_total]


def _dedupe_patches_by_target_paragraph(patches: list[ProtocolEditPatch]) -> list[ProtocolEditPatch]:
    if len(patches) <= 1:
        return patches
    grouped: dict[str, list[ProtocolEditPatch]] = {}
    ordered_targets: list[str] = []
    for patch in patches:
        target = patch.target_paragraph_id.strip()
        if target not in grouped:
            grouped[target] = []
            ordered_targets.append(target)
        grouped[target].append(patch)

    action_priority = {"replace": 0, "insert_after": 1, "insert_before": 2}
    deduped: list[ProtocolEditPatch] = []
    for target in sorted(ordered_targets, key=_paragraph_id_sort_key):
        candidates = grouped[target]
        if len(candidates) == 1:
            deduped.append(candidates[0])
            continue
        chosen = sorted(
            candidates,
            key=lambda patch: (
                action_priority.get(patch.action, 9),
                -len(patch.new_text.strip()),
            ),
        )[0]
        if chosen.action != "replace":
            deduped.append(chosen)
            continue
        merged_reason = _summarize_patch_reasons(candidates)
        merged_risk_title = _summarize_patch_risks(candidates)
        deduped.append(
            ProtocolEditPatch(
                risk_title=merged_risk_title,
                action=chosen.action,
                target_paragraph_id=chosen.target_paragraph_id,
                new_text=chosen.new_text,
                reason=merged_reason,
            )
        )
    return deduped


def _chunk_risks(risks: list[RiskAssessment], batch_size: int) -> list[list[RiskAssessment]]:
    if not risks:
        return []
    return [risks[i : i + batch_size] for i in range(0, len(risks), batch_size)]


def _chunk_contexts(contexts: list[_RiskEditContext], batch_size: int) -> list[list[_RiskEditContext]]:
    if not contexts:
        return []
    return [contexts[i : i + batch_size] for i in range(0, len(contexts), batch_size)]


def _serialize_risk_contexts(contexts: list[_RiskEditContext]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for context in contexts:
        payload = context.risk.model_dump()
        payload["anchor_paragraph_ids"] = list(context.anchor_paragraph_ids)
        payload["evidence_anchor_paragraph_ids"] = list(context.evidence_anchor_paragraph_ids)
        payload["candidate_paragraph_ids"] = list(context.candidate_paragraph_ids)
        payload["must_cover_mitigations"] = list(context.must_cover_mitigations)
        serialized.append(payload)
    return serialized


def _paragraph_id_sort_key(paragraph_id: str) -> tuple[int, str]:
    match = re.match(r"^P(\d+)$", paragraph_id)
    if match:
        return (int(match.group(1)), paragraph_id)
    return (10**9, paragraph_id)


def _normalize_match_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _evidence_match_score(paragraph: ProtocolParagraph, evidence_key: str) -> int:
    paragraph_key = _normalize_match_text(paragraph.text)
    if not paragraph_key:
        return -1000
    if evidence_key in paragraph_key:
        return 100 + len(evidence_key.split())
    evidence_tokens = {token for token in evidence_key.split() if len(token) > 2}
    if not evidence_tokens:
        return 0
    paragraph_tokens = set(paragraph_key.split())
    token_hits = len(evidence_tokens.intersection(paragraph_tokens))
    if token_hits == 0:
        return 0
    return token_hits * 5


def _missing_coverage_risks(
    paragraphs: list[ProtocolParagraph],
    selected_risks: list[RiskAssessment],
    patches: list[ProtocolEditPatch],
) -> list[RiskAssessment]:
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    risk_by_key = {_normalize_risk_key(risk.title): risk for risk in selected_risks}
    covered_keys = {
        risk_key
        for patch in patches
        for risk_key in [_normalize_risk_key(patch.risk_title)]
        if risk_key in risk_by_key
        and _patch_effectively_mitigates_risk(patch, paragraph_by_id, risk_by_key[risk_key])
    }
    return [
        risk
        for risk in selected_risks
        if _normalize_risk_key(risk.title) not in covered_keys
    ]


def _normalize_risk_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _build_fallback_patches_for_missing_risks(
    paragraphs: list[ProtocolParagraph],
    missing_risks: list[RiskAssessment],
) -> list[ProtocolEditPatch]:
    fallback: list[ProtocolEditPatch] = []
    for risk in missing_risks:
        target_id = _select_target_paragraph_for_risk(paragraphs, risk)
        if not target_id:
            continue
        target_paragraph = next(
            (paragraph for paragraph in paragraphs if paragraph.paragraph_id == target_id),
            paragraphs[0],
        )
        mitigation_text = _build_fallback_mitigation_text(risk, target_paragraph.text)
        fallback.append(
            ProtocolEditPatch(
                risk_title=risk.title,
                action="replace",
                target_paragraph_id=target_id,
                new_text=mitigation_text,
                reason="Fallback mitigation patch to ensure selected risk coverage.",
            )
        )
    return fallback


def _select_target_paragraph_for_risk(
    paragraphs: list[ProtocolParagraph],
    risk: RiskAssessment,
) -> str | None:
    editable_paragraphs = [paragraph for paragraph in paragraphs if paragraph.is_editable]
    if not editable_paragraphs:
        editable_paragraphs = paragraphs
    risk_tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", f"{risk.title} {risk.category}".lower())
        if len(token) > 3
    }
    evidence_pages: set[int] = set()
    for ev in risk.evidence:
        evidence_pages.update(range(ev.page_start, ev.page_end + 1))
    best_id: str | None = None
    best_score = -1000
    for paragraph in editable_paragraphs:
        if not _paragraph_contains_risk_tokens(paragraph, risk_tokens):
            continue
        score = _paragraph_match_score(paragraph, evidence_pages, risk_tokens)
        if score > best_score:
            best_score = score
            best_id = paragraph.paragraph_id
    return best_id


def _paragraph_contains_risk_tokens(paragraph: ProtocolParagraph, risk_tokens: set[str]) -> bool:
    if not risk_tokens:
        return False
    paragraph_text = _normalize_match_text(paragraph.text)
    if not paragraph_text:
        return False
    return any(token in paragraph_text for token in risk_tokens)


def _build_fallback_mitigation_text(risk: RiskAssessment, base_text: str = "") -> str:
    mitigations = [m.strip().rstrip(".") for m in risk.mitigations if m.strip()]
    if not mitigations:
        mitigations = [
            "implement targeted monitoring and safeguards",
            "add explicit operational controls and escalation criteria",
        ]
    selected = mitigations[:3]
    actions = "; ".join(selected)
    lead_sentence = _build_fallback_lead_sentence(risk, base_text)
    if lead_sentence:
        return (
            f"{lead_sentence} The protocol will {actions}. "
            "These requirements will be documented in trial procedures, monitored prospectively, "
            "and escalated when predefined criteria are met."
        )
    return (
        f"The protocol will {actions}. "
        "These requirements will be documented in trial procedures, monitored prospectively, "
        "and escalated when predefined criteria are met."
    )


def _patch_effectively_mitigates_risk(
    patch: ProtocolEditPatch,
    paragraph_by_id: dict[str, ProtocolParagraph],
    risk: RiskAssessment,
) -> bool:
    if patch.action != "replace":
        return False
    paragraph = paragraph_by_id.get(patch.target_paragraph_id)
    if paragraph is None or not paragraph.is_editable:
        return False
    original_text = _normalize_match_text(paragraph.text)
    new_text = _normalize_match_text(patch.new_text)
    if not original_text or not new_text or original_text == new_text:
        return False
    similarity = SequenceMatcher(None, original_text, new_text).ratio()
    length_delta = abs(len(new_text) - len(original_text))
    if similarity >= 0.97 and length_delta < 40:
        return False
    if not (_contains_control_language(new_text) or _mentions_risk_mitigations(risk, new_text)):
        return False
    return True


def _contains_control_language(text: str) -> bool:
    control_markers = (
        " will ",
        " must ",
        " required ",
        " criteria ",
        " monitor",
        " assessment",
        " document",
        " review",
        " schedule",
        " training",
        " eligibility",
        " threshold",
        " confirm",
        " escalate",
    )
    padded = f" {text} "
    return any(marker in padded for marker in control_markers)


def _mentions_risk_mitigations(risk: RiskAssessment, normalized_new_text: str) -> bool:
    for mitigation in risk.mitigations:
        mitigation_tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", mitigation.lower())
            if len(token) > 4
        ]
        if mitigation_tokens and any(token in normalized_new_text for token in mitigation_tokens):
            return True
    return False


def _build_fallback_lead_sentence(risk: RiskAssessment, base_text: str) -> str:
    cleaned = base_text.strip()
    if cleaned:
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
        normalized = _normalize_match_text(first_sentence)
        if (
            normalized
            and len(first_sentence.split()) >= 8
            and len(first_sentence) <= 240
            and not _looks_risk_heavy(normalized)
        ):
            softened = re.sub(r"\b(may|might|could|should|can)\b", "will", first_sentence, flags=re.IGNORECASE)
            return softened if softened.endswith((".", "!", "?")) else f"{softened}."
    return _domain_mitigation_lead(risk)


def _looks_risk_heavy(text: str) -> bool:
    risk_markers = (
        " risk ",
        " concern ",
        " unknown ",
        " may ",
        " might ",
        " could ",
        " toxicity",
        " dropout",
        " insufficient",
        " limited ",
        " burden",
        " adverse event",
        " failure ",
    )
    padded = f" {text} "
    return any(marker in padded for marker in risk_markers)


def _domain_mitigation_lead(risk: RiskAssessment) -> str:
    domain_leads = {
        "health": "Participants will follow explicit safety monitoring and management requirements.",
        "financial": "Study operations will follow explicit budget, feasibility, and execution controls.",
        "ethical": "The protocol will apply explicit consent, oversight, and participant-protection requirements.",
        "eligibility": "Participants must meet explicit screening, eligibility, and enrollment criteria.",
        "intervention": "Intervention administration will follow explicit dosing and treatment controls.",
    }
    return domain_leads.get(
        risk.domain.value,
        "The protocol will apply explicit control and implementation requirements.",
    )


def _sanitize_xml_text(text: str) -> str:
    # XML 1.0 valid chars: tab, LF, CR, and most printable Unicode ranges.
    return "".join(
        ch
        for ch in text
        if ch == "\t"
        or ch == "\n"
        or ch == "\r"
        or (0x20 <= ord(ch) <= 0xD7FF)
        or (0xE000 <= ord(ch) <= 0xFFFD)
        or (0x10000 <= ord(ch) <= 0x10FFFF)
    )
