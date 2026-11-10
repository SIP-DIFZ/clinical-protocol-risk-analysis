from docx import Document

from protocol_risk_analyzer.editor import (
    _RenderedParagraph,
    _apply_protocol_patches,
    _build_docx_protocol_paragraphs,
    _build_fallback_patches_for_missing_risks,
    _build_fallback_mitigation_text,
    _consolidate_patches_for_application,
    _consolidate_patches_for_preview,
    _filter_high_risk_expansion_patches,
    _build_protocol_paragraphs,
    _build_risk_edit_contexts,
    _chunk_risks,
    _chunk_contexts,
    _evidence_match_score,
    _find_anchor_paragraph_ids,
    format_protocol_paragraph_reference,
    _missing_coverage_risks,
    _normalize_match_text,
    _normalize_risk_key,
    _paragraph_id_sort_key,
    _reduce_patch_set,
    _render_patch_log,
    _render_markdown_protocol,
    _sanitize_xml_text,
    _select_target_paragraph_for_risk,
    _serialize_risk_contexts,
    _validate_generated_patches,
    _write_preserved_docx,
    ProtocolParagraph,
)
from protocol_risk_analyzer.io import PageText
from protocol_risk_analyzer.models import EvidenceSnippet, ProtocolEditPatch, RiskAssessment, RiskDomain


class _FakeJointRewriteLlm:
    def editor_agent_rewrite_overlapping_paragraph(
        self,
        target_paragraph: dict[str, str | int],
        overlapping_risks: list[dict[str, object]],
        candidate_rewrites: list[dict[str, str]],
    ) -> tuple[str, str]:
        assert target_paragraph["paragraph_id"] == "P2"
        assert len(overlapping_risks) == 2
        assert len(candidate_rewrites) == 2
        return (
            "Original paragraph text. The protocol will require weekly safety assessments. "
            "Dose-hold criteria will be documented before escalation.",
            "Combines overlapping mitigations into one rewrite.",
        )


class _FailingJointRewriteLlm:
    def editor_agent_rewrite_overlapping_paragraph(
        self,
        target_paragraph: dict[str, str | int],
        overlapping_risks: list[dict[str, object]],
        candidate_rewrites: list[dict[str, str]],
    ) -> tuple[str, str]:
        del target_paragraph, overlapping_risks, candidate_rewrites
        raise RuntimeError("rewrite failed")


def test_build_protocol_paragraphs_splits_blocks_and_assigns_ids() -> None:
    pages = [PageText(page_number=1, text="First paragraph.\n\nSecond paragraph.")]
    paragraphs = _build_protocol_paragraphs(pages)
    assert [p.paragraph_id for p in paragraphs] == ["P1", "P2"]
    assert paragraphs[0].text == "First paragraph."
    assert paragraphs[1].text == "Second paragraph."
    assert paragraphs[0].paragraph_number_on_page == 1
    assert paragraphs[1].paragraph_number_on_page == 2


def test_format_protocol_paragraph_reference_uses_page_and_page_local_number() -> None:
    paragraph = ProtocolParagraph("P8", 62, "Protocol text.", False, True, 3)

    assert format_protocol_paragraph_reference(paragraph) == "Page 62 - Paragraph 3"


def test_apply_protocol_patches_replaces_and_inserts() -> None:
    pages = [PageText(page_number=1, text="Alpha.\n\nBeta.")]
    paragraphs = _build_protocol_paragraphs(pages)
    patches = [
        ProtocolEditPatch(
            risk_title="Alpha risk",
            action="replace",
            target_paragraph_id="P1",
            new_text="Alpha revised with additional safety monitoring requirements.",
            reason="Mitigates safety risk.",
        ),
        ProtocolEditPatch(
            risk_title="Beta risk",
            action="insert_after",
            target_paragraph_id="P2",
            new_text="Beta mitigation addendum.",
            reason="Adds monitoring language.",
        ),
    ]
    rendered = _apply_protocol_patches(paragraphs, patches)
    assert rendered[0] == _RenderedParagraph(
        text="Alpha revised with additional safety monitoring requirements.",
        edited=True,
        is_heading=False,
    )
    assert rendered[1] == _RenderedParagraph(text="Beta.", edited=False, is_heading=False)
    assert rendered[2] == _RenderedParagraph(
        text="Beta mitigation addendum.",
        edited=True,
        is_heading=False,
    )


def test_render_markdown_protocol_colors_edited_text() -> None:
    rendered = [
        _RenderedParagraph(text="Original text.", edited=False, is_heading=False),
        _RenderedParagraph(text="Edited text.", edited=True, is_heading=False),
    ]
    md = _render_markdown_protocol(rendered)
    assert "Original text." in md
    assert "<span style=\"color:#B00020\">Edited text.</span>" in md


def test_render_patch_log_includes_original_and_new_text() -> None:
    paragraphs = [ProtocolParagraph("P1", 1, "Original protocol text.", False, True)]
    patches = [
        ProtocolEditPatch(
            risk_title="Alpha risk",
            action="replace",
            target_paragraph_id="P1",
            new_text="Updated protocol text with mitigation.",
            reason="Mitigates risk.",
        )
    ]
    log = _render_patch_log(paragraphs, patches)
    assert "### Original Text" in log
    assert "Original protocol text." in log
    assert "### New Text" in log
    assert "Updated protocol text with mitigation." in log


def test_render_patch_log_includes_validation_notes() -> None:
    paragraphs = [ProtocolParagraph("P1", 1, "Original protocol text.", False, True)]
    patches = [
        ProtocolEditPatch(
            risk_title="Alpha risk",
            action="replace",
            target_paragraph_id="P1",
            new_text="Updated protocol text with mitigation.",
            reason="Mitigates risk.",
        )
    ]
    log = _render_patch_log(
        paragraphs,
        patches,
        validation_notes=["Discarded patch for P1 because it did not validate as a mitigation."],
    )
    assert "Validation Notes" in log
    assert "did not validate as a mitigation" in log


def test_validate_generated_patches_discards_non_mitigating_replace() -> None:
    paragraphs = [
        ProtocolParagraph(
            paragraph_id="P1",
            page_number=1,
            text="Participants may be enrolled monthly.",
            is_heading=False,
            is_editable=True,
        )
    ]
    selected = [
        RiskAssessment(
            title="Enrollment cadence risk",
            domain=RiskDomain.HEALTH,
            category="safety",
            likelihood=3,
            impact=4,
            confidence=0.8,
            rationale="Monthly enrollment may not protect participants well enough.",
            mitigations=["Require weekly safety review"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="monthly", page_start=1, page_end=1)],
        )
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Enrollment cadence risk",
            action="replace",
            target_paragraph_id="P1",
            new_text="The protocol will require weekly safety monitoring and escalation criteria.",
            reason="Simple rewrite.",
        )
    ]
    validated, notes = _validate_generated_patches(paragraphs, selected, patches)
    assert validated == []
    assert notes
    assert "strayed from the original paragraph topic" in notes[0]


def test_validate_generated_patches_discards_off_topic_replace() -> None:
    paragraphs = [
        ProtocolParagraph(
            paragraph_id="P1",
            page_number=1,
            text="Participants may be enrolled monthly.",
            is_heading=False,
            is_editable=True,
        )
    ]
    selected = [
        RiskAssessment(
            title="Enrollment cadence risk",
            domain=RiskDomain.HEALTH,
            category="safety",
            likelihood=3,
            impact=4,
            confidence=0.8,
            rationale="Monthly enrollment may not protect participants well enough.",
            mitigations=["Require weekly safety review"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="monthly", page_start=1, page_end=1)],
        )
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Enrollment cadence risk",
            action="replace",
            target_paragraph_id="P1",
            new_text="The protocol will require weekly safety monitoring and escalation criteria.",
            reason="Off topic rewrite.",
        )
    ]
    validated, notes = _validate_generated_patches(paragraphs, selected, patches)
    assert validated == []
    assert notes
    assert "strayed from the original paragraph topic" in notes[0]


def test_validate_generated_patches_discards_off_topic_insert() -> None:
    paragraphs = [
        ProtocolParagraph(
            paragraph_id="P1",
            page_number=1,
            text="Dose escalation is observed in clinic before administration.",
            is_heading=False,
            is_editable=True,
        )
    ]
    selected = [
        RiskAssessment(
            title="Dose observation risk",
            domain=RiskDomain.HEALTH,
            category="safety",
            likelihood=3,
            impact=4,
            confidence=0.8,
            rationale="Observation standards may be unclear.",
            mitigations=["Define monitoring criteria for dose escalation"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="Dose escalation", page_start=1, page_end=1)],
        )
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Dose observation risk",
            action="insert_after",
            target_paragraph_id="P1",
            new_text="The protocol will require vendor budget reconciliation and invoice approval workflows.",
            reason="Adds generic control language.",
        )
    ]
    validated, notes = _validate_generated_patches(paragraphs, selected, patches)
    assert validated == []
    assert notes
    assert "strayed from the original paragraph topic" in notes[0]


def test_sanitize_xml_text_removes_null_and_control_chars() -> None:
    raw = "Alpha\x00Beta\x01Gamma\tDelta\n"
    cleaned = _sanitize_xml_text(raw)
    assert cleaned == "AlphaBetaGamma\tDelta\n"


def test_build_docx_protocol_paragraphs_preserves_order(tmp_path) -> None:
    path = tmp_path / "source.docx"
    doc = Document()
    doc.add_heading("1. Title", level=1)
    doc.add_paragraph("A first protocol paragraph.")
    doc.add_paragraph("A second protocol paragraph.")
    doc.save(path)

    paragraphs = _build_docx_protocol_paragraphs(path)
    assert [p.paragraph_id for p in paragraphs] == ["P1", "P2", "P3"]
    assert paragraphs[0].is_heading is True
    assert paragraphs[1].text == "A first protocol paragraph."


def test_find_anchor_paragraph_ids_prefers_evidence_matching_paragraph() -> None:
    paragraphs = [
        ProtocolParagraph(
            paragraph_id="P1",
            page_number=4,
            text="Participants attend a baseline visit.",
            is_heading=False,
            is_editable=True,
        ),
        ProtocolParagraph(
            paragraph_id="P2",
            page_number=5,
            text="Participants must discontinue prior therapy for 14 days before enrollment.",
            is_heading=False,
            is_editable=True,
        ),
        ProtocolParagraph(
            paragraph_id="P3",
            page_number=5,
            text="General operational note.",
            is_heading=False,
            is_editable=True,
        ),
    ]
    risk = RiskAssessment(
        title="Prior therapy washout risk",
        domain=RiskDomain.ELIGIBILITY,
        category="eligibility",
        likelihood=4,
        impact=4,
        confidence=0.9,
        rationale="Washout may limit enrollment.",
        mitigations=["Allow shorter washout"],
        factor_triggers=[],
        evidence=[
            EvidenceSnippet(
                excerpt="discontinue prior therapy for 14 days before enrollment",
                page_start=5,
                page_end=5,
            )
        ],
    )
    anchors = _find_anchor_paragraph_ids(paragraphs, risk, {5})
    assert anchors[0] == "P2"


def test_build_risk_edit_contexts_keeps_local_candidates_near_anchor() -> None:
    paragraphs = [
        ProtocolParagraph("P1", 5, "Heading", True, False),
        ProtocolParagraph("P2", 5, "Participants must discontinue prior therapy for 14 days before enrollment.", False, True),
        ProtocolParagraph("P3", 5, "Laboratory checks will occur before first dose.", False, True),
        ProtocolParagraph("P4", 6, "Long-term follow-up procedures.", False, True),
    ]
    risk = RiskAssessment(
        title="Prior therapy washout risk",
        domain=RiskDomain.ELIGIBILITY,
        category="eligibility",
        likelihood=4,
        impact=4,
        confidence=0.9,
        rationale="Washout may limit enrollment.",
        mitigations=["Allow shorter washout"],
        factor_triggers=[],
        evidence=[
            EvidenceSnippet(
                excerpt="discontinue prior therapy for 14 days before enrollment",
                page_start=5,
                page_end=5,
            )
        ],
    )
    contexts = _build_risk_edit_contexts(paragraphs, [risk])
    assert len(contexts) == 1
    assert contexts[0].anchor_paragraph_ids[0] == "P2"
    assert "P2" in contexts[0].candidate_paragraph_ids
    assert "P3" not in contexts[0].candidate_paragraph_ids
    assert "P1" not in contexts[0].candidate_paragraph_ids


def test_select_target_paragraph_for_risk_skips_irrelevant_paragraphs() -> None:
    paragraphs = [
        ProtocolParagraph("P1", 1, "General overview.", False, True),
        ProtocolParagraph("P2", 2, "Participants will return for follow-up visits.", False, True),
    ]
    risk = RiskAssessment(
        title="Enrollment washout risk",
        domain=RiskDomain.ELIGIBILITY,
        category="eligibility",
        likelihood=4,
        impact=4,
        confidence=0.9,
        rationale="Washout may limit enrollment.",
        mitigations=["Allow shorter washout"],
        factor_triggers=[],
        evidence=[],
    )

    assert _select_target_paragraph_for_risk(paragraphs, risk) is None


def test_write_preserved_docx_keeps_unedited_paragraphs_and_order(tmp_path) -> None:
    source = tmp_path / "source.docx"
    output = tmp_path / "output.docx"
    doc = Document()
    doc.add_heading("Study Overview", level=1)
    doc.add_paragraph("Paragraph one stays unchanged.")
    doc.add_paragraph("Paragraph two will be updated.")
    doc.save(source)

    patches = [
        ProtocolEditPatch(
            risk_title="Monitoring risk",
            action="replace",
            target_paragraph_id="P3",
            new_text="Paragraph two updated with additional safety monitoring controls.",
            reason="Mitigates monitoring gaps.",
        )
    ]
    _write_preserved_docx(source, patches, output)

    updated = Document(output)
    texts = [p.text for p in updated.paragraphs if p.text.strip()]
    assert texts[0] == "Study Overview"
    assert texts[1] == "Paragraph one stays unchanged."
    assert texts[2] == "Paragraph two updated with additional safety monitoring controls."


def test_consolidate_patches_for_application_merges_same_target_replacements() -> None:
    paragraphs = [
        ProtocolParagraph("P2", 2, "Original paragraph text.", False, True),
    ]
    selected_risks = [
        RiskAssessment(
            title="Risk one",
            domain=RiskDomain.HEALTH,
            category="safety",
            likelihood=4,
            impact=4,
            confidence=0.8,
            rationale="First overlapping rationale text.",
            mitigations=["require weekly safety assessments"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="Original paragraph text", page_start=2, page_end=2)],
        ),
        RiskAssessment(
            title="Risk two",
            domain=RiskDomain.HEALTH,
            category="safety",
            likelihood=4,
            impact=4,
            confidence=0.8,
            rationale="Second overlapping rationale text.",
            mitigations=["document dose-hold criteria before escalation"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="Original paragraph text", page_start=2, page_end=2)],
        ),
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Risk one",
            action="replace",
            target_paragraph_id="P2",
            new_text="Original paragraph text. The protocol will require weekly safety assessments.",
            reason="Adds monitoring.",
        ),
        ProtocolEditPatch(
            risk_title="Risk two",
            action="replace",
            target_paragraph_id="P2",
            new_text="Original paragraph text. Dose-hold criteria will be documented before escalation.",
            reason="Adds dose criteria.",
        ),
    ]
    consolidated = _consolidate_patches_for_application(
        paragraphs,
        patches,
        selected_risks,
        _FakeJointRewriteLlm(),
    )
    assert len(consolidated) == 1
    assert consolidated[0].target_paragraph_id == "P2"
    assert "weekly safety assessments" in consolidated[0].new_text
    assert "Dose-hold criteria will be documented" in consolidated[0].new_text


def test_consolidate_patches_for_application_falls_back_when_rewrite_fails() -> None:
    paragraphs = [
        ProtocolParagraph("P2", 2, "Original paragraph text.", False, True),
    ]
    selected_risks = [
        RiskAssessment(
            title="Risk one",
            domain=RiskDomain.HEALTH,
            category="safety",
            likelihood=4,
            impact=4,
            confidence=0.8,
            rationale="First overlapping rationale text.",
            mitigations=["require weekly safety assessments"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="Original paragraph text", page_start=2, page_end=2)],
        ),
        RiskAssessment(
            title="Risk two",
            domain=RiskDomain.HEALTH,
            category="safety",
            likelihood=4,
            impact=4,
            confidence=0.8,
            rationale="Second overlapping rationale text.",
            mitigations=["document dose-hold criteria before escalation"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="Original paragraph text", page_start=2, page_end=2)],
        ),
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Risk one",
            action="replace",
            target_paragraph_id="P2",
            new_text="Original paragraph text. The protocol will require weekly safety assessments.",
            reason="Adds monitoring.",
        ),
        ProtocolEditPatch(
            risk_title="Risk two",
            action="replace",
            target_paragraph_id="P2",
            new_text="Original paragraph text. Dose-hold criteria will be documented before escalation.",
            reason="Adds dose criteria.",
        ),
    ]
    notes: list[str] = []
    consolidated = _consolidate_patches_for_application(
        paragraphs,
        patches,
        selected_risks,
        _FailingJointRewriteLlm(),
        notes,
    )
    assert len(consolidated) == 1
    assert "weekly safety assessments" in consolidated[0].new_text
    assert "Dose-hold criteria will be documented" in consolidated[0].new_text
    assert notes
    assert "deterministic merge" in notes[0].lower()


def test_consolidate_patches_for_preview_merges_without_llm_rewrite() -> None:
    paragraphs = [
        ProtocolParagraph("P2", 2, "Original paragraph text.", False, True),
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Risk one",
            action="replace",
            target_paragraph_id="P2",
            new_text="Original paragraph text. The protocol will require weekly safety assessments.",
            reason="Adds monitoring.",
        ),
        ProtocolEditPatch(
            risk_title="Risk two",
            action="replace",
            target_paragraph_id="P2",
            new_text="Original paragraph text. Dose-hold criteria will be documented before escalation.",
            reason="Adds dose criteria.",
        ),
    ]

    consolidated = _consolidate_patches_for_preview(paragraphs, patches)

    assert len(consolidated) == 1
    assert consolidated[0].target_paragraph_id == "P2"
    assert "weekly safety assessments" in consolidated[0].new_text
    assert "Dose-hold criteria will be documented" in consolidated[0].new_text


def test_filter_high_risk_expansion_patches_drops_bloated_speculative_rewrite() -> None:
    paragraphs = [
        ProtocolParagraph("P1", 1, "Dose escalation is observed in clinic.", False, True),
    ]
    risky_patch = ProtocolEditPatch(
        risk_title="Speculative risk",
        action="replace",
        target_paragraph_id="P1",
        new_text=(
            "Dose escalation may cause severe adverse issues and may cause burden and dropout, "
            "and this could create risk in many scenarios. "
            "Additional narrative context is provided to explain broad uncertainties."
        ),
        reason="Adds broad language.",
    )
    safe_patch = ProtocolEditPatch(
        risk_title="Monitoring risk",
        action="replace",
        target_paragraph_id="P1",
        new_text=(
            "Dose escalation will be observed in clinic with defined monitoring criteria and "
            "emergency response requirements."
        ),
        reason="Adds concrete controls.",
    )
    filtered = _filter_high_risk_expansion_patches(
        paragraphs,
        [risky_patch, safe_patch],
        required_risks=None,
    )
    assert safe_patch in filtered
    assert risky_patch not in filtered


def test_filter_high_risk_expansion_patches_allows_long_but_on_topic_rewrite() -> None:
    paragraphs = [
        ProtocolParagraph("P1", 1, "Dose escalation is observed in clinic.", False, True),
    ]
    expanded_patch = ProtocolEditPatch(
        risk_title="Monitoring risk",
        action="replace",
        target_paragraph_id="P1",
        new_text=(
            "Dose escalation will be observed in clinic with defined monitoring criteria, "
            "emergency response requirements, documented review before escalation, and dose-hold "
            "criteria when clinical findings indicate that continued escalation is no longer appropriate."
        ),
        reason="Adds concrete controls.",
    )
    filtered = _filter_high_risk_expansion_patches(
        paragraphs,
        [expanded_patch],
        required_risks=None,
    )
    assert expanded_patch in filtered


def test_filter_high_risk_expansion_patches_drops_explanatory_rationale_rewrite() -> None:
    paragraphs = [
        ProtocolParagraph("P1", 1, "Infusions are administered at the study site.", False, True),
    ]
    rationale_patch = ProtocolEditPatch(
        risk_title="Infusion monitoring risk",
        action="replace",
        target_paragraph_id="P1",
        new_text=(
            "To mitigate infusion reaction risk, infusions will be administered at the study site "
            "with direct observation and emergency medication availability."
        ),
        reason="Adds rationale in the edit text.",
    )
    filtered = _filter_high_risk_expansion_patches(
        paragraphs,
        [rationale_patch],
        required_risks=None,
    )
    assert rationale_patch not in filtered


def test_chunk_risks_splits_into_batches() -> None:
    sample = [f"r{i}" for i in range(45)]
    batches = _chunk_risks(sample, batch_size=20)
    assert len(batches) == 3
    assert len(batches[0]) == 20
    assert len(batches[1]) == 20
    assert len(batches[2]) == 5


def test_chunk_contexts_splits_into_batches() -> None:
    risk = RiskAssessment(
        title="Risk 1",
        domain=RiskDomain.HEALTH,
        category="cat",
        likelihood=3,
        impact=3,
        confidence=0.8,
        rationale="Some rationale text here.",
        mitigations=["Do X"],
        factor_triggers=[],
        evidence=[EvidenceSnippet(excerpt="evidence", page_start=1, page_end=1)],
    )
    contexts = [
        _build_risk_edit_contexts(
            [ProtocolParagraph(f"P{i+1}", 1, "body text", False, True)],
            [risk.model_copy(update={"title": f"Risk {i}"})],
        )[0]
        for i in range(3)
    ]
    batches = _chunk_contexts(contexts, batch_size=2)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_reduce_patch_set_keeps_multiple_valid_patches() -> None:
    paragraphs = [
        ProtocolParagraph(paragraph_id="P1", page_number=1, text="a", is_heading=False, is_editable=True),
        ProtocolParagraph(paragraph_id="P2", page_number=1, text="b", is_heading=False, is_editable=True),
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="risk one",
            action="replace",
            target_paragraph_id="P1",
            new_text="a" * 30,
            reason="x" * 12,
        ),
        ProtocolEditPatch(
            risk_title="risk two",
            action="insert_after",
            target_paragraph_id="P1",
            new_text="b" * 30,
            reason="y" * 12,
        ),
        ProtocolEditPatch(
            risk_title="risk three",
            action="insert_before",
            target_paragraph_id="P1",
            new_text="c" * 30,
            reason="z" * 12,
        ),
        ProtocolEditPatch(
            risk_title="risk four",
            action="replace",
            target_paragraph_id="P2",
            new_text="d" * 30,
            reason="w" * 12,
        ),
    ]
    reduced = _reduce_patch_set(paragraphs, patches, max_total=10)
    assert len(reduced) == 4
    assert reduced[0].target_paragraph_id == "P1"
    assert reduced[-1].target_paragraph_id == "P2"


def test_paragraph_id_sort_key_orders_numerically() -> None:
    ids = ["P10", "P2", "P1"]
    assert sorted(ids, key=_paragraph_id_sort_key) == ["P1", "P2", "P10"]


def test_missing_coverage_risks_detects_uncovered() -> None:
    paragraphs = [
        ProtocolParagraph(
            paragraph_id="P1",
            page_number=1,
            text="Baseline text that may create a study risk if not revised.",
            is_heading=False,
            is_editable=True,
        )
    ]
    selected = [
        RiskAssessment(
            title="Risk A",
            domain=RiskDomain.HEALTH,
            category="cat",
            likelihood=3,
            impact=3,
            confidence=0.8,
            rationale="Some rationale text here.",
            mitigations=["Do X"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="evidence", page_start=1, page_end=1)],
        ),
        RiskAssessment(
            title="Risk B",
            domain=RiskDomain.HEALTH,
            category="cat",
            likelihood=3,
            impact=3,
            confidence=0.8,
            rationale="Some rationale text here.",
            mitigations=["Do Y"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="evidence", page_start=1, page_end=1)],
        ),
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Risk A",
            action="replace",
            target_paragraph_id="P1",
            new_text="The protocol will require weekly safety monitoring and documented review before dosing.",
            reason="Mitigate risk A.",
        )
    ]
    missing = _missing_coverage_risks(paragraphs, selected, patches)
    assert len(missing) == 1
    assert missing[0].title == "Risk B"


def test_build_fallback_patches_for_missing_risks_creates_patch_per_risk() -> None:
    paragraphs = [
        ProtocolParagraph(
            paragraph_id="P1",
            page_number=1,
            text="baseline text",
            is_heading=False,
            is_editable=True,
        )
    ]
    missing = [
        RiskAssessment(
            title="Risk C",
            domain=RiskDomain.HEALTH,
            category="cat",
            likelihood=3,
            impact=3,
            confidence=0.8,
            rationale="Some rationale text here.",
            mitigations=["Add monitoring visits"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="evidence", page_start=1, page_end=1)],
        )
    ]
    fallback = _build_fallback_patches_for_missing_risks(paragraphs, missing)
    assert fallback == []


def test_missing_coverage_risks_treats_insert_only_patch_as_uncovered() -> None:
    paragraphs = [
        ProtocolParagraph(
            paragraph_id="P1",
            page_number=4,
            text="Participants may discontinue prior therapy immediately before dosing.",
            is_heading=False,
            is_editable=True,
        )
    ]
    selected = [
        RiskAssessment(
            title="Washout risk",
            domain=RiskDomain.ELIGIBILITY,
            category="eligibility",
            likelihood=4,
            impact=4,
            confidence=0.8,
            rationale="Washout period is unclear.",
            mitigations=["require a 14-day washout period before first dose"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="prior therapy", page_start=4, page_end=4)],
        )
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="Washout risk",
            action="insert_after",
            target_paragraph_id="P1",
            new_text="Participants will be monitored closely before dosing.",
            reason="Adds mitigation language.",
        )
    ]
    missing = _missing_coverage_risks(paragraphs, selected, patches)
    assert len(missing) == 1
    assert missing[0].title == "Washout risk"


def test_build_fallback_mitigation_text_rewrites_risk_heavy_base_text() -> None:
    risk = RiskAssessment(
        title="Safety risk",
        domain=RiskDomain.HEALTH,
        category="safety",
        likelihood=4,
        impact=5,
        confidence=0.9,
        rationale="Monitoring may be insufficient.",
        mitigations=["require weekly safety assessments", "define dose-hold criteria"],
        factor_triggers=[],
        evidence=[EvidenceSnippet(excerpt="monitoring", page_start=2, page_end=2)],
    )
    text = _build_fallback_mitigation_text(
        risk,
        "Safety monitoring may be insufficient and adverse events could be missed.",
    )
    assert "Safety monitoring may be insufficient" not in text
    assert "weekly safety assessments" in text


def test_normalize_risk_key_is_stable() -> None:
    assert _normalize_risk_key("Risk: A/B?") == "risk a b"
    assert _normalize_match_text("Evidence: A/B?") == "evidence a b"
    assert "the protocol will" in _build_fallback_mitigation_text(
        RiskAssessment(
            title="Risk D",
            domain=RiskDomain.HEALTH,
            category="cat",
            likelihood=3,
            impact=3,
            confidence=0.8,
            rationale="Some rationale text here.",
            mitigations=["Action one"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="evidence", page_start=1, page_end=1)],
        )
    ).lower()


def test_build_docx_protocol_paragraphs_locks_front_matter_and_tracks_pages(tmp_path) -> None:
    path = tmp_path / "front-matter.docx"
    doc = Document()
    doc.add_paragraph("Clinical Study Protocol")
    doc.add_paragraph("Prepared by Sponsor Team, Jane Doe; John Smith")
    doc.add_page_break()
    doc.add_heading("1. Introduction", level=1)
    doc.add_paragraph("This section describes the study design and conduct.")
    doc.save(path)

    paragraphs = _build_docx_protocol_paragraphs(path)
    assert paragraphs[0].is_editable is False
    assert paragraphs[1].is_editable is False
    assert paragraphs[2].page_number == 2
    assert paragraphs[3].is_editable is True


def test_build_docx_protocol_paragraphs_marks_toc_as_non_editable(tmp_path) -> None:
    path = tmp_path / "toc.docx"
    doc = Document()
    toc_heading = doc.add_paragraph("Table of Contents")
    toc_heading.style = doc.styles["Title"]
    toc_entry = doc.add_paragraph("1. Introduction ........ 3")
    doc.add_paragraph("1. Introduction")
    doc.add_paragraph("Study procedures are described here.")
    doc.save(path)

    paragraphs = _build_docx_protocol_paragraphs(path)
    assert paragraphs[0].is_editable is False
    assert paragraphs[1].is_editable is False


def test_build_docx_protocol_paragraphs_locks_investigator_roster_sections(tmp_path) -> None:
    path = tmp_path / "investigator-roster.docx"
    doc = Document()
    doc.add_paragraph("Cover")
    doc.add_page_break()
    doc.add_paragraph("Preface")
    doc.add_page_break()
    doc.add_paragraph("Contents")
    doc.add_page_break()
    doc.add_heading("1. Investigators", level=1)
    doc.add_paragraph("Principal Investigator: Jane Doe, MD; Co-Investigator: John Smith, PhD")
    doc.add_heading("2. Study Procedures", level=1)
    doc.add_paragraph("Participants will complete scheduled safety monitoring visits.")
    doc.save(path)

    paragraphs = _build_docx_protocol_paragraphs(path)
    assert paragraphs[3].is_editable is False
    assert paragraphs[4].is_editable is False
    assert paragraphs[5].is_editable is True
    assert paragraphs[6].is_editable is True


def test_build_docx_protocol_paragraphs_locks_contact_blocks(tmp_path) -> None:
    path = tmp_path / "contacts.docx"
    doc = Document()
    doc.add_paragraph("Cover")
    doc.add_page_break()
    doc.add_paragraph("Preface")
    doc.add_page_break()
    doc.add_paragraph("Contents")
    doc.add_page_break()
    doc.add_paragraph("Study contact information")
    doc.add_paragraph("Sponsor Contact: Alice Lee, phone: 555-1100, email: alice@example.com")
    doc.add_paragraph("Dose escalation will follow predefined clinical criteria.")
    doc.save(path)

    paragraphs = _build_docx_protocol_paragraphs(path)
    assert paragraphs[3].is_editable is False
    assert paragraphs[4].is_editable is False
    assert paragraphs[5].is_editable is True


def test_build_docx_protocol_paragraphs_locks_name_rows_inside_staff_section(tmp_path) -> None:
    path = tmp_path / "staff-roster.docx"
    doc = Document()
    doc.add_heading("Study Personnel", level=1)
    doc.add_paragraph("Jane Doe, MD")
    doc.add_paragraph("John Smith, PhD")
    doc.add_heading("2. Study Procedures", level=1)
    doc.add_paragraph("Participants will undergo informed consent and screening procedures.")
    doc.save(path)

    paragraphs = _build_docx_protocol_paragraphs(path)
    assert paragraphs[0].is_editable is False
    assert paragraphs[1].is_editable is False
    assert paragraphs[2].is_editable is False
    assert paragraphs[3].is_editable is False
    assert paragraphs[4].is_editable is True


def test_reduce_patch_set_drops_non_editable_targets() -> None:
    paragraphs = [
        ProtocolParagraph(paragraph_id="P1", page_number=1, text="title", is_heading=True, is_editable=False),
        ProtocolParagraph(paragraph_id="P2", page_number=2, text="body", is_heading=False, is_editable=True),
    ]
    patches = [
        ProtocolEditPatch(
            risk_title="risk one",
            action="replace",
            target_paragraph_id="P1",
            new_text="x" * 30,
            reason="y" * 12,
        ),
        ProtocolEditPatch(
            risk_title="risk two",
            action="replace",
            target_paragraph_id="P2",
            new_text="z" * 30,
            reason="w" * 12,
        ),
    ]
    reduced = _reduce_patch_set(paragraphs, patches, max_total=10)
    assert len(reduced) == 1
    assert reduced[0].target_paragraph_id == "P2"


def test_serialize_risk_contexts_includes_anchor_and_candidate_ids() -> None:
    risk = RiskAssessment(
        title="Risk E",
        domain=RiskDomain.HEALTH,
        category="cat",
        likelihood=3,
        impact=3,
        confidence=0.8,
        rationale="Some rationale text here.",
        mitigations=["Action one"],
        factor_triggers=[],
        evidence=[EvidenceSnippet(excerpt="evidence", page_start=1, page_end=1)],
    )
    contexts = _build_risk_edit_contexts(
        [ProtocolParagraph("P1", 1, "evidence in body text", False, True)],
        [risk],
    )
    serialized = _serialize_risk_contexts(contexts)
    assert serialized[0]["anchor_paragraph_ids"] == ["P1"]
    assert serialized[0]["evidence_anchor_paragraph_ids"] == ["P1"]
    assert serialized[0]["candidate_paragraph_ids"] == ["P1"]
    assert serialized[0]["must_cover_mitigations"] == ["Action one"]


def test_evidence_match_score_rewards_direct_match() -> None:
    paragraph = ProtocolParagraph("P1", 1, "This paragraph contains evidence text.", False, True)
    direct = _evidence_match_score(paragraph, "contains evidence text")
    partial = _evidence_match_score(paragraph, "contains missing words")
    assert direct > partial
