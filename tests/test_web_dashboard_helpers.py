from pathlib import Path

from protocol_risk_analyzer.frontend.assistant_popup import _handle_editor_proposal_edit_reference
from protocol_risk_analyzer.frontend.mitigate_protocol import (
    _count_generation_ready_proposals,
    _format_proposal_label,
    _paginate_editor_proposals,
    _render_editor_job_progress,
    _visible_review_proposals,
)
from protocol_risk_analyzer.backend.editor_service import generate_editor_outputs_from_proposals
from protocol_risk_analyzer.models import (
    AnalysisResult,
    EvidenceSnippet,
    ProtocolMetadata,
    RiskAssessment,
    RiskDomain,
)
from protocol_risk_analyzer.web_app import (
    _add_saved_project,
    _append_chat_turn,
    _build_saved_analysis_archive,
    _ai_assistant_popup_input_key,
    _build_editor_risk_entries,
    _build_risk_dataframe,
    _clean_editor_progress_message,
    _clear_editor_job,
    _create_saved_analysis_snapshot,
    _delete_saved_analysis,
    _delete_saved_project,
    _editor_job_state_path,
    _estimate_editor_eta_seconds,
    _format_eta,
    _filter_risks,
    _format_domain_label,
    _initial_ai_assistant_message,
    _is_docx_protocol_upload,
    _load_saved_analysis_index,
    _load_saved_analysis_state,
    _empty_revision_session_state,
    _add_revision_proposals_to_session,
    _approve_pending_revision_changes,
    _discard_awaiting_revision_changes,
    _undo_last_revision_change,
    _clear_revision_plan,
    _ensure_revision_session_state,
    _sanitize_revision_session_state,
    _pending_ai_assistant_message,
    _persist_editor_job,
    _rename_saved_project,
    _rename_saved_analysis,
    _resolve_selected_risk_id_from_checkboxes,
    _read_editor_job,
    _reset_ai_assistant_conversation,
    _saved_analyses_index_path,
    _saved_project_names,
    _set_saved_analysis_project,
    _ensure_flat_chat_history_state,
    _sanitize_accepted_risk_ids,
    _sanitize_accepted_risk_notes,
    _sanitize_chat_history,
    _truncate_text_for_label,
    _score_band,
    _is_general_conversation_message,
)
from protocol_risk_analyzer.ui_helpers import (
    _apply_revision_item_edit_request,
    _approve_revision_change,
    _decline_revision_change,
    _edit_revision_change,
    _extract_revision_target_index,
)


class _FakeProgressSlot:
    def __init__(self) -> None:
        self.last_markdown: str | None = None
        self.last_caption: str | None = None
        self.was_emptied = False

    def markdown(self, text: str) -> None:
        self.last_markdown = text

    def caption(self, text: str) -> None:
        self.last_caption = text

    def empty(self) -> None:
        self.was_emptied = True


def _sample_result() -> AnalysisResult:
    return AnalysisResult(
        metadata=ProtocolMetadata(
            source_path="x.pdf",
            document_name="x.pdf",
            page_count=10,
            chunk_count=2,
            generated_at_utc=ProtocolMetadata.now_iso(),
        ),
        executive_summary="Sample summary that is long enough to satisfy schema.",
        key_assumptions=[],
        risks=[
            RiskAssessment(
                title="Recruitment delay risk",
                domain=RiskDomain.FINANCIAL,
                category="Eligibility pressure",
                likelihood=4,
                impact=4,
                confidence=0.8,
                rationale="Eligibility may be restrictive and reduce recruitment speed.",
                mitigation_status="unmitigated",
                mitigation_status_reason="No concrete enrollment contingency controls were provided in the protocol.",
                factor_triggers=["Highly restrictive inclusion criteria"],
                mitigations=["Add backup sites"],
                evidence=[
                    EvidenceSnippet(
                        excerpt="Participants must meet strict criteria.",
                        page_start=3,
                        page_end=3,
                    )
                ],
            ),
            RiskAssessment(
                title="Adverse reaction risk",
                domain=RiskDomain.HEALTH,
                category="Safety uncertainty",
                likelihood=3,
                impact=5,
                confidence=0.9,
                rationale="Prior severe reactions indicate potential participant harm.",
                mitigation_status="partially_mitigated",
                mitigation_status_reason="The protocol defines frequent safety labs and stopping rules tied to adverse signals.",
                factor_triggers=["Serious adverse events in prior studies"],
                mitigations=["Increase monitoring frequency"],
                evidence=[
                    EvidenceSnippet(
                        excerpt="Severe adverse reactions were previously observed.",
                        page_start=5,
                        page_end=5,
                    )
                ],
            ),
        ],
    )


def test_build_risk_dataframe_contains_expected_columns() -> None:
    df = _build_risk_dataframe(_sample_result())
    assert len(df) == 2
    assert "risk_id" in df.columns
    assert "title" in df.columns
    assert "score" in df.columns
    assert "factor_triggers" in df.columns
    assert "domain_label" in df.columns
    assert "mitigation_status" in df.columns
    assert "mitigation_status_label" in df.columns
    assert "mitigation_status_reason" in df.columns
    assert "severity" in df.columns
    assert "report_markdown" in df.columns


def test_filter_risks_applies_domain_score_and_query() -> None:
    df = _build_risk_dataframe(_sample_result())
    filtered = _filter_risks(df, selected_domain="financial", min_score=10.0, query="recruitment")
    assert len(filtered) == 1
    assert filtered.iloc[0]["domain"] == "financial"


def test_domain_and_score_helpers() -> None:
    assert _format_domain_label("health") == "Health"
    assert _score_band(15.0) == "Very High"
    assert _score_band(12.0) == "High"
    assert _score_band(8.0) == "Moderate"
    assert _score_band(3.0) == "Low"
    assert _format_eta(0) == "00:00"
    assert _format_eta(65) == "01:05"
    assert _estimate_editor_eta_seconds(20, 2, 5) == 30.0


def test_is_docx_protocol_upload_accepts_docx_only() -> None:
    assert _is_docx_protocol_upload("protocol.docx") is True
    assert _is_docx_protocol_upload("PROTOC OL.DOCX") is True
    assert _is_docx_protocol_upload("protocol.pdf") is False
    assert _is_docx_protocol_upload("protocol.txt") is False
    assert _is_docx_protocol_upload("") is False


def test_build_risk_dataframe_uses_model_mitigation_status() -> None:
    df = _build_risk_dataframe(_sample_result())
    status_by_title = {row["title"]: row["mitigation_status"] for _, row in df.iterrows()}
    assert status_by_title["Recruitment delay risk"] == "unmitigated"
    assert status_by_title["Adverse reaction risk"] == "partially_mitigated"


def test_build_risk_dataframe_keeps_full_mitigation_reason() -> None:
    result = _sample_result()
    long_reason = " ".join(["This reason should remain visible."] * 8)
    result.risks[0] = result.risks[0].model_copy(update={"mitigation_status_reason": long_reason})

    df = _build_risk_dataframe(result)
    reason_by_title = {row["title"]: row["mitigation_status_reason"] for _, row in df.iterrows()}

    assert reason_by_title["Recruitment delay risk"] == long_reason


def test_build_risk_dataframe_maps_legacy_complete_status_to_adequate() -> None:
    result = _sample_result()
    result.risks[1] = result.risks[1].model_copy(update={"mitigation_status": "completely_mitigated"})

    df = _build_risk_dataframe(result)
    status_by_title = {row["title"]: row["mitigation_status"] for _, row in df.iterrows()}
    label_by_title = {row["title"]: row["mitigation_status_label"] for _, row in df.iterrows()}

    assert status_by_title["Adverse reaction risk"] == "adequately_mitigated"
    assert label_by_title["Adverse reaction risk"] == "Adequately Mitigated"


def test_clean_editor_progress_message_removes_prefix_and_substep_fraction() -> None:
    message = "Mitigate Protocol: Editor Agent generating patches (1/7)..."
    assert _clean_editor_progress_message(message) == "Editor Agent generating patches ..."


def test_render_editor_job_progress_uses_step_when_processed_steps_missing() -> None:
    progress_slot = _FakeProgressSlot()
    note_slot = _FakeProgressSlot()

    _render_editor_job_progress(
        {
            "status": "running",
            "step": 3,
            "total_steps": 5,
            "message": "Mitigate Protocol: Editor Agent generating patches (2/3)...",
            "eta_seconds": 42.0,
            "selected_count": 5,
            "proposals": [{}, {}],
        },
        progress_slot,
        note_slot,
    )

    assert progress_slot.last_markdown == "**2 proposals ready**"
    assert note_slot.last_caption == "Editor Agent generating patches ..."


def test_paginate_editor_proposals_uses_five_items_per_page() -> None:
    proposals = [{"proposal_id": f"p{i}"} for i in range(1, 13)]

    page_items, current_page, total_pages = _paginate_editor_proposals(proposals, current_page=2)

    assert total_pages == 3
    assert current_page == 2
    assert [idx for idx, _ in page_items] == [6, 7, 8, 9, 10]
    assert [proposal["proposal_id"] for _, proposal in page_items] == ["p6", "p7", "p8", "p9", "p10"]


def test_visible_review_proposals_hide_declined_items() -> None:
    proposals = [
        {"proposal_id": "p1", "status": "pending"},
        {"proposal_id": "p2", "status": "declined"},
        {"proposal_id": "p3", "status": "user_edited"},
    ]

    visible = _visible_review_proposals(proposals)

    assert [proposal["proposal_id"] for proposal in visible] == ["p1", "p3"]


def test_count_generation_ready_proposals_includes_user_edited() -> None:
    proposals = [
        {"proposal_id": "p1", "status": "pending"},
        {"proposal_id": "p2", "status": "accepted"},
        {"proposal_id": "p3", "status": "user_edited"},
        {"proposal_id": "p4", "status": "declined"},
    ]

    assert _count_generation_ready_proposals(proposals) == 2


def test_build_editor_risk_entries_and_sanitize_accepted_ids() -> None:
    result = _sample_result()
    entries = _build_editor_risk_entries(result)
    assert len(entries) == 2
    first_id = entries[0]["risk_id"]
    second_id = entries[1]["risk_id"]
    assert first_id.startswith("financial:1:")
    assert second_id.startswith("health:2:")

    sanitized = _sanitize_accepted_risk_ids(
        [first_id, "missing-id", first_id, second_id],
        {first_id, second_id},
    )
    assert sanitized == [first_id, second_id]


def test_sanitize_accepted_risk_notes_filters_invalid_ids_and_empty_notes() -> None:
    valid_ids = {"financial:1:Recruitment delay risk", "health:2:Adverse reaction risk"}
    sanitized = _sanitize_accepted_risk_notes(
        {
            "financial:1:Recruitment delay risk": "Operational constraints acknowledged.",
            "health:2:Adverse reaction risk": "  ",
            "unknown:risk:id": "Should be removed",
        },
        valid_ids,
    )
    assert sanitized == {"financial:1:Recruitment delay risk": "Operational constraints acknowledged."}


def test_resolve_selected_risk_id_from_checkboxes_prefers_new_selection() -> None:
    selected = _resolve_selected_risk_id_from_checkboxes(
        displayed_risk_ids=["risk_a", "risk_b", "risk_c"],
        selected_flags=[True, True, False],
        current_selected_risk_id="risk_a",
    )
    assert selected == "risk_b"


def test_resolve_selected_risk_id_from_checkboxes_keeps_current_when_clicked_off() -> None:
    selected = _resolve_selected_risk_id_from_checkboxes(
        displayed_risk_ids=["risk_a", "risk_b"],
        selected_flags=[False, False],
        current_selected_risk_id="risk_a",
    )
    assert selected == "risk_a"


def test_resolve_selected_risk_id_from_checkboxes_defaults_to_first_when_missing() -> None:
    selected = _resolve_selected_risk_id_from_checkboxes(
        displayed_risk_ids=["risk_a", "risk_b"],
        selected_flags=[False, False],
        current_selected_risk_id="risk_x",
    )
    assert selected == "risk_a"


def test_format_proposal_label_uses_edit_number_and_target_reference() -> None:
    assert _format_proposal_label(3, "Page 4 - Paragraph 2") == "Edit 3: Page 4 - Paragraph 2"
    assert _format_proposal_label(7, "") == "Edit 7"


def test_extract_revision_target_index_matches_edit_number_references() -> None:
    assert _extract_revision_target_index("Edit #4, make it more detailed.") == 4
    assert _extract_revision_target_index("Change 2 to clarify the safety language.") == 2
    assert _extract_revision_target_index("Please rewrite the protocol.") is None


def test_apply_revision_item_edit_request_updates_targeted_change() -> None:
    session = _sanitize_revision_session_state(
        {
            "pending_changes": [
                {
                    "id": "r1",
                    "summary": "Add safety monitoring",
                    "requested_change": "Add safety monitoring during treatment.",
                    "rationale": "This reduces risk.",
                    "affected_sections": ["Safety Monitoring"],
                    "estimated_edits": 2,
                    "focus": "health",
                }
            ],
            "approved_changes": [],
        }
    )

    handled, message = _apply_revision_item_edit_request(session, "Edit #1, make it more detailed.")

    assert handled is True
    assert "updated" in message.lower()
    assert "Include additional detail" in session["pending_changes"][0]["requested_change"]
    assert session["pending_changes"][0]["status"] == "user_edited"


def test_apply_revision_item_edit_request_requests_clarification_when_needed() -> None:
    session = _sanitize_revision_session_state(
        {
            "pending_changes": [
                {
                    "id": "r1",
                    "summary": "Add safety monitoring",
                    "requested_change": "Add safety monitoring during treatment.",
                    "rationale": "This reduces risk.",
                    "affected_sections": ["Safety Monitoring"],
                    "estimated_edits": 2,
                    "focus": "health",
                }
            ],
            "approved_changes": [],
        }
    )

    handled, message = _apply_revision_item_edit_request(session, "Edit #1")

    assert handled is True
    assert "specific instruction" in message.lower()
    assert session["pending_changes"][0]["requested_change"] == "Add safety monitoring during treatment."


def test_handle_editor_proposal_edit_reference_explains_existing_edit() -> None:
    analysis_state = {
        "editor_review": {
            "proposals": [
                {
                    "proposal_id": "p1",
                    "target_label": "Page 4 - Paragraph 2",
                    "risk_title": "Safety monitoring gap",
                    "reason": "Adds clearer monitoring language to reduce the unresolved safety gap.",
                    "original_text": "Subjects will return for safety follow-up after dosing.",
                    "proposed_text": "Subjects will return for scheduled safety follow-up after dosing with clearer monitoring expectations.",
                    "accepted_text": "",
                    "status": "pending",
                }
            ]
        }
    }

    handled, message = _handle_editor_proposal_edit_reference(analysis_state, "Tell me more about Edit 1.")

    assert handled is True
    assert "Edit #1: Page 4 - Paragraph 2" in message
    assert "Safety monitoring gap" in message
    assert "Current proposed paragraph" in message


def test_handle_editor_proposal_edit_reference_updates_proposed_paragraph(monkeypatch) -> None:
    analysis_state = {
        "chosen_model": "test-model",
        "editor_review": {
            "proposals": [
                {
                    "proposal_id": "p1",
                    "target_label": "Page 4 - Paragraph 2",
                    "risk_title": "Safety monitoring gap",
                    "reason": "Adds clearer monitoring language to reduce the unresolved safety gap.",
                    "original_text": "Subjects will return for safety follow-up after dosing.",
                    "proposed_text": "Subjects will return for safety follow-up after dosing.",
                    "accepted_text": "",
                    "status": "pending",
                }
            ]
        },
    }

    def _fake_run_protocol_proposal_edit(**kwargs):
        assert kwargs["target_label"] == "Page 4 - Paragraph 2"
        assert kwargs["risk_title"] == "Safety monitoring gap"
        assert kwargs["original_text"] == "Subjects will return for safety follow-up after dosing."
        assert kwargs["current_text"] == "Subjects will return for safety follow-up after dosing."
        assert "more detailed" in kwargs["user_request"].lower()
        assert kwargs["model"] == "test-model"
        return (
            "Subjects will return for safety follow-up after dosing, including added monitoring detail.",
            "Clarifies the monitoring expectations while preserving the paragraph's original purpose.",
        )

    monkeypatch.setattr(
        "protocol_risk_analyzer.frontend.assistant_popup.run_protocol_proposal_edit",
        _fake_run_protocol_proposal_edit,
    )

    handled, message = _handle_editor_proposal_edit_reference(analysis_state, "Edit #1, make it more detailed.")

    proposal = analysis_state["editor_review"]["proposals"][0]
    assert handled is True
    assert "I updated **Edit #1: Page 4 - Paragraph 2**" in message
    assert proposal["accepted_text"].startswith("Subjects will return for safety follow-up after dosing, including")
    assert proposal["reason"].startswith("Clarifies the monitoring expectations")
    assert proposal["status"] == "user_edited"


def test_read_editor_job_falls_back_to_snapshot_and_output_files(tmp_path: Path) -> None:
    job_id = "editor-test"
    _persist_editor_job(
        tmp_path,
        {
            "job_id": job_id,
            "status": "completed",
            "step": 6,
            "total_steps": 6,
            "message": "Mitigated protocol ready.",
            "eta_seconds": 0.0,
            "selected_count": 4,
            "chosen_model": "test-model",
        },
    )
    (tmp_path / "mitigated_protocol.md").write_text("# Mitigated", encoding="utf-8")
    (tmp_path / "mitigated_protocol.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "mitigated_protocol.docx").write_bytes(b"DOCX")
    (tmp_path / "mitigated_protocol_patches.txt").write_text("patch log", encoding="utf-8")

    job = _read_editor_job(job_id, run_dir=tmp_path)

    assert job is not None
    assert job["status"] == "completed"
    assert job["result"]["chosen_model"] == "test-model"
    assert job["result"]["selected_count"] == 4
    assert job["result"]["mitigated_md_text"] == "# Mitigated"


def test_clear_editor_job_removes_snapshot_file(tmp_path: Path) -> None:
    job_id = "editor-clear"
    _persist_editor_job(
        tmp_path,
        {
            "job_id": job_id,
            "status": "running",
            "step": 1,
            "total_steps": 6,
            "message": "Loading protocol document...",
            "eta_seconds": 12.0,
        },
    )
    state_path = _editor_job_state_path(tmp_path, job_id)
    assert state_path.exists()

    _clear_editor_job(job_id, run_dir=tmp_path)

    assert not state_path.exists()


def test_generate_editor_outputs_from_proposals_includes_user_edited(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_protocol_editor_from_patches(*, input_path, output_dir, patches, model):
        assert input_path == tmp_path / "protocol.docx"
        assert output_dir == tmp_path
        assert model == "test-model"
        assert [patch.target_paragraph_id for patch in patches] == ["P1", "P2"]
        assert [patch.new_text for patch in patches] == ["Accepted replacement", "User-edited replacement"]

        class _Paths:
            mitigated_md_path = tmp_path / "mitigated_protocol.md"
            mitigated_pdf_path = tmp_path / "mitigated_protocol.pdf"
            mitigated_docx_path = tmp_path / "mitigated_protocol.docx"
            patch_log_path = tmp_path / "mitigated_protocol_patches.txt"

        _Paths.mitigated_md_path.write_text("md", encoding="utf-8")
        _Paths.mitigated_pdf_path.write_bytes(b"pdf")
        _Paths.mitigated_docx_path.write_bytes(b"docx")
        _Paths.patch_log_path.write_text("patches", encoding="utf-8")
        return "test-model", _Paths()

    monkeypatch.setattr(
        "protocol_risk_analyzer.backend.editor_service.run_protocol_editor_from_patches",
        _fake_run_protocol_editor_from_patches,
    )

    result = generate_editor_outputs_from_proposals(
        input_path=tmp_path / "protocol.docx",
        run_dir=tmp_path,
        proposals=[
            {
                "proposal_id": "p1",
                "status": "accepted",
                "target_paragraph_id": "P1",
                "risk_title": "Risk A",
                "action": "replace",
                "accepted_text": "Accepted replacement",
                "reason": "Accepted replacement text.",
            },
            {
                "proposal_id": "p2",
                "status": "user_edited",
                "target_paragraph_id": "P2",
                "risk_title": "Risk B",
                "action": "replace",
                "accepted_text": "User-edited replacement",
                "reason": "User-edited replacement text.",
            },
            {
                "proposal_id": "p3",
                "status": "declined",
                "target_paragraph_id": "P3",
                "risk_title": "Risk C",
                "action": "replace",
                "accepted_text": "Declined replacement",
                "reason": "Declined replacement text.",
            },
        ],
        model="test-model",
    )

    assert result["chosen_model"] == "test-model"
    assert result["mitigated_md_text"] == "md"
    assert result["patch_log_text"] == "patches"


def test_saved_analysis_snapshot_roundtrip_and_lifecycle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "protocol.pdf"
    input_path.write_bytes(b"%PDF-1.4")
    result = _sample_result()
    analysis_state = {
        "chosen_model": "test-model",
        "analyzed_pages": 10,
        "total_pages": 10,
        "chunk_count": 2,
        "input_path": str(input_path),
        "run_dir": str(run_dir),
        "json_text": result.model_dump_json(indent=2),
        "md_text": "report text",
        "pdf_bytes": b"%PDF-1.4",
        "editor_state": None,
        "accepted_risk_ids": ["financial:1:Recruitment delay risk"],
        "accepted_risk_notes": {"financial:1:Recruitment delay risk": "Tolerance accepted by sponsor."},
        "chat_threads": [],
        "chat_selected_thread_id": "",
        "chat_history": [
            {"role": "user", "content": "Summarize the analysis."},
            {
                "role": "assistant",
                "content": "Summary response",
                "can_answer": True,
                "cited_source_ids": [1],
                "follow_up_questions": ["What are the highest risks?"],
                "sources": [
                    {
                        "source_id": 1,
                        "kind": "analysis_risk",
                        "label": "Risk: Recruitment delay risk",
                    }
                ],
            },
        ],
        "revision_session": {
            "pending_changes": [
                {
                    "id": "rev-1",
                    "summary": "Reduce participant burden",
                    "requested_change": "Reduce in-person visit frequency and allow remote follow-up visits.",
                    "rationale": "Lower visit burden can improve retention and recruitment feasibility.",
                    "affected_sections": ["Study Visits", "Schedule of Assessments"],
                    "estimated_edits": 4,
                    "focus": "operational",
                }
            ],
            "approved_changes": [],
            "awaiting_confirmation_change_ids": ["rev-1"],
        },
        "editor_review": {
            "proposals": [
                {
                    "proposal_id": "P2:1",
                    "risk_title": "Recruitment delay risk",
                    "action": "replace",
                    "target_paragraph_id": "P2",
                    "reason": "Strengthens enrollment controls.",
                    "original_text": "Original paragraph text.",
                    "proposed_text": "Original paragraph text with enrollment controls.",
                    "accepted_text": "Original paragraph text with enrollment controls.",
                    "status": "user_edited",
                }
            ],
            "validation_notes": ["Use deterministic merge for same paragraph."],
            "ready": True,
        },
        "saved_analysis_project": "Oncology Study A",
    }
    saved_root = tmp_path / "saved"
    _create_saved_analysis_snapshot(analysis_state, saved_root)

    session_id = str(analysis_state.get("saved_analysis_id", ""))
    assert session_id
    index_path = _saved_analyses_index_path(saved_root)
    assert index_path.exists()
    index = _load_saved_analysis_index(saved_root)
    assert len(index["sessions"]) == 1
    assert index["sessions"][0]["project"] == "Oncology Study A"

    loaded_state = _load_saved_analysis_state(saved_root, session_id)
    assert loaded_state is not None
    assert loaded_state["json_text"]
    assert loaded_state["accepted_risk_ids"] == ["financial:1:Recruitment delay risk"]
    assert loaded_state["accepted_risk_notes"] == {"financial:1:Recruitment delay risk": "Tolerance accepted by sponsor."}
    assert loaded_state["chat_selected_thread_id"] == ""
    assert loaded_state["chat_threads"] == []
    assert loaded_state["chat_history"][0]["role"] == "user"
    assert loaded_state["chat_history"][1]["sources"][0]["label"] == "Risk: Recruitment delay risk"
    assert loaded_state["saved_analysis_project"] == "Oncology Study A"
    assert loaded_state["chat_history"][1]["content"] == "Summary response"
    assert loaded_state["revision_session"]["pending_changes"][0]["summary"] == "Reduce participant burden"
    assert loaded_state["editor_review"]["proposals"][0]["status"] == "user_edited"
    assert loaded_state["editor_review"]["validation_notes"][0] == "Use deterministic merge for same paragraph."

    archive = _build_saved_analysis_archive(saved_root, session_id)
    assert archive is not None
    assert len(archive) > 0

    _rename_saved_analysis(saved_root, session_id, "Renamed Analysis")
    renamed_index = _load_saved_analysis_index(saved_root)
    assert renamed_index["sessions"][0]["title"] == "Renamed Analysis"

    _delete_saved_analysis(saved_root, session_id)
    deleted_index = _load_saved_analysis_index(saved_root)
    assert deleted_index["sessions"] == []


def test_saved_project_names_include_default_and_sort_projects() -> None:
    sessions = [
        {"id": "a", "project": "Cardiology"},
        {"id": "b", "project": "oncology"},
        {"id": "c"},
    ]
    projects = _saved_project_names(sessions)
    assert "Cardiology" in projects
    assert "oncology" in projects


def test_saved_analysis_project_crud_updates_index_and_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_b"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "protocol.pdf"
    input_path.write_bytes(b"%PDF-1.4")
    result = _sample_result()
    analysis_state = {
        "chosen_model": "test-model",
        "analyzed_pages": 10,
        "total_pages": 10,
        "chunk_count": 2,
        "input_path": str(input_path),
        "run_dir": str(run_dir),
        "json_text": result.model_dump_json(indent=2),
        "md_text": "report text",
        "pdf_bytes": b"%PDF-1.4",
        "editor_state": None,
        "accepted_risk_ids": [],
        "saved_analysis_project": "",
    }
    saved_root = tmp_path / "saved"
    _create_saved_analysis_snapshot(analysis_state, saved_root)
    session_id = str(analysis_state["saved_analysis_id"])

    _add_saved_project(saved_root, "Immunology")
    _set_saved_analysis_project(saved_root, session_id, "Immunology")

    index = _load_saved_analysis_index(saved_root)
    assert "Immunology" in index["projects"]
    assert index["sessions"][0]["project"] == "Immunology"

    _rename_saved_project(saved_root, "Immunology", "Respiratory")
    renamed_index = _load_saved_analysis_index(saved_root)
    assert "Respiratory" in renamed_index["projects"]
    assert renamed_index["sessions"][0]["project"] == "Respiratory"

    _delete_saved_project(saved_root, "Respiratory")
    deleted_project_index = _load_saved_analysis_index(saved_root)
    assert "Respiratory" not in deleted_project_index["projects"]
    assert deleted_project_index["sessions"][0]["project"] == ""

    reloaded_state = _load_saved_analysis_state(saved_root, session_id)
    assert reloaded_state is not None
    assert reloaded_state["saved_analysis_project"] == ""


def test_truncate_text_for_label_handles_long_text() -> None:
    text = "This is a very long chat history label that should be shortened nicely"
    truncated = _truncate_text_for_label(text, 30)
    assert len(truncated) <= 30
    assert truncated.endswith("…")


def test_ai_assistant_popup_input_key_isolation_by_analysis() -> None:
    key_a = _ai_assistant_popup_input_key({"saved_analysis_id": "analysis-1"})
    key_b = _ai_assistant_popup_input_key({"saved_analysis_id": "analysis-2"})
    assert key_a != key_b
    assert key_a.startswith("ai_assistant_popup_input_")
    assert key_b.startswith("ai_assistant_popup_input_")


def test_ensure_flat_chat_history_state_seeds_initial_assistant_intro() -> None:
    analysis_state = {"chat_history": [], "chat_threads": [], "chat_selected_thread_id": ""}

    history = _ensure_flat_chat_history_state(analysis_state)

    assert history == [_initial_ai_assistant_message()]
    assert analysis_state["chat_history"] == history
    assert analysis_state["chat_threads"] == []
    assert analysis_state["chat_selected_thread_id"] == ""


def test_sanitize_chat_history_preserves_pending_assistant_placeholder() -> None:
    sanitized = _sanitize_chat_history([_pending_ai_assistant_message()])

    assert len(sanitized) == 1
    assert sanitized[0]["content"] == "..."
    assert sanitized[0]["pending_response"] is True


def test_append_chat_turn_replaces_pending_assistant_placeholder() -> None:
    analysis_state = {
        "chat_history": [
            _initial_ai_assistant_message(),
            {"role": "user", "content": "Summarize the top risks."},
            _pending_ai_assistant_message(),
        ],
        "chat_threads": [],
        "chat_selected_thread_id": "",
    }

    _append_chat_turn(
        analysis_state=analysis_state,
        user_prompt="Summarize the top risks.",
        assistant_payload={
            "answer_markdown": "Here are the top risks.",
            "can_answer": True,
            "show_sources": True,
            "cited_source_ids": [1],
            "follow_up_questions": ["Which one is highest priority?"],
        },
        sources=[{"source_id": 1, "label": "Risk: Recruitment delay risk"}],
    )

    assert len(analysis_state["chat_history"]) == 3
    assert analysis_state["chat_history"][-1]["role"] == "assistant"
    assert analysis_state["chat_history"][-1]["content"] == "Here are the top risks."
    assert analysis_state["chat_history"][-1].get("pending_response") is None
    assert analysis_state["chat_history"][-1]["show_sources"] is True
    assert analysis_state["chat_history"][-1]["sources"][0]["label"] == "Risk: Recruitment delay risk"


def test_sanitize_chat_history_removes_consecutive_duplicate_messages() -> None:
    sanitized = _sanitize_chat_history(
        [
            {"role": "user", "content": "What are the top risks?"},
            {"role": "user", "content": "What are the top risks?"},
            {"role": "assistant", "content": "The top risks are..."},
            {"role": "assistant", "content": "The top risks are..."},
        ]
    )
    assert len(sanitized) == 2
    assert sanitized[0]["role"] == "user"
    assert sanitized[0]["content"] == "What are the top risks?"
    assert sanitized[1]["role"] == "assistant"
    assert sanitized[1]["content"] == "The top risks are..."


def test_is_general_conversation_message_detects_small_talk() -> None:
    assert _is_general_conversation_message("Thanks!") is True
    assert _is_general_conversation_message("You're very welcome") is True
    assert _is_general_conversation_message("Which risks are high severity?") is False


def test_revision_session_lifecycle_helpers() -> None:
    analysis_state = {"revision_session": _empty_revision_session_state()}
    session = _ensure_revision_session_state(analysis_state)
    added = _add_revision_proposals_to_session(
        session,
        [
            {
                "summary": "Reduce visit burden",
                "requested_change": "Reduce in-person visits and allow telehealth follow-up visits where clinically acceptable.",
                "rationale": "Lower burden can reduce dropouts and improve operational feasibility.",
                "affected_sections": ["Schedule of Assessments", "Study Visits"],
                "estimated_edits": 4,
                "focus": "operational",
            }
        ],
    )
    assert added == 1
    assert len(session["pending_changes"]) == 1
    assert session["awaiting_confirmation_change_ids"]
    approved = _approve_pending_revision_changes(session, only_awaiting=True)
    assert approved == 1
    assert len(session["pending_changes"]) == 0
    assert len(session["approved_changes"]) == 1
    undo_message, undo_ok = _undo_last_revision_change(session)
    assert undo_ok is True
    assert "last approved revision change" in undo_message
    assert len(session["approved_changes"]) == 0
    assert _clear_revision_plan(session) is False


def test_revision_session_item_review_helpers_cover_accept_decline_and_edit() -> None:
    session = _sanitize_revision_session_state(
        {
            "pending_changes": [
                {
                    "id": "c1",
                    "summary": "Reduce visit burden",
                    "requested_change": "Reduce in-person visits.",
                    "rationale": "Improve retention.",
                    "affected_sections": ["Visits"],
                    "estimated_edits": 2,
                    "focus": "operational",
                },
                {
                    "id": "c2",
                    "summary": "Add safety labs",
                    "requested_change": "Add monthly safety labs.",
                    "rationale": "Improve monitoring.",
                    "affected_sections": ["Safety"],
                    "estimated_edits": 2,
                    "focus": "health",
                },
            ],
        }
    )

    edited = _edit_revision_change(
        session,
        "c1",
        summary="Reduce participant burden",
        requested_change="Reduce in-person visits and add telehealth follow-up.",
        rationale="Better retention and feasibility.",
    )
    assert edited is True
    assert session["pending_changes"][0]["status"] == "user_edited"
    assert session["pending_changes"][0]["requested_change"] == "Reduce in-person visits and add telehealth follow-up."

    approved = _approve_revision_change(session, "c1")
    assert approved == 1
    assert len(session["approved_changes"]) == 1
    assert session["approved_changes"][0]["requested_change"] == "Reduce in-person visits and add telehealth follow-up."

    declined = _decline_revision_change(session, "c2")
    assert declined == 1
    assert len(session["declined_changes"]) == 1
    assert session["declined_changes"][0]["id"] == "c2"


def test_discard_awaiting_revision_changes_only_removes_pending_batch() -> None:
    session = _sanitize_revision_session_state(
        {
            "pending_changes": [
                {
                    "id": "c1",
                    "summary": "Add monthly ECG",
                    "requested_change": "Add monthly ECG monitoring for high-risk participants.",
                    "rationale": "Improves ongoing safety surveillance.",
                    "affected_sections": ["Safety Monitoring"],
                    "estimated_edits": 2,
                    "focus": "health",
                },
                {
                    "id": "c2",
                    "summary": "Simplify eligibility",
                    "requested_change": "Simplify exclusion criteria language for consistency.",
                    "rationale": "Reduces screening confusion and recruitment friction.",
                    "affected_sections": ["Eligibility Criteria"],
                    "estimated_edits": 2,
                    "focus": "eligibility",
                },
            ],
            "awaiting_confirmation_change_ids": ["c1"],
        }
    )
    removed = _discard_awaiting_revision_changes(session)
    assert removed == 1
    assert len(session["pending_changes"]) == 1
    assert session["pending_changes"][0]["id"] == "c2"


def test_reset_ai_assistant_conversation_clears_revision_plan_state() -> None:
    analysis_state = {
        "chat_threads": [{"id": "t1", "messages": []}],
        "chat_history": [{"role": "user", "content": "Reduce burden"}],
        "chat_selected_thread_id": "t1",
        "revision_session": {
            "pending_changes": [
                {
                    "id": "r1",
                    "summary": "Reduce visit burden",
                    "requested_change": "Reduce in-person visits.",
                    "rationale": "Improve retention.",
                    "affected_sections": ["Visits"],
                    "estimated_edits": 2,
                    "focus": "operational",
                }
            ],
            "approved_changes": [],
            "generated_output": {"output_dir": "C:\\temp\\x"},
        },
    }
    _reset_ai_assistant_conversation(analysis_state)
    assert analysis_state["chat_threads"] == []
    assert analysis_state["chat_history"] == []
    assert analysis_state["chat_selected_thread_id"] == ""
    revision_session = analysis_state["revision_session"]
    assert revision_session["pending_changes"] == []
    assert revision_session["approved_changes"] == []
    assert revision_session["declined_changes"] == []
    assert revision_session["generated_output"] is None
