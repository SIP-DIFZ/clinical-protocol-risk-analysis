from __future__ import annotations

import html
import re
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from protocol_risk_analyzer.backend.editor_service import (
    _clear_editor_job,
    _read_editor_job,
    _start_editor_job,
    generate_editor_outputs_from_proposals,
)
from protocol_risk_analyzer.backend.revision_service import build_revision_change_risks
from protocol_risk_analyzer.models import AnalysisResult, RiskAssessment
from protocol_risk_analyzer.ui_helpers import (
    _build_editor_risk_entries,
    _clean_editor_progress_message,
    _ensure_revision_session_state,
    _sanitize_accepted_risk_ids,
)


def _mitigate_protocol_tab_guidance(revision_session: dict[str, Any]) -> str:
    approved_changes = revision_session.get("approved_changes", [])
    approved_count = len(approved_changes) if isinstance(approved_changes, list) else 0
    if approved_count <= 0:
        return (
            "No approved revision changes are available yet. Approve one or more AI Assistant edits first, "
            "then open the Mitigate Protocol tab to generate the mitigated protocol."
        )
    return (
        f"You have **{approved_count}** approved AI Assistant edit(s). "
        "Open the **Mitigate Protocol** tab to choose whether to include those edits, protocol risk mitigations, "
        "or both, then generate and download the mitigated protocol there."
    )


def render_editor_agent_builder(
    result: AnalysisResult,
    analysis_state: dict[str, Any],
    *,
    render_editor_job_monitor_fragment,
) -> None:
    st.subheader("Mitigate Protocol")
    st.caption(
        "Create a mitigated protocol draft using structured patch operations. "
        "You can include approved AI Assistant edits, non-accepted protocol risks, or both."
    )

    run_dir = Path(analysis_state["run_dir"])
    editor_job = _read_editor_job(analysis_state.get("editor_job_id"), run_dir=run_dir)
    is_running = editor_job is not None and editor_job.get("status") == "running"
    has_existing_mitigated_protocol = _has_mitigated_protocol_output(analysis_state.get("editor_state"))
    _render_editor_error_banner(analysis_state)
    revision_session = _ensure_revision_session_state(analysis_state)
    approved_changes = revision_session.get("approved_changes", [])
    if not isinstance(approved_changes, list):
        approved_changes = []
    pending_changes = revision_session.get("pending_changes", [])
    if not isinstance(pending_changes, list):
        pending_changes = []

    if approved_changes or pending_changes:
        with st.container(border=True):
            st.markdown("### AI Assistant Revision Plan")
            if approved_changes:
                st.caption(f"{len(approved_changes)} approved AI Assistant edit(s) are ready for mitigation generation.")
                approved_changes_expanded = bool(st.session_state.get("mitigate_protocol_revision_plan_expanded", False))
                visible_approved_changes = (
                    approved_changes
                    if approved_changes_expanded or len(approved_changes) <= 3
                    else approved_changes[:3]
                )
                for change in visible_approved_changes:
                    summary = str(change.get("summary", "")).strip()
                    sections = [
                        str(item).strip()
                        for item in change.get("affected_sections", [])
                        if str(item).strip()
                    ] if isinstance(change.get("affected_sections", []), list) else []
                    edit_count = int(change.get("estimated_edits", 0) or 0)
                    detail_bits: list[str] = []
                    if sections:
                        detail_bits.append(", ".join(sections[:3]))
                    if edit_count > 0:
                        detail_bits.append(f"{edit_count} estimated edit(s)")
                    st.markdown(f"- **{summary or 'Approved revision change'}**")
                    if detail_bits:
                        st.caption(" | ".join(detail_bits))
                if len(approved_changes) > 3:
                    toggle_col, _ = st.columns([1.2, 4.8], gap="small")
                    with toggle_col:
                        if approved_changes_expanded:
                            if st.button("↑ Show Less", key="mitigate_protocol_revision_plan_show_less", use_container_width=True):
                                st.session_state["mitigate_protocol_revision_plan_expanded"] = False
                                st.rerun()
                        else:
                            remaining_count = len(approved_changes) - len(visible_approved_changes)
                            if st.button(
                                f"↓ Show More ({remaining_count})",
                                key="mitigate_protocol_revision_plan_show_more",
                                use_container_width=True,
                            ):
                                st.session_state["mitigate_protocol_revision_plan_expanded"] = True
                                st.rerun()
            if pending_changes:
                st.caption(
                    f"{len(pending_changes)} pending AI Assistant revision item(s) are still awaiting approval in the AI Assistant."
                )

    entries = _build_editor_risk_entries(result)
    valid_ids = {entry["risk_id"] for entry in entries}
    accepted_risk_ids = _sanitize_accepted_risk_ids(analysis_state.get("accepted_risk_ids"), valid_ids)
    if accepted_risk_ids != analysis_state.get("accepted_risk_ids", []):
        analysis_state["accepted_risk_ids"] = accepted_risk_ids
        st.session_state["analysis_state"] = analysis_state
    accepted_risk_id_set = set(accepted_risk_ids)
    available_entries = [entry for entry in entries if entry["risk_id"] not in accepted_risk_id_set]
    accepted_count = len(entries) - len(available_entries)
    if accepted_count > 0:
        st.caption(
            f"{accepted_count} accepted risk(s) are excluded from editor patch generation. "
            "Manage acceptance in the Dashboard risk tables."
        )

    option_col_1, option_col_2 = st.columns(2, gap="small")
    with option_col_1:
        include_ai_assistant_edits = st.checkbox(
            "Include approved AI Assistant edits",
            value=bool(approved_changes),
            disabled=not approved_changes,
            key="editor_include_ai_assistant_edits",
        )
    with option_col_2:
        include_protocol_risks = st.checkbox(
            "Include protocol risk mitigations",
            value=bool(entries),
            disabled=not entries,
            key="editor_include_protocol_risks",
        )

    options = [entry["label"] for entry in available_entries]
    default_selection = options[: min(10, len(options))]
    selected_options: list[str] = []
    selected_risks: list[RiskAssessment] = []
    if include_protocol_risks:
        selection_col, summary_col = st.columns([2.2, 1])
        with selection_col:
            scope = st.radio(
                "Mitigation Scope",
                options=["Mitigate all risks", "Choose specific risks"],
                index=0,
                horizontal=True,
                key="editor_scope",
            )
            mitigate_all = scope.startswith("Mitigate all")
        with summary_col:
            st.empty()
        if mitigate_all:
            selected_options = options
        else:
            selected_options = st.multiselect(
                "Risks to mitigate",
                options=options,
                default=default_selection,
                key="editor_selected_risks",
                help="Choose the specific non-accepted risks you want the Editor Agent to address in the revised protocol.",
            )
        selected_titles = set(options if mitigate_all else selected_options)
        selected_risks = [
            entry["risk"]
            for entry in available_entries
            if entry["label"] in selected_titles
        ]
    elif options:
        st.caption("Protocol risk mitigations are currently excluded from this generation run.")

    assistant_edit_risks: list[RiskAssessment] = []
    if include_ai_assistant_edits and approved_changes:
        assistant_edit_risks = build_revision_change_risks(approved_changes)

    combined_selected_risks = selected_risks + assistant_edit_risks
    selected_count = len(selected_risks)
    assistant_edit_count = len(assistant_edit_risks)
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Selected Risks", selected_count)
    metric_col_2.metric("AI Assistant Edits", assistant_edit_count)
    metric_col_3.metric("Total Mitigation Inputs", len(combined_selected_risks))
    if not options and not approved_changes:
        st.info("No protocol risks or approved AI Assistant edits are currently available for mitigation.")
    elif not options and include_protocol_risks:
        st.info("All risks are currently accepted. Unaccept at least one risk to include protocol risks in mitigation.")
    elif approved_changes and not include_ai_assistant_edits:
        st.caption("Approved AI Assistant edits are available but not included in this generation run.")

    if st.button(
        "Generate Proposed Changes",
        type="primary",
        disabled=not combined_selected_risks or is_running,
    ):
        job_id = _start_editor_job(
            input_path=Path(analysis_state["input_path"]),
            run_dir=run_dir,
            selected_risks=combined_selected_risks,
            model=analysis_state["chosen_model"],
        )
        analysis_state["editor_job_id"] = job_id
        analysis_state["editor_state"] = None
        analysis_state["editor_review"] = {"proposals": [], "ready": False}
        analysis_state.pop("editor_error", None)
        st.session_state["analysis_state"] = analysis_state
        editor_job = _read_editor_job(job_id, run_dir=run_dir)
        is_running = True

    if analysis_state.get("editor_job_id"):
        render_editor_job_monitor_fragment()
        refreshed_state = st.session_state.get("analysis_state", analysis_state)
        if refreshed_state.get("editor_job_id"):
            return

    _render_protocol_change_review_ui(st.session_state.get("analysis_state", analysis_state))

    editor_state = st.session_state.get("analysis_state", analysis_state).get("editor_state")
    if not editor_state:
        return

    st.success("Mitigated protocol is ready.")
    st.caption(f"Editor model: {editor_state['chosen_model']}")
    if editor_state.get("selected_count") is not None:
        st.caption(f"Risks targeted in this run: {editor_state['selected_count']}")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            label="Download mitigated_protocol.md",
            data=editor_state["mitigated_md_text"].encode("utf-8"),
            file_name="mitigated_protocol.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="Download mitigated_protocol.pdf",
            data=editor_state["mitigated_pdf_bytes"],
            file_name="mitigated_protocol.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            label="Download mitigated_protocol.docx",
            data=editor_state["mitigated_docx_bytes"],
            file_name="mitigated_protocol.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    st.download_button(
        label="Download patch log",
        data=editor_state["patch_log_text"].encode("utf-8"),
        file_name="mitigated_protocol_patches.txt",
        mime="text/plain",
        use_container_width=True,
    )


def _render_editor_job_progress(
    editor_job: dict[str, Any],
    status_slot,
    detail_slot,
) -> None:
    status = str(editor_job.get("status", "running"))
    message = str(editor_job.get("message", "Running Editor Agent..."))
    cleaned_message = _clean_editor_progress_message(message)
    proposals = editor_job.get("proposals", [])
    ready_count = len(proposals) if isinstance(proposals, list) else 0

    if status in {"completed", "review_ready"}:
        status_slot.markdown(f"**{ready_count} proposals ready**")
        detail_slot.caption("All proposed changes are ready for review.")
        return

    if status == "failed":
        status_slot.empty()
        detail_slot.empty()
        return
    status_slot.markdown(f"**{ready_count} proposals ready**")
    detail_slot.caption(cleaned_message)


@st.fragment(run_every=1.0)
def _render_editor_job_monitor_fragment() -> None:
    analysis_state = st.session_state.get("analysis_state")
    if not analysis_state:
        return
    run_dir = Path(analysis_state["run_dir"])
    editor_job = _read_editor_job(analysis_state.get("editor_job_id"), run_dir=run_dir)
    if editor_job is None:
        return

    with st.container():
        status_slot = st.empty()
        detail_slot = st.empty()
        _render_editor_job_progress(editor_job, status_slot, detail_slot)
        incoming = editor_job.get("proposals", [])
        if isinstance(incoming, list):
            review_state = analysis_state.get("editor_review", {})
            if not isinstance(review_state, dict):
                review_state = {}
            existing = review_state.get("proposals", [])
            if not isinstance(existing, list):
                existing = []
            review_state["proposals"] = _merge_review_proposals(existing, incoming)
            review_state["ready"] = editor_job.get("status") == "review_ready"
            analysis_state["editor_review"] = review_state
            st.session_state["analysis_state"] = analysis_state
        _render_protocol_change_review_ui(analysis_state)
        if editor_job["status"] == "review_ready":
            analysis_state.pop("editor_job_id", None)
            st.session_state["analysis_state"] = analysis_state
            _clear_editor_job(editor_job["job_id"], run_dir=run_dir)
            st.rerun()
        elif editor_job["status"] == "failed":
            analysis_state["editor_error"] = {
                "message": str(editor_job.get("error", "Editor Agent failed.")),
                "traceback": str(editor_job.get("traceback", "")).strip(),
                "job_id": str(editor_job.get("job_id", "")).strip(),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            analysis_state.pop("editor_job_id", None)
            st.session_state["analysis_state"] = analysis_state
            _clear_editor_job(editor_job["job_id"], run_dir=run_dir)
            st.rerun()


def _merge_review_proposals(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_by_id: dict[str, dict[str, Any]] = {}
    for proposal in existing:
        proposal_id = str(proposal.get("proposal_id", "")).strip()
        if proposal_id:
            existing_by_id[proposal_id] = dict(proposal)

    merged: list[dict[str, Any]] = []
    for proposal in incoming:
        if not isinstance(proposal, dict):
            continue
        proposal_id = str(proposal.get("proposal_id", "")).strip()
        if not proposal_id:
            continue
        prior = existing_by_id.get(proposal_id, {})
        proposed_text = str(proposal.get("proposed_text", "")).strip()
        accepted_text = str(prior.get("accepted_text", "")).strip() or proposed_text
        status = str(prior.get("status", "pending")).strip().lower() or "pending"
        normalized_status = status if status in {"pending", "accepted", "declined", "user_edited"} else "pending"
        merged.append(
            {
                "proposal_id": proposal_id,
                "risk_title": str(proposal.get("risk_title", "")).strip(),
                "action": str(proposal.get("action", "replace")).strip() or "replace",
                "target_paragraph_id": str(proposal.get("target_paragraph_id", "")).strip(),
                "target_label": str(proposal.get("target_label", "")).strip(),
                "page_number": proposal.get("page_number"),
                "paragraph_number_on_page": proposal.get("paragraph_number_on_page"),
                "reason": str(proposal.get("reason", "")).strip(),
                "original_text": str(proposal.get("original_text", "")).strip(),
                "proposed_text": proposed_text,
                "accepted_text": accepted_text,
                "status": normalized_status,
            }
        )
    return merged


def _render_protocol_change_review_ui(analysis_state: dict[str, Any]) -> None:
    review_state = analysis_state.get("editor_review", {})
    if not isinstance(review_state, dict):
        return
    proposals = review_state.get("proposals", [])
    if not isinstance(proposals, list):
        return
    if not proposals:
        if bool(review_state.get("ready")):
            st.info("No proposed protocol edits were generated for the selected inputs.")
        return

    running = bool(analysis_state.get("editor_job_id"))
    has_existing_mitigated_protocol = _has_mitigated_protocol_output(analysis_state.get("editor_state"))
    visible_proposals = _visible_review_proposals(proposals)
    st.markdown("### Protocol Change Review")
    if running:
        st.caption("Proposed rewrites will appear here as they are generated.")
    elif has_existing_mitigated_protocol:
        st.caption("This protocol already has mitigated outputs, and generating again will replace those downloads with the newly accepted edits.")

    status_counts = {"pending": 0, "accepted": 0, "declined": 0, "user_edited": 0}
    for proposal in proposals:
        status = str(proposal.get("status", "pending")).strip().lower()
        if status in status_counts:
            status_counts[status] += 1
    metric_cols = st.columns(4)
    metric_cols[0].metric("Pending", status_counts["pending"])
    metric_cols[1].metric("Accepted", status_counts["accepted"])
    metric_cols[2].metric("Declined", status_counts["declined"])
    metric_cols[3].metric("User Edited", status_counts["user_edited"])

    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button(
            "Accept All",
            key="mitigate_accept_all",
            disabled=not visible_proposals,
        ):
            for proposal in visible_proposals:
                if str(proposal.get("status", "pending")).strip().lower() != "declined":
                    proposal["status"] = "accepted"
                    text = str(proposal.get("accepted_text", "")).strip()
                    if not text:
                        proposal["accepted_text"] = str(proposal.get("proposed_text", "")).strip()
            analysis_state["editor_review"] = review_state
            st.session_state["analysis_state"] = analysis_state
            st.rerun()
    with action_cols[1]:
        accepted_count = _count_generation_ready_proposals(proposals)
        generate_disabled = accepted_count == 0 or running
        if st.button("Generate Mitigated Protocol", key="mitigate_apply_accepted", type="primary", disabled=generate_disabled):
            try:
                with st.spinner("Generating mitigated protocol from accepted edits..."):
                    result = generate_editor_outputs_from_proposals(
                        input_path=Path(analysis_state["input_path"]),
                        run_dir=Path(analysis_state["run_dir"]),
                        proposals=proposals,
                        model=analysis_state["chosen_model"],
                    )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
            analysis_state["editor_state"] = result
            st.session_state["analysis_state"] = analysis_state
            st.rerun()

    if not visible_proposals:
        st.info("No visible proposed edits remain in review. Declined edits are hidden.")
        return

    page_key = "mitigate_protocol_proposal_page"
    current_page = int(st.session_state.get(page_key, 1) or 1)
    paginated_proposals, current_page, total_pages = _paginate_editor_proposals(visible_proposals, current_page)
    if st.session_state.get(page_key) != current_page:
        st.session_state[page_key] = current_page

    for idx, proposal in paginated_proposals:
        proposal_id = str(proposal.get("proposal_id", "")).strip() or f"proposal-{idx}"
        target = (
            str(proposal.get("target_label", "")).strip()
            or str(proposal.get("target_paragraph_id", "")).strip()
            or "Unknown paragraph"
        )
        proposal_label = _format_proposal_label(idx, target)
        risk_title = str(proposal.get("risk_title", "")).strip() or "Mitigation update"
        reason = str(proposal.get("reason", "")).strip()
        status = str(proposal.get("status", "pending")).strip().lower() or "pending"
        status_label = status.replace("_", " ").title()
        accepted_text = str(proposal.get("accepted_text", "")).strip() or str(proposal.get("proposed_text", "")).strip()
        original_text = str(proposal.get("original_text", "")).strip()
        proposed_text = str(proposal.get("proposed_text", "")).strip()
        edit_mode_key = f"edit_mode_{proposal_id}"
        edit_text_key = f"edit_text_{proposal_id}"
        is_edit_mode = bool(st.session_state.get(edit_mode_key, False))
        if is_edit_mode and edit_text_key not in st.session_state:
            st.session_state[edit_text_key] = accepted_text

        with st.container(border=True):
            label_col, _ = st.columns([3, 1], gap="small")
            with label_col:
                st.markdown(f"#### {proposal_label}")
            st.caption(f"Risk: {risk_title} · Status: {status_label}")
            if reason:
                st.caption(reason)

            left_col, right_col = st.columns(2, gap="small")
            left_html, right_html = _build_diff_html(original_text, accepted_text)
            with left_col:
                st.markdown("**Original Paragraph**")
                st.markdown(left_html, unsafe_allow_html=True)
            with right_col:
                st.markdown("**Proposed Paragraph**")
                if is_edit_mode:
                    st.text_area(
                        "Edit proposed paragraph",
                        key=edit_text_key,
                        height=160,
                        label_visibility="collapsed",
                    )
                else:
                    st.markdown(right_html, unsafe_allow_html=True)

            button_cols = st.columns([1, 1, 1], gap="small")
            with button_cols[0]:
                if st.button("Accept", key=f"accept_{proposal_id}", use_container_width=True):
                    proposal["status"] = "accepted"
                    proposal["accepted_text"] = accepted_text
                    analysis_state["editor_review"] = review_state
                    st.session_state["analysis_state"] = analysis_state
                    st.rerun()
            with button_cols[1]:
                if st.button("Decline", key=f"decline_{proposal_id}", use_container_width=True):
                    proposal["status"] = "declined"
                    analysis_state["editor_review"] = review_state
                    st.session_state["analysis_state"] = analysis_state
                    st.rerun()
            with button_cols[2]:
                if st.button("Edit", key=f"edit_{proposal_id}", use_container_width=True):
                    st.session_state[edit_mode_key] = True
                    st.session_state.setdefault(edit_text_key, accepted_text)
                    st.rerun()

            if is_edit_mode:
                edit_action_cols = st.columns([1, 1], gap="small")
                with edit_action_cols[0]:
                    if st.button("Save Edit", key=f"save_edit_{proposal_id}", use_container_width=True):
                        cleaned = str(st.session_state.get(edit_text_key, "")).strip()
                        if cleaned:
                            proposal["accepted_text"] = cleaned
                            proposal["status"] = "user_edited"
                            analysis_state["editor_review"] = review_state
                            st.session_state["analysis_state"] = analysis_state
                        st.session_state[edit_mode_key] = False
                        st.rerun()
                with edit_action_cols[1]:
                    if st.button("Cancel Edit", key=f"cancel_edit_{proposal_id}", use_container_width=True):
                        st.session_state[edit_mode_key] = False
                        st.rerun()

    _render_proposal_pagination_controls(page_key, current_page, total_pages)


def _format_proposal_label(index: int, target_label: str) -> str:
    cleaned_target = str(target_label or "").strip()
    if not cleaned_target:
        return f"Edit {index}"
    return f"Edit {index}: {cleaned_target}"


def _visible_review_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        proposal
        for proposal in proposals
        if str(proposal.get("status", "pending")).strip().lower() != "declined"
    ]


def _count_generation_ready_proposals(proposals: list[dict[str, Any]]) -> int:
    return sum(
        1
        for proposal in proposals
        if str(proposal.get("status", "")).strip().lower() in {"accepted", "user_edited"}
    )


def _build_diff_html(original_text: str, revised_text: str) -> tuple[str, str]:
    left_tokens = _tokenize_diff_text(original_text)
    right_tokens = _tokenize_diff_text(revised_text)
    matcher = SequenceMatcher(a=left_tokens, b=right_tokens)
    left_parts: list[str] = []
    right_parts: list[str] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        left_segment = "".join(left_tokens[i1:i2])
        right_segment = "".join(right_tokens[j1:j2])
        if op == "equal":
            escaped = html.escape(left_segment)
            left_parts.append(escaped)
            right_parts.append(escaped)
        elif op == "delete":
            left_parts.append(f"<span style='color:#B00020;text-decoration:line-through'>{html.escape(left_segment)}</span>")
        elif op == "insert":
            right_parts.append(f"<span style='color:#0A7A33'>{html.escape(right_segment)}</span>")
        elif op == "replace":
            left_parts.append(f"<span style='color:#B00020;text-decoration:line-through'>{html.escape(left_segment)}</span>")
            right_parts.append(f"<span style='color:#0A7A33'>{html.escape(right_segment)}</span>")
    left_html = "".join(left_parts).strip() or "<span style='color:#6B7280'>No original text available.</span>"
    right_html = "".join(right_parts).strip() or "<span style='color:#6B7280'>No proposed text available.</span>"
    return left_html, right_html


def _tokenize_diff_text(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\s+|[^\s]+", text)


def _paginate_editor_proposals(
    proposals: list[dict[str, Any]],
    current_page: int,
    page_size: int = 5,
) -> tuple[list[tuple[int, dict[str, Any]]], int, int]:
    total_pages = max(1, (len(proposals) + page_size - 1) // page_size)
    current_page = max(1, min(current_page, total_pages))
    start_index = (current_page - 1) * page_size
    page_items = list(enumerate(proposals[start_index : start_index + page_size], start=start_index + 1))
    return page_items, current_page, total_pages


def _render_proposal_pagination_controls(page_key: str, current_page: int, total_pages: int) -> None:
    if total_pages <= 1:
        return

    control_cols = st.columns([1] + [0.7] * total_pages + [1], gap="small")
    with control_cols[0]:
        if st.button("Previous", key=f"{page_key}_previous", disabled=current_page <= 1, use_container_width=True):
            st.session_state[page_key] = current_page - 1
            st.rerun()

    for page_number in range(1, total_pages + 1):
        with control_cols[page_number]:
            if page_number == current_page:
                st.button(
                    str(page_number),
                    key=f"{page_key}_page_{page_number}",
                    type="primary",
                    disabled=True,
                    use_container_width=True,
                )
            else:
                if st.button(str(page_number), key=f"{page_key}_page_{page_number}", use_container_width=True):
                    st.session_state[page_key] = page_number
                    st.rerun()

    with control_cols[-1]:
        if st.button("Next", key=f"{page_key}_next", disabled=current_page >= total_pages, use_container_width=True):
            st.session_state[page_key] = current_page + 1
            st.rerun()


def _render_editor_error_banner(analysis_state: dict[str, Any]) -> None:
    editor_error = analysis_state.get("editor_error")
    if not isinstance(editor_error, dict):
        return
    message = str(editor_error.get("message", "")).strip()
    traceback_text = str(editor_error.get("traceback", "")).strip()
    job_id = str(editor_error.get("job_id", "")).strip()
    if not message:
        return
    with st.container(border=True):
        st.error(f"Editor Agent failed: {message}")
        if job_id:
            st.caption(f"Job ID: {job_id}")
        if traceback_text:
            with st.expander("Show error details", expanded=False):
                st.code(traceback_text, language="text")
        if st.button("Dismiss error", key="dismiss_editor_error", use_container_width=True):
            analysis_state.pop("editor_error", None)
            st.session_state["analysis_state"] = analysis_state
            st.rerun()


def _has_mitigated_protocol_output(editor_state: object) -> bool:
    if not isinstance(editor_state, dict):
        return False
    required_keys = {
        "mitigated_md_text",
        "mitigated_pdf_bytes",
        "mitigated_docx_bytes",
        "patch_log_text",
    }
    return required_keys.issubset(editor_state.keys())


__all__ = ["render_editor_agent_builder", "_render_editor_job_monitor_fragment", "_render_editor_job_progress"]
