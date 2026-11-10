from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import html

import streamlit as st
import streamlit.components.v1 as components

from protocol_risk_analyzer.backend.chat_service import run_protocol_chat
from protocol_risk_analyzer.backend.revision_service import (
    run_protocol_proposal_edit,
    run_protocol_revision_planner,
)
from protocol_risk_analyzer.chunking import chunk_pages
from protocol_risk_analyzer.io import read_document
from protocol_risk_analyzer.models import AnalysisResult
from protocol_risk_analyzer.ui_helpers import (
    _add_revision_proposals_to_session,
    _ai_assistant_popup_input_key,
    _append_chat_turn,
    _apply_revision_item_edit_request,
    _approve_pending_revision_changes,
    _approve_revision_change,
    _clear_revision_plan,
    _decline_revision_change,
    _edit_revision_change,
    _discard_awaiting_revision_changes,
    _empty_revision_session_state,
    _ensure_flat_chat_history_state,
    _ensure_revision_session_state,
    _extract_revision_target_index,
    _initial_ai_assistant_message,
    _is_affirmative_message,
    _is_general_conversation_message,
    _is_negative_message,
    _looks_like_apply_plan_request,
    _looks_like_clear_plan_request,
    _looks_like_generate_revision_request,
    _looks_like_undo_plan_request,
    _pending_ai_assistant_message,
    _pending_revision_prompt,
    _safe_key,
    _truncate_text_for_label,
    _undo_last_revision_change,
)

DEFAULT_CHUNK_CHARS = 14000
DEFAULT_OVERLAP_CHARS = 1200


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


def render_ai_assistant_popup(
    result: AnalysisResult,
    analysis_state: dict[str, Any],
    saved_root: Path,
    sync_saved_analysis_snapshot: Callable[[dict[str, Any], Path], None],
) -> None:
    input_key = _ai_assistant_popup_input_key(analysis_state)
    pending_clear_key = str(st.session_state.get("ai_assistant_popup_clear_input_key", "")).strip()
    if pending_clear_key and pending_clear_key == input_key:
        st.session_state[input_key] = ""
        st.session_state["ai_assistant_popup_clear_input_key"] = ""
    empty_submit_key = str(st.session_state.get("ai_assistant_popup_empty_submit_key", "")).strip()
    show_empty_submit_warning = bool(empty_submit_key and empty_submit_key == input_key)
    if show_empty_submit_warning:
        st.session_state["ai_assistant_popup_empty_submit_key"] = ""
    chat_history = _ensure_flat_chat_history_state(analysis_state)
    minimized = bool(st.session_state.get("ai_assistant_popup_minimized", False))
    expanded = bool(st.session_state.get("ai_assistant_popup_expanded", False))

    if minimized:
        with st.container(key="ai_assistant_popup_launcher"):
            if st.button("AI Assistant", key="ai_assistant_popup_open_btn", type="primary", use_container_width=True):
                st.session_state["ai_assistant_popup_minimized"] = False
                st.rerun()
        return

    shell_key = "ai_assistant_popup_shell_expanded" if expanded else "ai_assistant_popup_shell"
    with st.container(key=shell_key):
        with st.container(key="ai_assistant_popup_body"):
            with st.container(key="ai_assistant_popup_header"):
                protocol_name = (
                    str(analysis_state.get("saved_analysis_title", "")).strip()
                    or str(result.metadata.document_name).strip()
                    or "Protocol"
                )
                protocol_name_max = 92 if expanded else 42
                protocol_label = _truncate_text_for_label(protocol_name, protocol_name_max)

                header_col, actions_col = st.columns([2.8, 2.2], gap="small")
                with header_col:
                    st.markdown("<div class='ai-assistant-popup-title'>AI Assistant</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='ai-assistant-popup-protocol'>{html.escape(protocol_label)}</div>",
                        unsafe_allow_html=True,
                    )
                with actions_col:
                    expand_col, min_col, clear_col = st.columns([1.0, 1.0, 1.5], gap="small")
                    with expand_col:
                        expand_label = "⤡" if expanded else "⤢"
                        if st.button(
                            expand_label,
                            key="ai_assistant_popup_expand_btn",
                            use_container_width=True,
                            help="Resize",
                        ):
                            st.session_state["ai_assistant_popup_expanded"] = not expanded
                            st.rerun()
                    with min_col:
                        if st.button("—", key="ai_assistant_popup_min_btn", use_container_width=True, help="Minimize"):
                            st.session_state["ai_assistant_popup_minimized"] = True
                            st.rerun()
                    with clear_col:
                        if st.button("Clear", key="ai_assistant_popup_clear_btn", use_container_width=True):
                            _reset_ai_assistant_conversation(analysis_state)
                            sync_saved_analysis_snapshot(analysis_state, saved_root)
                            st.session_state["ai_assistant_popup_clear_input_key"] = input_key
                            st.rerun()
            revision_session = _ensure_revision_session_state(analysis_state)
            pending_count = len(revision_session.get("pending_changes", []))
            approved_count = len(revision_session.get("approved_changes", []))
            declined_count = len(revision_session.get("declined_changes", []))
            estimated_edit_count = int(revision_session.get("estimated_edit_count", 0) or 0)
            show_revision_plan = pending_count > 0 or approved_count > 0 or declined_count > 0
            if show_revision_plan:
                with st.container(key="ai_assistant_popup_revision_plan"):
                    st.markdown("<div class='ai-revision-plan-title'>Revision Review</div>", unsafe_allow_html=True)
                    st.markdown(
                        (
                            "<div class='ai-revision-plan-meta'>"
                            f"<span>Pending: <strong>{pending_count}</strong></span>"
                            f"<span>Approved: <strong>{approved_count}</strong></span>"
                            f"<span>Declined: <strong>{declined_count}</strong></span>"
                            f"<span>Estimated edits: <strong>{estimated_edit_count}</strong></span>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    action_col_1, action_col_2, action_col_3 = st.columns(3, gap="small")
                    with action_col_1:
                        apply_clicked = st.button(
                            "Approve All Pending",
                            key="ai_revision_apply_changes_btn",
                            use_container_width=True,
                            disabled=pending_count == 0,
                        )
                    with action_col_2:
                        undo_clicked = st.button(
                            "Undo Last Change",
                            key="ai_revision_undo_change_btn",
                            use_container_width=True,
                            disabled=(pending_count + approved_count + declined_count) == 0,
                        )
                    with action_col_3:
                        clear_clicked = st.button(
                            "Clear Revision Plan",
                            key="ai_revision_clear_plan_btn",
                            use_container_width=True,
                            disabled=(pending_count + approved_count + declined_count) == 0,
                        )
                    if apply_clicked:
                        approved = _approve_pending_revision_changes(revision_session)
                        if approved > 0:
                            _append_chat_turn(
                                analysis_state=analysis_state,
                                user_prompt="Apply changes",
                                assistant_payload={
                                    "answer_markdown": f"Added **{approved}** change(s) to the approved revision plan.",
                                    "can_answer": True,
                                    "cited_source_ids": [],
                                    "follow_up_questions": [],
                                },
                                sources=[],
                            )
                        else:
                            _append_chat_turn(
                                analysis_state=analysis_state,
                                user_prompt="Apply changes",
                                assistant_payload={
                                    "answer_markdown": "There are no pending revision changes to apply.",
                                    "can_answer": True,
                                    "cited_source_ids": [],
                                    "follow_up_questions": [],
                                },
                                sources=[],
                            )
                        sync_saved_analysis_snapshot(analysis_state, saved_root)
                        st.rerun()
                    if undo_clicked:
                        undo_message, _ = _undo_last_revision_change(revision_session)
                        _append_chat_turn(
                            analysis_state=analysis_state,
                            user_prompt="Undo last change",
                            assistant_payload={
                                "answer_markdown": undo_message,
                                "can_answer": True,
                                "cited_source_ids": [],
                                "follow_up_questions": [],
                            },
                            sources=[],
                        )
                        sync_saved_analysis_snapshot(analysis_state, saved_root)
                        st.rerun()
                    if clear_clicked:
                        _clear_revision_plan(revision_session)
                        _append_chat_turn(
                            analysis_state=analysis_state,
                            user_prompt="Clear revision plan",
                            assistant_payload={
                                "answer_markdown": "Revision plan cleared.",
                                "can_answer": True,
                                "cited_source_ids": [],
                                "follow_up_questions": [],
                            },
                            sources=[],
                        )
                        sync_saved_analysis_snapshot(analysis_state, saved_root)
                        st.rerun()
                    if approved_count > 0:
                        st.markdown(
                            (
                                "<div class='ai-revision-plan-note'>"
                                "Approved AI Assistant edits are available in the Mitigate Protocol tab."
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                    _render_revision_review_items(
                        analysis_state=analysis_state,
                        revision_session=revision_session,
                        saved_root=saved_root,
                        sync_saved_analysis_snapshot=sync_saved_analysis_snapshot,
                    )
            with st.container(key="ai_assistant_popup_messages"):
                if chat_history:
                    for index, item in enumerate(chat_history):
                        role = str(item.get("role", "")).strip()
                        content = str(item.get("content", "")).strip()
                        if not content:
                            continue
                        turn_id = _chat_message_turn_id(chat_history, index)
                        st.markdown(f'<div id="{turn_id}"></div>', unsafe_allow_html=True)
                        with st.chat_message("user" if role == "user" else "assistant"):
                            role_label = "You" if role == "user" else "AI Assistant"
                            st.markdown(f"<div class='ai-chat-role'>{role_label}</div>", unsafe_allow_html=True)
                            if role == "assistant" and bool(item.get("pending_response", False)):
                                _render_ai_assistant_pending_indicator()
                            else:
                                st.markdown(content)
                            if (
                                role == "assistant"
                                and not bool(item.get("pending_response", False))
                                and bool(item.get("can_answer", True))
                                and bool(item.get("show_sources", True))
                            ):
                                _render_ai_assistant_sources(item)
                st.markdown('<div id="ai-assistant-chat-end"></div>', unsafe_allow_html=True)

            with st.container(key="ai_assistant_popup_form_shell"):
                with st.form("ai_assistant_popup_form", clear_on_submit=False):
                    input_col, send_col = st.columns([5.2, 0.8], gap="small")
                    with input_col:
                        st.text_area(
                            "Ask the assistant",
                            key=input_key,
                            placeholder="Ask a follow-up question about this analysis...",
                            height=96,
                            label_visibility="collapsed",
                        )
                    with send_col:
                        st.form_submit_button(
                            "➤",
                            type="primary",
                            use_container_width=False,
                            on_click=_queue_ai_assistant_popup_submit,
                            kwargs={
                                "analysis_state": analysis_state,
                                "saved_root": saved_root,
                                "input_key": input_key,
                                "sync_saved_analysis_snapshot": sync_saved_analysis_snapshot,
                            },
                        )

    if st.session_state.pop("ai_assistant_popup_scroll_pending", False):
        _scroll_ai_assistant_to_bottom()
    if show_empty_submit_warning:
        st.warning("Please enter a question.")
        return

    pending_prompts = st.session_state.get("ai_assistant_popup_pending_prompts", {})
    if not isinstance(pending_prompts, dict):
        pending_prompts = {}
    pending_prompt = str(pending_prompts.pop(input_key, "")).strip()
    st.session_state["ai_assistant_popup_pending_prompts"] = pending_prompts
    if pending_prompt:
        _submit_ai_assistant_prompt(
            analysis_state=analysis_state,
            saved_root=saved_root,
            user_prompt=pending_prompt,
            sync_saved_analysis_snapshot=sync_saved_analysis_snapshot,
        )
        st.session_state["ai_assistant_popup_scroll_pending"] = True
        st.rerun()


def _queue_ai_assistant_popup_submit(
    *,
    analysis_state: dict[str, Any],
    saved_root: Path,
    input_key: str,
    sync_saved_analysis_snapshot: Callable[[dict[str, Any], Path], None],
) -> None:
    cleaned_prompt = str(st.session_state.get(input_key, "")).strip()
    if not cleaned_prompt:
        st.session_state["ai_assistant_popup_empty_submit_key"] = input_key
        return
    messages = _ensure_flat_chat_history_state(analysis_state)
    if not (
        messages
        and str(messages[-1].get("role", "")).strip() == "user"
        and str(messages[-1].get("content", "")).strip() == cleaned_prompt
    ):
        messages.append({"role": "user", "content": cleaned_prompt})
    if not (
        messages
        and str(messages[-1].get("role", "")).strip() == "assistant"
        and bool(messages[-1].get("pending_response", False))
    ):
        messages.append(_pending_ai_assistant_message())
    analysis_state["chat_history"] = messages
    analysis_state["chat_threads"] = []
    analysis_state["chat_selected_thread_id"] = ""
    st.session_state["analysis_state"] = analysis_state
    pending_prompts = st.session_state.get("ai_assistant_popup_pending_prompts", {})
    if not isinstance(pending_prompts, dict):
        pending_prompts = {}
    pending_prompts[input_key] = cleaned_prompt
    st.session_state["ai_assistant_popup_pending_prompts"] = pending_prompts
    st.session_state["ai_assistant_popup_clear_input_key"] = input_key
    st.session_state["ai_assistant_popup_scroll_pending"] = True
    sync_saved_analysis_snapshot(analysis_state, saved_root)


def _render_ai_assistant_sources(message: dict[str, Any]) -> None:
    sources = message.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return
    with st.expander("Evidence and citations", expanded=False):
        for source in sources:
            if not isinstance(source, dict):
                continue
            label = str(source.get("label", "")).strip()
            page_start = source.get("page_start")
            page_end = source.get("page_end")
            risk_title = str(source.get("risk_title", "")).strip()
            excerpt = str(source.get("excerpt", "")).strip()
            why_relevant = str(source.get("why_relevant", "")).strip()
            st.markdown(f"**{label}**")
            meta_bits = []
            if risk_title:
                meta_bits.append(f"Related risk: {risk_title}")
            if isinstance(page_start, int):
                if isinstance(page_end, int) and page_end != page_start:
                    meta_bits.append(f"Pages {page_start}-{page_end}")
                else:
                    meta_bits.append(f"Page {page_start}")
            if meta_bits:
                st.caption(" | ".join(meta_bits))
            if excerpt:
                st.markdown(excerpt)
            if why_relevant:
                st.markdown(why_relevant)


def _render_ai_assistant_pending_indicator() -> None:
    st.markdown(
        """
        <div class="ai-assistant-typing-indicator" aria-label="AI Assistant is generating a response">
            <span class="ai-assistant-typing-dot"></span>
            <span class="ai-assistant-typing-dot"></span>
            <span class="ai-assistant-typing-dot"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chat_message_turn_id(chat_history: list[dict[str, Any]], message_index: int) -> str:
    turn_counter = 0
    for idx, item in enumerate(chat_history):
        role = str(item.get("role", "")).strip()
        if idx == message_index:
            if role == "assistant":
                return f"turn-{turn_counter}"
            if role == "user":
                return f"turn-{turn_counter}"
        if role == "assistant":
            turn_counter += 1
    return f"turn-{turn_counter}"


def _scroll_ai_assistant_to_bottom() -> None:
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const scrollPopupToBottom = () => {
          const target = doc.getElementById("ai-assistant-chat-end");
          if (!target) {
            return;
          }
          let scroller = target.closest('div[class*="st-key-ai_assistant_popup_messages"]');
          if (!scroller) {
            let node = target.parentElement;
            while (node) {
              if (node.scrollHeight > node.clientHeight + 8) {
                scroller = node;
                break;
              }
              node = node.parentElement;
            }
          }
          if (scroller) {
            scroller.scrollTop = scroller.scrollHeight + 9999;
          } else {
            target.scrollIntoView({behavior: "smooth", block: "end"});
          }
        };
        scrollPopupToBottom();
        setTimeout(scrollPopupToBottom, 60);
        setTimeout(scrollPopupToBottom, 180);
        </script>
        """,
        height=0,
    )


def _scroll_ai_assistant_to_target(target_id: str) -> None:
    components.html(
        f"""
        <script>
        const target = window.parent.document.getElementById("{target_id}");
        if (target) {{
          target.scrollIntoView({{behavior: "smooth", block: "start"}});
        }}
        </script>
        """,
        height=0,
    )


def _get_ai_assistant_pages(analysis_state: dict[str, Any]) -> list[Any]:
    current_path = str(analysis_state.get("input_path", "")).strip()
    if analysis_state.get("_ai_assistant_pages_cache_path") != current_path:
        pages = read_document(Path(current_path))
        analysis_state["_ai_assistant_pages_cache_path"] = current_path
        analysis_state["_ai_assistant_pages_cache"] = pages
        analysis_state.pop("_ai_assistant_chunks_cache", None)
    pages = analysis_state.get("_ai_assistant_pages_cache")
    if not isinstance(pages, list):
        pages = read_document(Path(current_path))
        analysis_state["_ai_assistant_pages_cache"] = pages
    return pages


def _get_ai_assistant_chunks(analysis_state: dict[str, Any], pages: list[Any]) -> list[Any]:
    current_path = str(analysis_state.get("input_path", "")).strip()
    if analysis_state.get("_ai_assistant_chunks_cache_path") != current_path:
        chunks = chunk_pages(pages, max_chars=DEFAULT_CHUNK_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS)
        analysis_state["_ai_assistant_chunks_cache_path"] = current_path
        analysis_state["_ai_assistant_chunks_cache"] = chunks
    chunks = analysis_state.get("_ai_assistant_chunks_cache")
    if not isinstance(chunks, list):
        chunks = chunk_pages(pages, max_chars=DEFAULT_CHUNK_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS)
        analysis_state["_ai_assistant_chunks_cache"] = chunks
    return chunks


def _handle_editor_proposal_edit_reference(
    analysis_state: dict[str, Any],
    user_prompt: str,
) -> tuple[bool, str]:
    prompt = " ".join(str(user_prompt or "").strip().split())
    if not prompt:
        return False, ""
    target_index = _extract_revision_target_index(prompt)
    if target_index is None:
        return False, ""
    editor_review = analysis_state.get("editor_review") or {}
    proposals = editor_review.get("proposals") or []
    if not proposals:
        return False, ""
    if target_index > len(proposals):
        return (
            True,
            f"I couldn't find Edit #{target_index} in the Protocol Change Review. "
            f"There {'is' if len(proposals) == 1 else 'are'} currently "
            f"{len(proposals)} proposal{'s' if len(proposals) != 1 else ''} listed.",
        )
    proposal = proposals[target_index - 1]
    target_label = str(proposal.get("target_label") or f"Paragraph {target_index}").strip()
    status = str(proposal.get("status") or "pending").strip()
    original_text = str(proposal.get("original_text") or "").strip()
    current_text = str(proposal.get("accepted_text") or proposal.get("proposed_text") or "").strip()
    current_reason = str(proposal.get("reason") or "").strip()
    risk_title = str(proposal.get("risk_title") or "Mitigation update").strip()
    normalized_prompt = prompt.lower()
    change_keywords = (
        "make it ",
        "make edit ",
        "change it",
        "change edit",
        "update it",
        "update edit",
        "revise it",
        "revise edit",
        "rewrite it",
        "rewrite edit",
        "more detailed",
        "more detail",
        "more specific",
        "clearer",
        "clarify",
        "simplify",
        "shorten",
        "expand",
        "reduce",
        "remove",
        "add ",
    )
    info_keywords = (
        "tell me more",
        "what is",
        "what does",
        "why is",
        "why was",
        "explain",
        "details",
        "describe",
        "about edit",
    )
    if any(keyword in normalized_prompt for keyword in change_keywords):
        if not current_text:
            return True, f"I found Edit #{target_index}, but it does not have a current proposed paragraph to revise yet."
        revised_text, revised_reason = run_protocol_proposal_edit(
            target_label=target_label,
            risk_title=risk_title,
            original_text=original_text,
            current_text=current_text,
            current_reason=current_reason,
            user_request=prompt,
            model=str(analysis_state.get("chosen_model", "")).strip() or None,
        )
        proposal["accepted_text"] = revised_text.strip()
        if revised_reason.strip():
            proposal["reason"] = revised_reason.strip()
        proposal["status"] = "user_edited"
        analysis_state["editor_review"] = editor_review
        return (
            True,
            f"I updated **Edit #{target_index}: {target_label}** based on your request.\n\n"
            f"**Current proposed paragraph**\n\n{proposal['accepted_text']}",
        )
    if any(keyword in normalized_prompt for keyword in info_keywords):
        original_preview = _truncate_text_for_label(original_text, 280) if original_text else "Not available."
        current_preview = _truncate_text_for_label(current_text, 320) if current_text else "Not available."
        reason_line = current_reason if current_reason else "No rationale was saved for this edit."
        return (
            True,
            f"**Edit #{target_index}: {target_label}**\n\n"
            f"- **Risk addressed:** {risk_title}\n"
            f"- **Status:** {status}\n"
            f"- **Why this edit exists:** {reason_line}\n"
            f"- **Original paragraph:** {original_preview}\n"
            f"- **Current proposed paragraph:** {current_preview}",
        )
    return (
        True,
        f"I found **Edit #{target_index}: {target_label}** (current status: {status}). "
        "If you want, ask me to explain it or tell me how to change the proposed paragraph.",
    )


def _submit_ai_assistant_prompt(
    *,
    analysis_state: dict[str, Any],
    saved_root: Path,
    user_prompt: str,
    sync_saved_analysis_snapshot: Callable[[dict[str, Any], Path], None],
) -> str:
    history = list(_ensure_flat_chat_history_state(analysis_state))
    if history and str(history[-1].get("role", "")).strip() == "assistant" and bool(history[-1].get("pending_response", False)):
        history = history[:-1]
    if not (
        history
        and str(history[-1].get("role", "")).strip() == "user"
        and str(history[-1].get("content", "")).strip() == user_prompt.strip()
    ):
        history.append({"role": "user", "content": user_prompt})
    revision_session = _ensure_revision_session_state(analysis_state)
    normalized_prompt = " ".join(user_prompt.strip().lower().split())
    assistant_payload: dict[str, Any]
    assistant_sources: list[dict[str, Any]] = []
    if _is_affirmative_message(normalized_prompt) and revision_session.get("awaiting_confirmation_change_ids"):
        approved = _approve_pending_revision_changes(revision_session, only_awaiting=True)
        assistant_payload = {
            "answer_markdown": (
                f"Great — I added **{approved}** change(s) to the approved revision plan."
                if approved > 0
                else "There were no pending revision items to approve."
            ),
            "can_answer": True,
            "cited_source_ids": [],
            "follow_up_questions": [],
        }
    elif _is_negative_message(normalized_prompt) and revision_session.get("awaiting_confirmation_change_ids"):
        removed = _discard_awaiting_revision_changes(revision_session)
        assistant_payload = {
            "answer_markdown": (
                f"Understood — I discarded **{removed}** pending revision change(s)."
                if removed > 0
                else "There were no pending revision items to discard."
            ),
            "can_answer": True,
            "cited_source_ids": [],
            "follow_up_questions": [],
        }
    elif _looks_like_apply_plan_request(normalized_prompt):
        approved = _approve_pending_revision_changes(revision_session)
        assistant_payload = {
            "answer_markdown": (
                f"Added **{approved}** pending change(s) to the approved revision plan."
                if approved > 0
                else "There are no pending revision changes to apply."
            ),
            "can_answer": True,
            "cited_source_ids": [],
            "follow_up_questions": [],
        }
    elif _looks_like_undo_plan_request(normalized_prompt):
        undo_message, _ = _undo_last_revision_change(revision_session)
        assistant_payload = {
            "answer_markdown": undo_message,
            "can_answer": True,
            "cited_source_ids": [],
            "follow_up_questions": [],
        }
    elif _looks_like_clear_plan_request(normalized_prompt):
        _clear_revision_plan(revision_session)
        assistant_payload = {
            "answer_markdown": "Revision plan cleared.",
            "can_answer": True,
            "cited_source_ids": [],
            "follow_up_questions": [],
        }
    elif _looks_like_generate_revision_request(normalized_prompt):
        assistant_payload = {
            "answer_markdown": _mitigate_protocol_tab_guidance(revision_session),
            "can_answer": True,
            "cited_source_ids": [],
            "follow_up_questions": [],
        }
    else:
        proposal_edit_handled, proposal_edit_message = _handle_editor_proposal_edit_reference(analysis_state, user_prompt)
        if proposal_edit_handled:
            assistant_payload = {
                "answer_markdown": proposal_edit_message,
                "can_answer": True,
                "cited_source_ids": [],
                "follow_up_questions": [],
            }
            assistant_sources = []
        else:
            targeted_edit_handled, targeted_edit_message = _apply_revision_item_edit_request(revision_session, user_prompt)
            if targeted_edit_handled:
                assistant_payload = {
                    "answer_markdown": targeted_edit_message,
                    "can_answer": True,
                    "cited_source_ids": [],
                    "follow_up_questions": [],
                }
                assistant_sources = []
            else:
                planner_result = run_protocol_revision_planner(
                    analysis_json_text=analysis_state["json_text"],
                    question=user_prompt,
                    history=[{"role": item.get("role", ""), "content": item.get("content", "")} for item in history],
                    revision_session=revision_session,
                    model=str(analysis_state.get("chosen_model", "")).strip() or None,
                )
                planner_payload = planner_result.payload
                if planner_payload.intent == "revision_request":
                    proposals = [
                        {
                            "summary": proposal.summary,
                            "requested_change": proposal.requested_change,
                            "rationale": proposal.rationale,
                            "affected_sections": list(proposal.affected_sections),
                            "estimated_edits": proposal.estimated_edits,
                            "focus": proposal.focus,
                        }
                        for proposal in planner_payload.proposals
                    ]
                    added = _add_revision_proposals_to_session(revision_session, proposals)
                    if added > 0:
                        response_text = _pending_revision_prompt(proposals)
                    else:
                        response_text = "I couldn't derive concrete protocol edits from that request. Please specify the section and intended change."
                    assistant_payload = {
                        "answer_markdown": response_text,
                        "can_answer": True,
                        "cited_source_ids": [],
                        "follow_up_questions": [],
                    }
                elif planner_payload.intent == "out_of_scope":
                    assistant_payload = {
                        "answer_markdown": planner_payload.assistant_reply.strip(),
                        "can_answer": False,
                        "cited_source_ids": [],
                        "follow_up_questions": [],
                    }
                else:
                    pages = _get_ai_assistant_pages(analysis_state)
                    chunks = _get_ai_assistant_chunks(analysis_state, pages)
                    chat_result = run_protocol_chat(
                        input_path=Path(str(analysis_state["input_path"])),
                        analysis_json_text=analysis_state["json_text"],
                        question=user_prompt,
                        history=history,
                        model=str(analysis_state.get("chosen_model", "")).strip() or None,
                        pages=pages,
                        chunks=chunks,
                    )
                    assistant_payload = {
                        "answer_markdown": chat_result.payload.answer_markdown.strip(),
                        "can_answer": chat_result.payload.can_answer,
                        "cited_source_ids": list(chat_result.payload.cited_source_ids),
                        "follow_up_questions": list(chat_result.payload.follow_up_questions),
                    }
                    assistant_sources = [vars(source) for source in chat_result.sources]
                    assistant_payload["show_sources"] = (
                        bool(assistant_sources) and not _is_general_conversation_message(normalized_prompt)
                    )
    _append_chat_turn(
        analysis_state=analysis_state,
        user_prompt=user_prompt,
        assistant_payload=assistant_payload,
        sources=assistant_sources,
    )
    updated_history = _ensure_flat_chat_history_state(analysis_state)
    assistant_turn_id = _chat_message_turn_id(updated_history, len(updated_history) - 1)
    st.session_state["analysis_state"] = analysis_state
    sync_saved_analysis_snapshot(analysis_state, saved_root)
    return assistant_turn_id


def _reset_ai_assistant_conversation(analysis_state: dict[str, Any]) -> None:
    analysis_state["chat_threads"] = []
    analysis_state["chat_history"] = []
    analysis_state["chat_selected_thread_id"] = ""
    analysis_state["revision_session"] = _empty_revision_session_state()
    st.session_state["analysis_state"] = analysis_state


def _render_revision_review_items(
    *,
    analysis_state: dict[str, Any],
    revision_session: dict[str, Any],
    saved_root: Path,
    sync_saved_analysis_snapshot: Callable[[dict[str, Any], Path], None],
) -> None:
    pending_changes = revision_session.get("pending_changes", [])
    approved_changes = revision_session.get("approved_changes", [])
    declined_changes = revision_session.get("declined_changes", [])
    if not isinstance(pending_changes, list):
        pending_changes = []
    if not isinstance(approved_changes, list):
        approved_changes = []
    if not isinstance(declined_changes, list):
        declined_changes = []

    if pending_changes:
        st.markdown("#### Pending review")
        for index, change in enumerate(pending_changes, start=1):
            change_id = str(change.get("id", "")).strip() or f"pending-{index}"
            status = str(change.get("status", "pending")).strip().lower() or "pending"
            status_label = status.replace("_", " ").title()
            summary = str(change.get("summary", "")).strip()
            requested_change = str(change.get("requested_change", "")).strip()
            rationale = str(change.get("rationale", "")).strip()
            sections = change.get("affected_sections", [])
            section_text = ", ".join([str(item).strip() for item in sections if str(item).strip()]) if isinstance(sections, list) else ""
            with st.container(border=True):
                st.markdown(f"**{index}. {summary or 'Revision proposal'}**")
                st.caption(f"Status: {status_label}")
                if section_text:
                    st.caption(f"Affected sections: {section_text}")
                if requested_change:
                    st.markdown(f"**Requested change**\n\n{requested_change}")
                if rationale:
                    st.caption(rationale)

                review_cols = st.columns([1, 1, 1], gap="small")
                with review_cols[0]:
                    if st.button("Accept", key=f"ai_rev_accept_{change_id}", use_container_width=True):
                        _approve_revision_change(revision_session, change_id)
                        analysis_state["revision_session"] = revision_session
                        st.session_state["analysis_state"] = analysis_state
                        sync_saved_analysis_snapshot(analysis_state, saved_root)
                        st.rerun()
                with review_cols[1]:
                    if st.button("Decline", key=f"ai_rev_decline_{change_id}", use_container_width=True):
                        _decline_revision_change(revision_session, change_id)
                        analysis_state["revision_session"] = revision_session
                        st.session_state["analysis_state"] = analysis_state
                        sync_saved_analysis_snapshot(analysis_state, saved_root)
                        st.rerun()
                with review_cols[2]:
                    if st.button("Edit", key=f"ai_rev_edit_{change_id}", use_container_width=True):
                        st.session_state[f"ai_rev_edit_mode_{change_id}"] = True
                        st.rerun()

                if bool(st.session_state.get(f"ai_rev_edit_mode_{change_id}", False)):
                    edited_summary = st.text_input(
                        "Summary",
                        value=summary,
                        key=f"ai_rev_summary_{change_id}",
                    )
                    edited_requested_change = st.text_area(
                        "Requested change",
                        value=requested_change,
                        key=f"ai_rev_requested_change_{change_id}",
                        height=120,
                    )
                    edited_rationale = st.text_input(
                        "Rationale",
                        value=rationale,
                        key=f"ai_rev_rationale_{change_id}",
                    )
                    edit_cols = st.columns([1, 1], gap="small")
                    with edit_cols[0]:
                        if st.button("Save Edit", key=f"ai_rev_save_{change_id}", use_container_width=True):
                            _edit_revision_change(
                                revision_session,
                                change_id,
                                summary=edited_summary,
                                requested_change=edited_requested_change,
                                rationale=edited_rationale,
                            )
                            analysis_state["revision_session"] = revision_session
                            st.session_state["analysis_state"] = analysis_state
                            st.session_state[f"ai_rev_edit_mode_{change_id}"] = False
                            sync_saved_analysis_snapshot(analysis_state, saved_root)
                            st.rerun()
                    with edit_cols[1]:
                        if st.button("Cancel Edit", key=f"ai_rev_cancel_{change_id}", use_container_width=True):
                            st.session_state[f"ai_rev_edit_mode_{change_id}"] = False
                            st.rerun()

    if approved_changes:
        with st.expander(f"Approved changes ({len(approved_changes)})", expanded=False):
            for change in approved_changes:
                st.markdown(f"- **{str(change.get('summary', '')).strip() or 'Approved change'}**")
    if declined_changes:
        with st.expander(f"Declined changes ({len(declined_changes)})", expanded=False):
            for change in declined_changes:
                st.markdown(f"- **{str(change.get('summary', '')).strip() or 'Declined change'}**")


__all__ = ["render_ai_assistant_popup"]
