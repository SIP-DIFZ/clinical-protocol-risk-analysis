from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from protocol_risk_analyzer.models import AnalysisResult, RiskAssessment


def _coerce_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower())


def _format_domain_label(domain: str) -> str:
    return domain.strip().title()


def _format_domain_tab_label(domain: str) -> str:
    icons = {
        "health": "🩺",
        "financial": "💰",
        "ethical": "⚖️",
        "eligibility": "✅",
        "intervention": "💊",
    }
    normalized = domain.strip().lower()
    icon = icons.get(normalized, "📌")
    return f"{_format_domain_label(normalized)} {icon}"


def _truncate_text_for_label(value: object, max_chars: int) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[: max_chars - 1].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0].rstrip()
    return truncated.rstrip(" -,:;") + "…"


def _format_eta(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes = total_seconds // 60
    remaining = total_seconds % 60
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def _clean_editor_progress_message(message: str) -> str:
    cleaned = re.sub(r"^\s*Mitigate Protocol:\s*", "", message or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(\d+\s*/\s*\d+\)\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "Running Editor Agent..."


def _risk_row_id(risk: RiskAssessment, index: int) -> str:
    return f"{risk.domain.value}:{index}:{risk.title}"


def _risk_selection_label(risk: RiskAssessment, index: int) -> str:
    return f"[{index:03d}] {risk.title} | {_format_domain_label(risk.domain.value)} | Score {risk.risk_score:.2f}"


def _build_editor_risk_entries(result: AnalysisResult) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, risk in enumerate(result.risks, start=1):
        entries.append(
            {
                "risk_id": _risk_row_id(risk, idx),
                "risk": risk,
                "label": _risk_selection_label(risk, idx),
            }
        )
    return entries


def _sanitize_accepted_risk_ids(accepted_risk_ids: object, valid_risk_ids: set[str]) -> list[str]:
    if not isinstance(accepted_risk_ids, list):
        return []
    sanitized: list[str] = []
    seen: set[str] = set()
    for item in accepted_risk_ids:
        risk_id = str(item).strip()
        if not risk_id or risk_id not in valid_risk_ids or risk_id in seen:
            continue
        seen.add(risk_id)
        sanitized.append(risk_id)
    return sanitized


def _sanitize_accepted_risk_notes(
    accepted_risk_notes: object,
    valid_risk_ids: set[str] | None = None,
) -> dict[str, str]:
    if not isinstance(accepted_risk_notes, dict):
        return {}
    sanitized: dict[str, str] = {}
    for raw_risk_id, raw_note in accepted_risk_notes.items():
        risk_id = str(raw_risk_id).strip()
        note = str(raw_note).strip()
        if not risk_id or not note:
            continue
        if valid_risk_ids is not None and risk_id not in valid_risk_ids:
            continue
        sanitized[risk_id] = note
    return sanitized


def _normalize_mitigation_status_for_ui(status: str | None) -> str:
    normalized = (status or "").strip().lower().replace(" ", "_")
    if normalized == "completely_mitigated":
        return "adequately_mitigated"
    if normalized == "mitigated":
        return "partially_mitigated"
    if normalized in {"unmitigated", "partially_mitigated", "adequately_mitigated"}:
        return normalized
    return "unmitigated"


def _mitigation_status_label(status: str) -> str:
    labels = {
        "unmitigated": "Unmitigated",
        "partially_mitigated": "Partially Mitigated",
        "adequately_mitigated": "Adequately Mitigated",
    }
    return labels.get(status, "Unmitigated")


def _mitigation_status_css_class(status: str | None) -> str:
    normalized = _normalize_mitigation_status_for_ui(status)
    suffix_map = {
        "unmitigated": "status-unmitigated",
        "partially_mitigated": "status-partially-mitigated",
        "adequately_mitigated": "status-adequately-mitigated",
    }
    return suffix_map.get(normalized, "status-unmitigated")


def _truncate_status_reason(reason: str | None) -> str:
    cleaned = " ".join((reason or "").strip().split())
    if not cleaned:
        return "No explicit mitigation rationale was returned for this risk."
    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if len(sentence_parts) <= 3:
        return " ".join(sentence_parts)
    return " ".join(sentence_parts[:3]).rstrip() + " …"


def _score_band(score: float) -> str:
    if score >= 15:
        return "Very High"
    if score >= 12:
        return "High"
    if score >= 8:
        return "Moderate"
    return "Low"


def _sanitize_chat_history(chat_history: object) -> list[dict[str, Any]]:
    if not isinstance(chat_history, list):
        return []
    sanitized: list[dict[str, Any]] = []
    previous_role = ""
    previous_content = ""
    for item in chat_history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if role == previous_role and content == previous_content:
            continue
        sanitized_item: dict[str, Any] = {"role": role, "content": content}
        if "can_answer" in item:
            sanitized_item["can_answer"] = bool(item.get("can_answer"))
        if "show_sources" in item:
            sanitized_item["show_sources"] = bool(item.get("show_sources"))
        if "pending_response" in item:
            sanitized_item["pending_response"] = bool(item.get("pending_response"))
        cited_source_ids = item.get("cited_source_ids", [])
        if isinstance(cited_source_ids, list):
            sanitized_item["cited_source_ids"] = [int(value) for value in cited_source_ids if str(value).strip().isdigit()]
        follow_up_questions = item.get("follow_up_questions", [])
        if isinstance(follow_up_questions, list):
            sanitized_item["follow_up_questions"] = [
                str(value).strip() for value in follow_up_questions if str(value).strip()
            ]
        sources = item.get("sources", [])
        if isinstance(sources, list):
            cleaned_sources: list[dict[str, Any]] = []
            for source in sources:
                if not isinstance(source, dict):
                    continue
                cleaned_source: dict[str, Any] = {}
                for key in (
                    "source_id",
                    "kind",
                    "label",
                    "page_start",
                    "page_end",
                    "risk_title",
                    "domain",
                    "excerpt",
                    "why_relevant",
                ):
                    if key in source:
                        cleaned_source[key] = source[key]
                if cleaned_source:
                    cleaned_sources.append(cleaned_source)
            sanitized_item["sources"] = cleaned_sources
        sanitized.append(sanitized_item)
        previous_role = role
        previous_content = content
    return sanitized


def _build_chat_thread_title(messages: list[dict[str, Any]], fallback_index: int) -> str:
    question = ""
    for message in messages:
        if str(message.get("role", "")).strip() == "user":
            question = str(message.get("content", "")).strip()
            if question:
                break
    if not question:
        return f"Chat {fallback_index}"
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", question) if token]
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "can",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "what",
        "which",
        "why",
        "with",
    }
    filtered: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in stopwords:
            continue
        filtered.append(token)
        if len(filtered) >= 3:
            break
    if not filtered:
        filtered = tokens[:3]
    title = " ".join(word.capitalize() for word in filtered).strip()
    return title or f"Chat {fallback_index}"


def _sanitize_chat_threads(chat_threads: object) -> list[dict[str, Any]]:
    if not isinstance(chat_threads, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for index, item in enumerate(chat_threads, start=1):
        if not isinstance(item, dict):
            continue
        thread_id = str(item.get("id", "")).strip() or f"thread-{index}"
        messages = _sanitize_chat_history(item.get("messages", []))
        title = str(item.get("title", "")).strip() or _build_chat_thread_title(messages, index)
        created_at = str(item.get("created_at", "")).strip()
        updated_at = str(item.get("updated_at", "")).strip()
        sanitized.append(
            {
                "id": thread_id,
                "title": title,
                "messages": messages,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
    return sanitized


def _initial_ai_assistant_message() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": (
            "Hello! I'm your AI Assistant for this clinical protocol analysis. "
            "I can explain identified risks, summarize findings, and help you build an approved revision plan "
            "before generating a revised protocol document."
        ),
        "can_answer": True,
        "cited_source_ids": [],
        "follow_up_questions": [],
        "sources": [],
    }


def _pending_ai_assistant_message() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "...",
        "pending_response": True,
        "can_answer": True,
        "cited_source_ids": [],
        "follow_up_questions": [],
        "sources": [],
    }


def _ensure_flat_chat_history_state(analysis_state: dict[str, Any]) -> list[dict[str, Any]]:
    history = _sanitize_chat_history(analysis_state.get("chat_history", []))
    if history:
        analysis_state["chat_threads"] = []
        analysis_state["chat_selected_thread_id"] = ""
        analysis_state["chat_history"] = history
        return history

    threads = _sanitize_chat_threads(analysis_state.get("chat_threads", []))
    flattened: list[dict[str, Any]] = []
    for thread in sorted(
        threads,
        key=lambda item: (
            str(item.get("created_at", "")).strip(),
            str(item.get("updated_at", "")).strip(),
            str(item.get("id", "")).strip(),
        ),
    ):
        flattened.extend(_sanitize_chat_history(thread.get("messages", [])))

    history = _sanitize_chat_history(flattened)
    if not history:
        history = [_initial_ai_assistant_message()]
    analysis_state["chat_threads"] = []
    analysis_state["chat_selected_thread_id"] = ""
    analysis_state["chat_history"] = history
    return history


def _empty_revision_session_state() -> dict[str, Any]:
    return {
        "pending_changes": [],
        "approved_changes": [],
        "declined_changes": [],
        "affected_sections": [],
        "estimated_edit_count": 0,
        "history": [],
        "awaiting_confirmation_change_ids": [],
        "generated_output": None,
    }


def _sanitize_revision_change(change: object) -> dict[str, Any] | None:
    if not isinstance(change, dict):
        return None
    change_id = str(change.get("id", "")).strip()
    summary = str(change.get("summary", "")).strip()
    requested_change = str(change.get("requested_change", "")).strip()
    rationale = str(change.get("rationale", "")).strip()
    focus = str(change.get("focus", "general")).strip().lower() or "general"
    status = str(change.get("status", "pending")).strip().lower() or "pending"
    if status not in {"pending", "approved", "declined", "user_edited"}:
        status = "pending"
    if not summary or not requested_change:
        return None
    sections_raw = change.get("affected_sections", [])
    affected_sections = (
        [str(item).strip() for item in sections_raw if str(item).strip()]
        if isinstance(sections_raw, list)
        else []
    )
    estimated_edits = _coerce_int(change.get("estimated_edits")) or 1
    estimated_edits = max(1, min(estimated_edits, 50))
    created_at = str(change.get("created_at", "")).strip() or datetime.now().isoformat(timespec="seconds")
    return {
        "id": change_id or f"rev-{uuid4().hex[:10]}",
        "summary": summary,
        "requested_change": requested_change,
        "rationale": rationale or "User-directed protocol revision item.",
        "affected_sections": affected_sections,
        "estimated_edits": estimated_edits,
        "focus": focus,
        "status": status,
        "created_at": created_at,
    }


def _sanitize_revision_session_state(revision_session: object) -> dict[str, Any]:
    if not isinstance(revision_session, dict):
        return _empty_revision_session_state()
    pending_changes_raw = revision_session.get("pending_changes", [])
    approved_changes_raw = revision_session.get("approved_changes", [])
    declined_changes_raw = revision_session.get("declined_changes", [])
    pending_changes: list[dict[str, Any]] = []
    approved_changes: list[dict[str, Any]] = []
    declined_changes: list[dict[str, Any]] = []
    pending_ids: set[str] = set()
    approved_ids: set[str] = set()
    declined_ids: set[str] = set()
    if isinstance(pending_changes_raw, list):
        for item in pending_changes_raw:
            cleaned = _sanitize_revision_change(item)
            if cleaned is None:
                continue
            if cleaned["id"] in pending_ids:
                continue
            pending_ids.add(cleaned["id"])
            pending_changes.append(cleaned)
    if isinstance(approved_changes_raw, list):
        for item in approved_changes_raw:
            cleaned = _sanitize_revision_change(item)
            if cleaned is None:
                continue
            if cleaned["id"] in approved_ids:
                continue
            approved_ids.add(cleaned["id"])
            approved_changes.append(cleaned)
    if isinstance(declined_changes_raw, list):
        for item in declined_changes_raw:
            cleaned = _sanitize_revision_change(item)
            if cleaned is None:
                continue
            if cleaned["id"] in declined_ids:
                continue
            declined_ids.add(cleaned["id"])
            declined_changes.append(cleaned)
    awaiting_raw = revision_session.get("awaiting_confirmation_change_ids", [])
    awaiting_ids = (
        [str(item).strip() for item in awaiting_raw if str(item).strip() in pending_ids]
        if isinstance(awaiting_raw, list)
        else []
    )
    generated_output_raw = revision_session.get("generated_output")
    generated_output: dict[str, Any] | None = None
    if isinstance(generated_output_raw, dict):
        generated_output = {
            "chosen_model": str(generated_output_raw.get("chosen_model", "")).strip(),
            "approved_change_count": _coerce_int(generated_output_raw.get("approved_change_count")) or 0,
            "output_dir": str(generated_output_raw.get("output_dir", "")).strip(),
            "generated_at": str(generated_output_raw.get("generated_at", "")).strip(),
        }
        if isinstance(generated_output_raw.get("revised_md_text"), str):
            generated_output["revised_md_text"] = str(generated_output_raw.get("revised_md_text", ""))
        if isinstance(generated_output_raw.get("patch_log_text"), str):
            generated_output["patch_log_text"] = str(generated_output_raw.get("patch_log_text", ""))
        revised_pdf_bytes = generated_output_raw.get("revised_pdf_bytes")
        if isinstance(revised_pdf_bytes, bytes):
            generated_output["revised_pdf_bytes"] = revised_pdf_bytes
        revised_docx_bytes = generated_output_raw.get("revised_docx_bytes")
        if isinstance(revised_docx_bytes, bytes):
            generated_output["revised_docx_bytes"] = revised_docx_bytes
    session = {
        "pending_changes": pending_changes,
        "approved_changes": approved_changes,
        "declined_changes": declined_changes,
        "affected_sections": sorted(
            {
                section
                for change in pending_changes + approved_changes
                for section in change.get("affected_sections", [])
                if str(section).strip()
            }
        ),
        "estimated_edit_count": sum(
            int(change.get("estimated_edits", 0) or 0) for change in pending_changes + approved_changes
        ),
        "history": list(revision_session.get("history", []))
        if isinstance(revision_session.get("history", []), list)
        else [],
        "awaiting_confirmation_change_ids": awaiting_ids,
        "generated_output": generated_output,
    }
    return session


def _ensure_revision_session_state(analysis_state: dict[str, Any]) -> dict[str, Any]:
    session = _sanitize_revision_session_state(analysis_state.get("revision_session"))
    generated_output = session.get("generated_output")
    if isinstance(generated_output, dict):
        output_dir_text = str(generated_output.get("output_dir", "")).strip()
        if output_dir_text and (
            "revised_pdf_bytes" not in generated_output
            or "revised_docx_bytes" not in generated_output
            or "revised_md_text" not in generated_output
            or "patch_log_text" not in generated_output
        ):
            output_dir = Path(output_dir_text)
            md_path = output_dir / "mitigated_protocol.md"
            pdf_path = output_dir / "mitigated_protocol.pdf"
            docx_path = output_dir / "mitigated_protocol.docx"
            patch_log_path = output_dir / "mitigated_protocol_patches.txt"
            if md_path.exists() and pdf_path.exists() and docx_path.exists() and patch_log_path.exists():
                generated_output["revised_md_text"] = md_path.read_text(encoding="utf-8")
                generated_output["revised_pdf_bytes"] = pdf_path.read_bytes()
                generated_output["revised_docx_bytes"] = docx_path.read_bytes()
                generated_output["patch_log_text"] = patch_log_path.read_text(encoding="utf-8")
        session["generated_output"] = generated_output
    analysis_state["revision_session"] = session
    return session


def _revision_session_for_metadata(revision_session: object) -> dict[str, Any]:
    session = _sanitize_revision_session_state(revision_session)
    generated_output = session.get("generated_output")
    if isinstance(generated_output, dict):
        generated_output = {
            "chosen_model": str(generated_output.get("chosen_model", "")).strip(),
            "approved_change_count": _coerce_int(generated_output.get("approved_change_count")) or 0,
            "output_dir": str(generated_output.get("output_dir", "")).strip(),
            "generated_at": str(generated_output.get("generated_at", "")).strip(),
        }
        session["generated_output"] = generated_output
    return session


def _append_revision_session_history(revision_session: dict[str, Any], event_type: str, detail: str) -> None:
    history = revision_session.get("history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "detail": detail.strip(),
        }
    )
    revision_session["history"] = history[-80:]


def _refresh_revision_session_rollups(revision_session: dict[str, Any]) -> None:
    all_changes = list(revision_session.get("pending_changes", [])) + list(revision_session.get("approved_changes", []))
    revision_session["affected_sections"] = sorted(
        {
            str(section).strip()
            for item in all_changes
            for section in item.get("affected_sections", [])
            if str(section).strip()
        }
    )
    revision_session["estimated_edit_count"] = sum(int(item.get("estimated_edits", 0) or 0) for item in all_changes)


def _is_affirmative_message(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    return normalized in {"yes", "y", "yeah", "yep", "confirm", "approved", "approve", "apply", "ok", "okay"}


def _is_negative_message(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    return normalized in {"no", "n", "nope", "cancel", "discard", "reject"}


def _is_general_conversation_message(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return True
    casual_exact = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "thank you!",
        "thanks!",
        "ok",
        "okay",
        "awesome",
        "great",
        "got it",
        "sounds good",
        "bye",
        "goodbye",
    }
    if normalized in casual_exact:
        return True
    tokens = re.findall(r"[a-z0-9']+", normalized)
    casual_tokens = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "you",
        "you're",
        "youre",
        "very",
        "ok",
        "okay",
        "awesome",
        "great",
        "bye",
        "goodbye",
        "cool",
        "nice",
        "welcome",
    }
    if tokens and len(tokens) <= 6 and all(token in casual_tokens for token in tokens):
        return True
    return any(phrase in normalized for phrase in ("how are you", "you're welcome", "you are welcome"))


def _looks_like_generate_revision_request(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    triggers = {
        "generate revised protocol",
        "generate revision",
        "create revised protocol",
        "build revised protocol",
        "apply revisions now",
        "generate document",
    }
    return any(trigger in normalized for trigger in triggers)


def _looks_like_apply_plan_request(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    return normalized in {"apply changes", "apply plan", "add to plan", "approve changes", "yes apply"}


def _looks_like_undo_plan_request(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    return "undo" in normalized and "change" in normalized


def _looks_like_clear_plan_request(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    return "clear revision plan" in normalized or "reset revision plan" in normalized


def _format_revision_plan_summary(revision_session: dict[str, Any]) -> str:
    pending_count = len(revision_session.get("pending_changes", []))
    approved_count = len(revision_session.get("approved_changes", []))
    affected_count = len(revision_session.get("affected_sections", []))
    edit_count = int(revision_session.get("estimated_edit_count", 0) or 0)
    return (
        f"Pending changes: **{pending_count}**\n\n"
        f"Approved changes: **{approved_count}**\n\n"
        f"Affected sections: **{affected_count}**\n\n"
        f"Estimated edits: **{edit_count}**"
    )


def _add_revision_proposals_to_session(
    revision_session: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> int:
    pending_changes = revision_session.get("pending_changes", [])
    if not isinstance(pending_changes, list):
        pending_changes = []
    added_ids: list[str] = []
    for proposal in proposals:
        cleaned = _sanitize_revision_change(proposal)
        if cleaned is None:
            continue
        if cleaned["id"] in {item.get("id") for item in pending_changes}:
            continue
        cleaned["status"] = "pending"
        pending_changes.append(cleaned)
        added_ids.append(cleaned["id"])
    revision_session["pending_changes"] = pending_changes
    revision_session["awaiting_confirmation_change_ids"] = added_ids
    _refresh_revision_session_rollups(revision_session)
    if added_ids:
        _append_revision_session_history(
            revision_session,
            "proposed",
            f"Proposed {len(added_ids)} change(s) awaiting approval.",
        )
    return len(added_ids)


def _approve_pending_revision_changes(revision_session: dict[str, Any], only_awaiting: bool = False) -> int:
    pending_changes = revision_session.get("pending_changes", [])
    approved_changes = revision_session.get("approved_changes", [])
    declined_changes = revision_session.get("declined_changes", [])
    awaiting_ids = set(revision_session.get("awaiting_confirmation_change_ids", []))
    if not isinstance(pending_changes, list) or not isinstance(approved_changes, list) or not isinstance(declined_changes, list):
        return 0
    remaining_pending: list[dict[str, Any]] = []
    moved = 0
    for change in pending_changes:
        should_move = not only_awaiting or str(change.get("id", "")).strip() in awaiting_ids
        if should_move:
            change["status"] = "approved"
            approved_changes.append(change)
            moved += 1
        else:
            remaining_pending.append(change)
    revision_session["approved_changes"] = approved_changes
    revision_session["pending_changes"] = remaining_pending if only_awaiting else []
    revision_session["awaiting_confirmation_change_ids"] = []
    _refresh_revision_session_rollups(revision_session)
    if moved:
        _append_revision_session_history(revision_session, "approved", f"Approved {moved} change(s).")
    return moved


def _approve_revision_change(revision_session: dict[str, Any], change_id: str) -> int:
    pending_changes = revision_session.get("pending_changes", [])
    approved_changes = revision_session.get("approved_changes", [])
    if not isinstance(pending_changes, list) or not isinstance(approved_changes, list):
        return 0
    normalized_id = str(change_id).strip()
    if not normalized_id:
        return 0
    remaining_pending: list[dict[str, Any]] = []
    moved = 0
    for change in pending_changes:
        if str(change.get("id", "")).strip() == normalized_id:
            change["status"] = "approved"
            approved_changes.append(change)
            moved += 1
        else:
            remaining_pending.append(change)
    revision_session["pending_changes"] = remaining_pending
    revision_session["approved_changes"] = approved_changes
    revision_session["awaiting_confirmation_change_ids"] = [
        item for item in revision_session.get("awaiting_confirmation_change_ids", []) if str(item).strip() != normalized_id
    ]
    _refresh_revision_session_rollups(revision_session)
    if moved:
        _append_revision_session_history(revision_session, "approved", f"Approved change {normalized_id}.")
    return moved


def _decline_revision_change(revision_session: dict[str, Any], change_id: str) -> int:
    pending_changes = revision_session.get("pending_changes", [])
    declined_changes = revision_session.get("declined_changes", [])
    if not isinstance(pending_changes, list) or not isinstance(declined_changes, list):
        return 0
    normalized_id = str(change_id).strip()
    if not normalized_id:
        return 0
    remaining_pending: list[dict[str, Any]] = []
    moved = 0
    for change in pending_changes:
        if str(change.get("id", "")).strip() == normalized_id:
            change["status"] = "declined"
            declined_changes.append(change)
            moved += 1
        else:
            remaining_pending.append(change)
    revision_session["pending_changes"] = remaining_pending
    revision_session["declined_changes"] = declined_changes
    revision_session["awaiting_confirmation_change_ids"] = [
        item for item in revision_session.get("awaiting_confirmation_change_ids", []) if str(item).strip() != normalized_id
    ]
    _refresh_revision_session_rollups(revision_session)
    if moved:
        _append_revision_session_history(revision_session, "declined", f"Declined change {normalized_id}.")
    return moved


def _edit_revision_change(
    revision_session: dict[str, Any],
    change_id: str,
    *,
    summary: str | None = None,
    requested_change: str | None = None,
    rationale: str | None = None,
) -> bool:
    pending_changes = revision_session.get("pending_changes", [])
    if not isinstance(pending_changes, list):
        return False
    normalized_id = str(change_id).strip()
    if not normalized_id:
        return False
    edited = False
    for change in pending_changes:
        if str(change.get("id", "")).strip() != normalized_id:
            continue
        if summary is not None:
            cleaned_summary = str(summary).strip()
            if cleaned_summary:
                change["summary"] = cleaned_summary
                edited = True
        if requested_change is not None:
            cleaned_requested_change = str(requested_change).strip()
            if cleaned_requested_change:
                change["requested_change"] = cleaned_requested_change
                edited = True
        if rationale is not None:
            cleaned_rationale = str(rationale).strip()
            if cleaned_rationale:
                change["rationale"] = cleaned_rationale
                edited = True
        if edited:
            change["status"] = "user_edited"
        break
    if edited:
        _refresh_revision_session_rollups(revision_session)
        _append_revision_session_history(revision_session, "edited", f"Edited change {normalized_id}.")
    return edited


def _discard_awaiting_revision_changes(revision_session: dict[str, Any]) -> int:
    awaiting_ids = set(revision_session.get("awaiting_confirmation_change_ids", []))
    if not awaiting_ids:
        return 0
    pending_changes = revision_session.get("pending_changes", [])
    if not isinstance(pending_changes, list):
        return 0
    remaining = [change for change in pending_changes if str(change.get("id", "")).strip() not in awaiting_ids]
    removed = len(pending_changes) - len(remaining)
    revision_session["pending_changes"] = remaining
    revision_session["awaiting_confirmation_change_ids"] = []
    _refresh_revision_session_rollups(revision_session)
    if removed:
        _append_revision_session_history(revision_session, "discarded", f"Discarded {removed} pending change(s).")
    return removed


def _undo_last_revision_change(revision_session: dict[str, Any]) -> tuple[str, bool]:
    approved_changes = revision_session.get("approved_changes", [])
    pending_changes = revision_session.get("pending_changes", [])
    if isinstance(approved_changes, list) and approved_changes:
        removed = approved_changes.pop()
        revision_session["approved_changes"] = approved_changes
        revision_session["awaiting_confirmation_change_ids"] = []
        _refresh_revision_session_rollups(revision_session)
        _append_revision_session_history(
            revision_session,
            "undo",
            f"Removed approved change: {str(removed.get('summary', '')).strip()}",
        )
        return "Removed the last approved revision change.", True
    if isinstance(pending_changes, list) and pending_changes:
        removed = pending_changes.pop()
        revision_session["pending_changes"] = pending_changes
        revision_session["awaiting_confirmation_change_ids"] = []
        _refresh_revision_session_rollups(revision_session)
        _append_revision_session_history(
            revision_session,
            "undo",
            f"Removed pending change: {str(removed.get('summary', '')).strip()}",
        )
        return "Removed the last pending revision change.", True
    return "There are no revision changes to undo.", False


def _clear_revision_plan(revision_session: dict[str, Any]) -> bool:
    had_changes = bool(revision_session.get("pending_changes")) or bool(revision_session.get("approved_changes"))
    revision_session["pending_changes"] = []
    revision_session["approved_changes"] = []
    revision_session["declined_changes"] = []
    revision_session["awaiting_confirmation_change_ids"] = []
    revision_session["generated_output"] = None
    _refresh_revision_session_rollups(revision_session)
    _append_revision_session_history(revision_session, "cleared", "Cleared revision plan.")
    return had_changes


def _pending_revision_prompt(proposals: list[dict[str, Any]]) -> str:
    lines = ["I identified the following protocol revisions:"]
    for proposal in proposals:
        summary = str(proposal.get("summary", "")).strip()
        sections = proposal.get("affected_sections", [])
        estimated_edits = int(proposal.get("estimated_edits", 0) or 0)
        if summary:
            lines.append(f"- **{summary}**")
        if isinstance(sections, list) and sections:
            lines.append(f"  - Affected sections: {', '.join(str(item).strip() for item in sections if str(item).strip())}")
        if estimated_edits > 0:
            lines.append(f"  - Estimated edits: {estimated_edits}")
    lines.append("")
    lines.append("Approve these changes into the revision plan? (You can click **Apply Changes** or reply **Yes**.)")
    return "\n".join(lines).strip()


def _extract_revision_target_index(text: str) -> int | None:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return None
    direct_match = re.search(r"(?:^|\s)(?:edit|change|revision|proposal|item|entry|line)\s*(?:#|no\.?|number\s*)?\s*(\d+)", normalized)
    if direct_match:
        candidate = int(direct_match.group(1))
        return candidate if candidate > 0 else None
    hash_match = re.search(r"#\s*(\d+)", normalized)
    if hash_match:
        candidate = int(hash_match.group(1))
        return candidate if candidate > 0 else None
    return None


def _ordered_revision_targets(revision_session: dict[str, Any]) -> list[dict[str, Any]]:
    pending_changes = revision_session.get("pending_changes", [])
    approved_changes = revision_session.get("approved_changes", [])
    ordered: list[dict[str, Any]] = []
    if isinstance(pending_changes, list):
        ordered.extend(change for change in pending_changes if isinstance(change, dict))
    if isinstance(approved_changes, list):
        ordered.extend(change for change in approved_changes if isinstance(change, dict))
    return ordered


def _apply_revision_item_edit_request(revision_session: dict[str, Any], user_prompt: str) -> tuple[bool, str]:
    prompt = " ".join(str(user_prompt or "").strip().split())
    if not prompt:
        return False, ""
    target_index = _extract_revision_target_index(prompt)
    if target_index is None:
        return False, ""
    targets = _ordered_revision_targets(revision_session)
    if not targets:
        return False, ""
    if target_index > len(targets):
        return True, f"I couldn't find Edit #{target_index}. I only see {len(targets)} revision item(s) in the plan."

    target = targets[target_index - 1]
    normalized_prompt = prompt.lower()
    if not any(keyword in normalized_prompt for keyword in ("more detailed", "more detail", "more specific", "detailed", "specific", "clearer", "clarify", "simplify", "shorten", "refine")):
        return True, f"I can update Edit #{target_index}, but I need a more specific instruction such as 'make it more detailed' or 'simplify it.'"

    current_requested_change = str(target.get("requested_change", "")).strip()
    summary = str(target.get("summary", "")).strip()
    rationale = str(target.get("rationale", "")).strip()
    if any(keyword in normalized_prompt for keyword in ("more detailed", "more detail", "detailed", "more specific", "specific", "clearer", "clarify")):
        if not re.search(r"more detailed|more detail|detailed|more specific|clearer|clarify", current_requested_change.lower()):
            revised_requested_change = (
                current_requested_change.rstrip(".")
                + ". Include additional detail on the intended operational steps, timing, and monitoring responsibilities."
            )
        else:
            revised_requested_change = current_requested_change
        revised_summary = str(summary or "Revision proposal")
        revised_rationale = rationale or "This change is intended to reduce protocol risk while preserving the original study intent."
        revised_rationale = revised_rationale.rstrip(".") + ". The wording was expanded to provide more explicit implementation details without altering the study's core objective."
    elif any(keyword in normalized_prompt for keyword in ("simplify", "shorten", "less verbose")):
        revised_requested_change = re.sub(r"\s+", " ", current_requested_change)
        revised_requested_change = re.sub(r"\s*\.(?=\s*[A-Z])", ". ", revised_requested_change)
        revised_requested_change = revised_requested_change[: max(180, len(revised_requested_change))]
        revised_summary = str(summary or "Revision proposal")
        revised_rationale = rationale or "This change is intended to reduce protocol risk while preserving the original study intent."
        revised_rationale = revised_rationale.rstrip(".") + ". The wording was streamlined for clarity while preserving the intended mitigation."
    else:
        revised_requested_change = current_requested_change
        revised_summary = summary
        revised_rationale = rationale

    if revised_requested_change.strip():
        target["requested_change"] = revised_requested_change.strip()
    if revised_summary.strip():
        target["summary"] = revised_summary.strip()
    if revised_rationale.strip():
        target["rationale"] = revised_rationale.strip()
    target["status"] = "user_edited"
    _refresh_revision_session_rollups(revision_session)
    _append_revision_session_history(revision_session, "edited", f"Updated Edit #{target_index} based on user instruction.")
    return True, f"I updated Edit #{target_index} to match that request."


def _is_docx_protocol_upload(filename: object) -> bool:
    candidate = str(filename).strip()
    if not candidate:
        return False
    return Path(candidate).suffix.lower() == ".docx"


def _ai_assistant_popup_input_key(analysis_state: dict[str, Any]) -> str:
    analysis_id = str(analysis_state.get("saved_analysis_id", "")).strip()
    if not analysis_id:
        analysis_id = str(analysis_state.get("run_dir", "")).strip()
    if not analysis_id:
        analysis_id = str(analysis_state.get("input_path", "")).strip()
    if not analysis_id:
        analysis_id = "active-analysis"
    return f"ai_assistant_popup_input_{_safe_key(analysis_id)}"


def _ensure_chat_thread_state(analysis_state: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    threads = _sanitize_chat_threads(analysis_state.get("chat_threads", []))
    if not threads:
        legacy_messages = _sanitize_chat_history(analysis_state.get("chat_history", []))
        if legacy_messages:
            threads = [
                {
                    "id": "thread-1",
                    "title": _build_chat_thread_title(legacy_messages, 1),
                    "messages": legacy_messages,
                    "created_at": "",
                    "updated_at": "",
                }
            ]
    selected_thread_id = str(analysis_state.get("chat_selected_thread_id", "")).strip()
    if threads:
        thread_ids = {str(thread.get("id", "")).strip() for thread in threads}
        if selected_thread_id not in thread_ids:
            selected_thread_id = str(threads[-1].get("id", "")).strip()
    else:
        selected_thread_id = ""
    analysis_state["chat_threads"] = threads
    analysis_state["chat_selected_thread_id"] = selected_thread_id
    analysis_state["chat_history"] = _selected_chat_thread_messages(threads, selected_thread_id)
    return threads, selected_thread_id


def _selected_chat_thread_messages(threads: list[dict[str, Any]], selected_thread_id: str) -> list[dict[str, Any]]:
    for thread in threads:
        if str(thread.get("id", "")).strip() == selected_thread_id:
            return _sanitize_chat_history(thread.get("messages", []))
    return []


def _find_chat_thread(threads: list[dict[str, Any]], thread_id: str) -> dict[str, Any] | None:
    for thread in threads:
        if str(thread.get("id", "")).strip() == thread_id:
            return thread
    return None


def _append_chat_turn(
    *,
    analysis_state: dict[str, Any],
    user_prompt: str,
    assistant_payload: dict[str, Any],
    sources: list[dict[str, Any]],
) -> None:
    messages = _ensure_flat_chat_history_state(analysis_state)
    assistant_message = {
        "role": "assistant",
        "content": str(assistant_payload.get("answer_markdown", "")).strip(),
        "can_answer": bool(assistant_payload.get("can_answer", True)),
        "show_sources": bool(assistant_payload.get("show_sources", True)),
        "cited_source_ids": list(assistant_payload.get("cited_source_ids", [])),
        "follow_up_questions": list(assistant_payload.get("follow_up_questions", [])),
        "sources": sources,
    }
    pending_placeholder_index: int | None = None
    if len(messages) >= 2:
        last_message = messages[-1]
        previous_message = messages[-2]
        if (
            str(last_message.get("role", "")).strip() == "assistant"
            and bool(last_message.get("pending_response", False))
            and str(previous_message.get("role", "")).strip() == "user"
            and str(previous_message.get("content", "")).strip() == user_prompt.strip()
        ):
            pending_placeholder_index = len(messages) - 1
    if pending_placeholder_index is None and not (
        messages
        and str(messages[-1].get("role", "")).strip() == "user"
        and str(messages[-1].get("content", "")).strip() == user_prompt.strip()
    ):
        messages.append({"role": "user", "content": user_prompt})
    if pending_placeholder_index is not None:
        messages[pending_placeholder_index] = assistant_message
    else:
        messages.append(assistant_message)
    analysis_state["chat_threads"] = []
    analysis_state["chat_selected_thread_id"] = ""
    analysis_state["chat_history"] = messages


def _summarize_chat_turn(question: str) -> str:
    title = _build_chat_thread_title([{"role": "user", "content": question}], 1)
    return title


def _create_new_chat_thread(analysis_state: dict[str, Any]) -> dict[str, Any]:
    threads, _ = _ensure_chat_thread_state(analysis_state)
    thread = {
        "id": f"thread-{uuid4().hex[:8]}",
        "title": f"Chat {len(threads) + 1}",
        "messages": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    threads.append(thread)
    analysis_state["chat_threads"] = threads
    analysis_state["chat_selected_thread_id"] = str(thread["id"])
    analysis_state["chat_history"] = []
    return thread
