from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import html
import io
import json
from pathlib import Path
import re
import shutil
from threading import Lock
from typing import Any
from uuid import uuid4
import zipfile

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from protocol_risk_analyzer.backend.analysis_service import run_protocol_analysis
from protocol_risk_analyzer.backend.chat_service import run_protocol_chat
from protocol_risk_analyzer.backend.editor_service import run_protocol_editor
from protocol_risk_analyzer.backend.revision_service import (
    build_revision_change_risks,
    run_protocol_revision_planner,
)
from protocol_risk_analyzer.chunking import chunk_pages
from protocol_risk_analyzer.config import get_default_model
from protocol_risk_analyzer.io import read_document
from protocol_risk_analyzer.models import AnalysisResult, RiskAssessment
from protocol_risk_analyzer.reporting import build_risk_markdown_section
from protocol_risk_analyzer.ui_helpers import (
    _add_revision_proposals_to_session,
    _ai_assistant_popup_input_key,
    _append_chat_turn,
    _append_revision_session_history,
    _approve_pending_revision_changes,
    _build_chat_thread_title,
    _build_editor_risk_entries,
    _clear_revision_plan,
    _create_new_chat_thread,
    _discard_awaiting_revision_changes,
    _empty_revision_session_state,
    _ensure_chat_thread_state,
    _ensure_flat_chat_history_state,
    _ensure_revision_session_state,
    _find_chat_thread,
    _format_domain_label,
    _format_domain_tab_label,
    _format_revision_plan_summary,
    _initial_ai_assistant_message,
    _is_affirmative_message,
    _is_docx_protocol_upload,
    _is_general_conversation_message,
    _is_negative_message,
    _looks_like_apply_plan_request,
    _looks_like_clear_plan_request,
    _looks_like_generate_revision_request,
    _looks_like_undo_plan_request,
    _pending_ai_assistant_message,
    _pending_revision_prompt,
    _refresh_revision_session_rollups,
    _revision_session_for_metadata,
    _risk_row_id,
    _risk_selection_label,
    _score_band,
    _safe_key,
    _clean_editor_progress_message,
    _mitigation_status_css_class,
    _mitigation_status_label,
    _normalize_mitigation_status_for_ui,
    _sanitize_accepted_risk_ids,
    _sanitize_accepted_risk_notes,
    _sanitize_chat_history,
    _sanitize_chat_threads,
    _sanitize_revision_session_state,
    _selected_chat_thread_messages,
    _summarize_chat_turn,
    _truncate_status_reason,
    _undo_last_revision_change,
)

DOMAIN_COLORS = {
    "health": "#D95F5F",
    "financial": "#4C78A8",
    "ethical": "#8E6BBE",
    "eligibility": "#2A9D8F",
    "intervention": "#F4A261",
}
MITIGATION_STATUS_COLORS = {
    "unmitigated": "#dc2626",
    "partially_mitigated": "#d97706",
    "adequately_mitigated": "#16a34a",
}

_EDITOR_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="editor-job")
_EDITOR_JOBS: dict[str, dict[str, Any]] = {}
_EDITOR_JOBS_LOCK = Lock()
DEFAULT_CHUNK_CHARS = 14000
DEFAULT_OVERLAP_CHARS = 1200
LEGACY_DEFAULT_SAVED_PROJECT = "Default Project"
UNASSIGNED_SAVED_PROJECT_LABEL = "Unassigned Analyses"


def main() -> None:
    from protocol_risk_analyzer.frontend.app import main as _main

    _main()


def _truncate_text_for_label(value: object, max_chars: int) -> str:
    from protocol_risk_analyzer.ui_helpers import _truncate_text_for_label as _shared_truncate_text_for_label

    return _shared_truncate_text_for_label(value, max_chars)


def _render_ai_assistant_popup(result: AnalysisResult, analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer.assistant_popup import render_ai_assistant_popup

    render_ai_assistant_popup(
        result,
        analysis_state,
        saved_root,
        _sync_saved_analysis_snapshot,
    )


def _run_analysis_with_progress(
    input_path: Path,
    run_dir: Path,
    model: str,
) -> tuple[str, Any, Any] | None:
    progress_bar = st.progress(0, text="Preparing analysis...")
    progress_note = st.empty()
    progress_note.caption("Upload complete. Starting protocol analysis...")

    def on_progress(
        processed_jobs: int,
        total_jobs: int,
        progress_ratio: float,
        eta_seconds: float,
        message: str | None = None,
    ) -> None:
        percent = int(round(progress_ratio * 100))
        phase_text = message or "Running analysis tasks..."
        progress_bar.progress(
            percent,
            text=(
                f"{phase_text} | {processed_jobs}/{total_jobs} ({percent}%)"
                if total_jobs > 0
                else phase_text
            ),
        )
        if processed_jobs <= 0:
            progress_note.caption("Estimated time remaining: Calculating...")
        elif processed_jobs >= total_jobs:
            progress_note.caption("Estimated time remaining: 00:00")
        elif eta_seconds <= 0:
            progress_note.caption("Estimated time remaining: Calculating...")
        elif "judge" in phase_text.lower() or "summary" in phase_text.lower():
            progress_note.caption(
                f"Estimated time remaining: {_format_eta(eta_seconds)} "
                "(AI synthesis phase can take longer per step)"
            )
        elif processed_jobs >= max(total_jobs - 4, 1):
            progress_note.caption(
                f"Estimated time remaining: {_format_eta(eta_seconds)} "
                "(finalizing aggregation and summary...)"
            )
        else:
            progress_note.caption(f"Estimated time remaining: {_format_eta(eta_seconds)}")

    with st.spinner("Analyzing protocol... this can take a few minutes for large files."):
        try:
            chosen_model, stats, outputs = run_protocol_analysis(
                input_path=input_path,
                output_dir=run_dir,
                model=model or None,
                chunk_chars=DEFAULT_CHUNK_CHARS,
                overlap_chars=DEFAULT_OVERLAP_CHARS,
                progress_callback=on_progress,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            st.session_state["analysis_error"] = str(exc)
            st.session_state.pop("analysis_state", None)
            progress_bar.empty()
            progress_note.empty()
            return None
    progress_bar.progress(100, text="Analysis complete (100%)")
    progress_note.caption("Estimated time remaining: 00:00")
    st.session_state["analysis_error"] = None
    return chosen_model, stats, outputs


def _build_analysis_state_from_outputs(
    input_path: Path,
    run_dir: Path,
    chosen_model: str,
    stats: Any,
    outputs: Any,
    saved_analysis_id: str,
    saved_analysis_title: str,
    saved_analysis_project: str,
) -> dict[str, Any]:
    state = {
        "chosen_model": chosen_model,
        "analyzed_pages": stats.analyzed_pages,
        "total_pages": stats.total_pages,
        "chunk_count": stats.chunk_count,
        "input_path": str(input_path),
        "run_dir": str(run_dir),
        "json_text": outputs.analysis_json_path.read_text(encoding="utf-8"),
        "md_text": outputs.report_md_path.read_text(encoding="utf-8"),
        "pdf_bytes": outputs.report_pdf_path.read_bytes(),
        "editor_state": None,
        "accepted_risk_ids": [],
        "accepted_risk_notes": {},
        "chat_threads": [],
        "chat_history": [],
        "chat_selected_thread_id": "",
        "revision_session": _empty_revision_session_state(),
        "saved_analysis_project": _normalize_saved_project_name(saved_analysis_project),
        "analysis_view": "Dashboard",
    }
    if saved_analysis_id.strip():
        state["saved_analysis_id"] = saved_analysis_id.strip()
    if saved_analysis_title.strip():
        state["saved_analysis_title"] = saved_analysis_title.strip()
    return state


def _render_saved_analyses_sidebar(saved_root: Path) -> None:
    from protocol_risk_analyzer.workspace_sidebar import render_saved_analyses_sidebar as _render_workspace_sidebar

    _render_workspace_sidebar(saved_root)


def _render_saved_analysis_picker(
    saved_root: Path,
    filtered_sessions: list[dict[str, Any]],
    all_sessions: list[dict[str, Any]],
    section_key: str,
    current_project: str,
    project_names: list[str],
) -> None:
    selected_default = str(st.session_state.get("saved_analysis_selected_id", "")).strip()
    for item in filtered_sessions:
        selected_id = str(item.get("id", "")).strip()
        if not selected_id:
            continue
        current_title = _saved_analysis_title(selected_id, all_sessions)
        item_label = current_title

        label_col, menu_col = st.columns([5.2, 1.0], gap="small")
        with label_col:
            if st.button(
                item_label,
                key=f"saved_open_{section_key}_{selected_id}",
                use_container_width=True,
                type="primary" if selected_id == selected_default else "secondary",
            ):
                _sync_current_open_saved_analysis(saved_root)
                if not _open_saved_analysis(saved_root, selected_id):
                    st.error("Unable to open this saved analysis.")
                else:
                    st.rerun()
        with menu_col:
            with st.container(key=f"saved_menu_container_{section_key}_{selected_id}"):
                with st.popover("⋯", use_container_width=True):
                    with st.form(f"saved_rename_form_{section_key}_{selected_id}", clear_on_submit=False):
                        rename_value = st.text_input(
                            "Rename analysis",
                            value=current_title,
                            key=f"saved_title_{section_key}_{selected_id}",
                        )
                        rename_clicked = st.form_submit_button("Rename", use_container_width=True)
                    if rename_clicked:
                        _rename_saved_analysis(saved_root, selected_id, rename_value)
                        analysis_state = st.session_state.get("analysis_state")
                        if isinstance(analysis_state, dict) and analysis_state.get("saved_analysis_id") == selected_id:
                            analysis_state["saved_analysis_title"] = rename_value.strip() or current_title
                            st.session_state["analysis_state"] = analysis_state
                        st.rerun()
                    if current_project:
                        if st.button(
                            "Move to Analysis",
                            key=f"saved_move_to_analysis_{section_key}_{selected_id}",
                            use_container_width=True,
                        ):
                            _set_saved_analysis_project(saved_root, selected_id, "")
                            analysis_state = st.session_state.get("analysis_state")
                            if isinstance(analysis_state, dict) and analysis_state.get("saved_analysis_id") == selected_id:
                                analysis_state["saved_analysis_project"] = ""
                                st.session_state["analysis_state"] = analysis_state
                            st.session_state["saved_analysis_selected_project"] = current_project
                            st.rerun()
                    else:
                        if project_names:
                            with st.container(border=True):
                                move_target = st.selectbox(
                                    "Move to Project",
                                    options=project_names,
                                    index=0,
                                    key=f"saved_move_target_{section_key}_{selected_id}",
                                )
                                if st.button(
                                    "Move",
                                    key=f"saved_move_to_project_{section_key}_{selected_id}",
                                    use_container_width=True,
                                ):
                                    target_project = _normalize_saved_project_name(move_target)
                                    if target_project:
                                        _set_saved_analysis_project(saved_root, selected_id, target_project)
                                        analysis_state = st.session_state.get("analysis_state")
                                        if isinstance(analysis_state, dict) and analysis_state.get("saved_analysis_id") == selected_id:
                                            analysis_state["saved_analysis_project"] = target_project
                                            st.session_state["analysis_state"] = analysis_state
                                        st.session_state["saved_analysis_selected_project"] = target_project
                                        st.rerun()
                    if st.button("Delete", key=f"saved_delete_button_{section_key}_{selected_id}", use_container_width=True):
                        _delete_saved_analysis(saved_root, selected_id)
                        current_state = st.session_state.get("analysis_state")
                        if isinstance(current_state, dict) and current_state.get("saved_analysis_id") == selected_id:
                            st.session_state.pop("analysis_state", None)
                        if st.session_state.get("saved_analysis_selected_id") == selected_id:
                            st.session_state["saved_analysis_selected_id"] = ""
                        st.rerun()


def _project_sessions_sorted(sessions: list[dict[str, Any]], project_name: str) -> list[dict[str, Any]]:
    return sorted(
        [
            item
            for item in sessions
            if _normalize_saved_project_name(item.get("project")) == project_name
        ],
        key=lambda item: (
            str(item.get("created_at", "")).strip()
            or str(item.get("updated_at", "")).strip(),
            str(item.get("title", "")).strip().lower(),
        ),
    )


def _open_saved_analysis(saved_root: Path, selected_id: str) -> bool:
    current_state = st.session_state.get("analysis_state")
    if isinstance(current_state, dict) and str(current_state.get("saved_analysis_id", "")).strip() == selected_id:
        st.session_state["saved_analysis_selected_id"] = selected_id
        st.session_state["saved_analysis_selected_project"] = _normalize_saved_project_name(
            current_state.get("saved_analysis_project")
        )
        return True

    loaded_state = _load_saved_analysis_state(saved_root, selected_id)
    if loaded_state is None:
        return False
    st.session_state["analysis_state"] = loaded_state
    st.session_state["analysis_error"] = None
    st.session_state["ai_assistant_input"] = ""
    st.session_state["ai_assistant_selected_thread_id"] = str(loaded_state.get("chat_selected_thread_id", "")).strip()
    st.session_state["saved_analysis_selected_id"] = selected_id
    st.session_state["saved_analysis_selected_project"] = _normalize_saved_project_name(
        loaded_state.get("saved_analysis_project")
    )
    saved_view = str(loaded_state.get("analysis_view", "Dashboard")).strip()
    if saved_view not in {"Dashboard", "Report", "Mitigate Protocol"}:
        saved_view = "Dashboard"
    st.session_state["analysis_view"] = saved_view
    return True


def _sync_current_open_saved_analysis(saved_root: Path) -> None:
    current_state = st.session_state.get("analysis_state")
    if isinstance(current_state, dict) and str(current_state.get("saved_analysis_id", "")).strip():
        _sync_saved_analysis_snapshot(current_state, saved_root)


def _workspace_project_ui_state(project_names: list[str]) -> tuple[str, str]:
    valid_modes = {"idle", "creating_project", "renaming_project", "confirm_delete_project"}
    mode = str(st.session_state.get("workspace_project_ui_mode", "idle")).strip()
    target_project = _normalize_saved_project_name(st.session_state.get("workspace_project_ui_target"))
    if mode not in valid_modes:
        mode = "idle"
    if mode in {"renaming_project", "confirm_delete_project"} and target_project not in project_names:
        mode = "idle"
        target_project = ""
    if mode in {"idle", "creating_project"}:
        target_project = ""
    st.session_state["workspace_project_ui_mode"] = mode
    st.session_state["workspace_project_ui_target"] = target_project
    return mode, target_project


def _set_workspace_project_ui_state(mode: str, target_project: str = "") -> None:
    st.session_state["workspace_project_ui_mode"] = mode
    st.session_state["workspace_project_ui_target"] = _normalize_saved_project_name(target_project)


def _normalize_saved_project_name(value: object) -> str:
    project = str(value or "").strip()
    if project.lower() == LEGACY_DEFAULT_SAVED_PROJECT.lower():
        return ""
    return project


def _saved_project_names(sessions: list[dict[str, Any]], extra_projects: list[str] | None = None) -> list[str]:
    return _ordered_project_names(sessions=sessions, extra_projects=extra_projects or [])


def _ordered_project_names(sessions: list[dict[str, Any]], extra_projects: list[str]) -> list[str]:
    project_time: dict[str, str] = {}
    session_order: list[str] = []
    seen_session_projects: set[str] = set()
    for item in sessions:
        project = _normalize_saved_project_name(item.get("project"))
        if not project:
            continue
        if project not in seen_session_projects:
            seen_session_projects.add(project)
            session_order.append(project)
        timestamp = str(item.get("created_at", "")).strip() or str(item.get("updated_at", "")).strip()
        if not timestamp:
            continue
        current = project_time.get(project)
        if current is None or timestamp < current:
            project_time[project] = timestamp

    ordered_projects: list[str] = []
    seen: set[str] = set()
    for project in extra_projects:
        normalized = _normalize_saved_project_name(project)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_projects.append(normalized)

    session_projects = sorted(project_time.keys(), key=lambda name: (project_time[name], name.lower()))
    for project in session_projects:
        if project not in seen:
            seen.add(project)
            ordered_projects.append(project)
    for project in session_order:
        if project not in seen:
            seen.add(project)
            ordered_projects.append(project)

    projects_with_time = [project for project in ordered_projects if project in project_time]
    projects_without_time = [project for project in ordered_projects if project not in project_time]
    projects_with_time.sort(key=lambda name: (project_time[name], name.lower()))
    return projects_with_time + projects_without_time


def _saved_analysis_option_label(session_id: str, sessions: list[dict[str, Any]]) -> str:
    if not session_id:
        return "Select a saved analysis..."
    for item in sessions:
        if str(item.get("id", "")).strip() != session_id:
            continue
        title = str(item.get("title", "Untitled Analysis")).strip() or "Untitled Analysis"
        updated = str(item.get("updated_at", "")).strip().replace("T", " ")
        input_filename = str(item.get("input_filename", "")).strip()
        if updated:
            return f"{title} · {updated[:16]} · {input_filename}"
        return f"{title} · {input_filename}"
    return session_id


def _saved_analysis_title(session_id: str, sessions: list[dict[str, Any]]) -> str:
    for item in sessions:
        if str(item.get("id", "")).strip() == session_id:
            return str(item.get("title", "Untitled Analysis")).strip() or "Untitled Analysis"
    return "Untitled Analysis"


def _saved_analysis_project(session_id: str, sessions: list[dict[str, Any]]) -> str:
    for item in sessions:
        if str(item.get("id", "")).strip() == session_id:
            return _normalize_saved_project_name(item.get("project"))
    return ""


def _saved_analyses_index_path(saved_root: Path) -> Path:
    return saved_root / "index.json"


def _load_saved_analysis_index(saved_root: Path) -> dict[str, Any]:
    index_path = _saved_analyses_index_path(saved_root)
    if not index_path.exists():
        return {"sessions": [], "projects": []}
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {"sessions": [], "projects": []}
    sessions = raw.get("sessions", [])
    if not isinstance(sessions, list):
        return {"sessions": [], "projects": []}
    raw_projects = raw.get("projects", [])
    if not isinstance(raw_projects, list):
        raw_projects = []
    projects: list[str] = []
    seen_projects: set[str] = set()
    for item in raw_projects:
        normalized = _normalize_saved_project_name(item)
        if not normalized or normalized in seen_projects:
            continue
        seen_projects.add(normalized)
        projects.append(normalized)
    sanitized: list[dict[str, Any]] = []
    for item in sessions:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized["project"] = _normalize_saved_project_name(item.get("project"))
        sanitized.append(normalized)
    return {"sessions": sanitized, "projects": _ordered_project_names(sanitized, projects)}


def _write_saved_analysis_index(saved_root: Path, index: dict[str, Any]) -> None:
    saved_root.mkdir(parents=True, exist_ok=True)
    index_path = _saved_analyses_index_path(saved_root)
    temp = index_path.with_suffix(".tmp")
    temp.write_text(json.dumps(index, indent=2), encoding="utf-8")
    temp.replace(index_path)


def _saved_analysis_dir(saved_root: Path, session_id: str) -> Path:
    return saved_root / session_id


def _create_saved_analysis_snapshot(analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer import saved_analysis_store as store

    store._create_saved_analysis_snapshot(analysis_state, saved_root)


def _sync_saved_analysis_snapshot(analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer import saved_analysis_store as store

    store._sync_saved_analysis_snapshot(analysis_state, saved_root)


def _saved_analysis_metadata(analysis_state: dict[str, Any], created_at: str, updated_at: str) -> dict[str, Any]:
    from protocol_risk_analyzer import saved_analysis_store as store

    return store._saved_analysis_metadata(analysis_state, created_at, updated_at)


def _persist_saved_analysis_files(session_dir: Path, analysis_state: dict[str, Any]) -> Path | None:
    from protocol_risk_analyzer import saved_analysis_store as store

    return store._persist_saved_analysis_files(session_dir, analysis_state)

def _load_saved_analysis_state(saved_root: Path, session_id: str) -> dict[str, Any] | None:
    from protocol_risk_analyzer import saved_analysis_store as store

    return store._load_saved_analysis_state(saved_root, session_id)


def _rename_saved_analysis(saved_root: Path, session_id: str, new_title: str) -> None:
    title = new_title.strip() or "Untitled Analysis"
    session_dir = _saved_analysis_dir(saved_root, session_id)
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            metadata = {}
        metadata["title"] = title
        metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    index = _load_saved_analysis_index(saved_root)
    sessions = []
    for item in index.get("sessions", []):
        if str(item.get("id", "")).strip() == session_id:
            updated = dict(item)
            updated["title"] = title
            updated["updated_at"] = datetime.now().isoformat(timespec="seconds")
            sessions.append(updated)
        else:
            sessions.append(item)
    index["sessions"] = sessions
    _write_saved_analysis_index(saved_root, index)


def _set_saved_analysis_project(saved_root: Path, session_id: str, project_name: str) -> None:
    normalized_project = _normalize_saved_project_name(project_name)
    session_dir = _saved_analysis_dir(saved_root, session_id)
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            metadata = {}
        metadata["project"] = normalized_project
        metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    index = _load_saved_analysis_index(saved_root)
    sessions: list[dict[str, Any]] = []
    for item in index.get("sessions", []):
        if str(item.get("id", "")).strip() == session_id:
            updated = dict(item)
            updated["project"] = normalized_project
            updated["updated_at"] = datetime.now().isoformat(timespec="seconds")
            sessions.append(updated)
        else:
            sessions.append(item)
    index["sessions"] = sessions
    projects = list(index.get("projects", []))
    if normalized_project and normalized_project not in projects:
        projects.append(normalized_project)
    index["projects"] = _ordered_project_names(index.get("sessions", []), [p for p in projects if p])
    _write_saved_analysis_index(saved_root, index)


def _add_saved_project(saved_root: Path, project_name: str) -> None:
    normalized_project = _normalize_saved_project_name(project_name)
    if not normalized_project:
        return
    index = _load_saved_analysis_index(saved_root)
    projects = list(index.get("projects", []))
    if normalized_project not in projects:
        projects.append(normalized_project)
    index["projects"] = _ordered_project_names(index.get("sessions", []), [p for p in projects if p])
    _write_saved_analysis_index(saved_root, index)


def _rename_saved_project(saved_root: Path, old_name: str, new_name: str) -> None:
    old_project = _normalize_saved_project_name(old_name)
    new_project = _normalize_saved_project_name(new_name)
    if not old_project or not new_project:
        return
    if old_project == new_project:
        return
    index = _load_saved_analysis_index(saved_root)
    sessions: list[dict[str, Any]] = []
    for item in index.get("sessions", []):
        updated = dict(item)
        project = _normalize_saved_project_name(updated.get("project"))
        if project == old_project:
            updated["project"] = new_project
            updated["updated_at"] = datetime.now().isoformat(timespec="seconds")
            session_id = str(updated.get("id", "")).strip()
            if session_id:
                session_dir = _saved_analysis_dir(saved_root, session_id)
                metadata_path = session_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError, TypeError, ValueError):
                        metadata = {}
                    metadata["project"] = new_project
                    metadata["updated_at"] = updated["updated_at"]
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        sessions.append(updated)
    index["sessions"] = sessions
    projects = [_normalize_saved_project_name(project) for project in index.get("projects", []) if _normalize_saved_project_name(project)]
    deduped_projects: list[str] = []
    for project in projects:
        if project not in deduped_projects:
            deduped_projects.append(project)
    if old_project in deduped_projects:
        old_index = deduped_projects.index(old_project)
        deduped_projects.pop(old_index)
        if new_project and new_project not in deduped_projects:
            deduped_projects.insert(old_index, new_project)
    elif new_project and new_project not in deduped_projects:
        deduped_projects.append(new_project)
    index["projects"] = _ordered_project_names(index.get("sessions", []), deduped_projects)
    _write_saved_analysis_index(saved_root, index)


def _delete_saved_project(saved_root: Path, project_name: str) -> None:
    normalized_project = _normalize_saved_project_name(project_name)
    if not normalized_project:
        return
    updated_at = datetime.now().isoformat(timespec="seconds")
    index = _load_saved_analysis_index(saved_root)
    sessions: list[dict[str, Any]] = []
    for item in index.get("sessions", []):
        updated = dict(item)
        project = _normalize_saved_project_name(updated.get("project"))
        if project == normalized_project:
            updated["project"] = ""
            updated["updated_at"] = updated_at
            session_id = str(updated.get("id", "")).strip()
            if session_id:
                session_dir = _saved_analysis_dir(saved_root, session_id)
                metadata_path = session_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError, TypeError, ValueError):
                        metadata = {}
                    metadata["project"] = ""
                    metadata["updated_at"] = updated_at
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        sessions.append(updated)
    index["sessions"] = sessions
    projects = [_normalize_saved_project_name(project) for project in index.get("projects", [])]
    index["projects"] = _ordered_project_names(
        index.get("sessions", []),
        [project for project in projects if project and project != normalized_project],
    )
    _write_saved_analysis_index(saved_root, index)


def _delete_saved_analysis(saved_root: Path, session_id: str) -> None:
    session_dir = _saved_analysis_dir(saved_root, session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    index = _load_saved_analysis_index(saved_root)
    index["sessions"] = [
        item for item in index.get("sessions", []) if str(item.get("id", "")).strip() != session_id
    ]
    _write_saved_analysis_index(saved_root, index)


def _build_saved_analysis_archive(saved_root: Path, session_id: str) -> bytes | None:
    session_dir = _saved_analysis_dir(saved_root, session_id)
    if not session_dir.exists():
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in session_dir.rglob("*"):
            if not path.is_file():
                continue
            archive.write(path, arcname=path.relative_to(session_dir))
    return buffer.getvalue()


def _render_analysis_results(analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer.analysis_shell import render_analysis_results as _render_analysis_shell

    _render_analysis_shell(analysis_state, saved_root)


def _render_risk_dashboard(result: AnalysisResult, analysis_state: dict[str, Any]) -> None:
    from protocol_risk_analyzer.frontend.dashboard import _render_risk_dashboard as _render_dashboard

    _render_dashboard(result, analysis_state)


def _build_risk_dataframe(result: AnalysisResult) -> pd.DataFrame:
    from protocol_risk_analyzer.frontend.dashboard import _build_risk_dataframe as _build

    return _build(result)


def _filter_risks(
    df: pd.DataFrame,
    selected_domain: str,
    min_score: float,
    query: str,
) -> pd.DataFrame:
    from protocol_risk_analyzer.frontend.dashboard import _filter_risks as _filter

    return _filter(df, selected_domain, min_score, query)


def _resolve_selected_risk_id_from_checkboxes(
    displayed_risk_ids: list[str],
    selected_flags: list[object],
    current_selected_risk_id: str | None,
) -> str | None:
    from protocol_risk_analyzer.frontend.dashboard import _resolve_selected_risk_id_from_checkboxes as _resolve

    return _resolve(displayed_risk_ids, selected_flags, current_selected_risk_id)


def _render_editor_agent_builder(
    result: AnalysisResult,
    analysis_state: dict[str, Any],
) -> None:
    from protocol_risk_analyzer.frontend.mitigate_protocol import render_editor_agent_builder

    render_editor_agent_builder(
        result,
        analysis_state,
        render_editor_job_monitor_fragment=_render_editor_job_monitor_fragment,
    )


def _render_ai_assistant_popup(result: AnalysisResult, analysis_state: dict[str, Any], saved_root: Path) -> None:
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
                            _sync_saved_analysis_snapshot(analysis_state, saved_root)
                            st.session_state["ai_assistant_popup_clear_input_key"] = input_key
                            st.rerun()
            revision_session = _ensure_revision_session_state(analysis_state)
            pending_count = len(revision_session.get("pending_changes", []))
            approved_count = len(revision_session.get("approved_changes", []))
            estimated_edit_count = int(revision_session.get("estimated_edit_count", 0) or 0)
            show_revision_plan = pending_count > 0 or approved_count > 0
            if show_revision_plan:
                with st.container(key="ai_assistant_popup_revision_plan"):
                    st.markdown("<div class='ai-revision-plan-title'>Revision Plan</div>", unsafe_allow_html=True)
                    st.markdown(
                        (
                            "<div class='ai-revision-plan-meta'>"
                            f"<span>Pending: <strong>{pending_count}</strong></span>"
                            f"<span>Approved: <strong>{approved_count}</strong></span>"
                            f"<span>Estimated edits: <strong>{estimated_edit_count}</strong></span>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    action_col_1, action_col_2, action_col_3 = st.columns(3, gap="small")
                    with action_col_1:
                        apply_clicked = st.button(
                            "Apply Changes",
                            key="ai_revision_apply_changes_btn",
                            use_container_width=True,
                            disabled=pending_count == 0,
                        )
                    with action_col_2:
                        undo_clicked = st.button(
                            "Undo Last Change",
                            key="ai_revision_undo_change_btn",
                            use_container_width=True,
                            disabled=(pending_count + approved_count) == 0,
                        )
                    with action_col_3:
                        clear_clicked = st.button(
                            "Clear Revision Plan",
                            key="ai_revision_clear_plan_btn",
                            use_container_width=True,
                            disabled=(pending_count + approved_count) == 0,
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
                        _sync_saved_analysis_snapshot(analysis_state, saved_root)
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
                        _sync_saved_analysis_snapshot(analysis_state, saved_root)
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
                        _sync_saved_analysis_snapshot(analysis_state, saved_root)
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
        )
        st.session_state["ai_assistant_popup_scroll_pending"] = True
        st.rerun()


def _queue_ai_assistant_popup_submit(
    *,
    analysis_state: dict[str, Any],
    saved_root: Path,
    input_key: str,
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
    _sync_saved_analysis_snapshot(analysis_state, saved_root)


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


def _build_chat_exchanges(chat_history: list[dict[str, Any]]) -> list[dict[str, str]]:
    exchanges: list[dict[str, str]] = []
    pending_question = ""
    turn_counter = 0
    for item in chat_history:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role == "user":
            pending_question = content
        elif role == "assistant":
            turn_id = f"turn-{turn_counter}"
            exchanges.append(
                {
                    "turn_id": turn_id,
                    "question": pending_question,
                    "answer": content,
                }
            )
            turn_counter += 1
            pending_question = ""
    if pending_question:
        exchanges.append(
            {
                "turn_id": f"turn-{turn_counter}",
                "question": pending_question,
                "answer": "",
            }
        )
    return exchanges


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


def _submit_ai_assistant_prompt(
    *,
    analysis_state: dict[str, Any],
    saved_root: Path,
    user_prompt: str,
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
    _sync_saved_analysis_snapshot(analysis_state, saved_root)
    return assistant_turn_id


def _reset_ai_assistant_conversation(analysis_state: dict[str, Any]) -> None:
    analysis_state["chat_threads"] = []
    analysis_state["chat_history"] = []
    analysis_state["chat_selected_thread_id"] = ""
    analysis_state["revision_session"] = _empty_revision_session_state()
    st.session_state["analysis_state"] = analysis_state


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


def _start_editor_job(
    input_path: Path,
    run_dir: Path,
    selected_risks: list[RiskAssessment],
    model: str,
) -> str:
    from protocol_risk_analyzer.backend import editor_service as editor

    return editor._start_editor_job(input_path, run_dir, selected_risks, model)


def _run_editor_job(
    job_id: str,
    input_path: Path,
    run_dir: Path,
    selected_risks: list[RiskAssessment],
    model: str,
) -> None:
    from protocol_risk_analyzer.backend import editor_service as editor

    editor._run_editor_job(job_id, input_path, run_dir, selected_risks, model)


def _read_editor_job(job_id: str | None, run_dir: Path | None = None) -> dict[str, Any] | None:
    from protocol_risk_analyzer.backend import editor_service as editor

    return editor._read_editor_job(job_id, run_dir=run_dir)


def _persist_editor_job(run_dir: Path, job_payload: dict[str, Any]) -> None:
    from protocol_risk_analyzer.backend import editor_service as editor

    editor._persist_editor_job(run_dir, job_payload)


def _read_editor_job_snapshot(run_dir: Path, job_id: str) -> dict[str, Any] | None:
    from protocol_risk_analyzer.backend import editor_service as editor

    return editor._read_editor_job_snapshot(run_dir, job_id)


def _load_editor_result_from_run_dir(
    run_dir: Path,
    chosen_model: str,
    selected_count: int | None,
) -> dict[str, Any] | None:
    md_path = run_dir / "mitigated_protocol.md"
    pdf_path = run_dir / "mitigated_protocol.pdf"
    docx_path = run_dir / "mitigated_protocol.docx"
    patch_log_path = run_dir / "mitigated_protocol_patches.txt"
    required_paths = [md_path, pdf_path, docx_path, patch_log_path]
    if any(not path.exists() for path in required_paths):
        return None
    return {
        "chosen_model": chosen_model,
        "mitigated_md_text": md_path.read_text(encoding="utf-8"),
        "mitigated_pdf_bytes": pdf_path.read_bytes(),
        "mitigated_docx_bytes": docx_path.read_bytes(),
        "patch_log_text": patch_log_path.read_text(encoding="utf-8"),
        "selected_count": selected_count,
    }


def _editor_job_state_path(run_dir: Path, job_id: str) -> Path:
    from protocol_risk_analyzer.backend import editor_service as editor

    return editor._editor_job_state_path(run_dir, job_id)


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clear_editor_job(job_id: str | None, run_dir: Path | None = None) -> None:
    from protocol_risk_analyzer.backend import editor_service as editor

    editor._clear_editor_job(job_id, run_dir=run_dir)


def _render_editor_job_progress(
    editor_job: dict[str, Any],
    progress_bar_slot: Any,
    progress_note_slot: Any,
) -> None:
    from protocol_risk_analyzer.frontend.mitigate_protocol import _render_editor_job_progress as _render

    _render(editor_job, progress_bar_slot, progress_note_slot)


def _render_editor_job_monitor_fragment() -> None:
    from protocol_risk_analyzer.frontend.mitigate_protocol import _render_editor_job_monitor_fragment as _render

    _render()


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
    from protocol_risk_analyzer.ui_helpers import _format_eta as _shared_format_eta

    return _shared_format_eta(seconds)


def _estimate_editor_eta_seconds(elapsed_seconds: float, completed_steps: int, total_steps: int) -> float:
    if completed_steps <= 0:
        return 0.0
    remaining = max(total_steps - completed_steps, 0)
    average_seconds_per_step = elapsed_seconds / completed_steps
    return max(remaining * average_seconds_per_step, 0.0)


def _render_styles() -> None:
    css = """
        <style>
        html, body {
            margin: 0 !important;
            padding: 0 !important;
        }
        button[data-testid="collapsedControl"] {
            display: inline-flex !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        .risk-card {
            background: #f7f9fc;
            border: 1px solid #dce6f2;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        .evidence-card {
            background: #ffffff;
            border-left: 4px solid #4C78A8;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
        }
        [data-testid="stSegmentedControl"] {
            margin-top: 0.35rem;
            margin-bottom: 0.75rem;
        }
        [data-testid="stSegmentedControl"] button {
            border-radius: 0.65rem 0.65rem 0 0 !important;
            padding: 0.55rem 1rem !important;
            font-weight: 600 !important;
            border-bottom: 2px solid transparent !important;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5eaf1;
            border-radius: 12px;
            padding: 0.4rem 0.75rem;
            min-height: 7.2rem;
        }
        .summary-status-metric-card {
            background: #ffffff;
            border: 1px solid #e5eaf1;
            border-radius: 12px;
            padding: 0.4rem 0.75rem;
            min-height: 7.2rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .summary-status-metric-label {
            color: rgb(49, 51, 63);
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.25;
            letter-spacing: -0.015em;
            padding-bottom: 0.15rem;
        }
        .summary-status-metric-value {
            color: rgb(49, 51, 63);
            font-size: 1.6rem;
            font-weight: 400;
            line-height: 1.05;
            letter-spacing: -0.04em;
            white-space: nowrap;
            font-variant-numeric: tabular-nums;
            margin-top: -0.05rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #f5f8fc 100%);
            border-right: 1px solid #dde5f0;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            letter-spacing: 0.02em;
        }
        .workspace-title {
            margin-top: -0.48rem;
            margin-bottom: 0.12rem;
            font-size: 1.95rem;
            font-weight: 700;
            line-height: 1.02;
            color: #1e293b;
            letter-spacing: -0.01em;
        }
        .workspace-subtitle {
            margin: 0 0 0.55rem 0.05rem;
            font-size: 0.8rem;
            color: #64748b;
        }
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 10px !important;
            border: 1px solid #d9e2ee !important;
            background: #ffffff !important;
            font-weight: 600 !important;
            min-height: 2.35rem !important;
            transition: all 120ms ease-in-out;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #aac0de !important;
            box-shadow: 0 1px 6px rgba(30, 64, 175, 0.08);
            transform: translateY(-0.5px);
        }
        [data-testid="stSidebar"] [data-testid="stPopover"] > button svg {
            display: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stPopover"] > div > button [aria-hidden="true"],
        [data-testid="stSidebar"] [data-testid="stPopover"] > div > button [data-testid="stIconMaterial"] {
            display: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stPopover"] > div > button p {
            line-height: 1 !important;
        }
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stSelectbox select {
            border-radius: 10px !important;
        }
        .workspace-section-title {
            margin-top: 0.78rem;
            margin-bottom: 0.28rem;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button,
        [data-testid="stSidebar"] [class*="st-key-saved_open_"] .stButton > button,
        [data-testid="stSidebar"] [class*="st-key-saved_project_rename_btn_"] .stButton > button {
            border-radius: 9px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-weight: 550 !important;
            padding: 0.42rem 0.7rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button {
            min-height: 2.3rem !important;
            font-size: 0.98rem !important;
            font-weight: 600 !important;
            border-width: 1.25px !important;
            position: relative !important;
            padding-left: 1.55rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] {
            margin-bottom: 0rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_"] .stButton > button {
            min-height: 2.02rem !important;
            font-size: 0.9rem !important;
            font-weight: 520 !important;
            padding: 0.3rem 0.58rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_"] .stButton > button p {
            margin: 0 !important;
            width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
            display: block !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button[kind="secondary"],
        [data-testid="stSidebar"] [class*="st-key-saved_open_"] .stButton > button[kind="secondary"] {
            border: 1px solid transparent !important;
            background: transparent !important;
            color: #1f2937 !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button[kind="secondary"] {
            border-color: #d5e1f2 !important;
            background: #f8fbff !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button[kind="secondary"]:hover,
        [data-testid="stSidebar"] [class*="st-key-saved_open_"] .stButton > button[kind="secondary"]:hover {
            background: #f2f6fc !important;
            border-color: #dce6f4 !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button[kind="primary"] {
            background: linear-gradient(to right, #3b82f6 0 0.16rem, #eef4ff 0.16rem 100%) !important;
            border: 1px solid #c6d8f5 !important;
            color: #0f172a !important;
            box-shadow: none !important;
            padding-left: 1.55rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button::before {
            content: "";
            position: absolute;
            left: 0.42rem;
            top: 50%;
            transform: translateY(-50%);
            width: 0.78rem;
            height: 0.54rem;
            border: 1px solid #111827;
            border-radius: 0.07rem;
            background: #d1d5db;
            pointer-events: none;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_project_select_"] .stButton > button::after {
            content: "";
            position: absolute;
            left: 0.48rem;
            top: calc(50% - 0.39rem);
            width: 0.3rem;
            height: 0.15rem;
            border: 1px solid #111827;
            border-bottom: none;
            border-radius: 0.08rem 0.08rem 0 0;
            background: #d1d5db;
            pointer-events: none;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_"] .stButton > button[kind="primary"] {
            background: #f3f7fd !important;
            border: 1px solid #d7e4f6 !important;
            color: #0f172a !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_workspace_project_"] .stButton > button {
            min-height: 1.86rem !important;
            font-size: 0.86rem !important;
            font-weight: 500 !important;
            border: 1px solid #e3ebf7 !important;
            background: #f9fbfe !important;
            color: #334155 !important;
            padding: 0.22rem 0.54rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_workspace_project_"] .stButton > button[kind="secondary"] {
            background: #f9fbfe !important;
            border-color: #e3ebf7 !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_workspace_project_"] .stButton > button[kind="secondary"]:hover {
            background: #f4f8fe !important;
            border-color: #d7e4f6 !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_workspace_project_"] .stButton > button[kind="primary"] {
            background: #eef4fb !important;
            border-color: #d3e0f2 !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_"] {
            margin-bottom: 0.06rem;
        }
        .workspace-analysis-meta {
            margin: -0.08rem 0 0.45rem 0.14rem;
            font-size: 0.72rem;
            color: #6b7280;
        }
        .workspace-project-children {
            margin: 0rem 0 0.2rem 1.05rem;
            padding: 0.02rem 0 0.12rem 0.85rem;
            border-left: 1px solid #d3e0f0;
            background: linear-gradient(90deg, rgba(241, 246, 254, 0.58) 0%, rgba(241, 246, 254, 0) 100%);
            border-radius: 8px;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_workspace_project_"] {
            position: relative;
            margin-left: 0.42rem;
            margin-top: 0rem !important;
            margin-bottom: 0.08rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_workspace_project_"]:first-of-type {
            margin-top: -0.26rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_open_workspace_project_"]::before {
            content: "•";
            position: absolute;
            left: -0.52rem;
            top: 0.63rem;
            color: #8aa0be;
            font-size: 0.68rem;
            line-height: 1;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_menu_"] .stButton > button {
            min-height: 2.05rem !important;
            padding: 0.3rem 0.45rem !important;
            border-radius: 8px !important;
            border-color: #dbe5f2 !important;
            background: #f8fbff !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_menu_container_workspace_project_"] {
            margin-top: 0rem !important;
            margin-bottom: 0.08rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_menu_container_workspace_project_"]:first-of-type {
            margin-top: -0.26rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_menu_container_workspace_project_"] [data-testid="stPopover"] {
            min-height: 1.86rem !important;
            height: 1.86rem !important;
            max-height: 1.86rem !important;
        }
        [data-testid="stSidebar"] [class*="st-key-saved_menu_container_workspace_project_"] [data-testid="stPopover"] > button,
        [data-testid="stSidebar"] [class*="st-key-saved_menu_skip_workspace_project_"] .stButton > button {
            min-height: 1.86rem !important;
            height: 1.86rem !important;
            max-height: 1.86rem !important;
            line-height: 1 !important;
            padding: 0rem 0.45rem !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 8px !important;
            border: 1px solid #e3ebf7 !important;
            background: #f9fbfe !important;
        }
        [data-testid="stSidebar"] [data-testid="stPopover"] > button {
            min-height: 2.3rem !important;
            border-radius: 10px !important;
            border: 1px solid #d8e2ef !important;
        }
        [data-testid="stAppViewContainer"] .main .block-container,
        [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"] {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] .main h1 {
            margin-top: 2.56rem !important;
            padding-top: 0rem !important;
        }
        .workspace-subsection-title {
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
            font-size: 0.74rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #6b7280;
            font-weight: 700;
        }
        .stDownloadButton button {
            border-radius: 10px !important;
            border: 1px solid #dbe4f0 !important;
            background: #f8fbff !important;
        }
        .compact-results-header {
            background: #edf7f0;
            border: 1px solid #cde6d3;
            color: #166534;
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .ai-assistant-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #2563eb 100%);
            color: #eff6ff;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
        }
        .ai-assistant-hero-kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            opacity: 0.82;
            margin-bottom: 0.25rem;
        }
        .ai-assistant-hero-title {
            font-size: 1.4rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }
        .ai-assistant-hero-subtitle {
            font-size: 0.9rem;
            line-height: 1.45;
            color: rgba(239, 246, 255, 0.9);
            max-width: 72ch;
        }
        .ai-assistant-panel-title {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.55rem;
        }
        .ai-assistant-empty-state {
            border: 1px dashed #cbd5e1;
            border-radius: 16px;
            padding: 1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        }
        .ai-assistant-empty-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .ai-assistant-empty-body {
            font-size: 0.92rem;
            color: #475569;
            line-height: 1.45;
        }
        .ai-assistant-selected-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: #e0ecff;
            color: #1d4ed8;
            border: 1px solid #bfd4ff;
            border-radius: 999px;
            padding: 0.25rem 0.6rem;
            font-size: 0.72rem;
            font-weight: 700;
            margin: 0.15rem 0 0.35rem 0;
        }
        .ai-assistant-input-shell {
            margin-top: 0.8rem;
        }
        div[class*="st-key-ai_assistant_popup_shell"] {
            position: fixed;
            right: 1rem;
            bottom: 1rem;
            width: min(390px, calc(100vw - 2rem));
            z-index: 999;
            background: #f1f5f9 !important;
            border: 1px solid #000000 !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18) !important;
            padding: 0.55rem !important;
            height: clamp(430px, 74vh, 620px) !important;
            max-height: calc(100vh - 5.4rem) !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
        }
        div[class*="st-key-ai_assistant_popup_shell_expanded"] {
            position: fixed;
            left: max(21rem, 3vw);
            right: 1.25rem;
            top: 3.5rem;
            bottom: 1rem;
            width: auto;
            z-index: 999;
            background: #f1f5f9 !important;
            border: 1px solid #000000 !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18) !important;
            padding: 0.8rem !important;
            height: auto !important;
            max-height: calc(100vh - 4.5rem) !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] > div,
        div[class*="st-key-ai_assistant_popup_shell_expanded"] > div,
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stVerticalBlock"],
        div[class*="st-key-ai_assistant_popup_shell_expanded"] [data-testid="stVerticalBlock"],
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stVerticalBlock"] > div,
        div[class*="st-key-ai_assistant_popup_shell_expanded"] [data-testid="stVerticalBlock"] > div {
            background: #f1f5f9 !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] > div,
        div[class*="st-key-ai_assistant_popup_shell_expanded"] > div {
            height: 100% !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] > div > [data-testid="stVerticalBlock"],
        div[class*="st-key-ai_assistant_popup_shell_expanded"] > div > [data-testid="stVerticalBlock"] {
            height: 100% !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] textarea {
            border-radius: 12px !important;
        }
        div[class*="st-key-ai_assistant_popup_shell_expanded"] textarea {
            border-radius: 12px !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stButton"] > button {
            min-height: 2.15rem;
            white-space: nowrap !important;
            padding-left: 0.55rem !important;
            padding-right: 0.55rem !important;
        }
        div[class*="st-key-ai_assistant_popup_shell_expanded"] [data-testid="stButton"] > button {
            min-height: 2.15rem;
            white-space: nowrap !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stCaptionContainer"] p {
            color: #475569 !important;
        }
        div[class*="st-key-ai_assistant_popup_shell_expanded"] [data-testid="stCaptionContainer"] p {
            color: #475569 !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stVerticalBlock"],
        div[class*="st-key-ai_assistant_popup_shell_expanded"] [data-testid="stVerticalBlock"] {
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] {
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] > div {
            flex: 0 0 auto !important;
            display: flex !important;
            flex-direction: column !important;
            height: auto !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] > div > [data-testid="stVerticalBlock"] {
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] > div:has(> div[class*="st-key-ai_assistant_popup_header"]) {
            flex: 0 0 auto !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] > div:has(> div[class*="st-key-ai_assistant_popup_messages"]) {
            flex: 1 1 auto !important;
            min-height: 0 !important;
            display: flex !important;
            flex-direction: column !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] > div:has(> div[class*="st-key-ai_assistant_popup_form_shell"]) {
            flex: 0 0 auto !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] > div:has(> div[class*="st-key-ai_assistant_popup_revision_plan"]) {
            flex: 0 0 auto !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] > div:has(> div[class*="st-key-ai_assistant_popup_messages"]) > [data-testid="stVerticalBlock"] {
            flex: 1 1 auto !important;
            height: 100% !important;
        }
        div[class*="st-key-ai_assistant_popup_header"] {
            flex: 0 0 auto !important;
        }
        div[class*="st-key-ai_assistant_popup_intro"] {
            flex: 0 0 auto !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_intro"] [data-testid="stCaptionContainer"] p {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            line-height: 1.25 !important;
        }
        .ai-assistant-popup-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
            margin-bottom: 0.05rem;
        }
        .ai-assistant-popup-protocol {
            font-size: 0.9rem;
            color: #475569;
            line-height: 1.25;
            margin-top: 0.05rem;
            margin-bottom: 0.15rem;
        }
        div[class*="st-key-ai_assistant_popup_revision_plan"] {
            background: #ffffff !important;
            border: 1px solid #dbe5f2 !important;
            border-radius: 10px !important;
            padding: 0.45rem 0.55rem !important;
            margin: 0.1rem 0 0.35rem 0 !important;
        }
        .ai-revision-plan-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.05rem;
        }
        .ai-revision-plan-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem 0.65rem;
            font-size: 0.72rem;
            color: #334155;
            margin-bottom: 0.1rem;
        }
        .ai-revision-plan-downloads {
            font-size: 0.74rem;
            color: #166534;
            font-weight: 600;
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }
        div[class*="st-key-ai_assistant_popup_revision_plan"] [data-testid="stButton"] > button {
            min-height: 1.58rem !important;
            font-size: 0.60rem !important;
            line-height: 1.1 !important;
            padding-left: 0.18rem !important;
            padding-right: 0.18rem !important;
            white-space: normal !important;
            text-align: center !important;
        }
        .ai-revision-plan-note {
            font-size: 0.68rem;
            color: #475569;
            line-height: 1.2;
            margin-top: 0.15rem;
        }
        div[class*="st-key-ai_assistant_popup_body"] div[class*="st-key-ai_assistant_popup_messages"] {
            flex: 1 1 auto !important;
            display: block !important;
            width: 100% !important;
            height: auto !important;
            margin-top: 0 !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
            padding-right: 0.25rem;
            margin-bottom: 0.35rem;
            scrollbar-gutter: stable;
            overscroll-behavior: contain;
            border-top: 1px solid transparent;
            border-bottom: 1px solid transparent;
        }
        div[class*="st-key-ai_assistant_popup_body"] div[class*="st-key-ai_assistant_popup_messages"] > div {
            display: block !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] div[class*="st-key-ai_assistant_popup_messages"] [data-testid="stVerticalBlock"] {
            min-height: 0 !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
            justify-content: flex-start !important;
            align-content: flex-start !important;
            align-items: stretch !important;
        }
        div[class*="st-key-ai_assistant_popup_body"] div[class*="st-key-ai_assistant_popup_messages"] [data-testid="stVerticalBlock"] > div {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] {
            flex: 0 0 auto !important;
            width: 100% !important;
            align-self: stretch !important;
            margin-top: auto !important;
            z-index: 3;
            background: #f1f5f9;
            padding-top: 0.3rem;
            padding-bottom: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stForm"] {
            margin-bottom: 0 !important;
            width: 100% !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stForm"] > div {
            width: 100% !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stHorizontalBlock"] {
            align-items: center !important;
            width: 100% !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stHorizontalBlock"] > div:last-child {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] textarea {
            border-radius: 10px !important;
            min-height: 6rem !important;
            max-height: 8rem !important;
            line-height: 1.35 !important;
            resize: none !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stTextArea"] textarea::-webkit-resizer {
            display: none !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="InputInstructions"] {
            display: none !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stFormSubmitButton"] > button {
            min-height: 2.5rem !important;
            width: 2.5rem !important;
            min-width: 2.5rem !important;
            max-width: 2.5rem !important;
            border-radius: 10px !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            margin: 0.35rem auto 0 auto !important;
            display: block !important;
        }
        div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stFormSubmitButton"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            height: 100% !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stButton"],
        div[class*="st-key-ai_assistant_popup_shell_expanded"] [data-testid="stButton"] {
            margin-right: 0 !important;
        }
        div[class*="st-key-ai_assistant_popup_shell_expanded"] [data-testid="stHorizontalBlock"] {
            gap: 0.35rem !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stHorizontalBlock"] {
            gap: 0.35rem !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stChatMessage"] p,
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stChatMessage"] li,
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stChatMessage"] h1,
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stChatMessage"] h2,
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stChatMessage"] h3,
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stChatMessage"] h4 {
            color: #0f172a !important;
        }
        div[class*="st-key-ai_assistant_popup_shell"] [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }
        div[class*="st-key-ai_assistant_popup_launcher"] {
            position: fixed;
            right: 1rem;
            bottom: 1rem;
            width: min(170px, calc(100vw - 1.2rem));
            max-width: calc(100vw - 1.2rem);
            z-index: 999;
        }
        div[class*="st-key-ai_assistant_popup_launcher"] button {
            min-height: 2.45rem !important;
        }
        @media (max-width: 980px) {
            div[class*="st-key-ai_assistant_popup_shell_expanded"] {
                left: 0.8rem;
                right: 0.8rem;
                top: 4rem;
                bottom: 0.8rem;
                border-radius: 12px !important;
                padding: 0.65rem !important;
                max-height: calc(100vh - 4.8rem) !important;
            }
        }
        @media (max-width: 760px) {
            div[class*="st-key-ai_assistant_popup_shell"] {
                left: 0.5rem;
                right: 0.5rem;
                width: auto !important;
                bottom: 0.5rem;
                max-height: calc(100vh - 5rem) !important;
            }
            div[class*="st-key-ai_assistant_popup_launcher"] {
                right: 0.5rem;
                left: auto;
                bottom: 0.5rem;
                width: min(170px, calc(100vw - 1rem));
                max-width: calc(100vw - 1rem);
            }
        }
        @media (max-height: 760px) {
            div[class*="st-key-ai_assistant_popup_shell"] {
                height: calc(100vh - 5rem) !important;
                max-height: calc(100vh - 5rem) !important;
                border-radius: 12px !important;
                padding: 0.5rem !important;
            }
            div[class*="st-key-ai_assistant_popup_shell_expanded"] {
                top: 4rem;
                bottom: 0.5rem;
                left: max(17rem, 1vw);
                border-radius: 12px !important;
                padding: 0.6rem !important;
            }
            div[class*="st-key-ai_assistant_popup_body"] div[class*="st-key-ai_assistant_popup_messages"] {
                min-height: 100px !important;
            }
            div[class*="st-key-ai_assistant_popup_form_shell"] [data-testid="stFormSubmitButton"] > button {
                min-height: 2.6rem !important;
            }
        }
        [data-testid="stChatMessage"] {
            border-radius: 10px !important;
            padding: 0.15rem 0 !important;
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stAvatarUser"]) {
            background: transparent !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stAvatarAssistant"]) {
            background: transparent !important;
        }
        div[class*="st-key-ai_assistant_popup_messages"] [data-testid="stChatMessage"] {
            max-width: 88% !important;
            padding: 0.55rem 0.7rem !important;
            border-radius: 16px !important;
            margin-bottom: 0.45rem !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
        }
        div[class*="st-key-ai_assistant_popup_messages"] [data-testid="stChatMessage"]:has([data-testid="stAvatarUser"]) {
            margin-left: auto !important;
            margin-right: 0 !important;
            background: #fee2e2 !important;
            border-color: #fecaca !important;
            flex-direction: row-reverse !important;
        }
        div[class*="st-key-ai_assistant_popup_messages"] [data-testid="stChatMessage"]:has([data-testid="stAvatarAssistant"]) {
            margin-right: auto !important;
            margin-left: 0 !important;
            background: #f8fafc !important;
            border-color: #dbe4f0 !important;
        }
        div[class*="st-key-ai_assistant_popup_messages"] [data-testid="stChatMessage"]:has([data-testid="stAvatarUser"]) [data-testid="stChatMessageContent"] {
            text-align: left !important;
        }
        .ai-chat-role {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.28rem;
        }
        .ai-assistant-typing-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            min-height: 1.2rem;
            padding: 0.05rem 0;
        }
        .ai-assistant-typing-dot {
            width: 0.46rem;
            height: 0.46rem;
            border-radius: 50%;
            background: #94a3b8;
            opacity: 0.32;
            animation: ai-assistant-typing-pulse 1.2s infinite ease-in-out;
        }
        .ai-assistant-typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        .ai-assistant-typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes ai-assistant-typing-pulse {
            0%, 80%, 100% {
                opacity: 0.28;
                transform: scale(0.82);
            }
            40% {
                opacity: 1;
                transform: scale(1);
            }
        }
        [data-testid="stChatMessage"] p {
            margin-bottom: 0.45rem;
        }
        [data-testid="stChatMessage"] ul,
        [data-testid="stChatMessage"] ol {
            margin-top: 0.25rem;
            margin-bottom: 0.45rem;
        }
        [data-testid="stChatMessage"] h1,
        [data-testid="stChatMessage"] h2,
        [data-testid="stChatMessage"] h3,
        [data-testid="stChatMessage"] h4 {
            margin-top: 0.35rem;
        }
        .st-key-editor_job_monitor [data-testid="stProgressBar"] {
            margin-bottom: 0.2rem !important;
        }
        .status-reason-chip {
            display: inline-flex;
            align-items: flex-start;
            gap: 0.45rem;
            margin-bottom: 0.55rem;
            background: #f8fbff;
            border: 1px solid #dbe7f5;
            border-radius: 10px;
            padding: 0.45rem 0.65rem;
            color: #334155;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        .status-dot-inline {
            width: 0.55rem;
            height: 0.55rem;
            margin-top: 0.32rem;
            border-radius: 50%;
            background: #4c78a8;
            flex: 0 0 auto;
        }
        .status-reason-chip.status-unmitigated {
            border-color: #fecaca;
            background: #fef2f2;
            color: #991b1b;
        }
        .status-reason-chip.status-partially-mitigated {
            border-color: #fed7aa;
            background: #fff7ed;
            color: #9a3412;
        }
        .status-reason-chip.status-adequately-mitigated {
            border-color: #bbf7d0;
            background: #f0fdf4;
            color: #166534;
        }
        .status-dot-inline.status-unmitigated {
            background: #dc2626;
        }
        .status-dot-inline.status-partially-mitigated {
            background: #d97706;
        }
        .status-dot-inline.status-adequately-mitigated {
            background: #16a34a;
        }
        .mitigation-key {
            display: grid;
            gap: 0.5rem;
            margin-top: 0.2rem;
        }
        .mitigation-key-item {
            display: grid;
            grid-template-columns: 0.75rem 1fr;
            gap: 0.55rem;
            align-items: start;
            background: #f8fbff;
            border: 1px solid #dbe7f5;
            border-radius: 10px;
            padding: 0.5rem 0.6rem;
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .mitigation-key-dot {
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 50%;
            margin-top: 0.26rem;
            display: inline-block;
        }
        .mitigation-key-dot-unmitigated {
            background: #dc2626;
        }
        .mitigation-key-dot-partial {
            background: #d97706;
        }
        .mitigation-key-dot-adequate {
            background: #16a34a;
        }
        .score-key {
            display: grid;
            gap: 0.45rem;
            margin-top: 0.2rem;
        }
        .score-key-item {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: #f8fbff;
            border: 1px solid #dbe7f5;
            border-radius: 10px;
            padding: 0.45rem 0.6rem;
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .score-key-dot {
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 50%;
            display: inline-block;
            flex: 0 0 auto;
        }
        .score-key-dot.score-low {
            background: #2563eb;
        }
        .score-key-dot.score-moderate {
            background: #d97706;
        }
        .score-key-dot.score-high {
            background: #ef4444;
        }
        .score-key-dot.score-very-high {
            background: #7f1d1d;
        }
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"] {
            --primary-color: #dc2626;
            --primary-color-rgb: 220, 38, 38;
        }
        [data-testid="stDataFrame"] input[type="checkbox"]:checked,
        [data-testid="stDataEditor"] input[type="checkbox"]:checked {
            accent-color: #dc2626 !important;
        }
        [data-testid="stDataFrame"] [role="checkbox"][aria-checked="true"],
        [data-testid="stDataEditor"] [role="checkbox"][aria-checked="true"],
        [data-testid="stDataFrame"] [data-baseweb="checkbox"] input:checked + div,
        [data-testid="stDataEditor"] [data-baseweb="checkbox"] input:checked + div {
            background-color: #dc2626 !important;
            border-color: #dc2626 !important;
            color: #ffffff !important;
            box-shadow: inset 0 0 0 1px #dc2626 !important;
        }
        [data-testid="stDataFrame"] [role="checkbox"][aria-checked="true"] svg,
        [data-testid="stDataEditor"] [role="checkbox"][aria-checked="true"] svg,
        [data-testid="stDataFrame"] [data-baseweb="checkbox"] input:checked + div svg,
        [data-testid="stDataEditor"] [data-baseweb="checkbox"] input:checked + div svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stDataFrame"] [role="checkbox"][aria-checked="true"] path,
        [data-testid="stDataEditor"] [role="checkbox"][aria-checked="true"] path,
        [data-testid="stDataFrame"] [data-baseweb="checkbox"] input:checked + div path,
        [data-testid="stDataEditor"] [data-baseweb="checkbox"] input:checked + div path {
            stroke: #ffffff !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def _truncate_text_for_label(value: object, max_chars: int) -> str:
    from protocol_risk_analyzer.ui_helpers import _truncate_text_for_label as _shared_truncate_text_for_label

    return _shared_truncate_text_for_label(value, max_chars)


def _render_ai_assistant_popup(result: AnalysisResult, analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer.assistant_popup import render_ai_assistant_popup

    render_ai_assistant_popup(
        result,
        analysis_state,
        saved_root,
        _sync_saved_analysis_snapshot,
    )


if __name__ == "__main__":
    main()
