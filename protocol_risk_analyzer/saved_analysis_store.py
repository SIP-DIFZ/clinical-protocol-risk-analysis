from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import streamlit as st

from protocol_risk_analyzer.config import get_default_model
from protocol_risk_analyzer.models import AnalysisResult
from protocol_risk_analyzer.ui_helpers import (
    _ensure_flat_chat_history_state,
    _ensure_revision_session_state,
    _revision_session_for_metadata,
    _sanitize_accepted_risk_notes,
    _sanitize_chat_history,
    _sanitize_revision_session_state,
)

LEGACY_DEFAULT_SAVED_PROJECT = "Default Project"
UNASSIGNED_SAVED_PROJECT_LABEL = "Unassigned Analyses"


def _normalize_saved_project_name(value: object) -> str:
    project = str(value or "").strip()
    if project.lower() == LEGACY_DEFAULT_SAVED_PROJECT.lower():
        return ""
    return project


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower())


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


def _saved_project_names(sessions: list[dict[str, Any]], extra_projects: list[str] | None = None) -> list[str]:
    return _ordered_project_names(sessions=sessions, extra_projects=extra_projects or [])


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


def _saved_analysis_title(session_id: str, sessions: list[dict[str, Any]]) -> str:
    for item in sessions:
        if str(item.get("id", "")).strip() == session_id:
            return str(item.get("title", "Untitled Analysis")).strip() or "Untitled Analysis"
    return "Untitled Analysis"


def _sync_current_open_saved_analysis(saved_root: Path) -> None:
    current_state = st.session_state.get("analysis_state")
    if isinstance(current_state, dict) and str(current_state.get("saved_analysis_id", "")).strip():
        from protocol_risk_analyzer.web_app import _sync_saved_analysis_snapshot

        _sync_saved_analysis_snapshot(current_state, saved_root)


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
    from protocol_risk_analyzer.web_app import _load_saved_analysis_state

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
    if not old_project or not new_project or old_project == new_project:
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


def _delete_saved_analysis(saved_root: Path, session_id: str) -> None:
    session_dir = _saved_analysis_dir(saved_root, session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    index = _load_saved_analysis_index(saved_root)
    index["sessions"] = [
        item for item in index.get("sessions", []) if str(item.get("id", "")).strip() != session_id
    ]
    _write_saved_analysis_index(saved_root, index)


def _create_saved_analysis_snapshot(analysis_state: dict[str, Any], saved_root: Path) -> None:
    session_id = str(analysis_state.get("saved_analysis_id", "")).strip() or f"analysis-{uuid4().hex[:10]}"
    now_iso = datetime.now().isoformat(timespec="seconds")
    title = str(analysis_state.get("saved_analysis_title", "")).strip()
    project = _normalize_saved_project_name(analysis_state.get("saved_analysis_project"))
    if not title:
        input_path = Path(str(analysis_state.get("input_path", "")))
        title = input_path.stem.strip() or "Untitled Analysis"
    analysis_state["saved_analysis_id"] = session_id
    analysis_state["saved_analysis_title"] = title
    analysis_state["saved_analysis_project"] = project

    session_dir = _saved_analysis_dir(saved_root, session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    saved_input_path = _persist_saved_analysis_files(session_dir, analysis_state)
    analysis_state["run_dir"] = str(session_dir)
    if saved_input_path is not None:
        analysis_state["input_path"] = str(saved_input_path)

    metadata_path = session_dir / "metadata.json"
    created_at = now_iso
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            created_at = str(existing.get("created_at", now_iso))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            created_at = now_iso
    metadata = _saved_analysis_metadata(analysis_state, created_at, now_iso)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    index = _load_saved_analysis_index(saved_root)
    sessions = [item for item in index.get("sessions", []) if str(item.get("id", "")) != session_id]
    sessions.append(
        {
            "id": session_id,
            "title": title,
            "project": project,
            "created_at": created_at,
            "updated_at": now_iso,
            "input_filename": str(metadata.get("input_filename", "")),
            "chosen_model": str(metadata.get("chosen_model", "")),
            "risk_count": int(metadata.get("risk_count", 0)),
        }
    )
    index["sessions"] = sessions
    projects = list(index.get("projects", []))
    if project and project not in projects:
        projects.append(project)
    index["projects"] = _ordered_project_names(index.get("sessions", []), projects)
    _write_saved_analysis_index(saved_root, index)


def _sync_saved_analysis_snapshot(analysis_state: dict[str, Any], saved_root: Path) -> None:
    session_id = str(analysis_state.get("saved_analysis_id", "")).strip()
    if not session_id:
        return
    _create_saved_analysis_snapshot(analysis_state, saved_root)


def _saved_analysis_metadata(analysis_state: dict[str, Any], created_at: str, updated_at: str) -> dict[str, Any]:
    json_text = str(analysis_state.get("json_text", ""))
    risk_count = 0
    input_filename = ""
    if json_text:
        try:
            result = AnalysisResult.model_validate_json(json_text)
            risk_count = len(result.risks)
            input_filename = result.metadata.document_name
        except Exception:
            risk_count = 0
    if not input_filename:
        input_filename = Path(str(analysis_state.get("input_path", ""))).name
    return {
        "id": str(analysis_state.get("saved_analysis_id", "")).strip(),
        "title": str(analysis_state.get("saved_analysis_title", "Untitled Analysis")).strip(),
        "project": _normalize_saved_project_name(analysis_state.get("saved_analysis_project")),
        "created_at": created_at,
        "updated_at": updated_at,
        "input_filename": input_filename,
        "chosen_model": str(analysis_state.get("chosen_model", "")),
        "analyzed_pages": int(analysis_state.get("analyzed_pages", 0)),
        "total_pages": int(analysis_state.get("total_pages", 0)),
        "chunk_count": int(analysis_state.get("chunk_count", 0)),
        "accepted_risk_ids": list(analysis_state.get("accepted_risk_ids", [])),
        "accepted_risk_notes": _sanitize_accepted_risk_notes(analysis_state.get("accepted_risk_notes", {})),
        "chat_threads": [],
        "chat_history": _sanitize_chat_history(analysis_state.get("chat_history", [])),
        "chat_selected_thread_id": "",
        "revision_session": _revision_session_for_metadata(analysis_state.get("revision_session")),
        "editor_review": _sanitize_editor_review_state(analysis_state.get("editor_review")),
        "analysis_view": str(analysis_state.get("analysis_view", "Dashboard")),
        "risk_count": risk_count,
    }


def _persist_saved_analysis_files(session_dir: Path, analysis_state: dict[str, Any]) -> Path | None:
    analysis_json_path = session_dir / "analysis.json"
    report_md_path = session_dir / "report.md"
    report_pdf_path = session_dir / "report.pdf"
    analysis_json_path.write_text(str(analysis_state.get("json_text", "")), encoding="utf-8")
    report_md_path.write_text(str(analysis_state.get("md_text", "")), encoding="utf-8")
    pdf_bytes = analysis_state.get("pdf_bytes", b"")
    if isinstance(pdf_bytes, bytes):
        report_pdf_path.write_bytes(pdf_bytes)
    else:
        report_pdf_path.write_bytes(bytes(pdf_bytes))

    source_input_path = Path(str(analysis_state.get("input_path", "")))
    if source_input_path.exists():
        target_name = source_input_path.name
        target_path = session_dir / target_name
        if source_input_path.resolve() != target_path.resolve():
            shutil.copy2(source_input_path, target_path)
        return target_path
    return None


def _load_saved_analysis_state(saved_root: Path, session_id: str) -> dict[str, Any] | None:
    session_dir = _saved_analysis_dir(saved_root, session_id)
    metadata_path = session_dir / "metadata.json"
    analysis_json_path = session_dir / "analysis.json"
    report_md_path = session_dir / "report.md"
    report_pdf_path = session_dir / "report.pdf"
    if not (metadata_path.exists() and analysis_json_path.exists() and report_md_path.exists() and report_pdf_path.exists()):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    input_filename = str(metadata.get("input_filename", "")).strip()
    input_path = session_dir / input_filename if input_filename else session_dir
    from protocol_risk_analyzer import web_app

    editor_state = web_app._load_editor_result_from_run_dir(
        run_dir=session_dir,
        chosen_model=str(metadata.get("chosen_model", "")).strip() or "Unknown",
        selected_count=None,
    )
    state = {
        "saved_analysis_id": session_id,
        "saved_analysis_title": str(metadata.get("title", "Untitled Analysis")).strip() or "Untitled Analysis",
        "saved_analysis_project": _normalize_saved_project_name(metadata.get("project")),
        "chosen_model": str(metadata.get("chosen_model", "")).strip() or get_default_model(),
        "analyzed_pages": _coerce_int(metadata.get("analyzed_pages")) or 0,
        "total_pages": _coerce_int(metadata.get("total_pages")) or 0,
        "chunk_count": _coerce_int(metadata.get("chunk_count")) or 0,
        "input_path": str(input_path),
        "run_dir": str(session_dir),
        "json_text": analysis_json_path.read_text(encoding="utf-8"),
        "md_text": report_md_path.read_text(encoding="utf-8"),
        "pdf_bytes": report_pdf_path.read_bytes(),
        "editor_state": editor_state,
        "accepted_risk_ids": list(metadata.get("accepted_risk_ids", [])),
        "accepted_risk_notes": _sanitize_accepted_risk_notes(metadata.get("accepted_risk_notes", {})),
        "chat_threads": list(metadata.get("chat_threads", [])),
        "chat_history": list(metadata.get("chat_history", [])),
        "chat_selected_thread_id": "",
        "revision_session": _sanitize_revision_session_state(metadata.get("revision_session")),
        "editor_review": _sanitize_editor_review_state(metadata.get("editor_review")),
        "analysis_view": str(metadata.get("analysis_view", "Dashboard")),
    }
    state["chat_history"] = _ensure_flat_chat_history_state(state)
    state["revision_session"] = _ensure_revision_session_state(state)
    return state


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _sanitize_editor_review_state(editor_review: object) -> dict[str, Any]:
    if not isinstance(editor_review, dict):
        return {"proposals": [], "validation_notes": [], "ready": False}
    proposals_raw = editor_review.get("proposals", [])
    proposals: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if isinstance(proposals_raw, list):
        for item in proposals_raw:
            if not isinstance(item, dict):
                continue
            proposal_id = str(item.get("proposal_id", "")).strip()
            if not proposal_id or proposal_id in seen_ids:
                continue
            seen_ids.add(proposal_id)
            status = str(item.get("status", "pending")).strip().lower() or "pending"
            if status not in {"pending", "accepted", "declined", "user_edited"}:
                status = "pending"
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "risk_title": str(item.get("risk_title", "")).strip(),
                    "action": str(item.get("action", "replace")).strip() or "replace",
                    "target_paragraph_id": str(item.get("target_paragraph_id", "")).strip(),
                    "target_label": str(item.get("target_label", "")).strip(),
                    "page_number": item.get("page_number"),
                    "paragraph_number_on_page": item.get("paragraph_number_on_page"),
                    "reason": str(item.get("reason", "")).strip(),
                    "original_text": str(item.get("original_text", "")).strip(),
                    "proposed_text": str(item.get("proposed_text", "")).strip(),
                    "accepted_text": str(item.get("accepted_text", "")).strip(),
                    "status": status,
                }
            )
    validation_notes_raw = editor_review.get("validation_notes", [])
    validation_notes = (
        [str(item).strip() for item in validation_notes_raw if str(item).strip()]
        if isinstance(validation_notes_raw, list)
        else []
    )
    return {
        "proposals": proposals,
        "validation_notes": validation_notes,
        "ready": bool(editor_review.get("ready", False)),
    }


def _editor_job_state_path(run_dir: Path, job_id: str) -> Path:
    return run_dir / f"{job_id}.status.json"


def _persist_editor_job(run_dir: Path, job_payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = _editor_job_state_path(run_dir, str(job_payload.get("job_id", "")))
    snapshot = {
        "job_id": job_payload.get("job_id"),
        "status": job_payload.get("status"),
        "step": int(job_payload.get("step", 0)),
        "total_steps": int(job_payload.get("total_steps", 1)),
        "message": str(job_payload.get("message", "")),
        "eta_seconds": float(job_payload.get("eta_seconds", 0.0)),
        "error": job_payload.get("error"),
        "selected_count": _coerce_int(job_payload.get("selected_count")),
        "chosen_model": job_payload.get("chosen_model"),
    }
    temp_path = state_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(snapshot), encoding="utf-8")
    temp_path.replace(state_path)


def _read_editor_job_snapshot(run_dir: Path, job_id: str) -> dict[str, Any] | None:
    state_path = _editor_job_state_path(run_dir, job_id)
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _clear_editor_job(job_id: str | None, run_dir: Path | None = None) -> None:
    if not job_id or run_dir is None:
        return
    state_path = _editor_job_state_path(run_dir, job_id)
    if state_path.exists():
        state_path.unlink()
