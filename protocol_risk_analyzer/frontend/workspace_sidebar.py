from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def render_saved_analyses_sidebar(saved_root: Path) -> None:
    from protocol_risk_analyzer.backend import persistence_service as store

    st.markdown('<div class="workspace-title">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-subtitle">Organize projects and analyses</div>', unsafe_allow_html=True)
    index = store._load_saved_analysis_index(saved_root)
    sessions = sorted(
        index.get("sessions", []),
        key=lambda item: str(item.get("updated_at", "")),
        reverse=True,
    )
    if st.button("New Analysis", use_container_width=True):
        st.session_state.pop("analysis_state", None)
        st.session_state.pop("analysis_error", None)
        st.session_state["analysis_view"] = "Dashboard"
        st.session_state["saved_analysis_selected_id"] = ""
        st.session_state["ai_assistant_input"] = ""
        st.session_state["ai_assistant_selected_thread_id"] = ""
        st.rerun()

    project_names = store._saved_project_names(sessions, index.get("projects", []))
    mode, mode_target_project = store._workspace_project_ui_state(project_names)
    selected_project = store._normalize_saved_project_name(st.session_state.get("saved_analysis_selected_project"))
    if selected_project and selected_project not in project_names:
        selected_project = ""
    st.session_state["saved_analysis_selected_project"] = selected_project
    mode_error = str(st.session_state.get("workspace_project_ui_error", "")).strip()

    if mode == "creating_project":
        st.markdown("Project Name")
        new_project_name = st.text_input(
            "Project Name",
            value=str(st.session_state.get("workspace_new_project_name", "")).strip(),
            key="workspace_new_project_name",
            label_visibility="collapsed",
        )
        cancel_col, confirm_col = st.columns(2, gap="small")
        with cancel_col:
            if st.button("Cancel", key="workspace_cancel_project_btn", use_container_width=True):
                st.session_state["workspace_project_ui_error"] = ""
                store._set_workspace_project_ui_state("idle")
                st.rerun()
        with confirm_col:
            if st.button("Confirm", key="workspace_create_project_btn", use_container_width=True):
                created_project = store._normalize_saved_project_name(new_project_name)
                project_exists = any(existing.lower() == created_project.lower() for existing in project_names)
                if not created_project:
                    st.session_state["workspace_project_ui_error"] = "Project name is required."
                elif project_exists:
                    st.session_state["workspace_project_ui_error"] = "A project with that name already exists."
                else:
                    store._add_saved_project(saved_root, created_project)
                    st.session_state["saved_analysis_selected_project"] = created_project
                    st.session_state["workspace_project_ui_error"] = ""
                    store._set_workspace_project_ui_state("idle")
                    st.rerun()
    elif mode == "idle":
        if st.button("New Project", key="workspace_new_project_btn", use_container_width=True):
            st.session_state["workspace_project_ui_error"] = ""
            st.session_state["workspace_new_project_name"] = ""
            store._set_workspace_project_ui_state("creating_project")
            st.rerun()
    if mode_error:
        st.caption(mode_error)

    st.markdown('<div class="workspace-section-title">Projects</div>', unsafe_allow_html=True)
    if not project_names:
        st.caption("No projects yet.")
    else:
        for project_index, project_name in enumerate(project_names):
            project_count = sum(
                1 for item in sessions if store._normalize_saved_project_name(item.get("project")) == project_name
            )
            is_selected = selected_project == project_name
            safe_project_key = f"{project_index}_{store._safe_key(project_name)}"

            if mode == "renaming_project" and mode_target_project == project_name:
                rename_value = st.text_input(
                    "Rename Project",
                    value=project_name,
                    key=f"workspace_project_rename_value_{safe_project_key}",
                    label_visibility="collapsed",
                )
                save_col, cancel_col = st.columns(2, gap="small")
                with save_col:
                    if st.button(
                        "Save",
                        key=f"workspace_project_rename_save_{safe_project_key}",
                        use_container_width=True,
                    ):
                        target_name = store._normalize_saved_project_name(rename_value)
                        project_exists = any(
                            existing.lower() == target_name.lower()
                            for existing in project_names
                            if existing != project_name
                        )
                        if not target_name:
                            st.session_state["workspace_project_ui_error"] = "Project name is required."
                        elif project_exists:
                            st.session_state["workspace_project_ui_error"] = "A project with that name already exists."
                        else:
                            store._rename_saved_project(saved_root, project_name, target_name)
                            if selected_project == project_name:
                                st.session_state["saved_analysis_selected_project"] = target_name
                            current_state = st.session_state.get("analysis_state")
                            if isinstance(current_state, dict):
                                current_project = store._normalize_saved_project_name(current_state.get("saved_analysis_project"))
                                if current_project == project_name:
                                    current_state["saved_analysis_project"] = target_name
                                    st.session_state["analysis_state"] = current_state
                            st.session_state["workspace_project_ui_error"] = ""
                            store._set_workspace_project_ui_state("idle")
                            st.rerun()
                with cancel_col:
                    if st.button(
                        "Cancel",
                        key=f"workspace_project_rename_cancel_{safe_project_key}",
                        use_container_width=True,
                    ):
                        st.session_state["workspace_project_ui_error"] = ""
                        store._set_workspace_project_ui_state("idle")
                        st.rerun()
                continue

            project_label = (
                f"{project_name} ({project_count})"
                if project_count > 0
                else project_name
            )
            project_label_col, project_menu_col = st.columns([5.2, 1.0], gap="small")
            with project_label_col:
                if st.button(
                    project_label,
                    key=f"saved_project_select_{safe_project_key}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    if is_selected:
                        st.session_state["saved_analysis_selected_project"] = ""
                        st.rerun()
                    store._sync_current_open_saved_analysis(saved_root)
                    st.session_state["saved_analysis_selected_project"] = project_name
                    project_sessions = store._project_sessions_sorted(
                        sessions=sessions,
                        project_name=project_name,
                    )
                    if project_sessions:
                        first_analysis_id = str(project_sessions[0].get("id", "")).strip()
                        if first_analysis_id:
                            opened = store._open_saved_analysis(saved_root, first_analysis_id)
                            if not opened:
                                st.error("Unable to open the first analysis in this project.")
                                st.session_state["saved_analysis_selected_id"] = ""
                        else:
                            st.session_state["saved_analysis_selected_id"] = ""
                    else:
                        store._sync_current_open_saved_analysis(saved_root)
                        st.session_state.pop("analysis_state", None)
                        st.session_state.pop("analysis_error", None)
                        st.session_state["analysis_view"] = "Dashboard"
                        st.session_state["saved_analysis_selected_id"] = ""
                        st.session_state["ai_assistant_input"] = ""
                        st.session_state["ai_assistant_selected_thread_id"] = ""
                    st.rerun()
            with project_menu_col:
                with st.popover("⋯", use_container_width=True):
                    if st.button("Rename", key=f"saved_project_rename_menu_{safe_project_key}", use_container_width=True):
                        st.session_state["workspace_project_ui_error"] = ""
                        store._set_workspace_project_ui_state("renaming_project", project_name)
                        st.rerun()
                    if st.button("Delete", key=f"saved_project_delete_menu_{safe_project_key}", use_container_width=True):
                        st.session_state["workspace_project_ui_error"] = ""
                        store._set_workspace_project_ui_state("confirm_delete_project", project_name)
                        st.rerun()

            if is_selected:
                project_sessions = store._project_sessions_sorted(
                    sessions=sessions,
                    project_name=project_name,
                )
                if project_sessions:
                    render_saved_analysis_picker(
                        saved_root,
                        project_sessions,
                        sessions,
                        section_key=f"workspace_project_{safe_project_key}",
                        current_project=project_name,
                        project_names=project_names,
                    )
                else:
                    st.caption("No analyses in this project yet.")

        if mode == "confirm_delete_project" and mode_target_project:
            st.warning(f'Delete project "{mode_target_project}"?')
            st.caption("Analyses in this project will be moved to Analysis.")
            confirm_col, cancel_col = st.columns(2, gap="small")
            with confirm_col:
                if st.button("Confirm", key="workspace_project_delete_confirm_btn", use_container_width=True):
                    store._delete_saved_project(saved_root, mode_target_project)
                    if selected_project == mode_target_project:
                        st.session_state["saved_analysis_selected_project"] = ""
                    current_state = st.session_state.get("analysis_state")
                    if isinstance(current_state, dict):
                        current_project = store._normalize_saved_project_name(current_state.get("saved_analysis_project"))
                        if current_project == mode_target_project:
                            current_state["saved_analysis_project"] = ""
                            st.session_state["analysis_state"] = current_state
                    st.session_state["workspace_project_ui_error"] = ""
                    store._set_workspace_project_ui_state("idle")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key="workspace_project_delete_cancel_btn", use_container_width=True):
                    st.session_state["workspace_project_ui_error"] = ""
                    store._set_workspace_project_ui_state("idle")
                    st.rerun()

    st.markdown('<div class="workspace-section-title">Unassigned Analysis</div>', unsafe_allow_html=True)
    filtered_sessions = [item for item in sessions if not store._normalize_saved_project_name(item.get("project"))]
    if filtered_sessions:
        render_saved_analysis_picker(
            saved_root,
            filtered_sessions,
            sessions,
            section_key="workspace_analyses",
            current_project="",
            project_names=project_names,
        )
    else:
        st.caption("No analyses in this section yet.")


def render_saved_analysis_picker(
    saved_root: Path,
    filtered_sessions: list[dict[str, Any]],
    all_sessions: list[dict[str, Any]],
    section_key: str,
    current_project: str,
    project_names: list[str],
) -> None:
    from protocol_risk_analyzer import saved_analysis_store as store

    selected_default = str(st.session_state.get("saved_analysis_selected_id", "")).strip()
    for item in filtered_sessions:
        selected_id = str(item.get("id", "")).strip()
        if not selected_id:
            continue
        current_title = store._saved_analysis_title(selected_id, all_sessions)
        item_label = current_title

        label_col, menu_col = st.columns([5.2, 1.0], gap="small")
        with label_col:
            if st.button(
                item_label,
                key=f"saved_open_{section_key}_{selected_id}",
                use_container_width=True,
                type="primary" if selected_id == selected_default else "secondary",
            ):
                store._sync_current_open_saved_analysis(saved_root)
                if not store._open_saved_analysis(saved_root, selected_id):
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
                        store._rename_saved_analysis(saved_root, selected_id, rename_value)
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
                            store._set_saved_analysis_project(saved_root, selected_id, "")
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
                                    target_project = store._normalize_saved_project_name(move_target)
                                    if target_project:
                                        store._set_saved_analysis_project(saved_root, selected_id, target_project)
                                        analysis_state = st.session_state.get("analysis_state")
                                        if isinstance(analysis_state, dict) and analysis_state.get("saved_analysis_id") == selected_id:
                                            analysis_state["saved_analysis_project"] = target_project
                                            st.session_state["analysis_state"] = analysis_state
                                        st.session_state["saved_analysis_selected_project"] = target_project
                                        st.rerun()
                    if st.button("Delete", key=f"saved_delete_button_{section_key}_{selected_id}", use_container_width=True):
                        store._delete_saved_analysis(saved_root, selected_id)
                        current_state = st.session_state.get("analysis_state")
                        if isinstance(current_state, dict) and current_state.get("saved_analysis_id") == selected_id:
                            st.session_state.pop("analysis_state", None)
                        if st.session_state.get("saved_analysis_selected_id") == selected_id:
                            st.session_state["saved_analysis_selected_id"] = ""
                        st.rerun()


__all__ = ["render_saved_analyses_sidebar", "render_saved_analysis_picker"]
