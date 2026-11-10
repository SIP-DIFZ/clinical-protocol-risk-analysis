from __future__ import annotations

from datetime import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4
import traceback

import streamlit as st

from protocol_risk_analyzer.backend.analysis_service import run_protocol_analysis
from protocol_risk_analyzer.config import get_default_model
from protocol_risk_analyzer.frontend.dashboard import render_analysis_results
from protocol_risk_analyzer.frontend.styles import render_styles
from protocol_risk_analyzer.frontend.workspace_sidebar import render_saved_analyses_sidebar
from protocol_risk_analyzer.ui_helpers import _empty_revision_session_state, _format_eta, _is_docx_protocol_upload

_ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis-job")
_ANALYSIS_JOBS: dict[str, dict[str, Any]] = {}
_ANALYSIS_JOBS_LOCK = Lock()


def _start_analysis_job(
    input_path: Path,
    run_dir: Path,
    model: str,
    *,
    saved_analysis_id: str = "",
    saved_analysis_title: str = "",
    saved_analysis_project: str = "",
) -> str:
    job_id = f"analysis-{uuid4().hex}"
    job_payload = {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "input_path": str(input_path),
        "status": "running",
        "step": 0,
        "processed_steps": 0,
        "total_steps": 1,
        "message": "Starting analysis...",
        "eta_seconds": 0.0,
        "error": None,
        "result": None,
        "chosen_model": model,
        "saved_analysis_id": saved_analysis_id,
        "saved_analysis_title": saved_analysis_title,
        "saved_analysis_project": saved_analysis_project,
    }
    with _ANALYSIS_JOBS_LOCK:
        _ANALYSIS_JOBS[job_id] = job_payload
    _ANALYSIS_EXECUTOR.submit(_run_analysis_job, job_id, input_path, run_dir, model)
    return job_id


def _run_analysis_job(job_id: str, input_path: Path, run_dir: Path, model: str) -> None:
    started_at = datetime.now().timestamp()

    def _set_job_progress(step: int, total_steps: int, message: str, eta_seconds: float) -> None:
        job_snapshot: dict[str, Any] | None = None
        with _ANALYSIS_JOBS_LOCK:
            job = _ANALYSIS_JOBS.get(job_id)
            if job is None:
                return
            job["step"] = step
            job["processed_steps"] = step
            job["total_steps"] = max(total_steps, 1)
            job["message"] = message
            job["eta_seconds"] = eta_seconds
            job_snapshot = dict(job)

    def on_progress(
        processed_jobs: int,
        total_jobs: int,
        progress_ratio: float,
        eta_seconds: float,
        message: str | None = None,
    ) -> None:
        del progress_ratio
        cleaned_message = message or "Running analysis tasks..."
        _set_job_progress(processed_jobs, total_jobs, cleaned_message, eta_seconds)

    try:
        chosen_model, stats, outputs = run_protocol_analysis(
            input_path=input_path,
            output_dir=run_dir,
            model=model or None,
            chunk_chars=14000,
            overlap_chars=1200,
            progress_callback=on_progress,
        )
    except Exception as exc:
        job_snapshot = None
        with _ANALYSIS_JOBS_LOCK:
            job = _ANALYSIS_JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["traceback"] = traceback.format_exc()
                job_snapshot = dict(job)
        return

    job_snapshot = None
    with _ANALYSIS_JOBS_LOCK:
        job = _ANALYSIS_JOBS.get(job_id)
        if job is not None:
            job["status"] = "completed"
            job["step"] = job.get("total_steps", 1)
            job["processed_steps"] = job.get("total_steps", 1)
            job["message"] = "Analysis complete."
            job["eta_seconds"] = 0.0
            job["result"] = (chosen_model, stats, outputs)
            job["chosen_model"] = chosen_model
            job["analyzed_pages"] = stats.analyzed_pages
            job["total_pages"] = stats.total_pages
            job["chunk_count"] = stats.chunk_count
            job_snapshot = dict(job)
    del started_at


def _read_analysis_job(job_id: str | None) -> dict[str, Any] | None:
    if not job_id:
        return None
    with _ANALYSIS_JOBS_LOCK:
        job = _ANALYSIS_JOBS.get(job_id)
        if job is None:
            return None
        return dict(job)


def _clear_analysis_job(job_id: str | None) -> None:
    if not job_id:
        return
    with _ANALYSIS_JOBS_LOCK:
        _ANALYSIS_JOBS.pop(job_id, None)


@st.fragment(run_every=1.0)
def _render_analysis_job_monitor_fragment(saved_root: Path) -> None:
    job_id = str(st.session_state.get("analysis_job_id", "")).strip()
    if not job_id:
        return
    job = _read_analysis_job(job_id)
    if job is None:
        return

    with st.container():
        progress_bar = st.empty()
        progress_note = st.empty()
        _render_analysis_job_progress(job, progress_bar, progress_note)
        if job.get("status") == "completed":
            result = job.get("result")
            if isinstance(result, tuple) and len(result) == 3:
                chosen_model, stats, outputs = result
                completed_state = _build_analysis_state_from_outputs(
                    input_path=Path(str(job.get("input_path", ""))),
                    run_dir=Path(str(job.get("run_dir", ""))),
                    chosen_model=chosen_model,
                    stats=stats,
                    outputs=outputs,
                    saved_analysis_id=str(job.get("saved_analysis_id", "")),
                    saved_analysis_title=str(job.get("saved_analysis_title", "")),
                    saved_analysis_project=str(job.get("saved_analysis_project", "")),
                )
                st.session_state["analysis_state"] = completed_state
                _create_saved_analysis_snapshot(st.session_state["analysis_state"], saved_root)
                st.session_state["saved_analysis_selected_id"] = str(completed_state.get("saved_analysis_id", "")).strip()
                st.session_state["saved_analysis_selected_project"] = _normalize_saved_project_name(
                    completed_state.get("saved_analysis_project")
                )
                st.session_state.pop("analysis_job_id", None)
                st.session_state["analysis_error"] = None
                _clear_analysis_job(job_id)
                st.rerun()
        elif job.get("status") == "failed":
            st.session_state["analysis_error"] = str(job.get("error", "Analysis failed."))
            st.session_state.pop("analysis_job_id", None)
            _clear_analysis_job(job_id)
            st.rerun()


def _render_analysis_job_progress(
    analysis_job: dict[str, Any],
    progress_bar_slot,
    progress_note_slot,
) -> None:
    step = int(analysis_job.get("processed_steps", analysis_job.get("step", 0)) or 0)
    total_steps = max(int(analysis_job.get("total_steps", 1)), 1)
    status = str(analysis_job.get("status", "running"))
    message = str(analysis_job.get("message", "Running analysis..."))
    eta_seconds = float(analysis_job.get("eta_seconds", 0.0))
    percent = min(max(int(round((step / total_steps) * 100)), 0), 100)
    if status == "completed":
        progress_bar_slot.progress(100, text=f"Completed | {total_steps}/{total_steps} (100%)")
        progress_note_slot.caption("Estimated time remaining: 00:00")
        return
    if status == "failed":
        progress_bar_slot.empty()
        progress_note_slot.empty()
        return
    progress_bar_slot.progress(
        percent,
        text=f"{message} | {min(max(step, 0), total_steps)}/{total_steps} ({percent}%)",
    )
    if step <= 0 or eta_seconds <= 0:
        progress_note_slot.caption("Estimated time remaining: Calculating...")
    else:
        progress_note_slot.caption(f"Estimated time remaining: {_format_eta(eta_seconds)}")


def main() -> None:
    default_model = get_default_model()
    model = default_model
    st.set_page_config(
        page_title="Clinical Protocol Risk Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_styles()
    st.title("Clinical Protocol Risk Analyzer")
    has_analysis = isinstance(st.session_state.get("analysis_state"), dict)
    if not has_analysis:
        st.caption("Upload a protocol document to assess trial risks and generate mitigated protocol drafts.")

    output_root = Path("web_runs")
    output_root.mkdir(parents=True, exist_ok=True)
    saved_root = output_root / "saved_analyses"
    saved_root.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        render_saved_analyses_sidebar(saved_root)

    _render_analysis_job_monitor_fragment(saved_root)
    analysis_job_id = str(st.session_state.get("analysis_job_id", "")).strip()
    analysis_running = bool(analysis_job_id)

    pending_reanalysis_id = str(st.session_state.get("pending_reanalysis_saved_analysis_id", "")).strip()
    if pending_reanalysis_id:
        st.session_state["pending_reanalysis_saved_analysis_id"] = ""
        existing_state = _load_saved_analysis_state(saved_root, pending_reanalysis_id)
        if existing_state is None:
            st.error("Unable to reanalyze this saved analysis.")
            st.stop()
        source_input_path = Path(str(existing_state.get("input_path", "")))
        if not source_input_path.exists() or not source_input_path.is_file():
            st.error("Unable to reanalyze because the saved input file was not found.")
            st.stop()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / source_input_path.name
        shutil.copy2(source_input_path, input_path)
        chosen_model = str(existing_state.get("chosen_model", model)).strip() or model
        st.session_state["analysis_job_id"] = _start_analysis_job(
            input_path=input_path,
            run_dir=run_dir,
            model=chosen_model,
            saved_analysis_id=pending_reanalysis_id,
            saved_analysis_title=str(existing_state.get("saved_analysis_title", "Untitled Analysis")),
            saved_analysis_project=_normalize_saved_project_name(existing_state.get("saved_analysis_project")),
        )
        st.session_state["analysis_error"] = None
        st.rerun()

    uploaded = None
    if not has_analysis:
        uploaded = st.file_uploader("Upload protocol (.docx)", type=["docx"])

    if not has_analysis and st.button(
        "Analyze Protocol",
        type="primary",
        disabled=uploaded is None or analysis_running,
    ):
        if uploaded is None:
            st.error("Please upload a protocol file first.")
            return
        if not _is_docx_protocol_upload(uploaded.name):
            st.error("Please upload a .docx protocol file.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / uploaded.name
        input_path.write_bytes(uploaded.getvalue())
        st.session_state["analysis_job_id"] = _start_analysis_job(
            input_path=input_path,
            run_dir=run_dir,
            model=model,
        )
        st.session_state["analysis_error"] = None
        st.rerun()

    if st.session_state.get("analysis_error"):
        st.error(f"Analysis failed: {st.session_state['analysis_error']}")

    analysis_state = st.session_state.get("analysis_state")
    if analysis_state:
        _sync_saved_analysis_snapshot(analysis_state, saved_root)
        render_analysis_results(analysis_state, saved_root)
    elif analysis_running:
        st.info("Analysis is running in the background. You can keep working in the workspace while it continues.")


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
                chunk_chars=14000,
                overlap_chars=1200,
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


def _normalize_saved_project_name(value: object) -> str:
    project = str(value or "").strip()
    if project.lower() == "default project":
        return ""
    return project


def _create_saved_analysis_snapshot(analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer.saved_analysis_store import _create_saved_analysis_snapshot as _create

    _create(analysis_state, saved_root)


def _sync_saved_analysis_snapshot(analysis_state: dict[str, Any], saved_root: Path) -> None:
    from protocol_risk_analyzer.saved_analysis_store import _sync_saved_analysis_snapshot as _sync

    _sync(analysis_state, saved_root)


def _load_saved_analysis_state(saved_root: Path, session_id: str) -> dict[str, Any] | None:
    from protocol_risk_analyzer.saved_analysis_store import _load_saved_analysis_state as _load

    return _load(saved_root, session_id)


__all__ = ["main"]
