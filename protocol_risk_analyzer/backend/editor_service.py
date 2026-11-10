from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import traceback
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from protocol_risk_analyzer.app_service import (
    RevisedProtocolPaths,
    run_protocol_editor,
    run_protocol_editor_from_patches,
    run_protocol_editor_proposals,
)
from protocol_risk_analyzer.backend import persistence_service as store
from protocol_risk_analyzer.editor import (
    ProtocolEditPatch,
    ProtocolParagraph,
    RevisedProtocolOutputPaths,
    create_revised_protocol_outputs,
    format_protocol_paragraph_reference,
)
from protocol_risk_analyzer.models import RiskAssessment

_EDITOR_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="editor-job")
_EDITOR_JOBS: dict[str, dict[str, Any]] = {}
_EDITOR_JOBS_LOCK = Lock()


__all__ = [
    "ProtocolParagraph",
    "RevisedProtocolOutputPaths",
    "RevisedProtocolPaths",
    "create_revised_protocol_outputs",
    "_clear_editor_job",
    "_editor_job_state_path",
    "_load_editor_result_from_run_dir",
    "_persist_editor_job",
    "_read_editor_job",
    "_read_editor_job_snapshot",
    "_run_editor_job",
    "_start_editor_job",
    "generate_editor_outputs_from_proposals",
    "run_protocol_editor",
]


def _start_editor_job(
    input_path: Path,
    run_dir: Path,
    selected_risks: list[RiskAssessment],
    model: str,
) -> str:
    job_id = f"editor-{uuid4().hex}"
    job_payload = {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "status": "running",
        "step": 0,
        "processed_steps": 0,
        "total_steps": 1,
        "message": "Starting Editor Agent...",
        "eta_seconds": 0.0,
        "error": None,
        "result": None,
        "proposals": [],
        "selected_count": len(selected_risks),
        "chosen_model": model,
    }
    with _EDITOR_JOBS_LOCK:
        _EDITOR_JOBS[job_id] = job_payload
    _persist_editor_job(run_dir, job_payload)
    _EDITOR_EXECUTOR.submit(
        _run_editor_job,
        job_id,
        input_path,
        run_dir,
        selected_risks,
        model,
    )
    return job_id


def _run_editor_job(
    job_id: str,
    input_path: Path,
    run_dir: Path,
    selected_risks: list[RiskAssessment],
    model: str,
) -> None:
    started_at = datetime.now().timestamp()

    def _set_job_progress(step: int, total_steps: int, message: str) -> None:
        now = datetime.now().timestamp()
        elapsed_seconds = max(now - started_at, 0.0)
        eta_seconds = _estimate_editor_eta_seconds(elapsed_seconds, step, total_steps)
        job_snapshot: dict[str, Any] | None = None
        with _EDITOR_JOBS_LOCK:
            job = _EDITOR_JOBS.get(job_id)
            if job is None:
                return
            job["step"] = step
            job["processed_steps"] = step
            job["total_steps"] = max(total_steps, 1)
            job["message"] = message
            job["eta_seconds"] = eta_seconds
            job_snapshot = dict(job)
        if job_snapshot is not None:
            _persist_editor_job(run_dir, job_snapshot)

    def on_editor_progress(step: int, total_steps: int, message: str) -> None:
        _set_job_progress(step, total_steps, message)

    try:
        _set_job_progress(1, 8, "Loading protocol document...")

        def on_proposals(
            paragraphs: list[ProtocolParagraph],
            patches: list[ProtocolEditPatch],
        ) -> None:
            job_snapshot: dict[str, Any] | None = None
            with _EDITOR_JOBS_LOCK:
                job = _EDITOR_JOBS.get(job_id)
                if job is None:
                    return
                job["proposals"] = _serialize_patch_proposals(paragraphs, patches)
                job_snapshot = dict(job)
            if job_snapshot is not None:
                _persist_editor_job(run_dir, job_snapshot)

        chosen_model, paragraphs, patches, validation_notes = run_protocol_editor_proposals(
            input_path=input_path,
            selected_risks=selected_risks,
            model=model,
            progress_callback=on_editor_progress,
            proposal_callback=on_proposals,
        )
    except Exception as exc:
        job_snapshot = None
        with _EDITOR_JOBS_LOCK:
            job = _EDITOR_JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["traceback"] = traceback.format_exc()
                job_snapshot = dict(job)
        if job_snapshot is not None:
            _persist_editor_job(run_dir, job_snapshot)
        return

    proposals = _serialize_patch_proposals(paragraphs, patches)
    job_snapshot = None
    with _EDITOR_JOBS_LOCK:
        job = _EDITOR_JOBS.get(job_id)
        if job is not None:
            job["status"] = "review_ready"
            job["step"] = job.get("total_steps", 1)
            job["message"] = "Protocol change proposals are ready for review."
            job["eta_seconds"] = 0.0
            job["proposals"] = proposals
            job["validation_notes"] = validation_notes
            job["result"] = None
            job["chosen_model"] = chosen_model
            job_snapshot = dict(job)
    if job_snapshot is not None:
        _persist_editor_job(run_dir, job_snapshot)


def _read_editor_job(job_id: str | None, run_dir: Path | None = None) -> dict[str, Any] | None:
    if not job_id:
        return None
    with _EDITOR_JOBS_LOCK:
        job = _EDITOR_JOBS.get(job_id)
        if job is not None:
            return dict(job)
    if run_dir is None:
        return None
    snapshot = _read_editor_job_snapshot(run_dir, job_id)
    if snapshot is None:
        return None
    if snapshot.get("status") == "completed":
        result = _load_editor_result_from_run_dir(
            run_dir=run_dir,
            chosen_model=str(snapshot.get("chosen_model", "")).strip() or "Unknown",
            selected_count=_coerce_int(snapshot.get("selected_count")),
        )
        if result is None:
            return None
        snapshot["result"] = result
    return snapshot


def _persist_editor_job(run_dir: Path, job_payload: dict[str, Any]) -> None:
    store._persist_editor_job(run_dir, job_payload)


def _read_editor_job_snapshot(run_dir: Path, job_id: str) -> dict[str, Any] | None:
    return store._read_editor_job_snapshot(run_dir, job_id)


def _editor_job_state_path(run_dir: Path, job_id: str) -> Path:
    return store._editor_job_state_path(run_dir, job_id)


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clear_editor_job(job_id: str | None, run_dir: Path | None = None) -> None:
    if job_id:
        with _EDITOR_JOBS_LOCK:
            _EDITOR_JOBS.pop(job_id, None)
    store._clear_editor_job(job_id, run_dir=run_dir)


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


def _estimate_editor_eta_seconds(elapsed_seconds: float, step: int, total_steps: int) -> float:
    if step <= 0:
        return 0.0
    if total_steps <= 0:
        return 0.0
    progress = step / max(total_steps, 1)
    if progress <= 0:
        return 0.0
    estimated_total = elapsed_seconds / progress
    return max(estimated_total - elapsed_seconds, 0.0)


def _serialize_patch_proposals(
    paragraphs: list[ProtocolParagraph],
    patches: list[ProtocolEditPatch],
) -> list[dict[str, Any]]:
    paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
    serialized: list[dict[str, Any]] = []
    for idx, patch in enumerate(patches, start=1):
        paragraph = paragraph_by_id.get(patch.target_paragraph_id)
        original_text = paragraph.text if paragraph is not None else ""
        serialized.append(
            {
                "proposal_id": f"{patch.target_paragraph_id}:{idx}",
                "risk_title": patch.risk_title,
                "action": patch.action,
                "target_paragraph_id": patch.target_paragraph_id,
                "target_label": format_protocol_paragraph_reference(paragraph, patch.target_paragraph_id),
                "page_number": paragraph.page_number if paragraph is not None else None,
                "paragraph_number_on_page": paragraph.paragraph_number_on_page if paragraph is not None else None,
                "reason": patch.reason,
                "original_text": original_text,
                "proposed_text": patch.new_text,
            }
        )
    return serialized


def generate_editor_outputs_from_proposals(
    input_path: Path,
    run_dir: Path,
    proposals: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    accepted_patches: list[ProtocolEditPatch] = []
    for proposal in proposals:
        status = str(proposal.get("status", "pending")).strip().lower()
        if status not in {"accepted", "user_edited"}:
            continue
        text = str(proposal.get("accepted_text", proposal.get("proposed_text", ""))).strip()
        if not text:
            continue
        accepted_patches.append(
            ProtocolEditPatch(
                risk_title=str(proposal.get("risk_title", "Selected mitigation")).strip() or "Selected mitigation",
                action=str(proposal.get("action", "replace")).strip() or "replace",
                target_paragraph_id=str(proposal.get("target_paragraph_id", "")).strip(),
                new_text=text,
                reason=str(proposal.get("reason", "Accepted in review workflow")).strip()
                or "Accepted in review workflow",
            )
        )
    if not accepted_patches:
        raise ValueError("No accepted protocol changes are available to generate the mitigated protocol.")
    chosen_model, outputs = run_protocol_editor_from_patches(
        input_path=input_path,
        output_dir=run_dir,
        patches=accepted_patches,
        model=model,
    )
    return {
        "chosen_model": chosen_model,
        "mitigated_md_text": outputs.mitigated_md_path.read_text(encoding="utf-8"),
        "mitigated_pdf_bytes": outputs.mitigated_pdf_path.read_bytes(),
        "mitigated_docx_bytes": outputs.mitigated_docx_path.read_bytes(),
        "patch_log_text": outputs.patch_log_path.read_text(encoding="utf-8"),
        "selected_count": len(accepted_patches),
    }
