from typing import Any

from protocol_risk_analyzer.ui_helpers import (
    _empty_revision_session_state,
    _ensure_flat_chat_history_state,
    _ensure_revision_session_state,
    _sanitize_accepted_risk_ids,
    _sanitize_accepted_risk_notes,
    _sanitize_chat_history,
    _sanitize_revision_session_state,
)


def empty_revision_session_state() -> dict[str, Any]:
    return _empty_revision_session_state()


__all__ = [
    "_ensure_flat_chat_history_state",
    "_ensure_revision_session_state",
    "_sanitize_accepted_risk_ids",
    "_sanitize_accepted_risk_notes",
    "_sanitize_chat_history",
    "_sanitize_revision_session_state",
    "empty_revision_session_state",
]
