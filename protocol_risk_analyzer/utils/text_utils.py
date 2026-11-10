from protocol_risk_analyzer.editor import _sanitize_xml_text as sanitize_xml_text
from protocol_risk_analyzer.ui_helpers import _truncate_text_for_label


def truncate_label_text(value: object, max_chars: int) -> str:
    return _truncate_text_for_label(value, max_chars)


__all__ = ["sanitize_xml_text", "truncate_label_text"]
