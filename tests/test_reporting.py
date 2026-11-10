from typing import Literal

from protocol_risk_analyzer.models import EvidenceSnippet, RiskAssessment, RiskDomain
from protocol_risk_analyzer.reporting import build_risk_markdown_lines


def _risk(
    status: Literal[
        "unmitigated",
        "partially_mitigated",
        "adequately_mitigated",
    ],
) -> RiskAssessment:
    return RiskAssessment(
        title="Example risk",
        domain=RiskDomain.HEALTH,
        category="Safety",
        likelihood=3,
        impact=4,
        confidence=0.8,
        rationale="A meaningful rationale that satisfies schema requirements.",
        mitigation_status=status,
        mitigations=["Add explicit escalation threshold."],
        factor_triggers=[],
        evidence=[EvidenceSnippet(excerpt="Evidence text.", page_start=1, page_end=1)],
    )


def test_build_risk_markdown_lines_lists_mitigations_without_priority_sections() -> None:
    risk = _risk("unmitigated").model_copy(
        update={
            "mitigations": [
                "Must define explicit stopping criteria for Grade 3 events.",
                "Clarify review cadence for safety committee decisions.",
                "Consider optional refresher training for backup staff.",
            ]
        }
    )
    lines = build_risk_markdown_lines(risk)
    assert "#### Suggested Mitigations" in lines
    assert "- Must define explicit stopping criteria for Grade 3 events." in lines
    assert "- Clarify review cadence for safety committee decisions." in lines
    assert "- Consider optional refresher training for backup staff." in lines
    assert "##### High Priority" not in lines
    assert "##### Medium Priority" not in lines
    assert "##### Nice to Have Enhancements" not in lines


def test_build_risk_markdown_lines_lists_partial_status_mitigations() -> None:
    risk = _risk("partially_mitigated").model_copy(
        update={"mitigations": ["Clarify review cadence for safety committee decisions."]}
    )
    lines = build_risk_markdown_lines(risk)
    assert "#### Suggested Mitigations" in lines
    assert "- Clarify review cadence for safety committee decisions." in lines


def test_build_risk_markdown_lines_lists_adequate_status_mitigations() -> None:
    risk = _risk("adequately_mitigated").model_copy(
        update={
            "mitigations": [
                "Must define immediate escalation criteria.",
                "Increase review cadence for committee oversight.",
            ]
        }
    )
    lines = build_risk_markdown_lines(risk)
    assert "#### Suggested Mitigations" in lines
    assert "- Must define immediate escalation criteria." in lines
    assert "- Increase review cadence for committee oversight." in lines


def test_build_risk_markdown_lines_keeps_full_evidence_excerpt() -> None:
    long_excerpt = "Evidence " + ("text " * 70).strip()
    risk = _risk("unmitigated").model_copy(
        update={
            "evidence": [EvidenceSnippet(excerpt=long_excerpt, page_start=2, page_end=2)],
        }
    )
    lines = build_risk_markdown_lines(risk)
    evidence_lines = [line for line in lines if line.startswith("- **p.2-2:**")]
    assert evidence_lines
    assert long_excerpt in evidence_lines[0]


def test_risk_score_reduces_when_controls_and_mitigations_are_present() -> None:
    base = RiskAssessment(
        title="Residual recruitment risk",
        domain=RiskDomain.HEALTH,
        category="Operational",
        likelihood=4,
        impact=5,
        confidence=0.9,
        rationale="The protocol has a meaningful residual risk that requires close monitoring.",
        mitigation_status="unmitigated",
        mitigations=[],
        factor_triggers=[],
        evidence=[EvidenceSnippet(excerpt="Recruitment will proceed as planned.", page_start=1, page_end=1)],
    )
    partial = base.model_copy(
        update={
            "mitigation_status": "partially_mitigated",
            "mitigations": [
                "Add enrollment review triggers.",
                "Add contingency recruitment planning.",
                "Clarify escalation criteria.",
            ],
        }
    )
    adequate = base.model_copy(
        update={
            "mitigation_status": "adequately_mitigated",
            "mitigations": [
                "Add enrollment review triggers.",
                "Add contingency recruitment planning.",
                "Clarify escalation criteria.",
                "Document rescue pathways for delayed recruitment.",
            ],
        }
    )

    assert partial.risk_score < base.risk_score
    assert adequate.risk_score < partial.risk_score
    assert adequate.risk_score < base.risk_score
