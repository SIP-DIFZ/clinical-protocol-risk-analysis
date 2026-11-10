from pathlib import Path

from protocol_risk_analyzer.analysis import analyze_protocol
from protocol_risk_analyzer.chunking import DocumentChunk
from protocol_risk_analyzer.models import EvidenceSnippet, RiskAssessment, RiskDomain


class _FakeLlm:
    def __init__(self) -> None:
        self.judge_calls = 0
        self.summary_calls = 0

    def extract_chunk_health_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return [
            RiskAssessment(
                title=f"Health risk {chunk.chunk_id}",
                domain=RiskDomain.HEALTH,
                category="safety",
                likelihood=3,
                impact=4,
                confidence=0.8,
                rationale="Health monitoring may be insufficient for adverse event detection.",
                mitigations=["Increase monitoring visits"],
                factor_triggers=["Limited long-term safety information"],
                evidence=[
                    EvidenceSnippet(
                        excerpt="Monitoring is performed monthly.",
                        page_start=chunk.start_page,
                        page_end=chunk.end_page,
                    )
                ],
            )
        ]

    def extract_chunk_financial_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return [
            RiskAssessment(
                title=f"Financial risk {chunk.chunk_id}",
                domain=RiskDomain.FINANCIAL,
                category="operations",
                likelihood=4,
                impact=3,
                confidence=0.7,
                rationale="Visit burden may increase dropout and hurt enrollment economics.",
                mitigations=["Reduce visit burden"],
                factor_triggers=["Many required visits"],
                evidence=[
                    EvidenceSnippet(
                        excerpt="Participants attend weekly clinic visits.",
                        page_start=chunk.start_page,
                        page_end=chunk.end_page,
                    )
                ],
            )
        ]

    def extract_chunk_ethical_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return [
            RiskAssessment(
                title=f"Ethical risk {chunk.chunk_id}",
                domain=RiskDomain.ETHICAL,
                category="informed consent",
                likelihood=3,
                impact=4,
                confidence=0.75,
                rationale="Consent comprehension may be limited for some participants.",
                mitigations=["Improve consent process"],
                factor_triggers=["Adequacy of informed consent"],
                evidence=[
                    EvidenceSnippet(
                        excerpt="Complex consent language may affect understanding.",
                        page_start=chunk.start_page,
                        page_end=chunk.end_page,
                    )
                ],
            )
        ]

    def extract_chunk_eligibility_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return [
            RiskAssessment(
                title=f"Eligibility risk {chunk.chunk_id}",
                domain=RiskDomain.ELIGIBILITY,
                category="eligibility criteria",
                likelihood=4,
                impact=4,
                confidence=0.78,
                rationale="Restrictive criteria may reduce enrollment feasibility.",
                mitigations=["Revisit exclusion criteria"],
                factor_triggers=["Long lists of exclusion criteria"],
                evidence=[
                    EvidenceSnippet(
                        excerpt="Multiple restrictive exclusions are listed.",
                        page_start=chunk.start_page,
                        page_end=chunk.end_page,
                    )
                ],
            )
        ]

    def extract_chunk_intervention_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return [
            RiskAssessment(
                title=f"Intervention risk {chunk.chunk_id}",
                domain=RiskDomain.INTERVENTION,
                category="intervention complexity",
                likelihood=4,
                impact=5,
                confidence=0.82,
                rationale="Complex intervention schedule may increase implementation risk.",
                mitigations=["Provide administration checklists"],
                factor_triggers=["Complex administration schedules"],
                evidence=[
                    EvidenceSnippet(
                        excerpt="Dosing schedule includes frequent modifications.",
                        page_start=chunk.start_page,
                        page_end=chunk.end_page,
                    )
                ],
            )
        ]

    def judge_aggregate_risks(self, risks: list[RiskAssessment]) -> list[RiskAssessment]:
        self.judge_calls += 1
        return risks

    def summarize(self, risks: list[RiskAssessment]):
        self.summary_calls += 1
        return type(
            "Summary",
            (),
            {
                "executive_summary": "Summary of analyzed health, financial, ethical, eligibility, and intervention risks.",
                "key_assumptions": ["Assumes source text is complete."],
            },
        )()


def test_analyze_protocol_runs_multi_agent_extraction() -> None:
    llm = _FakeLlm()
    chunks = [
        DocumentChunk(chunk_id=1, start_page=1, end_page=2, text="chunk1"),
        DocumentChunk(chunk_id=2, start_page=3, end_page=4, text="chunk2"),
    ]
    progress_events: list[tuple[int, int]] = []

    def progress_callback(processed: int, total: int, _ratio: float, _eta: float) -> None:
        progress_events.append((processed, total))

    result = analyze_protocol(
        source_path=Path("protocol.pdf"),
        chunks=chunks,
        llm=llm,
        progress_callback=progress_callback,
        max_workers=2,
    )

    assert len(result.risks) == 10
    assert llm.judge_calls == 1
    assert llm.summary_calls == 1
    assert progress_events[0] == (0, 18)
    assert progress_events[-1] == (18, 18)
