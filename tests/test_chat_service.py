from __future__ import annotations

from types import SimpleNamespace

from protocol_risk_analyzer.chat_service import ProtocolChatService
from protocol_risk_analyzer.io import PageText
from protocol_risk_analyzer.models import AnalysisResult, EvidenceSnippet, ProtocolMetadata, RiskAssessment, RiskDomain


def _sample_analysis() -> AnalysisResult:
    return AnalysisResult(
        metadata=ProtocolMetadata(
            source_path="protocol.pdf",
            document_name="protocol.pdf",
            page_count=4,
            chunk_count=2,
            generated_at_utc=ProtocolMetadata.now_iso(),
        ),
        executive_summary="The protocol has high safety and eligibility concerns.",
        key_assumptions=["Sites will follow safety monitoring procedures."],
        risks=[
            RiskAssessment(
                title="Safety monitoring gap",
                domain=RiskDomain.HEALTH,
                category="Safety",
                likelihood=4,
                impact=5,
                confidence=0.9,
                rationale="Safety monitoring is limited despite a high-risk intervention.",
                evidence=[
                    EvidenceSnippet(
                        excerpt="Safety assessments are only performed monthly.",
                        page_start=2,
                        page_end=2,
                    )
                ],
                mitigations=["Increase monitoring frequency"],
            ),
            RiskAssessment(
                title="Eligibility restriction",
                domain=RiskDomain.ELIGIBILITY,
                category="Eligibility",
                likelihood=3,
                impact=3,
                confidence=0.7,
                rationale="The eligibility criteria are restrictive.",
                evidence=[
                    EvidenceSnippet(
                        excerpt="Participants must meet multiple lab thresholds.",
                        page_start=3,
                        page_end=3,
                    )
                ],
                mitigations=["Broaden criteria"],
            ),
        ],
    )


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class _FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


def test_protocol_chat_service_answers_with_relevant_context() -> None:
    payload = """
    {
      "can_answer": true,
      "answer_markdown": "## Answer\\nThe main concern is safety monitoring.",
      "cited_source_ids": [1, 2],
      "follow_up_questions": ["What monitoring happens after dosing?"]
    }
    """.strip()
    service = ProtocolChatService(client=_FakeClient(payload), model="test-model")
    analysis = _sample_analysis()
    pages = [
        PageText(page_number=1, text="Introduction."),
        PageText(page_number=2, text="Safety assessments are only performed monthly."),
        PageText(page_number=3, text="Participants must meet multiple lab thresholds."),
        PageText(page_number=4, text="Conclusion."),
    ]

    result = service.answer_question(
        analysis=analysis,
        question="Explain the patient safety concerns.",
        history=[{"role": "user", "content": "Summarize the analysis."}],
        pages=pages,
        input_path=None,
    )

    assert result.payload.can_answer is True
    assert "safety monitoring" in result.payload.answer_markdown.lower()
    assert result.sources
    assert service.client.responses.calls
    system_prompt_text = service.client.responses.calls[0]["input"][0]["content"]
    prompt_text = service.client.responses.calls[0]["input"][1]["content"]
    assert "friendly and conversational" in system_prompt_text.lower()
    assert "citations are required only for protocol- or risk-analysis-related answers" in system_prompt_text.lower()
    assert "keep answers tight" in system_prompt_text.lower()
    assert "do not provide evidence excerpts or citations for regular conversational messages" in prompt_text.lower()
    assert "no more than three citations" in prompt_text.lower()
    assert "Explain the patient safety concerns." in prompt_text
    assert "Safety monitoring gap" in prompt_text


def test_protocol_chat_service_refuses_unrelated_questions() -> None:
    payload = """
    {
      "can_answer": false,
      "answer_markdown": "I can only provide answers related to the clinical protocol and risk analysis content provided.",
      "cited_source_ids": [],
      "follow_up_questions": []
    }
    """.strip()
    fake_client = _FakeClient(payload)
    service = ProtocolChatService(client=fake_client, model="test-model")
    analysis = _sample_analysis()
    pages = [PageText(page_number=1, text="Introduction.")]

    result = service.answer_question(
        analysis=analysis,
        question="What is the weather today?",
        history=[],
        pages=pages,
        input_path=None,
    )

    assert result.payload.can_answer is False
    assert "only provide answers related to the clinical protocol" in result.payload.answer_markdown.lower()
    assert result.sources == []
    assert fake_client.responses.calls


def test_protocol_chat_service_allows_friendly_greeting() -> None:
    payload = """
    {
      "can_answer": true,
      "answer_markdown": "Hello! I can help with your protocol and risk-analysis questions.",
      "cited_source_ids": [],
      "follow_up_questions": []
    }
    """.strip()
    fake_client = _FakeClient(payload)
    service = ProtocolChatService(client=fake_client, model="test-model")
    analysis = _sample_analysis()
    pages = [PageText(page_number=1, text="Introduction.")]

    result = service.answer_question(
        analysis=analysis,
        question="Hello",
        history=[],
        pages=pages,
        input_path=None,
    )

    assert result.payload.can_answer is True
    assert "hello!" in result.payload.answer_markdown.lower()
    assert result.payload.cited_source_ids == []
    assert result.payload.follow_up_questions == []
    assert result.sources == []
    assert fake_client.responses.calls


def test_protocol_chat_service_allows_contextual_follow_up_with_history() -> None:
    payload = """
    {
      "can_answer": true,
      "answer_markdown": "## Clarification\\nThe concern is the limited monitoring cadence.",
      "cited_source_ids": [1],
      "follow_up_questions": []
    }
    """.strip()
    fake_client = _FakeClient(payload)
    service = ProtocolChatService(client=fake_client, model="test-model")
    analysis = _sample_analysis()
    pages = [
        PageText(page_number=1, text="Introduction."),
        PageText(page_number=2, text="Safety assessments are only performed monthly."),
    ]

    result = service.answer_question(
        analysis=analysis,
        question="Can you expand on that?",
        history=[
            {"role": "user", "content": "What are the key safety risks?"},
            {"role": "assistant", "content": "Safety monitoring is a key concern."},
        ],
        pages=pages,
        input_path=None,
    )

    assert result.payload.can_answer is True
    assert "clarification" in result.payload.answer_markdown.lower()
    assert fake_client.responses.calls


def test_protocol_chat_service_filters_invalid_and_duplicate_citations() -> None:
    payload = """
    {
      "can_answer": true,
      "answer_markdown": "## Answer\\nThe main concern is safety monitoring and eligibility burden.",
      "cited_source_ids": [1, 999, 1, 2, 2, 3, 4],
      "follow_up_questions": []
    }
    """.strip()
    fake_client = _FakeClient(payload)
    service = ProtocolChatService(client=fake_client, model="test-model")
    analysis = _sample_analysis()
    pages = [
        PageText(page_number=1, text="Introduction."),
        PageText(page_number=2, text="Safety assessments are only performed monthly."),
        PageText(page_number=3, text="Participants must meet multiple lab thresholds."),
        PageText(page_number=4, text="Conclusion."),
    ]

    result = service.answer_question(
        analysis=analysis,
        question="Explain the patient safety concerns.",
        history=[],
        pages=pages,
        input_path=None,
    )

    assert result.payload.can_answer is True
    assert result.payload.cited_source_ids == [1, 2]


def test_protocol_chat_service_suppresses_sources_when_model_cannot_answer() -> None:
    payload = """
    {
      "can_answer": false,
      "answer_markdown": "I cannot answer from the supplied protocol and analysis.",
      "cited_source_ids": [1, 2],
      "follow_up_questions": ["What else?"]
    }
    """.strip()
    service = ProtocolChatService(client=_FakeClient(payload), model="test-model")
    analysis = _sample_analysis()
    pages = [
        PageText(page_number=1, text="Introduction."),
        PageText(page_number=2, text="Safety assessments are only performed monthly."),
    ]

    result = service.answer_question(
        analysis=analysis,
        question="Explain the patient safety concerns.",
        history=[],
        pages=pages,
        input_path=None,
    )

    assert result.payload.can_answer is False
    assert result.payload.cited_source_ids == []
    assert result.payload.follow_up_questions == []
    assert result.sources == []
