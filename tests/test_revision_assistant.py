import json

from protocol_risk_analyzer.models import AnalysisResult, ProtocolMetadata
from protocol_risk_analyzer.revision_assistant import (
    ProtocolProposalEditService,
    ProtocolRevisionPlannerService,
    _normalize_revision_planner_payload,
)


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _FakeResponsesApi:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs

    def create(self, **kwargs):
        del kwargs
        return _FakeResponse(self._outputs.pop(0))


class _FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.responses = _FakeResponsesApi(outputs)


def _sample_analysis() -> AnalysisResult:
    return AnalysisResult(
        metadata=ProtocolMetadata(
            source_path="C:\\tmp\\protocol.docx",
            document_name="protocol.docx",
            page_count=12,
            chunk_count=3,
            generated_at_utc=ProtocolMetadata.now_iso(),
        ),
        executive_summary="Sample executive summary long enough to satisfy validation requirements.",
        key_assumptions=["Sample assumption"],
        risks=[],
    )


def test_normalize_revision_planner_payload_coerces_invalid_values() -> None:
    payload = _normalize_revision_planner_payload(
        {
            "intent": "edit",
            "assistant_reply": "",
            "proposals": [
                {
                    "summary": "Add monthly ECG monitoring",
                    "requested_change": "Add monthly ECG monitoring for participants receiving active treatment.",
                    "rationale": "",
                    "affected_sections": "Safety Monitoring",
                    "estimated_edits": "not-a-number",
                    "focus": "cardiac",
                }
            ],
        },
        question="Add monthly ECG monitoring.",
    )
    assert payload["intent"] == "revision_request"
    assert payload["assistant_reply"]
    assert len(payload["proposals"]) == 1
    assert payload["proposals"][0]["focus"] == "general"
    assert payload["proposals"][0]["estimated_edits"] == 3


def test_normalize_revision_planner_payload_prefers_revision_intent_for_edit_requests() -> None:
    payload = _normalize_revision_planner_payload(
        {
            "intent": "analysis_question",
            "assistant_reply": "I can analyze that request using the protocol and generated risk analysis.",
            "proposals": [],
        },
        question="Please rewrite section 3 to add weekly safety monitoring.",
    )
    assert payload["intent"] == "revision_request"
    assert payload["proposals"]


def test_plan_turn_repairs_malformed_json() -> None:
    client = _FakeClient(
        outputs=[
            '{"intent":"revision_request","assistant_reply":"test","proposals":[',
            json.dumps(
                {
                    "intent": "revision_request",
                    "assistant_reply": "I identified a protocol edit for review.",
                    "proposals": [
                        {
                            "summary": "Reduce visit burden",
                            "requested_change": "Reduce in-person visit frequency where clinically acceptable.",
                            "rationale": "Lower burden can improve retention and operational feasibility.",
                            "affected_sections": ["Study Visits"],
                            "estimated_edits": 3,
                            "focus": "operational",
                        }
                    ],
                }
            ),
        ]
    )
    service = ProtocolRevisionPlannerService(client=client, model="test-model")
    result = service.plan_turn(
        analysis=_sample_analysis(),
        question="Reduce participant burden.",
        history=[],
        revision_session={},
    )
    assert result.payload.intent == "revision_request"
    assert len(result.payload.proposals) == 1
    assert result.payload.proposals[0].summary == "Reduce visit burden"


def test_proposal_edit_service_repairs_malformed_json() -> None:
    client = _FakeClient(
        outputs=[
            '{"updated_text":"Updated paragraph"',
            json.dumps(
                {
                    "updated_text": (
                        "Subjects will return for scheduled safety follow-up after dosing with added monitoring detail."
                    ),
                    "reason": "Clarifies the requested monitoring detail while preserving the paragraph topic.",
                }
            ),
        ]
    )
    service = ProtocolProposalEditService(client=client, model="test-model")

    payload = service.revise_proposal(
        target_label="Page 4 - Paragraph 2",
        risk_title="Safety monitoring gap",
        original_text="Subjects will return for safety follow-up after dosing.",
        current_text="Subjects will return for safety follow-up after dosing.",
        current_reason="Adds clearer monitoring language.",
        user_request="Edit #1, make it more detailed.",
    )

    assert payload.updated_text.startswith("Subjects will return for scheduled safety follow-up")
    assert payload.reason.startswith("Clarifies the requested monitoring detail")
