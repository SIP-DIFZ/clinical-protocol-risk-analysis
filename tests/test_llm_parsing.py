import json

import pytest

from protocol_risk_analyzer.llm import (
    RiskLlmClient,
    build_openai_client_with_resilience,
    _merge_duplicate_risks,
    _merge_legacy_domain_title_risks,
    _normalize_chunk_risk_payload,
    _normalize_editor_patch_action,
    _parse_model_json,
    _salvage_partial_risks_json,
)
from protocol_risk_analyzer.models import EvidenceSnippet, RiskAssessment, RiskDomain


def test_parse_model_json_plain_json() -> None:
    payload = {"risks": []}
    text = json.dumps(payload)
    assert _parse_model_json(text) == payload


def test_parse_model_json_fenced_json() -> None:
    payload = {"risks": [{"title": "Example"}]}
    text = "```json\n" + json.dumps(payload) + "\n```"
    assert _parse_model_json(text) == payload


def test_parse_model_json_finds_embedded_json_object() -> None:
    payload = {"risks": [{"title": "Example"}]}
    text = "Here is the output:\n" + json.dumps(payload) + "\nThanks."
    assert _parse_model_json(text) == payload


def test_parse_model_json_raises_on_invalid_payload() -> None:
    with pytest.raises(json.JSONDecodeError):
        _parse_model_json("not-json")


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _FakeResponsesApi:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._outputs.pop(0))


class _FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.responses = _FakeResponsesApi(outputs)


def test_build_openai_client_with_resilience_uses_timeout_and_retries(monkeypatch) -> None:
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("protocol_risk_analyzer.llm.OpenAI", _FakeOpenAI)
    build_openai_client_with_resilience(
        api_key="k",
        base_url="https://example.test/v1",
        timeout_seconds=75,
        max_retries=3,
    )
    assert captured["api_key"] == "k"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["timeout"] == 75.0
    assert captured["max_retries"] == 3


def test_build_openai_client_with_resilience_applies_min_timeout(monkeypatch) -> None:
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("protocol_risk_analyzer.llm.OpenAI", _FakeOpenAI)
    build_openai_client_with_resilience(
        api_key="k",
        base_url=None,
        timeout_seconds=1,
        max_retries=-1,
    )
    assert captured["timeout"] == 10.0
    assert captured["max_retries"] == 0


def test_complete_json_repairs_malformed_payload() -> None:
    client = _FakeClient(
        outputs=[
            '{"risks":[{"title":"Incomplete"',
            '{"risks":[]}',
        ]
    )
    llm = RiskLlmClient(client=client, model="test-model")

    payload = llm._complete_json(
        system_prompt="system",
        user_prompt="user",
        expected_json_shape='{"risks":[...]}',
        max_output_tokens=100,
    )

    assert payload == {"risks": []}
    assert len(client.responses.calls) == 2


def test_complete_json_raises_after_failed_repair() -> None:
    client = _FakeClient(outputs=['{"risks":[', "still-not-json"])
    llm = RiskLlmClient(client=client, model="test-model")

    with pytest.raises(RuntimeError):
        llm._complete_json(
            system_prompt="system",
            user_prompt="user",
            expected_json_shape='{"risks":[...]}',
            max_output_tokens=100,
        )


def test_editor_agent_generate_patches_drops_invalid_patch_entries(monkeypatch) -> None:
    llm = RiskLlmClient(client=_FakeClient(outputs=[]), model="test-model")
    monkeypatch.setattr(
        llm,
        "_complete_json",
        lambda **_: {
            "patches": [
                {
                    "risk_title": "Infusion reaction monitoring gap",
                    "action": "replace",
                    "target_paragraph_id": "P-12",
                    "new_text": (
                        "Infusion visits will include protocol-defined observation periods and immediate "
                        "escalation steps for suspected hypersensitivity reactions."
                    ),
                    "reason": "Adds explicit monitoring and escalation controls.",
                },
                {
                    "risk_title": "Infusion reaction monitoring gap",
                    "action": "none",
                    "target_paragraph_id": "",
                    "new_text": "",
                    "reason": "",
                },
            ]
        },
    )

    patches = llm.editor_agent_generate_patches(
        protocol_paragraphs=[
            {
                "paragraph_id": "P-12",
                "page_number": 4,
                "paragraph_number": 2,
                "text": "Participants will receive the study infusion in clinic.",
            }
        ],
        selected_risks=[
            {
                "risk_title": "Infusion reaction monitoring gap",
                "candidate_paragraph_ids": ["P-12"],
            }
        ],
    )

    assert len(patches) == 1
    assert patches[0].action == "replace"
    assert patches[0].target_paragraph_id == "P-12"


def test_normalize_editor_patch_action_handles_noop_values() -> None:
    assert _normalize_editor_patch_action("insert after") == "insert_after"
    assert _normalize_editor_patch_action("insert-before") == "insert_before"
    assert _normalize_editor_patch_action("none") is None
    assert _normalize_editor_patch_action("") is None


def test_classify_mitigation_statuses_uses_model_output() -> None:
    client = _FakeClient(
        outputs=[
        json.dumps(
            {
                "statuses": [
                    {
                        "risk_index": 0,
                        "mitigation_status": "partially_mitigated",
                        "status_reason": "Concrete controls are present and reduce residual severity.",
                    },
                    {
                        "risk_index": 1,
                        "mitigation_status": "adequately_mitigated",
                        "status_reason": "Controls are complete and no meaningful mitigation gaps remain.",
                    },
                ]
            }
        )
        ]
    )
    llm = RiskLlmClient(client=client, model="test-model")
    risks = [
        RiskAssessment(
        title="Risk A",
        domain=RiskDomain.HEALTH,
        category="Safety",
        likelihood=3,
        impact=4,
        confidence=0.8,
        rationale="Controls are present.",
        mitigations=["Monitor participants"],
        factor_triggers=[],
        evidence=[EvidenceSnippet(excerpt="Monitoring required.", page_start=1, page_end=1)],
        ),
        RiskAssessment(
        title="Risk B",
        domain=RiskDomain.HEALTH,
        category="Safety",
        likelihood=4,
        impact=5,
        confidence=0.9,
        rationale="Escalation criteria are unclear.",
        mitigations=[],
        factor_triggers=[],
        evidence=[EvidenceSnippet(excerpt="Criteria are not defined.", page_start=2, page_end=2)],
        ),
    ]

    classified = llm.classify_mitigation_statuses(risks)

    assert classified[0].mitigation_status == "partially_mitigated"
    assert classified[1].mitigation_status == "adequately_mitigated"
    assert "reduce residual severity" in classified[0].mitigation_status_reason
    assert "no meaningful mitigation gaps" in classified[1].mitigation_status_reason
    assert classified[1].mitigations == []


def test_classify_mitigation_statuses_promotes_to_adequate_when_no_suggested_mitigations() -> None:
    client = _FakeClient(
        outputs=[
            json.dumps(
                {
                    "statuses": [
                        {
                            "risk_index": 0,
                            "mitigation_status": "adequately_mitigated",
                            "status_reason": "Controls are strong but further actions are listed.",
                        },
                        {
                            "risk_index": 1,
                            "mitigation_status": "partially_mitigated",
                            "status_reason": "Some controls exist and no more actions are needed.",
                        },
                    ]
                }
            )
        ]
    )
    llm = RiskLlmClient(client=client, model="test-model")
    risks = [
        RiskAssessment(
            title="Risk A",
            domain=RiskDomain.HEALTH,
            category="Safety",
            likelihood=3,
            impact=4,
            confidence=0.8,
            rationale="Controls are present.",
            mitigations=["Add an additional safeguard"],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="Monitoring required.", page_start=1, page_end=1)],
        ),
        RiskAssessment(
            title="Risk B",
            domain=RiskDomain.HEALTH,
            category="Safety",
            likelihood=3,
            impact=3,
            confidence=0.8,
            rationale="Controls are complete.",
            mitigations=[],
            factor_triggers=[],
            evidence=[EvidenceSnippet(excerpt="Controls are fully specified.", page_start=2, page_end=2)],
        ),
    ]

    classified = llm.classify_mitigation_statuses(risks)

    assert classified[0].mitigation_status == "adequately_mitigated"
    assert classified[1].mitigation_status == "adequately_mitigated"


def test_salvage_partial_risks_json_recovers_complete_objects() -> None:
    text = (
        '{"risks":[{"title":"A","domain":"health","category":"x","likelihood":3,"impact":4,'
        '"confidence":0.8,"rationale":"Detailed rationale text here.","mitigations":[],"factor_triggers":[],'
        '"evidence":[]},{"title":"B","domain":"financial","category":"y","likelihood":4'
    )
    salvaged = _salvage_partial_risks_json(text)
    assert salvaged is not None
    assert "risks" in salvaged
    assert len(salvaged["risks"]) == 1
    assert salvaged["risks"][0]["title"] == "A"


def test_normalize_chunk_risk_payload_clamps_invalid_scores() -> None:
    payload = {
        "risks": [
            {
                "title": "Example risk",
                "domain": "health",
                "category": "safety",
                "likelihood": 0,
                "impact": 0,
                "confidence": 1.8,
                "rationale": "This is a meaningful rationale about the risk context.",
                "mitigations": ["Mitigation A"],
                "factor_triggers": ["Factor A"],
                "evidence": [{"excerpt": "Evidence text", "page_start": 0, "page_end": -2}],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="health")
    assert normalized["risks"][0]["likelihood"] == 1
    assert normalized["risks"][0]["impact"] == 1
    assert normalized["risks"][0]["confidence"] == 1.0
    assert normalized["risks"][0]["evidence"][0]["page_start"] == 1
    assert normalized["risks"][0]["evidence"][0]["page_end"] == 1


def test_normalize_chunk_risk_payload_filters_fully_mitigated_statements() -> None:
    payload = {
        "risks": [
            {
                "title": "Allergic symptoms during escalation",
                "domain": "health",
                "category": "allergic reactions",
                "likelihood": 3,
                "impact": 4,
                "confidence": 0.8,
                "rationale": (
                    "All dose escalations occur under clinical observation with emergency medications "
                    "and trained personnel available to mitigate reactions."
                ),
                "mitigations": ["Continue protocol-defined monitoring and emergency readiness."],
                "factor_triggers": ["Unknown or poorly characterized side effects"],
                "evidence": [
                    {
                        "excerpt": "Observation and emergency medications are required during escalation.",
                        "page_start": 4,
                        "page_end": 4,
                    }
                ],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="health")
    assert normalized["risks"] == []


def test_normalize_chunk_risk_payload_keeps_explicit_unresolved_gap() -> None:
    payload = {
        "risks": [
            {
                "title": "Monitoring gap during home dosing",
                "domain": "health",
                "category": "safety monitoring",
                "likelihood": 4,
                "impact": 4,
                "confidence": 0.85,
                "rationale": (
                    "Monitoring is described, but criteria for overnight escalation are unclear and "
                    "emergency transfer thresholds are not defined."
                ),
                "residual_gap": "Overnight escalation thresholds remain unclear and not defined.",
                "mitigations": ["Define explicit escalation thresholds for home events."],
                "factor_triggers": ["Unknown or poorly characterized side effects"],
                "evidence": [
                    {
                        "excerpt": "Participants are observed, but overnight escalation criteria are unclear.",
                        "page_start": 7,
                        "page_end": 7,
                    }
                ],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="health")
    assert len(normalized["risks"]) == 1


def test_normalize_chunk_risk_payload_filters_controlled_dropout_statement() -> None:
    payload = {
        "risks": [
            {
                "title": "Dropout concern during long maintenance",
                "domain": "financial",
                "category": "dropout and retention",
                "likelihood": 3,
                "impact": 3,
                "confidence": 0.8,
                "rationale": (
                    "Participants will receive retention support, scheduled follow-up, and escalation "
                    "procedures to mitigate dropout burden during maintenance."
                ),
                "residual_gap": "",
                "mitigations": [
                    "Retention outreach and flexible scheduling will be implemented.",
                    "Follow-up monitoring and escalation rules are required for missed visits.",
                ],
                "factor_triggers": ["High anticipated participant dropout percentage"],
                "evidence": [
                    {
                        "excerpt": (
                            "Retention support, follow-up monitoring, and escalation criteria are "
                            "defined to mitigate dropout risk."
                        ),
                        "page_start": 12,
                        "page_end": 12,
                    }
                ],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="financial")
    assert normalized["risks"] == []


def test_normalize_chunk_risk_payload_keeps_controlled_text_when_residual_gap_explicit() -> None:
    payload = {
        "risks": [
            {
                "title": "Emergency escalation ambiguity",
                "domain": "health",
                "category": "safety monitoring",
                "likelihood": 3,
                "impact": 4,
                "confidence": 0.8,
                "rationale": (
                    "Monitoring and emergency readiness are described, but the protocol text does not "
                    "define escalation thresholds for prolonged symptoms."
                ),
                "residual_gap": "Escalation thresholds for prolonged symptoms are not defined.",
                "mitigations": ["Define explicit escalation thresholds and transfer triggers."],
                "factor_triggers": ["Unknown or poorly characterized side effects"],
                "evidence": [
                    {
                        "excerpt": (
                            "Emergency medications are available, but escalation thresholds for prolonged "
                            "symptoms are not specified."
                        ),
                        "page_start": 20,
                        "page_end": 20,
                    }
                ],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="health")
    assert len(normalized["risks"]) == 1


def test_normalize_chunk_risk_payload_calibrates_residual_scores_when_controls_exist() -> None:
    payload = {
        "risks": [
            {
                "title": "Escalation threshold ambiguity",
                "domain": "health",
                "category": "safety monitoring",
                "likelihood": 5,
                "impact": 5,
                "confidence": 0.95,
                "rationale": (
                    "Monitoring, emergency readiness, and follow-up are present, but escalation "
                    "thresholds for persistent symptoms are not clearly defined."
                ),
                "residual_gap": "Escalation thresholds for persistent symptoms are not clearly defined.",
                "mitigations": [
                    "Continue monitoring, emergency preparedness, and required follow-up checks.",
                ],
                "factor_triggers": ["Unknown or poorly characterized side effects"],
                "evidence": [
                    {
                        "excerpt": (
                            "Monitoring and emergency medications are required, but escalation "
                            "thresholds are not specified."
                        ),
                        "page_start": 9,
                        "page_end": 9,
                    }
                ],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="health")
    assert len(normalized["risks"]) == 1
    risk = normalized["risks"][0]
    assert risk["likelihood"] <= 3
    assert risk["impact"] <= 4
    assert risk["confidence"] <= 0.8


def test_normalize_chunk_risk_payload_does_not_calibrate_without_controls() -> None:
    payload = {
        "risks": [
            {
                "title": "Uncontrolled severe reaction management gap",
                "domain": "health",
                "category": "emergency response",
                "likelihood": 5,
                "impact": 5,
                "confidence": 0.9,
                "rationale": (
                    "No emergency procedures are described and transfer pathways are missing for severe "
                    "reactions during home dosing."
                ),
                "residual_gap": "Emergency procedures and transfer pathways are missing.",
                "mitigations": [],
                "factor_triggers": ["Unknown or poorly characterized side effects"],
                "evidence": [
                    {
                        "excerpt": "The protocol does not specify emergency response or transfer pathways.",
                        "page_start": 11,
                        "page_end": 11,
                    }
                ],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="health")
    assert len(normalized["risks"]) == 1
    risk = normalized["risks"][0]
    assert risk["likelihood"] == 5
    assert risk["impact"] == 5
    assert risk["confidence"] == 0.9


def test_normalize_chunk_risk_payload_raises_severe_uncontrolled_residual_scores() -> None:
    payload = {
        "risks": [
            {
                "title": "Life-threatening anaphylaxis management gap",
                "domain": "health",
                "category": "emergency response",
                "likelihood": 2,
                "impact": 3,
                "confidence": 0.4,
                "rationale": (
                    "Participants may experience severe anaphylaxis and life-threatening reactions, "
                    "but emergency response pathways are unclear and missing."
                ),
                "residual_gap": "Emergency transfer criteria are missing and not defined for severe anaphylaxis.",
                "mitigations": [],
                "factor_triggers": ["Unknown or poorly characterized side effects"],
                "evidence": [
                    {
                        "excerpt": "Life-threatening anaphylaxis is possible and emergency transfer criteria are not specified.",
                        "page_start": 15,
                        "page_end": 15,
                    }
                ],
            }
        ]
    }
    normalized = _normalize_chunk_risk_payload(payload, forced_domain="health")
    assert len(normalized["risks"]) == 1
    risk = normalized["risks"][0]
    assert risk["likelihood"] >= 4
    assert risk["impact"] == 5
    assert risk["confidence"] >= 0.88


def test_merge_duplicate_risks_preserves_distinct_items() -> None:
    risks = [
        RiskAssessment(
            title="Enrollment criteria too restrictive",
            domain=RiskDomain.FINANCIAL,
            category="participant eligibility",
            likelihood=4,
            impact=4,
            confidence=0.9,
            rationale="Criteria may reduce enrollment speed.",
            mitigations=["Broaden inclusion criteria"],
            factor_triggers=["Highly restrictive inclusion criteria"],
            evidence=[EvidenceSnippet(excerpt="Strict criteria listed.", page_start=2, page_end=2)],
        ),
        RiskAssessment(
            title="Enrollment criteria too restrictive",
            domain=RiskDomain.FINANCIAL,
            category="participant eligibility",
            likelihood=3,
            impact=4,
            confidence=0.8,
            rationale="Criteria may reduce enrollment speed.",
            mitigations=["Open additional sites"],
            factor_triggers=["Highly restrictive exclusion criteria"],
            evidence=[EvidenceSnippet(excerpt="Multiple exclusions listed.", page_start=3, page_end=3)],
        ),
        RiskAssessment(
            title="Frequent blood draws may increase participant burden",
            domain=RiskDomain.HEALTH,
            category="procedure burden",
            likelihood=3,
            impact=3,
            confidence=0.85,
            rationale="Invasive schedule may reduce tolerance.",
            mitigations=["Reduce draw frequency"],
            factor_triggers=["Frequent invasive procedures"],
            evidence=[EvidenceSnippet(excerpt="Weekly blood collection required.", page_start=10, page_end=10)],
        ),
    ]

    merged = _merge_duplicate_risks(risks)
    assert len(merged) == 2
    financial = [r for r in merged if r.domain == RiskDomain.FINANCIAL][0]
    assert len(financial.mitigations) == 2
    assert len(financial.evidence) == 2


def test_merge_duplicate_risks_merges_near_duplicate_titles() -> None:
    risks = [
        RiskAssessment(
            title="Risk of Allergic Reactions During Oral Immunotherapy",
            domain=RiskDomain.HEALTH,
            category="allergic reactions",
            likelihood=4,
            impact=5,
            confidence=0.9,
            rationale="Oral immunotherapy can provoke allergic responses.",
            mitigations=["Monitor participants closely"],
            factor_triggers=["Unknown or poorly characterized side effects"],
            evidence=[EvidenceSnippet(excerpt="Allergic reactions observed.", page_start=4, page_end=4)],
        ),
        RiskAssessment(
            title="Severe allergic reactions during oral immunotheraphy",
            domain=RiskDomain.HEALTH,
            category="allergic reactions",
            likelihood=4,
            impact=5,
            confidence=0.85,
            rationale="Severe allergic events are possible during dosing.",
            mitigations=["Emergency response protocol"],
            factor_triggers=["Unknown or poorly characterized side effects"],
            evidence=[EvidenceSnippet(excerpt="Severe reactions can occur.", page_start=5, page_end=5)],
        ),
    ]

    merged = _merge_duplicate_risks(risks)
    assert len(merged) == 1
    assert len(merged[0].mitigations) == 2
    assert len(merged[0].evidence) == 2


def test_merge_duplicate_risks_merges_across_related_categories() -> None:
    risks = [
        RiskAssessment(
            title="Risk of Allergic Reactions During Oral Immunotherapy",
            domain=RiskDomain.HEALTH,
            category="allergic reactions",
            likelihood=4,
            impact=5,
            confidence=0.9,
            rationale="Oral immunotherapy can provoke allergic responses.",
            mitigations=["Monitor participants closely"],
            factor_triggers=["Unknown or poorly characterized side effects"],
            evidence=[EvidenceSnippet(excerpt="Allergic reactions observed.", page_start=4, page_end=4)],
        ),
        RiskAssessment(
            title="Severe allergic reactions during oral immunotheraphy",
            domain=RiskDomain.HEALTH,
            category="adverse events",
            likelihood=4,
            impact=5,
            confidence=0.85,
            rationale="Severe allergic events are possible during dosing.",
            mitigations=["Emergency response protocol"],
            factor_triggers=["Unknown or poorly characterized side effects"],
            evidence=[EvidenceSnippet(excerpt="Severe reactions can occur.", page_start=5, page_end=5)],
        ),
    ]

    merged = _merge_duplicate_risks(risks)
    assert len(merged) == 1
    assert len(merged[0].mitigations) == 2


def test_merge_duplicate_risks_merges_transitive_near_duplicates() -> None:
    risks = [
        RiskAssessment(
            title="Placebo use may limit direct participant benefit",
            domain=RiskDomain.ETHICAL,
            category="participant benefit and placebo",
            likelihood=3,
            impact=4,
            confidence=0.88,
            rationale="Placebo exposure can reduce perceived direct benefit.",
            mitigations=["Clarify consent language"],
            factor_triggers=["Use of placebo when an effective therapy exists"],
            evidence=[EvidenceSnippet(excerpt="Randomization includes placebo arm.", page_start=10, page_end=10)],
        ),
        RiskAssessment(
            title="Direct participant benefit may be uncertain under placebo control",
            domain=RiskDomain.ETHICAL,
            category="participant benefit and placebo",
            likelihood=3,
            impact=4,
            confidence=0.85,
            rationale="Direct benefit cannot be assured in placebo-controlled assignment.",
            mitigations=["Improve participant counseling"],
            factor_triggers=["Use of placebo when an effective therapy exists"],
            evidence=[EvidenceSnippet(excerpt="No guaranteed therapeutic benefit.", page_start=11, page_end=11)],
        ),
        RiskAssessment(
            title="Uncertain direct participant benefit due to placebo assignment",
            domain=RiskDomain.HEALTH,
            category="participant benefit uncertainty",
            likelihood=3,
            impact=3,
            confidence=0.82,
            rationale="Placebo assignment introduces uncertainty of clinical benefit.",
            mitigations=["Set explicit expectations"],
            factor_triggers=["Uncertain direct participant benefit due to placebo control"],
            evidence=[EvidenceSnippet(excerpt="Placebo may provide limited direct benefit.", page_start=12, page_end=12)],
        ),
    ]

    merged = _merge_duplicate_risks(risks)
    assert len(merged) == 1
    assert len(merged[0].mitigations) == 3


def test_legacy_merge_groups_placebo_participant_benefit_near_duplicates() -> None:
    risks = [
        RiskAssessment(
            title="Risk of placebo assignment limiting direct participant benefit",
            domain=RiskDomain.HEALTH,
            category="placebo use",
            likelihood=3,
            impact=4,
            confidence=0.9,
            rationale="Participants may receive placebo and not direct benefit.",
            mitigations=["Clarify consent expectations"],
            factor_triggers=["Uncertain direct participant benefit"],
            evidence=[EvidenceSnippet(excerpt="Participants may be randomized to placebo.", page_start=12, page_end=12)],
        ),
        RiskAssessment(
            title="Risk of Placebo Use Affecting Participant Benefit",
            domain=RiskDomain.HEALTH,
            category="participant benefit uncertainty",
            likelihood=3,
            impact=4,
            confidence=0.85,
            rationale="Placebo exposure may limit direct therapeutic effect.",
            mitigations=["Increase participant counseling"],
            factor_triggers=["Uncertain direct participant benefit due to placebo control"],
            evidence=[EvidenceSnippet(excerpt="Direct benefit cannot be guaranteed due to placebo control.", page_start=13, page_end=13)],
        ),
        RiskAssessment(
            title="Uncertain direct participant benefit",
            domain=RiskDomain.HEALTH,
            category="benefit uncertainty",
            likelihood=3,
            impact=3,
            confidence=0.8,
            rationale="Individual benefit is uncertain.",
            mitigations=["Discuss risks during consent"],
            factor_triggers=["Uncertain direct participant benefit"],
            evidence=[EvidenceSnippet(excerpt="No guaranteed direct benefit.", page_start=14, page_end=14)],
        ),
        RiskAssessment(
            title="Uncertain direct participant benefit due to placebo control",
            domain=RiskDomain.HEALTH,
            category="placebo control",
            likelihood=3,
            impact=3,
            confidence=0.82,
            rationale="Placebo assignment can reduce expected benefit.",
            mitigations=["Set participant expectations"],
            factor_triggers=["Uncertain direct participant benefit due to placebo control"],
            evidence=[EvidenceSnippet(excerpt="Placebo-control introduces uncertainty of benefit.", page_start=15, page_end=15)],
        ),
    ]

    merged = _merge_legacy_domain_title_risks(risks)
    assert len(merged) == 1
    assert len(merged[0].mitigations) == 4
