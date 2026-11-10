from protocol_risk_analyzer.analysis import (
    _canonicalize_category,
    _harmonize_similar_categories,
    _reconcile_risk_domains,
    aggregate_risks,
    sort_risks,
)
from protocol_risk_analyzer.models import EvidenceSnippet, RiskAssessment, RiskDomain


def _risk(
    title: str,
    domain: RiskDomain,
    likelihood: int,
    impact: int,
    confidence: float,
    page: int,
    factor_triggers: list[str] | None = None,
) -> RiskAssessment:
    return RiskAssessment(
        title=title,
        domain=domain,
        category="operational",
        likelihood=likelihood,
        impact=impact,
        confidence=confidence,
        rationale="Recruitment assumptions may be overly optimistic for target enrollment.",
        mitigations=["Add backup sites"],
        factor_triggers=factor_triggers or [],
        evidence=[
            EvidenceSnippet(
                excerpt="Enrollment is expected to complete in 2 months.",
                page_start=page,
                page_end=page,
            )
        ],
    )


def test_aggregate_risks_merges_by_domain_and_title() -> None:
    risks = [
        _risk("Recruitment delay", RiskDomain.FINANCIAL, 4, 5, 0.9, 12),
        _risk("recruitment delay", RiskDomain.FINANCIAL, 3, 4, 0.7, 13),
        _risk("Adverse event burden", RiskDomain.HEALTH, 3, 5, 0.8, 30),
    ]

    merged = aggregate_risks(risks)

    assert len(merged) == 2
    financial = [r for r in merged if r.domain == RiskDomain.FINANCIAL][0]
    assert financial.title.lower() == "recruitment delay"
    assert len(financial.evidence) == 2


def test_aggregate_risks_dedupes_factor_triggers() -> None:
    risks = [
        _risk(
            "Recruitment delay",
            RiskDomain.FINANCIAL,
            4,
            5,
            0.9,
            12,
            factor_triggers=["Many required visits", "High anticipated participant dropout percentage"],
        ),
        _risk(
            "Recruitment delay",
            RiskDomain.FINANCIAL,
            3,
            4,
            0.7,
            13,
            factor_triggers=["many required visits", "Very long participation periods"],
        ),
    ]
    merged = aggregate_risks(risks)
    assert len(merged) == 1
    assert merged[0].factor_triggers == [
        "Many required visits",
        "High anticipated participant dropout percentage",
        "Very long participation periods",
    ]


def test_aggregate_risks_can_merge_cross_domain_duplicates_before_reclassification() -> None:
    health = _risk("Strict eligibility limits enrollment", RiskDomain.HEALTH, 3, 4, 0.8, 2).model_copy(
        update={"category": "Eligibility And Enrollment"}
    )
    eligibility = _risk(
        "Strict eligibility limits enrollment",
        RiskDomain.ELIGIBILITY,
        4,
        4,
        0.9,
        3,
    ).model_copy(update={"category": "Eligibility And Enrollment"})

    merged = aggregate_risks([health, eligibility], include_domain=False)

    assert len(merged) == 1
    assert len(merged[0].evidence) == 2


def test_sort_risks_orders_highest_score_first() -> None:
    low = _risk("Low", RiskDomain.HEALTH, 2, 2, 0.5, 1)
    high = _risk("High", RiskDomain.HEALTH, 5, 5, 1.0, 2)
    ordered = sort_risks([low, high])
    assert ordered[0].title == "High"


def test_aggregate_risks_normalizes_category_and_removes_risk_suffix() -> None:
    r1 = _risk("Trial design issue", RiskDomain.FINANCIAL, 3, 3, 0.8, 1)
    r2 = _risk("Trial design issue", RiskDomain.FINANCIAL, 4, 4, 0.8, 2)
    r1 = r1.model_copy(update={"category": "trial design"})
    r2 = r2.model_copy(update={"category": "trial design risk"})

    merged = aggregate_risks([r1, r2])
    assert len(merged) == 1
    assert merged[0].category == "Trial Design"


def test_canonicalize_category_groups_common_synonyms_and_variants() -> None:
    assert _canonicalize_category("Unknown Side Effects") == "Unknown Effects"
    assert (
        _canonicalize_category("Unknown or Poorly Characterized Side Effects")
        == "Unknown Effects"
    )
    assert _canonicalize_category("Adverse Events") == "Adverse Events"
    assert _canonicalize_category("Adverse Reactions") == "Adverse Events"
    assert _canonicalize_category("Allergic Reaction") == "Allergic Reactions"
    assert _canonicalize_category("Allergic Reactions") == "Allergic Reactions"
    assert _canonicalize_category("Adverse Event Management") == "Adverse Events"
    assert _canonicalize_category("Adverse Event Management Costs") == "Adverse Events"
    assert _canonicalize_category("Informed Consent Comprehension") == "Informed Consent"
    assert _canonicalize_category("Data Privacy Concern") == "Privacy Confidentiality"
    assert _canonicalize_category("Travel And Lodging Burden") == "Cost Reimbursement"
    assert _canonicalize_category("Rare Biomarker Requirement") == "Eligibility Criteria"
    assert _canonicalize_category("Gene Therapy") == "Intervention Modality"


def test_harmonize_similar_categories_merges_extremely_close_labels() -> None:
    left = _risk("Label A", RiskDomain.HEALTH, 3, 3, 0.8, 1).model_copy(
        update={"category": "Participant Benefit And Placebo"}
    )
    right = _risk("Label B", RiskDomain.HEALTH, 3, 3, 0.8, 2).model_copy(
        update={"category": "Participant Benefit And Placebo Control"}
    )
    merged = _harmonize_similar_categories([left, right])
    assert merged[0].category == merged[1].category


def test_harmonize_similar_categories_merges_shared_driver_topic_into_broad_group() -> None:
    adherence = _risk("Treatment adherence concern", RiskDomain.HEALTH, 3, 3, 0.8, 1).model_copy(
        update={"category": "Treatment Adherence"}
    )
    efficacy = _risk("Treatment efficacy concern", RiskDomain.HEALTH, 3, 3, 0.8, 2).model_copy(
        update={"category": "Treatment Efficacy"}
    )
    merged = _harmonize_similar_categories([adherence, efficacy])
    assert merged[0].category == "Treatment"
    assert merged[1].category == "Treatment"


def test_harmonize_similar_categories_keeps_distinct_umbrella_topics_separate() -> None:
    side_effect = _risk("Side effect concern", RiskDomain.HEALTH, 3, 3, 0.8, 1).model_copy(
        update={"category": "Side Effects"}
    )
    adverse_effect = _risk("Adverse effect concern", RiskDomain.HEALTH, 3, 3, 0.8, 2).model_copy(
        update={"category": "Adverse Effects"}
    )
    merged = _harmonize_similar_categories([side_effect, adverse_effect])
    categories = {risk.category for risk in merged}
    assert len(categories) == 2


def test_reconcile_risk_domains_moves_eligibility_category_out_of_ethical() -> None:
    risk = _risk("Recruitment pressure", RiskDomain.ETHICAL, 3, 3, 0.9, 1).model_copy(
        update={
            "category": "Eligibility And Enrollment",
            "factor_triggers": ["Very narrow inclusion criteria"],
            "rationale": "Restrictive inclusion and exclusion criteria may reduce enrollment feasibility.",
        }
    )
    reconciled = _reconcile_risk_domains([risk])
    assert reconciled[0].domain == RiskDomain.ELIGIBILITY


def test_reconcile_risk_domains_keeps_true_ethical_risk_in_ethical() -> None:
    risk = _risk("Consent quality concern", RiskDomain.ETHICAL, 3, 4, 0.9, 1).model_copy(
        update={
            "category": "Informed Consent",
            "factor_triggers": ["Adequacy of informed consent"],
            "rationale": "Consent language is complex and may not ensure participant understanding.",
        }
    )
    reconciled = _reconcile_risk_domains([risk])
    assert reconciled[0].domain == RiskDomain.ETHICAL


def test_aggregate_risks_caps_mitigations_and_evidence_to_ten() -> None:
    evidence = [
        EvidenceSnippet(excerpt=f"Evidence {idx}", page_start=idx + 1, page_end=idx + 1)
        for idx in range(15)
    ]
    risk = _risk("High burden protocol", RiskDomain.FINANCIAL, 4, 4, 0.9, 1).model_copy(
        update={
            "mitigations": [f"Mitigation {idx}" for idx in range(15)],
            "evidence": evidence,
        }
    )

    merged = aggregate_risks([risk])

    assert len(merged) == 1
    assert len(merged[0].mitigations) == 10
    assert len(merged[0].evidence) == 10
