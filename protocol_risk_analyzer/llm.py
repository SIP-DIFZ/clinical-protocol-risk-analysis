from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from string import punctuation
from collections import defaultdict
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from protocol_risk_analyzer.chunking import DocumentChunk
from protocol_risk_analyzer.models import ProtocolEditPatch, RiskAssessment, RiskDomain
from protocol_risk_analyzer.rubric import (
    ELIGIBILITY_RISK_FACTORS,
    ETHICAL_RISK_FACTORS,
    FINANCIAL_RISK_FACTORS,
    HEALTH_RISK_FACTORS,
    INTERVENTION_RISK_FACTORS,
)


class ChunkRiskResponse(BaseModel):
    risks: list[RiskAssessment] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    executive_summary: str = Field(min_length=20)
    key_assumptions: list[str] = Field(default_factory=list)


class MitigationStatusItem(BaseModel):
    risk_index: int = Field(ge=0)
    mitigation_status: Literal[
        "unmitigated",
        "partially_mitigated",
        "adequately_mitigated",
        "completely_mitigated",
        "mitigated",
    ]
    status_reason: str = Field(min_length=10)


class MitigationStatusResponse(BaseModel):
    statuses: list[MitigationStatusItem] = Field(default_factory=list)


class EditorPatchResponse(BaseModel):
    patches: list[ProtocolEditPatch] = Field(default_factory=list)


class ParagraphRewriteResponse(BaseModel):
    new_text: str = Field(min_length=20)
    reason: str = Field(min_length=10)


SHARED_RISK_EXTRACTION_SYSTEM_PROMPT = (
    "You are a clinical trial risk analyst. Return strict JSON only."
    " Identify concrete {domain_name} risks from the provided protocol text."
    " Use evidence-based findings and include page references from the text markers."
    " Evaluate all listed rubric factors and only report risks grounded in the chunk text."
)


SHARED_RISK_EXTRACTION_USER_PROMPT = """
Analyze this chunk of a clinical trial protocol and return JSON object:

{

  "risks": [

    {

      "title": "short risk title",

      "domain": "{domain_name}",

      "category": "specific category",

      "likelihood": 1-5,

      "impact": 1-5,

      "confidence": 0.0-1.0,

      "rationale": "why this is a risk",

      "residual_gap": "specific unresolved gap that remains after existing controls",

      "mitigations": ["action 1", "action 2"],

      "factor_triggers": ["specific rubric factors that apply"],

      "evidence": [

        {

          "excerpt": "quote or close paraphrase",

          "page_start": integer >= 1,

          "page_end": integer >= page_start

        }

      ]

    }

  ]

}

If no meaningful risks are found, return {"risks":[]}.

RISK IDENTIFICATION:

Evaluate the protocol exactly as written.

- For every meaningful risk:
- Identify the protocol language related to the risk.
- Evaluate what controls already exist.
- Assess how effectively those controls reduce the risk.
- Identify any remaining limitation or opportunity for further improvement.

A risk may still be reported even if existing controls substantially mitigate it.

Do not invent risks that are unsupported by the protocol text.

Do not infer missing evidence.

CATEGORY CREATION:
The category is a high-level organizational topic used to group similar risks.

It is NOT a description of the individual finding.

The title explains the specific issue.

The category identifies the broader subject area.

Choose the broadest category that accurately represents the risk.

Before creating a new category, ask:

"Could this risk reasonably belong within an existing broader topic?"

If yes, use the broader topic.

Only create a new category when the issue clearly represents a genuinely different organizational area.

Prefer fewer, broader categories over many narrowly defined ones.

Prefer under-segmentation rather than over-segmentation.

Category names should generally:

- contain 1–3 words
- prefer 1 word when it still accurately captures the broader topic
- describe a broad organizational topic
- be reusable across many different protocols
- naturally group multiple related risks

Avoid creating categories based on:

- specific procedures
- individual assessments
- compliance subtypes
- timing differences
- wording variations
- protocol-specific phrasing

Mentally remove unnecessary qualifiers before choosing the category.

Examples:
Treatment Adherence → Treatment
Treatment Compliance → Treatment
Treatment Administration → Treatment
Dose Modifications → Treatment
Eligibility Exceptions → Eligibility
ECG Timing → Assessments
PK Sampling → Assessments
Safety Reporting → Safety
Informed Consent Comprehension → Consent
Emergency/Deferred Consent Process → Consent
Confidentiality Controls → Privacy
Undue Influence from Compensation → Compensation
Long Participation Duration Costs → Cost
Travel and Lodging Burden → Logistics
Site Startup Delays → Operations
Vendor Coordination Risk → Operations
Rare Biomarker Requirement → Eligibility
Multiple Lab Threshold Conflicts → Eligibility
Complex Administration Schedule → Administration
Frequent Dose Interruptions → Dosing

These examples are illustrative, not an exhaustive or required category set.
Do not force a risk into these example groups when another broader topic is a better fit.
You may use other broad categories when appropriate.

The title and rationale should contain the specificity.

The category should remain one level broader than the title.

RISK SCORING:

Likelihood, impact, and confidence should reflect the risk AFTER considering existing protocol controls.

When strong controls are present:

- Reduce likelihood appropriately.
- Adjust confidence based on how clearly the protocol addresses the issue.
- Avoid assigning high likelihood solely because the underlying hazard exists.

Impact should reflect the severity if the risk were to occur despite existing controls.

Do not assign high likelihood or high confidence from hazard mentions alone when controls materially reduce the remaining risk.

RISK UNIQUENESS:

Each reported risk must represent a distinct remaining concern.

Do not report multiple risks that differ only by:

- wording
- severity
- timing
- consequence
- supporting evidence

If two findings describe the same remaining concern, report only the most complete version.

If one protocol statement creates multiple genuinely different risks, report each separately.

RATIONALE:

Each rationale should be approximately 4–6 sentences and explain:

- what the protocol states
- why the issue represents a risk
- what controls are already present
- how those controls reduce the risk
- any remaining limitation or opportunity for further strengthening

Avoid generic explanations.

RESIDUAL GAP:

The residual_gap describes any meaningful remaining limitation after considering existing protocol controls.

Examples include:

- ambiguity
- missing detail
- incomplete procedures
- insufficient monitoring
- implementation uncertainty
- opportunities to strengthen existing safeguards

If no meaningful limitation remains, return an empty string.

MITIGATIONS:
Return at most 10 mitigations.

Mitigations should be practical protocol improvements. Avoid generic mitigation text; each mitigation should be implementation-ready as a protocol requirement.

Mitigations must be concrete protocol actions (who does what, when/trigger, and specific criteria/thresholds when available).

When existing controls already adequately address the risk, mitigations should focus on optional enhancements rather than implying the protocol is inadequate.

If no meaningful limitation remains, return mitigations as [].

Each mitigation should clearly describe:

- what should be done
- who performs it when appropriate
- when it should occur
- any relevant criteria or thresholds

Avoid vague recommendations.

EVIDENCE:
Return at most 10 evidence snippets.

Each evidence excerpt must:

- be a direct quote or close paraphrase
- remain under 260 characters
- include page_start and page_end
- directly support the identified risk

GENERAL RULES:

- Return all meaningful distinct risks you find, up to 20 for this chunk.
- Do not duplicate the same issue under different wording.
- Do not collapse different risks into one broad umbrella risk.
- Include factor_triggers only using labels from the rubric above.
- The domain field must always be "{domain_name}".
- Do not infer evidence that is absent from the text.
- Do not report a standalone risk when the chunk already contains concrete control language that closes the gap (for example explicit monitoring, stopping criteria, emergency readiness, and procedural safeguards).
- Keep risk titles specific and descriptive.
- Categories should organize findings, while titles explain the individual issue.
""".strip()


DOMAIN_RISK_EXTRACTION_ADDENDA = {
    RiskDomain.HEALTH: """
DOMAIN-SPECIFIC FOCUS (HEALTH):
- Prioritize participant safety risks from adverse reactions, clinical deterioration, toxicity, and monitoring gaps.
- Emphasize unresolved safety uncertainty, emergency-readiness limitations, or safeguards that may fail under realistic conditions.

Use these risk factors as guidance. They are common examples of risk indicators but are not exhaustive or complete. They are provided as examples only and should not be treated as a checklist or as limiting the identification of other relevant risks.
Health risk factors:
{rubric}
""".strip(),
    RiskDomain.FINANCIAL: """
DOMAIN-SPECIFIC FOCUS (FINANCIAL):
- Prioritize trial participation burden, operational cost drivers, and protocol design choices that can increase dropout or enrollment inefficiency.
- Emphasize unresolved cost/logistics constraints that can materially affect feasibility, timeline, or sponsor/site resource load.

Use these risk factors as guidance. They are common examples of risk indicators but are not exhaustive or complete. They are provided as examples only and should not be treated as a checklist or as limiting the identification of other relevant risks.
Financial risk factors:
{rubric}
""".strip(),
    RiskDomain.ETHICAL: """
DOMAIN-SPECIFIC FOCUS (ETHICAL):
- Prioritize participant rights, autonomy, fairness, vulnerability protections, and consent quality concerns.
- Emphasize unresolved ethical gaps where protocol language may be insufficient to ensure equitable and informed participation.

Use these risk factors as guidance. They are common examples of risk indicators but are not exhaustive or complete. They are provided as examples only and should not be treated as a checklist or as limiting the identification of other relevant risks.
Ethical risk factors:
{rubric}
""".strip(),
    RiskDomain.ELIGIBILITY: """
DOMAIN-SPECIFIC FOCUS (ELIGIBILITY):
- Prioritize inclusion/exclusion complexity, restrictive criteria, and threshold interactions that can limit generalizability or recruitment.
- Emphasize unresolved eligibility ambiguity likely to create inconsistent screening decisions or unnecessary participant exclusion.

Use these risk factors as guidance. They are common examples of risk indicators but are not exhaustive or complete. They are provided as examples only and should not be treated as a checklist or as limiting the identification of other relevant risks.
Eligibility risk factors:
{rubric}
""".strip(),
    RiskDomain.INTERVENTION: """
DOMAIN-SPECIFIC FOCUS (INTERVENTION):
- Prioritize intervention-delivery risks from complexity, escalation, combination regimens, administration burden, and treatment modifications.
- Emphasize unresolved intervention execution gaps that can affect safety, adherence, treatment fidelity, or protocol consistency.

Use these risk factors as guidance. They are common examples of risk indicators but are not exhaustive or complete. They are provided as examples only and should not be treated as a checklist or as limiting the identification of other relevant risks.
Intervention risk factors:
{rubric}
""".strip(),
}


@dataclass
class RiskLlmClient:
    client: OpenAI
    model: str

    def extract_chunk_health_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return self._extract_chunk_risks_for_domain(
            chunk=chunk,
            domain=RiskDomain.HEALTH,
            rubric_factors=HEALTH_RISK_FACTORS,
        )

    def extract_chunk_financial_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return self._extract_chunk_risks_for_domain(
            chunk=chunk,
            domain=RiskDomain.FINANCIAL,
            rubric_factors=FINANCIAL_RISK_FACTORS,
        )

    def extract_chunk_ethical_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return self._extract_chunk_risks_for_domain(
            chunk=chunk,
            domain=RiskDomain.ETHICAL,
            rubric_factors=ETHICAL_RISK_FACTORS,
        )

    def extract_chunk_eligibility_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return self._extract_chunk_risks_for_domain(
            chunk=chunk,
            domain=RiskDomain.ELIGIBILITY,
            rubric_factors=ELIGIBILITY_RISK_FACTORS,
        )

    def extract_chunk_intervention_risks(self, chunk: DocumentChunk) -> list[RiskAssessment]:
        return self._extract_chunk_risks_for_domain(
            chunk=chunk,
            domain=RiskDomain.INTERVENTION,
            rubric_factors=INTERVENTION_RISK_FACTORS,
        )

    def _extract_chunk_risks_for_domain(
        self,
        chunk: DocumentChunk,
        domain: RiskDomain,
        rubric_factors: list[str],
    ) -> list[RiskAssessment]:
        rubric = self._render_domain_rubric(rubric_factors)
        domain_name = domain.value
        system_prompt = SHARED_RISK_EXTRACTION_SYSTEM_PROMPT.replace("{domain_name}", domain_name)
        domain_addendum_template = DOMAIN_RISK_EXTRACTION_ADDENDA.get(domain, "")
        domain_addendum = domain_addendum_template.replace("{rubric}", rubric)
        prompt_template = (
            SHARED_RISK_EXTRACTION_USER_PROMPT
            + "\n\n"
            + domain_addendum
            + "\n\nChunk id: {chunk_id}\nPages: {page_range}\nText:\n{chunk_text}"
        )
        user_prompt = (
            prompt_template
            .replace("{domain_name}", domain_name)
            .replace("{chunk_id}", str(chunk.chunk_id))
            .replace("{page_range}", str(f"{chunk.start_page}-{chunk.end_page}"))
            .replace("{chunk_text}", str(chunk.text))
        )

        raw = self._complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expected_json_shape='{"risks":[...]}',
            max_output_tokens=6000,
        )
        parsed = ChunkRiskResponse.model_validate(
            _normalize_chunk_risk_payload(raw, forced_domain=domain.value)
        )
        return parsed.risks

    def judge_aggregate_risks(self, risks: list[RiskAssessment]) -> list[RiskAssessment]:
        if not risks:
            return []
        merged = _merge_duplicate_risks(risks)
        merged = _merge_legacy_domain_title_risks(merged)
        return _merge_duplicate_risks(merged)

    def summarize(self, risks: list[RiskAssessment]) -> SummaryResponse:
        risk_payload = json.dumps([r.model_dump() for r in risks], ensure_ascii=True)
        system_prompt = (
            "You are a clinical trial risk analyst. Return strict JSON only."
            " Create an executive summary from structured risks."
        )
        user_prompt = f"""
Given these risks, produce JSON:
{{
  "executive_summary": "concise but complete summary",
  "key_assumptions": ["assumption 1", "assumption 2"]
}}

Risks JSON:
{risk_payload}
""".strip()
        raw = self._complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expected_json_shape='{"executive_summary":"...","key_assumptions":[...]}',
            max_output_tokens=1200,
        )
        return SummaryResponse.model_validate(raw)

    def classify_mitigation_statuses(self, risks: list[RiskAssessment]) -> list[RiskAssessment]:
        if not risks:
            return []

        updated_risks = list(risks)
        batch_size = 30
        for start in range(0, len(risks), batch_size):
            batch = risks[start : start + batch_size]
            payload = [
                {
                    "risk_index": start + idx,
                    "title": risk.title,
                    "domain": risk.domain.value,
                    "category": risk.category,
                    "likelihood": risk.likelihood,
                    "impact": risk.impact,
                    "confidence": risk.confidence,
                    "risk_score": risk.risk_score,
                    "rationale": risk.rationale,
                    "mitigations": risk.mitigations,
                    "evidence": [ev.model_dump() for ev in risk.evidence],
                }
                for idx, risk in enumerate(batch)
            ]
            risks_payload = json.dumps(payload, ensure_ascii=True)
            system_prompt = (
                "You are a clinical trial risk analyst. Return strict JSON only."
                " Classify each risk as unmitigated, partially mitigated, or adequately mitigated using semantic analysis of the risk text."
            )
            user_prompt = f"""
Classify each risk item as unmitigated, partially mitigated, or adequately mitigated.

Return JSON only:

{{
  "statuses": [
    {{
      "risk_index": integer,
      "mitigation_status": "unmitigated|partially_mitigated|adequately_mitigated",
      "status_reason": "brief evidence-based explanation grounded in the protocol"
    }}
  ]
}}

Definitions:

A risk is "unmitigated" if no adequate controls are identified for the specific risk.

A risk is "partially_mitigated" if some controls exist and reduce risk, but important residual gaps remain.

A risk is "adequately_mitigated" if existing controls appear sufficient. Any listed mitigations should be optional enhancements that further strengthen the protocol but are not required for acceptable control.

The following alone do NOT count as mitigation:
- Simply acknowledging that a risk exists
- Describing the risk without a control
- Generic statements such as "subjects will be monitored" without meaningful detail
- Controls that are unrelated to the identified risk

Rules:
- Evaluate only information explicitly provided in the protocol and the supplied risk item.
- Do not assume standard clinical practice unless it is documented.
- Judge whether the documented controls meaningfully address the specific risk.
- A serious risk can still be partially or adequately mitigated if controls are appropriate.
- If controls exist but are clearly insufficient for the identified risk, classify as partially_mitigated.
- Use adequately_mitigated only when controls are sufficient and no meaningful required controls are missing.
- If mitigation suggestions are listed, only use adequately_mitigated when those suggestions are clearly optional enhancements rather than required fixes.
- Base the decision on the meaning of the evidence, not keyword matching.
- Use the rationale, evidence excerpts, and scored fields together when available.
- Return exactly one status object for every input risk_index.
- Do not omit any risk indices.
- Keep status_reason concise (1–2 sentences) and cite the specific protocol control or the missing control.

Risk items JSON:
{risks_payload}
""".strip()
            raw = self._complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                expected_json_shape='{"statuses":[{"risk_index":0,"mitigation_status":"partially_mitigated","status_reason":"..."}]}',
                max_output_tokens=2600,
            )
            parsed = MitigationStatusResponse.model_validate(raw)
            status_by_index = {item.risk_index: item for item in parsed.statuses}
            for idx in range(start, start + len(batch)):
                item = status_by_index.get(idx)
                if item is None:
                    raise RuntimeError(
                        "Mitigation status classification output is incomplete. "
                        f"Missing risk_index {idx}."
                    )
                normalized_status = _normalize_mitigation_status_value(item.mitigation_status)
                normalized_status = _reconcile_status_with_mitigation_suggestions(
                    normalized_status,
                    updated_risks[idx].mitigations,
                )
                update_payload: dict[str, object] = {
                    "mitigation_status": normalized_status,
                    "mitigation_status_reason": item.status_reason.strip(),
                }
                updated_risks[idx] = updated_risks[idx].model_copy(
                    update=update_payload
                )
        return updated_risks

    def editor_agent_generate_patches(
        self,
        protocol_paragraphs: list[dict[str, str | int]],
        selected_risks: list[dict[str, object]],
    ) -> list[ProtocolEditPatch]:
        if not protocol_paragraphs or not selected_risks:
            return []

        protocol_payload = json.dumps(protocol_paragraphs, ensure_ascii=True)
        risks_payload = json.dumps(selected_risks, ensure_ascii=True)
        system_prompt = (
            "You are Editor Agent for clinical trial protocols."
            " Return strict JSON only."
            " Create precise protocol text edits that mitigate selected risks while preserving structure,"
            " regulatory tone, and style."
        )
        target_patch_count = max(1, min(len(selected_risks), 30))
        user_prompt = f"""
Generate protocol patch operations as JSON:

{{
  "patches": [
    {{
      "risk_title": "exact risk title",
      "action": "replace|insert_before|insert_after",
      "target_paragraph_id": "P",
      "new_text": "revised protocol paragraph text",
      "reason": "brief reason linked to risk mitigation"
    }}
  ]
}}

## Requirements

### 1. Scope and target selection

- Mitigate **only** the selected risks.
- Use only paragraph IDs from the provided candidate protocol paragraphs.
- Treat `evidence_anchor_paragraph_ids` as the primary **evidence locations** for understanding where and how each risk was identified. Do **not** assume that an evidence paragraph is automatically the correct location for mitigation.
- Evaluate whether an evidence paragraph is actually an appropriate mitigation target before editing it.
- Prefer an evidence paragraph as the edit target only when the paragraph itself contains language that can be meaningfully modified to reduce the selected risk.
- If an evidence paragraph identifies or demonstrates the risk but does not contain an appropriate control, requirement, procedure, or condition that can be modified to reduce the risk, do not force a patch into that paragraph.
- When appropriate, use a different `candidate_paragraph_id` that provides a more direct and topic-appropriate location for the mitigation.
- Prefer the most appropriate mitigation target among the provided evidence and candidate paragraphs based on direct relevance to the selected risk and the paragraph's existing topic and purpose.
- Do not select a paragraph solely because it was identified as evidence of the risk.
- Return at most **1 patch per selected risk**.
- Include a patch only when it produces a net reduction in overall protocol risk.
- Every selected risk includes `must_cover_mitigations`. Implement every required mitigation intent that is applicable to the selected mitigation target, unless equivalent control language already exists.
- Do not force a `must_cover_mitigations` intent into a paragraph when doing so would violate the paragraph's topic, purpose, preservation, or source-grounding requirements.
- If all applicable required mitigation intents cannot reasonably be implemented in one paragraph, use the most appropriate candidate paragraph that can complete the mitigation while remaining within that paragraph's existing topic and purpose.
- Do not address unselected risks.

### 2. Paragraph topic and purpose

The original paragraph's topic, purpose, and role in the protocol are hard boundaries.

- The revised paragraph must remain primarily about the same subject as the original.
- Every modification or addition must be relevant to **both**:
  1. the selected risk; and
  2. the original paragraph's subject and function.
- Do not expand a paragraph into unrelated areas such as eligibility, treatment administration, monitoring, recruitment, dropout, staffing, statistics, or other protocol topics unless that subject already belongs to the paragraph and is directly relevant to the selected risk.
- Ask: **"Would this information naturally belong in this paragraph even if the risk label were removed?"** If not, do not include it.
- If the risk cannot be meaningfully mitigated within the paragraph's existing topic, use a more appropriate candidate paragraph rather than forcing unrelated content into it.

### 3. Risk-to-paragraph relevance gate

Before generating a patch, determine whether the selected paragraph is an appropriate place to mitigate the selected risk.

A paragraph is an appropriate mitigation target only when its existing content directly contributes to the selected risk or contains a protocol decision, requirement, procedure, or condition that can reasonably be changed to reduce that risk.

Do NOT patch a paragraph merely because:

- the paragraph is loosely or indirectly related to the risk;
- changing the paragraph could theoretically reduce the risk;
- the paragraph contains a factor that might contribute to the risk;
- the paragraph is operationally adjacent to the risk;
- another risk-reduction strategy could theoretically be inserted into the paragraph.

The relationship must be substantive, not merely plausible.

Ask:

1. What specific existing statement in this paragraph contributes to the selected risk?
2. Can that statement be meaningfully changed to reduce the selected risk?
3. Would the paragraph still naturally be about the same subject after the change?
4. Is this paragraph a more appropriate mitigation location than the other provided candidate paragraphs?

If there is no clear existing statement that can be meaningfully modified to reduce the selected risk, do not generate a patch for that paragraph.

If another provided candidate paragraph is a more appropriate mitigation location, use that paragraph instead.

If no provided paragraph is sufficiently relevant to the selected risk, return no patch for that risk rather than creating an unrelated edit.

### 4. Preserve existing protocol information

A patch is an **edit to the existing protocol**, not a newly written replacement.

- Preserve all unrelated factual, procedural, eligibility, dosing, treatment, timing, population, endpoint, safety, and operational information.
- Do not remove, generalize, or compress information merely to make room for mitigation language.
- If an existing statement is directly related to the selected risk, modify it when doing so is the most direct way to reduce the risk.
- Do not preserve risk-related wording unchanged when it should be revised.
- When a sentence contains both risk-related and unrelated information, modify only the portion necessary to mitigate the risk and preserve the unrelated information.
- Do not rewrite unrelated sentences for style, consistency, or concision.
- Do not edit titles, author lists, sponsor metadata, or unrelated front matter.

### 5. Minimal, targeted editing

Make the **smallest substantive change that adequately mitigates the selected risk** while preserving the paragraph's original purpose.

- Prefer revising existing risk-related wording over adding new sentences.
- If the risk trigger appears in the target paragraph, directly revise or replace the risk-related language whenever feasible. Do not default to adding new language when the existing wording can be changed to achieve the mitigation.
- Do not leave risky wording unchanged and merely append mitigation language nearby when a direct revision is feasible.
- Add new sentences only when existing language cannot reasonably implement the mitigation.
- When new language is necessary, add the minimum amount required.
- Combine related mitigation intents into the smallest coherent revision rather than creating one sentence per mitigation item.
- Do not make a token deletion solely to create the appearance of an edit while adding disproportionately large new language.
- A small, localized risk should normally result in a small, localized edit.
- Larger rewrites are acceptable only when genuinely necessary to implement the required mitigation while preserving substantive original information.
- "Surgical" means narrowly targeted to the selected risk, not necessarily shorter than the original.

### 6. Replacement and insertion rules

- If the risk trigger appears in a paragraph, prefer `replace` with a minimally revised version of that paragraph.
- A replacement must remain recognizable as a revised version of the original paragraph.
- Do not rewrite an entire paragraph when a targeted sentence- or phrase-level revision is sufficient.
- Prefer replacement over insertion when both can achieve the mitigation without loss of relevant content.
- Use `insert_before` or `insert_after` only when direct revision of an existing paragraph is not feasible or would violate the paragraph's purpose or preservation requirements.

### 7. Source grounding

Do not invent protocol facts or operational requirements.

New factual or procedural content must be directly supported by at least one of:

- the provided protocol text;
- the selected risk;
- its `must_cover_mitigations`; or
- the provided candidate rewrite context.

Do not invent:

- numerical thresholds;
- time periods;
- eligibility criteria;
- treatment or dosing rules;
- monitoring schedules;
- staffing requirements;
- dropout or replacement-subject procedures;
- amendment procedures;
- new visits, tests, equipment, committees, or workflows;
- other operational requirements.

If the context does not support a specific detail, use a less specific but still actionable mitigation rather than inventing a precise requirement.

Do not add generic "good clinical practice" measures merely because they seem beneficial.

### 8. Risk-balance check

Before returning a patch, assess whether the proposed change could introduce participant-safety, operational, statistical, compliance, ethical, or data-quality risks.

- The mitigation should be expected to reduce the selected risk.
- Do not implement a mitigation if its newly introduced risks are likely to equal or exceed its benefit.
- Prefer simple, practical, and proportionate controls.
- Do not add procedures or requirements unless they are clearly necessary and source-grounded.
- Do not solve one risk by unnecessarily creating another.

### 9. Writing style

- Match the original paragraph's tone, terminology, scientific meaning, and general style.
- Use formal protocol language.
- Preserve the original scientific and procedural intent except where a change is necessary to mitigate the selected risk.
- Use direct requirement language such as "will," "must," or defined criteria only when supported by the provided context.
- Do not include explanations, justifications, or risk descriptions inside `new_text`.
- Do not add background narrative or broad contextual discussion.

### 10. Length and topic discipline

The revised paragraph must stay close in length and topic to the original.

- Aim for roughly the same word count as the original paragraph.
- If the original paragraph is N words, the revised paragraph should normally be no more than N + 30% words.
- Substantially exceeding this range is a strong signal that off-topic content has been added or that the mitigation is being forced into a paragraph where it does not belong.
- If achieving the mitigation requires adding significantly more text, reconsider whether this is the right paragraph for the mitigation. If it is, add only the minimum required and cut equivalent off-topic language elsewhere.
- Every word added must be directly necessary for the selected risk mitigation and must belong in this paragraph.
- Do not pad the paragraph with background context, rationale, or general protocol boilerplate.
- Do not add generic, wide-ranging, or extreme mitigation language simply to sound more protective; that language is off-topic and should be rejected.
- Any mitigation that increases the implied risk score or expands the paragraph beyond its original subject is not acceptable.

Before returning each patch, compare the proposed paragraph against the original target paragraph.

Verify all of the following:

**Target selection**

- The selected paragraph is an appropriate mitigation location, not merely an evidence location.
- The paragraph contains existing content that directly contributes to the selected risk or can reasonably be modified to reduce it.
- A more appropriate provided candidate paragraph was not overlooked.
- The patch does not force an unrelated mitigation into the paragraph.

**Preservation**

- Every unrelated factual or procedural statement is preserved.
- Any removed or modified information is directly related to the selected risk and intentionally changed for mitigation.
- No substantive information was omitted merely for brevity.

**Risk mitigation**

- The risk-related language has been meaningfully revised when revision is appropriate.
- Every applicable `must_cover_mitigations` intent is implemented or already covered by equivalent control language.
- The patch is expected to produce a net reduction in the selected risk.

**Topic and relevance**

- The revised paragraph remains within its original topic and purpose.
- Every new or substantially modified sentence directly contributes to mitigating the selected risk.
- Every new sentence naturally belongs in the paragraph even without the risk label.
- No unrelated risks or protocol topics have been introduced.

**Edit proportionality and length**

- The amount of new or modified language is proportionate to the mitigation need.
- A smaller targeted edit would not adequately achieve the same mitigation.
- No token edit is being used to justify an unnecessarily large addition.
- The word count of the revised paragraph must be within roughly 30% of the original's word count.
- If the revised paragraph is significantly longer, verify that every added word is directly necessary for the mitigation and belongs in this paragraph. If not, trim the paragraph before returning.

**Source grounding**

- Every new factual or procedural requirement is supported by the protocol text, selected risk, `must_cover_mitigations`, or candidate rewrite context.
- No unsupported thresholds, timelines, procedures, or operational requirements have been invented.

If any validation condition fails, revise the patch before returning it.

### 11. Output constraints

- Target about `{target_patch_count}` patches unless there is genuinely no editable text.
- Return at most 1 patch per selected risk.
- Return **valid JSON only**.
- Do not include markdown fences, commentary, explanations, or text outside the JSON object.

Selected risks JSON:
{risks_payload}

Candidate protocol paragraphs JSON:
{protocol_payload}
""".strip()
        raw = self._complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expected_json_shape='{"patches":[...]}',
            max_output_tokens=8000,
        )
        normalized_payload = _normalize_editor_patch_payload(
            raw,
            protocol_paragraphs=protocol_paragraphs,
            selected_risks=selected_risks,
        )
        parsed = EditorPatchResponse.model_validate(normalized_payload)
        return parsed.patches

    def editor_agent_rewrite_overlapping_paragraph(
        self,
        target_paragraph: dict[str, str | int],
        overlapping_risks: list[dict[str, object]],
        candidate_rewrites: list[dict[str, str]],
    ) -> tuple[str, str]:
        if not overlapping_risks or not candidate_rewrites:
            raise ValueError("Overlapping paragraph rewrite requires risks and candidate rewrites.")

        target_payload = json.dumps(target_paragraph, ensure_ascii=True)
        risks_payload = json.dumps(overlapping_risks, ensure_ascii=True)
        rewrites_payload = json.dumps(candidate_rewrites, ensure_ascii=True)
        system_prompt = (
            "You are Editor Agent for clinical trial protocols."
            " Return strict JSON only."
            " Rewrite one existing protocol paragraph so it mitigates all specified overlapping risks"
            " while preserving protocol tone, structure, and scientific intent."
        )
        user_prompt = f"""
Rewrite this one protocol paragraph as JSON:
{{
  "new_text": "single rewritten paragraph",
  "reason": "brief explanation of how the combined rewrite mitigates the applicable overlapping risks"
}}

## Objective
Produce one coherent replacement paragraph that addresses the overlapping risks **only to the extent they can appropriately be mitigated within this paragraph**.

The rewrite must:

- preserve the original paragraph's subject, purpose, role, scientific meaning, and necessary substantive information;
- incorporate only mitigations that naturally belong in the paragraph;
- produce a net reduction in applicable risk without unnecessarily shifting risk elsewhere; and
- read as one integrated protocol requirement, not as multiple risk rewrites merged together.
A listed overlapping risk does **not** mean its candidate rewrite must be included. If a risk cannot appropriately be mitigated within this paragraph, do not force it into the rewrite.

## 1. Paragraph scope
The original paragraph's topic and purpose are hard boundaries.

Every new or substantially modified statement must:

1. relate to the original paragraph's subject and purpose; and
2. directly mitigate an applicable overlapping risk.
For each proposed change, ask:

**"Would this information naturally belong in this paragraph even if the risk label were removed?"**

If not, exclude it.

Do not introduce unrelated protocol topics or controls merely because they could be beneficial.

## 2. Risk applicability
For each overlapping risk, first determine whether it can meaningfully be mitigated within this paragraph.

Consider:

1. What existing statement is relevant to the risk?
2. Can it be meaningfully modified to reduce the risk?
3. Would the modification remain within the paragraph's original subject and purpose?
4. Is the proposed mitigation supported by the provided context?
Only apply a mitigation when the risk can appropriately be addressed here.

Do not make a token edit merely to show that every listed risk was addressed.

## 3. Preserve the original
This is a **targeted edit**, not a rewrite from scratch.

Preserve all unrelated substantive information, including factual, scientific, procedural, eligibility, dosing, treatment, timing, population, endpoint, safety, and operational information.

- Do not remove information merely to make room for mitigation.
- Do not generalize or compress substantive information solely for concision.
- If a sentence contains both risk-related and unrelated information, modify only what is necessary and preserve the rest.
- Prefer minimal cuts to the original paragraph; when the original language can be revised in place, do that instead of replacing most of the paragraph.
- Allow a longer rewrite only when the mitigation genuinely requires it and the paragraph still remains clearly about the same topic.
- Do not change scientific or procedural meaning unless necessary for mitigation.
- Keep the final paragraph close in word count to the original. If the original is N words, aim for no more than N + 30% words. A significantly longer paragraph almost always signals forced mitigations or off-topic additions that were not in the original.
- If the rewrite exceeds this range, verify that every added word is directly necessary for the mitigation and belongs in this paragraph. If not, remove it before returning.
- Do not add generic, inflated, or cross-topic mitigation language simply to sound more protective; it should be rejected if it feels like a new paragraph rather than a revision of the original.
- A better mitigation is a directly relevant one, not a longer or more dramatic one.
Before output, classify every substantive original statement as:

1. **Preserved**;
2. **Intentionally modified** because it directly relates to an applicable mitigation; or
3. **Intentionally removed** because it directly conflicts with a necessary mitigation.
If unrelated original information is missing, restore it.

## 4. Integrate; do not accumulate
Treat `rewrites_payload` as **candidate solutions, not instructions**.

A candidate rewrite is proposed source material, not text that must be preserved.

- Evaluate each candidate for relevance, necessity, source support, and compatibility with the paragraph.
- Do not mechanically combine or append candidate rewrites.
- Prefer modifying existing risk-related language over adding new sentences.
- Consolidate overlapping or compatible mitigations into the smallest coherent change.
- Add new language only when the existing paragraph cannot reasonably implement the mitigation.
- Do not create one sentence per risk unless genuinely necessary.
The final paragraph should read as though it was originally written as one coherent protocol paragraph.

## 5. Source grounding
Do not invent protocol facts or operational requirements.

Every new factual or procedural requirement must be supported by the target paragraph, overlapping risks, or candidate rewrites.

Do not invent:

- numerical thresholds or time periods;
- eligibility, dosing, treatment, or monitoring requirements;
- staffing, dropout, replacement-subject, amendment, or workflow procedures;
- new visits, tests, equipment, committees, or other operational steps.
Do not add generic best-practice controls solely because they seem beneficial.

If a specific mitigation detail is unsupported, do not invent precision.

## 6. Proportionality and risk balance
Make the **smallest substantive change that adequately addresses the applicable risks**.

- Prefer modifying existing wording over adding new requirements.
- Prefer simple, practical, proportional controls.
- Do not add language merely because multiple risks are present.
- Do not solve one risk by unnecessarily creating another.
- Do not introduce new risk domains or hazard topics that are not present in the target paragraph or overlapping risks.

## 7. Writing requirements

- Return valid JSON only.
- `new_text` must contain exactly one replacement paragraph.
- Do not use multiple paragraphs or bullet points in `new_text`.
- Use formal protocol language and preserve the original terminology and tone.
- Prefer "will" or "must" when supported by the context.
- Do not use unsupported mandatory language.
- Do not include risk explanations, justifications, or commentary inside `new_text`.

## 8. Final validation
Before returning the JSON, verify:

- all necessary original information is preserved;
- unrelated information has not been removed;
- the paragraph remains about the same subject and function;
- every new or substantially modified statement directly addresses an applicable overlapping risk;
- each risk that cannot appropriately be mitigated here has been left out;
- candidate rewrites have been treated as suggestions and integrated rather than concatenated;
- no unsupported protocol requirements were invented;
- the rewrite is proportionate to the mitigation need;
- redundant controls have been consolidated;
- the final paragraph is coherent and reads as one original protocol requirement; and
- the overall risk profile is improved rather than merely shifted.
If any condition fails, revise the rewrite before returning it.

Target paragraph JSON:
{target_payload}

Overlapping risks JSON:
{risks_payload}

Candidate rewrites JSON:
{rewrites_payload}
""".strip()
        raw = self._complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expected_json_shape='{"new_text":"...","reason":"..."}',
            max_output_tokens=2200,
        )
        parsed = ParagraphRewriteResponse.model_validate(raw)
        return parsed.new_text.strip(), parsed.reason.strip()

    def _complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_json_shape: str,
        max_output_tokens: int,
    ) -> dict:
        text = self._request_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )

        try:
            return _parse_model_json(text)
        except json.JSONDecodeError:
            repaired_text = self._repair_json_response(
                malformed_text=text,
                expected_json_shape=expected_json_shape,
            )
            try:
                return _parse_model_json(repaired_text)
            except json.JSONDecodeError:
                salvaged_original = _salvage_partial_risks_json(text)
                if salvaged_original is not None:
                    return salvaged_original
                salvaged_repaired = _salvage_partial_risks_json(repaired_text)
                if salvaged_repaired is not None:
                    return salvaged_repaired
                exc = json.JSONDecodeError("Invalid JSON", repaired_text, 0)
                raise RuntimeError(
                    "Model did not return valid JSON after repair attempt: "
                    f"{repaired_text[:500]}"
                ) from exc

    def _request_text(self, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_output_tokens=max_output_tokens,
        )
        return response.output_text.strip()

    def _repair_json_response(self, malformed_text: str, expected_json_shape: str) -> str:
        repair_system_prompt = (
            "You repair malformed JSON. Return valid JSON only with no markdown and no commentary."
            " Preserve the original content where possible."
            " If the trailing item is incomplete, remove it instead of inventing missing values."
        )
        repair_user_prompt = f"""
Repair this malformed JSON so it becomes valid JSON matching this shape:
{expected_json_shape}

Malformed JSON:
{malformed_text}
""".strip()
        return self._request_text(
            system_prompt=repair_system_prompt,
            user_prompt=repair_user_prompt,
            max_output_tokens=3500,
        )

    def _render_domain_rubric(self, factors: list[str]) -> str:
        return "\n".join(f"- {factor}" for factor in factors)


def build_openai_client(api_key: str, base_url: str | None) -> OpenAI:
    return build_openai_client_with_resilience(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=120.0,
        max_retries=2,
    )


def build_openai_client_with_resilience(
    api_key: str,
    base_url: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> OpenAI:
    timeout = max(float(timeout_seconds), 10.0)
    retries = max(int(max_retries), 0)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries)
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=retries)


def _parse_model_json(text: str) -> dict:
    cleaned = text.strip()

    # Handle fenced markdown JSON blocks: ```json ... ```
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: isolate the largest JSON object-like region.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _normalize_chunk_risk_payload(payload: dict, forced_domain: str) -> dict:
    risks_raw = payload.get("risks", [])
    if not isinstance(risks_raw, list):
        return {"risks": []}

    normalized_risks: list[dict] = []
    for item in risks_raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        category = str(item.get("category", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        residual_gap = str(item.get("residual_gap", "")).strip()
        if len(title) < 3 or len(category) < 2 or len(rationale) < 10:
            continue

        mitigations_raw = item.get("mitigations", [])
        factor_triggers_raw = item.get("factor_triggers", [])
        evidence_raw = item.get("evidence", [])

        mitigations = [str(m).strip() for m in mitigations_raw if str(m).strip()] if isinstance(
            mitigations_raw, list
        ) else []
        factor_triggers = [
            str(f).strip() for f in factor_triggers_raw if str(f).strip()
        ] if isinstance(factor_triggers_raw, list) else []

        normalized_evidence: list[dict] = []
        if isinstance(evidence_raw, list):
            for ev in evidence_raw:
                if not isinstance(ev, dict):
                    continue
                excerpt = str(ev.get("excerpt", "")).strip()
                if not excerpt:
                    continue
                page_start = _clamp_int(_coerce_int(ev.get("page_start"), 1), 1, 10**6)
                page_end = _clamp_int(_coerce_int(ev.get("page_end"), page_start), 1, 10**6)
                if page_end < page_start:
                    page_end = page_start
                normalized_evidence.append(
                    {
                        "excerpt": excerpt,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )

        likelihood = _clamp_int(_coerce_int(item.get("likelihood"), 3), 1, 5)
        impact = _clamp_int(_coerce_int(item.get("impact"), 3), 1, 5)
        confidence = round(_clamp_float(_coerce_float(item.get("confidence"), 0.7), 0.0, 1.0), 2)
        likelihood, impact, confidence = _calibrate_residual_risk_scores(
            likelihood=likelihood,
            impact=impact,
            confidence=confidence,
            rationale=rationale,
            residual_gap=residual_gap,
            mitigations=mitigations,
            evidence=normalized_evidence,
        )

        normalized_risks.append(
            {
                "title": title,
                "domain": forced_domain,
                "category": category,
                "likelihood": likelihood,
                "impact": impact,
                "confidence": confidence,
                "rationale": rationale,
                "residual_gap": residual_gap,
                "mitigations": mitigations[:10],
                "factor_triggers": factor_triggers,
                "evidence": normalized_evidence[:10],
            }
        )
    filtered_risks = [
        risk
        for risk in normalized_risks
        if not _looks_like_fully_mitigated_statement(risk)
    ]
    return {"risks": filtered_risks}


def _normalize_editor_patch_payload(
    payload: dict,
    *,
    protocol_paragraphs: list[dict[str, str | int]],
    selected_risks: list[dict[str, object]],
) -> dict:
    patches_raw = payload.get("patches", [])
    if not isinstance(patches_raw, list):
        return {"patches": []}

    valid_paragraph_ids = {
        str(item.get("paragraph_id", "")).strip()
        for item in protocol_paragraphs
        if isinstance(item, dict) and str(item.get("paragraph_id", "")).strip()
    }
    selected_risk_titles = {
        _normalize_text(str(item.get("risk_title", item.get("title", ""))))
        for item in selected_risks
        if isinstance(item, dict) and str(item.get("risk_title", item.get("title", ""))).strip()
    }
    normalized_patches: list[dict[str, str]] = []
    for item in patches_raw:
        if not isinstance(item, dict):
            continue
        risk_title = str(item.get("risk_title", "")).strip()
        normalized_risk_title = _normalize_text(risk_title)
        action = _normalize_editor_patch_action(item.get("action"))
        target_paragraph_id = str(item.get("target_paragraph_id", "")).strip()
        new_text = str(item.get("new_text", "")).strip()
        reason = str(item.get("reason", "")).strip()

        if selected_risk_titles and normalized_risk_title not in selected_risk_titles:
            continue
        if action is None:
            continue
        if target_paragraph_id not in valid_paragraph_ids:
            continue
        if len(risk_title) < 3 or len(new_text) < 20:
            continue
        if len(reason) < 10:
            reason = f"Mitigates {risk_title.lower()}."

        normalized_patches.append(
            {
                "risk_title": risk_title,
                "action": action,
                "target_paragraph_id": target_paragraph_id,
                "new_text": new_text,
                "reason": reason,
            }
        )
    return {"patches": normalized_patches}


def _calibrate_residual_risk_scores(
    *,
    likelihood: int,
    impact: int,
    confidence: float,
    rationale: str,
    residual_gap: str,
    mitigations: list[str],
    evidence: list[dict],
) -> tuple[int, int, float]:
    gap_text = residual_gap.lower().strip()
    rationale_text = rationale.lower()
    if not gap_text:
        return _calibrate_uncontrolled_severe_risk_scores(
            likelihood=likelihood,
            impact=impact,
            confidence=confidence,
            rationale=rationale_text,
            residual_gap=gap_text,
            mitigations=mitigations,
            evidence=evidence,
        )

    context = " ".join(
        [
            rationale_text,
            gap_text,
            " ".join(m.lower() for m in mitigations),
            " ".join(str(ev.get("excerpt", "")).lower() for ev in evidence if isinstance(ev, dict)),
        ]
    )
    control_hits = _effective_control_hits(context, gap_text)
    if control_hits < 2:
        return _calibrate_uncontrolled_severe_risk_scores(
            likelihood=likelihood,
            impact=impact,
            confidence=confidence,
            rationale=rationale_text,
            residual_gap=gap_text,
            mitigations=mitigations,
            evidence=evidence,
        )

    gap_strength = _residual_gap_signal_strength(gap_text)
    adjusted_likelihood = likelihood
    adjusted_impact = impact
    adjusted_confidence = confidence

    adjusted_likelihood = min(adjusted_likelihood, 3)
    adjusted_confidence = min(adjusted_confidence, 0.8)

    if control_hits >= 4 and gap_strength <= 1:
        adjusted_likelihood = min(adjusted_likelihood, 2)
        adjusted_confidence = min(adjusted_confidence, 0.65)

    if _is_procedural_residual_gap(gap_text) and control_hits >= 3 and gap_strength <= 2:
        adjusted_impact = min(adjusted_impact, 4)
        if control_hits >= 4:
            adjusted_impact = min(adjusted_impact, 3)

    return (
        _clamp_int(adjusted_likelihood, 1, 5),
        _clamp_int(adjusted_impact, 1, 5),
        round(_clamp_float(adjusted_confidence, 0.0, 1.0), 2),
    )


def _calibrate_uncontrolled_severe_risk_scores(
    *,
    likelihood: int,
    impact: int,
    confidence: float,
    rationale: str,
    residual_gap: str,
    mitigations: list[str],
    evidence: list[dict],
) -> tuple[int, int, float]:
    context = " ".join(
        [
            rationale,
            residual_gap,
            " ".join(m.lower() for m in mitigations),
            " ".join(str(ev.get("excerpt", "")).lower() for ev in evidence if isinstance(ev, dict)),
        ]
    )
    severe_hits = _count_severe_risk_markers(context)
    unresolved_hits = _residual_gap_signal_strength(residual_gap)
    control_hits = _effective_control_hits(context, residual_gap)
    if severe_hits <= 0 or unresolved_hits <= 0 or control_hits >= 2:
        return likelihood, impact, confidence

    adjusted_likelihood = max(likelihood, 3)
    adjusted_impact = max(impact, 4)
    adjusted_confidence = max(confidence, 0.75)
    if severe_hits >= 2:
        adjusted_likelihood = max(adjusted_likelihood, 4)
        adjusted_confidence = max(adjusted_confidence, 0.82)
    if _has_critical_severity_marker(context):
        adjusted_impact = 5
        adjusted_confidence = max(adjusted_confidence, 0.88)

    return (
        _clamp_int(adjusted_likelihood, 1, 5),
        _clamp_int(adjusted_impact, 1, 5),
        round(_clamp_float(adjusted_confidence, 0.0, 1.0), 2),
    )


def _looks_like_fully_mitigated_statement(risk_payload: dict[str, object]) -> bool:
    rationale = str(risk_payload.get("rationale", "")).lower()
    residual_gap = str(risk_payload.get("residual_gap", "")).lower().strip()
    mitigations_raw = risk_payload.get("mitigations", [])
    mitigations_list = mitigations_raw if isinstance(mitigations_raw, list) else []
    mitigations = " ".join(str(item).lower() for item in mitigations_list)
    evidence = " ".join(
        str(item.get("excerpt", "")).lower()
        for item in (risk_payload.get("evidence", []) if isinstance(risk_payload.get("evidence", []), list) else [])
        if isinstance(item, dict)
    )
    context = " ".join([rationale, residual_gap, mitigations, evidence])
    if not context.strip():
        return False

    if residual_gap and _text_has_unresolved_gap_signal(residual_gap):
        return False

    unresolved_markers = (
        "insufficient",
        "inadequate",
        "unclear",
        "not defined",
        "undefined",
        "missing",
        "no clear",
        "no documented",
        "not specified",
        "not available",
        "not addressed",
        "not monitored",
        "without ",
        "lack ",
        "lacks ",
        "limited ",
        "unknown ",
        "uncertain",
        "however",
        "despite",
        "remains",
        "still ",
        "yet ",
        "gap",
    )
    if any(marker in context for marker in unresolved_markers):
        return False

    residual_risk_markers = (
        "high risk",
        "elevated risk",
        "unacceptable risk",
        "potential harm",
        "likely to cause",
        "risk remains",
    )
    if any(marker in context for marker in residual_risk_markers):
        return False

    control_hits = _effective_control_hits(context, residual_gap)
    if control_hits >= 3:
        return True

    # If controls are present and no unresolved gap is explicit, treat as mitigated.
    if control_hits >= 2 and " will " in f" {context} ":
        return True
    if not residual_gap and control_hits >= 1:
        return True
    return False


def _text_has_unresolved_gap_signal(text: str) -> bool:
    unresolved_markers = (
        "insufficient",
        "inadequate",
        "unclear",
        "not defined",
        "undefined",
        "missing",
        "no clear",
        "no documented",
        "not specified",
        "not available",
        "not addressed",
        "not monitored",
        "without ",
        "lack ",
        "lacks ",
        "limited ",
        "unknown ",
        "uncertain",
        "however",
        "despite",
        "remains",
        "still ",
        "yet ",
        "gap",
    )
    return any(marker in text for marker in unresolved_markers)


def _residual_gap_signal_strength(text: str) -> int:
    unresolved_markers = (
        "insufficient",
        "inadequate",
        "unclear",
        "not defined",
        "undefined",
        "missing",
        "no clear",
        "no documented",
        "not specified",
        "not available",
        "not addressed",
        "not monitored",
        "without ",
        "lack ",
        "lacks ",
        "limited ",
        "unknown ",
        "uncertain",
        "however",
        "despite",
        "remains",
        "still ",
        "yet ",
        "gap",
    )
    return sum(1 for marker in unresolved_markers if marker in text)


def _is_procedural_residual_gap(text: str) -> bool:
    procedural_markers = (
        "criteria",
        "threshold",
        "timing",
        "frequency",
        "schedule",
        "workflow",
        "process",
        "documentation",
        "defined",
        "specified",
        "clarity",
        "unclear",
    )
    return any(marker in text for marker in procedural_markers)


def _count_control_markers(text: str) -> int:
    control_markers = (
        "to mitigate",
        "will be monitored",
        "will be assessed",
        "will be documented",
        "will be managed",
        "will be reviewed",
        "monitoring",
        "observation",
        "stopping criteria",
        "stopping rule",
        "emergency",
        "rescue medication",
        "trained personnel",
        "physician present",
        "protocol requires",
        "required",
        "criteria",
        "safeguard",
        "dose adjustment",
        "escalation rule",
        "follow-up",
    )
    return sum(1 for marker in control_markers if marker in text)


def _effective_control_hits(context: str, residual_gap: str) -> int:
    control_hits = _count_control_markers(context)
    if control_hits <= 0:
        return 0
    gap_penalty = min(_residual_gap_signal_strength(residual_gap), control_hits)
    return max(control_hits - gap_penalty, 0)


def _count_severe_risk_markers(text: str) -> int:
    severe_markers = (
        "severe",
        "life-threatening",
        "death",
        "fatal",
        "anaphylaxis",
        "critical",
        "hemorrhage",
        "organ failure",
        "irreversible",
        "icu",
        "hospitalization",
        "serious adverse",
        "major protocol violation",
    )
    return sum(1 for marker in severe_markers if marker in text)


def _has_critical_severity_marker(text: str) -> bool:
    critical_markers = (
        "death",
        "fatal",
        "life-threatening",
        "anaphylaxis",
        "organ failure",
        "irreversible",
        "icu",
    )
    return any(marker in text for marker in critical_markers)


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _salvage_partial_risks_json(text: str) -> dict | None:
    start_idx = text.find("[")
    if start_idx == -1:
        return None

    objects: list[dict] = []
    in_string = False
    escape = False
    brace_depth = 0
    obj_start = -1

    for i in range(start_idx, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if brace_depth == 0:
                obj_start = i
            brace_depth += 1
        elif ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0 and obj_start != -1:
                    candidate = text[obj_start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    obj_start = -1
        elif ch == "]" and brace_depth == 0:
            break

    if objects:
        return {"risks": objects}
    return None


def _merge_duplicate_risks(risks: list[RiskAssessment]) -> list[RiskAssessment]:
    if not risks:
        return []

    adjacency: list[set[int]] = [set() for _ in risks]
    for i in range(len(risks)):
        for j in range(i + 1, len(risks)):
            if _are_near_duplicate_risks(risks[i], risks[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)

    merged: list[RiskAssessment] = []
    visited = [False] * len(risks)
    for i in range(len(risks)):
        if visited[i]:
            continue
        stack = [i]
        group_indices: list[int] = []
        visited[i] = True
        while stack:
            idx = stack.pop()
            group_indices.append(idx)
            for neighbor in adjacency[idx]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        merged.append(_merge_risk_group([risks[idx] for idx in group_indices]))

    return merged


def _merge_legacy_domain_title_risks(risks: list[RiskAssessment]) -> list[RiskAssessment]:
    grouped: list[list[RiskAssessment]] = []
    for risk in risks:
        title_key = _normalize_risk_title(risk.title)

        matched_group: list[RiskAssessment] | None = None
        for group in grouped:
            if _title_matches_group(risk, title_key, group):
                matched_group = group
                break

        if matched_group is None:
            grouped.append([risk])
        else:
            matched_group.append(risk)

    merged: list[RiskAssessment] = []
    for group in grouped:
        merged.append(_merge_risk_group(group))
    return merged


def _are_near_duplicate_risks(a: RiskAssessment, b: RiskAssessment) -> bool:
    a_title = _normalize_risk_title(a.title)
    b_title = _normalize_risk_title(b.title)
    sequence_similarity = SequenceMatcher(
        None,
        a_title,
        b_title,
    ).ratio()
    token_overlap = _token_jaccard_similarity(a_title, b_title)
    category_similarity = _category_similarity(a.category, b.category)
    factor_overlap = _list_overlap_score(a.factor_triggers, b.factor_triggers)
    same_domain = a.domain == b.domain
    a_tokens = set(a_title.split())
    b_tokens = set(b_title.split())

    # Strong title match alone is enough to merge near-duplicate phrasings.
    if sequence_similarity >= 0.9 or token_overlap >= 0.8:
        return True

    # Domain-specific concept merge for placebo/direct-benefit phrasing variants.
    if {"placebo", "benefit"}.issubset(a_tokens) and {"placebo", "benefit"}.issubset(b_tokens):
        if token_overlap >= 0.5 or sequence_similarity >= 0.64:
            return True

    # Slightly weaker title match can still merge when category/factor context aligns.
    if (sequence_similarity >= 0.74 or token_overlap >= 0.58) and (
        same_domain or category_similarity >= 0.5 or factor_overlap >= 0.34
    ):
        return True

    # For cross-domain duplicates, require stronger title plus contextual alignment.
    if (sequence_similarity >= 0.8 or token_overlap >= 0.66) and (
        category_similarity >= 0.54 or factor_overlap >= 0.38
    ):
        return True

    return False


def _merge_risk_group(group: list[RiskAssessment]) -> RiskAssessment:
    if len(group) == 1:
        return group[0]

    head = _select_group_head(group)
    seen_mitigations = set()
    mitigations: list[str] = []
    seen_factors = set()
    factors: list[str] = []
    seen_evidence = set()
    evidences = []

    likelihood_sum = 0.0
    impact_sum = 0.0
    confidence_sum = 0.0
    weight_sum = 0.0

    for risk in group:
        weight = max(risk.confidence, 0.01)
        likelihood_sum += risk.likelihood * weight
        impact_sum += risk.impact * weight
        confidence_sum += risk.confidence
        weight_sum += weight

        for mitigation in risk.mitigations:
            normalized = _normalize_text(mitigation)
            if normalized not in seen_mitigations:
                seen_mitigations.add(normalized)
                mitigations.append(mitigation.strip())

        for factor in risk.factor_triggers:
            normalized = _normalize_text(factor)
            if normalized not in seen_factors:
                seen_factors.add(normalized)
                factors.append(factor.strip())

        for ev in risk.evidence:
            key = (ev.page_start, ev.page_end, _normalize_text(ev.excerpt))
            if key not in seen_evidence:
                seen_evidence.add(key)
                evidences.append(ev)

    return RiskAssessment(
        title=head.title,
        domain=head.domain,
        category=head.category,
        likelihood=_clamp_1_to_5(round(likelihood_sum / weight_sum)),
        impact=_clamp_1_to_5(round(impact_sum / weight_sum)),
        confidence=round(min(confidence_sum / len(group), 1.0), 2),
        rationale=head.rationale,
        mitigations=mitigations,
        factor_triggers=factors,
        evidence=evidences,
    )


def _select_group_head(group: list[RiskAssessment]) -> RiskAssessment:
    domain_priority = {
        RiskDomain.INTERVENTION: 5,
        RiskDomain.ELIGIBILITY: 4,
        RiskDomain.ETHICAL: 3,
        RiskDomain.HEALTH: 2,
        RiskDomain.FINANCIAL: 1,
    }
    return max(
        group,
        key=lambda r: (
            domain_priority.get(r.domain, 0),
            r.confidence,
            r.risk_score,
            len(r.evidence),
        ),
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _normalize_mitigation_status_value(status: str) -> str:
    normalized = _normalize_text(status).replace(" ", "_")
    if normalized == "completely_mitigated":
        return "adequately_mitigated"
    if normalized == "mitigated":
        return "partially_mitigated"
    if normalized in {"unmitigated", "partially_mitigated", "adequately_mitigated"}:
        return normalized
    return "unmitigated"


def _reconcile_status_with_mitigation_suggestions(status: str, mitigations: list[str]) -> str:
    has_suggestions = any(_normalize_text(item) for item in mitigations)
    if (not has_suggestions) and status in {"partially_mitigated", "mitigated"}:
        return "adequately_mitigated"
    return status


def _normalize_editor_patch_action(action: object) -> str | None:
    normalized = _normalize_text(str(action or "")).replace("-", "_").replace(" ", "_")
    if normalized in {"replace", "insert_before", "insert_after"}:
        return normalized
    if normalized in {"", "none", "no_patch", "no_edit", "skip", "n_a", "na"}:
        return None
    return None


def _clamp_1_to_5(value: int) -> int:
    if value < 1:
        return 1
    if value > 5:
        return 5
    return value


def _normalize_risk_title(title: str) -> str:
    normalized = _normalize_text(title).translate(str.maketrans("", "", punctuation))
    tokens = normalized.split()
    ignore_tokens = {
        "risk",
        "of",
        "during",
        "severe",
        "serious",
        "possible",
        "potential",
        "major",
        "minor",
        "due",
        "to",
        "with",
        "without",
        "from",
        "assignment",
        "use",
        "affecting",
        "limiting",
        "control",
        "controls",
        "concern",
        "concerns",
        "issue",
        "issues",
        "uncertainty",
        "uncertainties",
        "associated",
        "related",
        "may",
        "under",
        "limit",
        "limited",
        "guaranteed",
        "guarantee",
        "lack",
        "absence",
    }
    canonical_tokens = [_normalize_title_token(t) for t in tokens if t not in ignore_tokens]
    canonical_tokens = [t for t in canonical_tokens if t]
    return " ".join(canonical_tokens)


def _stem_token(token: str) -> str:
    if token.endswith("phie"):
        return token[:-4] + "phy"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _normalize_title_token(token: str) -> str:
    stemmed = _stem_token(token)
    canonical_map = {
        "immunotheraphy": "immunotherapy",
        "anaphylaxi": "anaphylaxis",
        "allergi": "allergic",
        "advers": "adverse",
        "event": "event",
        "events": "event",
        "reaction": "event",
        "reactions": "event",
        "toxicity": "event",
        "toxicities": "event",
        "placebo": "placebo",
        "participant": "participant",
        "benefit": "benefit",
        "direct": "benefit",
        "uncertain": "uncertain",
    }
    return canonical_map.get(stemmed, stemmed)


def _token_jaccard_similarity(a_title: str, b_title: str) -> float:
    a_tokens = set(a_title.split())
    b_tokens = set(b_title.split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens.intersection(b_tokens))
    union = len(a_tokens.union(b_tokens))
    return intersection / union


def _category_similarity(a_category: str, b_category: str) -> float:
    a_norm = _normalize_text(a_category)
    b_norm = _normalize_text(b_category)
    if a_norm == b_norm:
        return 1.0
    seq = SequenceMatcher(None, a_norm, b_norm).ratio()
    token = _list_overlap_score(a_norm.split(), b_norm.split())
    return max(seq, token)


def _list_overlap_score(a_items: list[str], b_items: list[str]) -> float:
    a_norm = {_normalize_text(x) for x in a_items if _normalize_text(x)}
    b_norm = {_normalize_text(x) for x in b_items if _normalize_text(x)}
    if not a_norm or not b_norm:
        return 0.0
    intersection = len(a_norm.intersection(b_norm))
    union = len(a_norm.union(b_norm))
    return intersection / union


def _title_matches_group(
    candidate: RiskAssessment,
    title_key: str,
    group: list[RiskAssessment],
) -> bool:
    title_tokens = set(title_key.split())
    for member in group:
        member_title_key = _normalize_risk_title(member.title)
        sequence_similarity = SequenceMatcher(None, title_key, member_title_key).ratio()
        token_similarity = _token_jaccard_similarity(title_key, member_title_key)
        category_similarity = _category_similarity(candidate.category, member.category)
        factor_overlap = _list_overlap_score(candidate.factor_triggers, member.factor_triggers)
        same_domain = candidate.domain == member.domain
        member_tokens = set(member_title_key.split())
        subset_overlap = 0.0
        if title_tokens and member_tokens:
            subset_overlap = min(
                len(title_tokens.intersection(member_tokens)) / len(title_tokens),
                len(title_tokens.intersection(member_tokens)) / len(member_tokens),
            )

        # Always merge unmistakably near-identical titles.
        if sequence_similarity >= 0.82 or token_similarity >= 0.68 or subset_overlap >= 0.88:
            return True
        # For weaker title similarity, require contextual alignment to avoid over-merging.
        if (sequence_similarity >= 0.68 or token_similarity >= 0.48 or subset_overlap >= 0.75) and (
            same_domain or category_similarity >= 0.56 or factor_overlap >= 0.4
        ):
            return True
    return False
