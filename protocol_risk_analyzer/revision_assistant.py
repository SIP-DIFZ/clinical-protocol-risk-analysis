from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from protocol_risk_analyzer.models import AnalysisResult


class RevisionProposal(BaseModel):
    summary: str = Field(min_length=6)
    requested_change: str = Field(min_length=8)
    rationale: str = Field(min_length=12)
    affected_sections: list[str] = Field(default_factory=list)
    estimated_edits: int = Field(ge=1, le=50)
    focus: Literal["health", "financial", "ethical", "eligibility", "intervention", "operational", "general"] = "general"


class RevisionPlannerPayload(BaseModel):
    intent: Literal["analysis_question", "revision_request", "out_of_scope"]
    assistant_reply: str = Field(min_length=1)
    proposals: list[RevisionProposal] = Field(default_factory=list)


class ProposalEditPayload(BaseModel):
    updated_text: str = Field(min_length=20)
    reason: str = Field(min_length=8)


@dataclass(frozen=True)
class RevisionPlannerResult:
    payload: RevisionPlannerPayload


_REVISION_PLANNER_RESPONSE_SHAPE = """
{
  "intent": "analysis_question | revision_request | out_of_scope",
  "assistant_reply": "markdown response text",
  "proposals": [
    {
      "summary": "short plan item title",
      "requested_change": "specific protocol change request",
      "rationale": "why this change reduces protocol risk",
      "affected_sections": ["Section A", "Section B"],
      "estimated_edits": 3,
      "focus": "health|financial|ethical|eligibility|intervention|operational|general"
    }
  ]
}
""".strip()

_PROPOSAL_EDIT_RESPONSE_SHAPE = """
{
  "updated_text": "single revised paragraph",
  "reason": "brief explanation of how the updated rewrite addresses the user's requested change while staying on topic"
}
""".strip()

_REVISION_PLANNER_SYSTEM_PROMPT = """
You are an AI Protocol Revision Assistant for a clinical trial protocol risk analysis application.
Return strict JSON only.
You must stay grounded in:
1) uploaded protocol analysis content,
2) identified risks,
3) current conversation context.
Never act as a general-purpose chatbot.
If the user is asking to rewrite, edit, add, remove, update, or otherwise change protocol text, classify that turn as a revision_request.
""".strip()


class ProtocolRevisionPlannerService:
    def __init__(self, client: OpenAI, model: str, max_history_messages: int = 8) -> None:
        self.client = client
        self.model = model
        self.max_history_messages = max_history_messages

    def plan_turn(
        self,
        *,
        analysis: AnalysisResult,
        question: str,
        history: list[dict[str, str]] | None = None,
        revision_session: dict[str, Any] | None = None,
    ) -> RevisionPlannerResult:
        revision_session = revision_session or {}
        prompt = self._build_user_prompt(
            analysis=analysis,
            question=question,
            history=history or [],
            revision_session=revision_session,
        )
        raw = self._request_text(
            system_prompt=_REVISION_PLANNER_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_output_tokens=1400,
        )
        payload_data = self._complete_json(
            raw_text=raw,
            expected_json_shape=_REVISION_PLANNER_RESPONSE_SHAPE,
        )
        payload = RevisionPlannerPayload.model_validate(
            _normalize_revision_planner_payload(payload_data, question=question)
        )
        if payload.intent == "revision_request" and not payload.proposals:
            payload = payload.model_copy(
                update={
                    "proposals": [
                        RevisionProposal(
                            summary="Protocol revision request",
                            requested_change=question.strip(),
                            rationale="The user explicitly requested protocol modifications to improve the analyzed risk profile.",
                            affected_sections=["Protocol section to update"],
                            estimated_edits=3,
                            focus="general",
                        )
                    ]
                }
            )
        return RevisionPlannerResult(payload=payload)

    def _build_user_prompt(
        self,
        *,
        analysis: AnalysisResult,
        question: str,
        history: list[dict[str, str]],
        revision_session: dict[str, Any],
    ) -> str:
        trimmed_history = _trim_history(history, self.max_history_messages)
        risk_index = [
            {
                "title": risk.title,
                "domain": risk.domain.value,
                "category": risk.category,
                "score": risk.risk_score,
                "mitigation_status": risk.mitigation_status,
                "rationale": risk.rationale,
                "mitigations": risk.mitigations[:4],
            }
            for risk in analysis.risks[:50]
        ]
        revision_snapshot = {
            "pending_changes": [
                {
                    "id": str(item.get("id", "")).strip(),
                    "summary": str(item.get("summary", "")).strip(),
                    "affected_sections": list(item.get("affected_sections", [])),
                    "estimated_edits": int(item.get("estimated_edits", 0) or 0),
                }
                for item in list(revision_session.get("pending_changes", []))[:20]
                if isinstance(item, dict)
            ],
            "approved_changes": [
                {
                    "id": str(item.get("id", "")).strip(),
                    "summary": str(item.get("summary", "")).strip(),
                    "affected_sections": list(item.get("affected_sections", [])),
                    "estimated_edits": int(item.get("estimated_edits", 0) or 0),
                }
                for item in list(revision_session.get("approved_changes", []))[:20]
                if isinstance(item, dict)
            ],
        }
        analysis_payload = {
            "document_name": analysis.metadata.document_name,
            "executive_summary": analysis.executive_summary,
            "key_assumptions": analysis.key_assumptions[:8],
            "risk_index": risk_index,
        }
        return f"""
User message:
{question}

Recent conversation history (JSON):
{json.dumps(trimmed_history, ensure_ascii=False)}

Current revision session (JSON):
{json.dumps(revision_snapshot, ensure_ascii=False)}

Generated protocol risk analysis context (JSON):
{json.dumps(analysis_payload, ensure_ascii=False)}

Instructions:
- Classify intent:
  - analysis_question: user is asking for explanation/summary/interpretation and not requesting protocol text changes.
  - revision_request: user asks to modify/rewrite/add/remove/simplify/update protocol content.
  - out_of_scope: unrelated to protocol analysis/revision.
- If the user is clearly asking for protocol changes, prefer revision_request even when the message is phrased as a question.
- For analysis_question: assistant_reply should be brief and say you'll analyze the request based on the protocol and risk analysis context.
- For revision_request:
  - Provide 1-4 concrete proposals.
  - Keep proposals surgical and implementation-oriented.
  - affected_sections should be likely protocol sections impacted by the request.
  - estimated_edits should be realistic (1-50).
  - assistant_reply should summarize intended edits and ask for approval to add to revision plan.
- For out_of_scope:
  - assistant_reply must be exactly: "I can only help with this protocol's risk analysis and revision workflow."
  - proposals must be [].
- Do not claim edits have already been applied.
- Do not generate revised documents in this step.

Return strict JSON only with shape:
{_REVISION_PLANNER_RESPONSE_SHAPE}
""".strip()

    def _request_text(self, *, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
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

    def _complete_json(self, *, raw_text: str, expected_json_shape: str) -> dict[str, Any]:
        try:
            return _parse_json_response(raw_text)
        except json.JSONDecodeError:
            repaired_text = self._repair_json_response(
                malformed_text=raw_text,
                expected_json_shape=expected_json_shape,
            )
            try:
                return _parse_json_response(repaired_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Model did not return valid JSON for revision planner response.") from exc

    def _repair_json_response(self, *, malformed_text: str, expected_json_shape: str) -> str:
        repair_system_prompt = (
            "You repair malformed JSON. Return valid JSON only with no markdown and no commentary."
            " Preserve the original content where possible."
            " If a field is missing or malformed, replace it with a safe minimal value rather than inventing extra detail."
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
            max_output_tokens=1800,
        )


class ProtocolProposalEditService:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def revise_proposal(
        self,
        *,
        target_label: str,
        risk_title: str,
        original_text: str,
        current_text: str,
        current_reason: str,
        user_request: str,
    ) -> ProposalEditPayload:
        prompt = f"""
Rewrite this one protocol paragraph as JSON:

{_PROPOSAL_EDIT_RESPONSE_SHAPE}

Rules:
- `current_text` is the active proposed paragraph. Modify it to satisfy the user's request.
- Preserve the original paragraph's subject, role, and scientific meaning.
- Keep this as exactly one paragraph.
- Make the smallest change needed to satisfy the user's request.
- Keep the rewrite close in length to the original paragraph unless the requested change genuinely requires more detail.
- Do not add off-topic controls or unrelated protocol content.
- Preserve useful mitigation language already present in `current_text` when it still fits the paragraph.
- Do not invent unsupported protocol facts, numbers, procedures, visits, or thresholds.

Target label:
{target_label}

Risk title:
{risk_title}

Current rationale:
{current_reason}

Original paragraph:
{original_text}

Current proposed paragraph:
{current_text}

User requested change:
{user_request}
""".strip()
        payload = self._complete_json(
            system_prompt=(
                "You revise a single proposed clinical protocol paragraph based on a user's requested change. "
                "Return strict JSON only."
            ),
            user_prompt=prompt,
            expected_json_shape=_PROPOSAL_EDIT_RESPONSE_SHAPE,
            max_output_tokens=1200,
        )
        return ProposalEditPayload.model_validate(payload)

    def _complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        expected_json_shape: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        text = self._request_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )
        try:
            return _parse_json_response(text)
        except json.JSONDecodeError:
            repaired_text = self._repair_json_response(
                malformed_text=text,
                expected_json_shape=expected_json_shape,
            )
            try:
                return _parse_json_response(repaired_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Model did not return valid JSON for proposal edit response.") from exc

    def _request_text(self, *, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
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

    def _repair_json_response(self, *, malformed_text: str, expected_json_shape: str) -> str:
        repair_system_prompt = (
            "You repair malformed JSON. Return valid JSON only with no markdown and no commentary."
            " Preserve the original content where possible."
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
            max_output_tokens=1600,
        )


def _trim_history(history: list[dict[str, str]], max_messages: int) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for item in history[-max_messages:]:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            sanitized.append({"role": role, "content": content})
    return sanitized


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start > 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _normalize_revision_planner_payload(payload_data: object, *, question: str) -> dict[str, Any]:
    question_text = str(question).strip()
    question_is_revision_request = _looks_like_revision_request(question_text)
    if not isinstance(payload_data, dict):
        return _fallback_revision_planner_payload(question)

    intent_raw = str(payload_data.get("intent", "")).strip().lower().replace(" ", "_")
    intent_aliases = {
        "analysis_question": "analysis_question",
        "question": "analysis_question",
        "qa": "analysis_question",
        "revision_request": "revision_request",
        "revision": "revision_request",
        "edit_request": "revision_request",
        "edit": "revision_request",
        "out_of_scope": "out_of_scope",
        "unsupported": "out_of_scope",
        "unrelated": "out_of_scope",
    }
    intent = intent_aliases.get(intent_raw)
    if intent is None:
        intent = "revision_request" if question_is_revision_request else "analysis_question"
    elif question_is_revision_request and intent != "revision_request":
        intent = "revision_request"
    assistant_reply = str(payload_data.get("assistant_reply", "")).strip()
    if not assistant_reply:
        if intent == "out_of_scope":
            assistant_reply = "I can only help with this protocol's risk analysis and revision workflow."
        elif intent == "revision_request":
            assistant_reply = "I identified protocol edits for your review and can add them to the revision plan."
        else:
            assistant_reply = "I can help analyze that request using the protocol and generated risk analysis."

    proposals_raw = payload_data.get("proposals", [])
    proposals: list[dict[str, Any]] = []
    if isinstance(proposals_raw, list):
        for item in proposals_raw[:4]:
            normalized = _normalize_revision_proposal(item)
            if normalized is not None:
                proposals.append(normalized)

    normalized_payload = {
        "intent": intent,
        "assistant_reply": assistant_reply,
        "proposals": [] if intent == "out_of_scope" else proposals,
    }
    if intent == "revision_request" and not normalized_payload["proposals"]:
        normalized_payload["proposals"] = _fallback_revision_planner_payload(question)["proposals"]
    return normalized_payload


def _looks_like_revision_request(question: str) -> bool:
    normalized = " ".join(question.strip().lower().split())
    if not normalized:
        return False
    direct_triggers = (
        "rewrite the protocol",
        "rewrite this",
        "rewrite section",
        "edit the protocol",
        "edit this",
        "modify the protocol",
        "modify this",
        "update the protocol",
        "update this",
        "change the protocol",
        "change this",
        "add a section",
        "remove a section",
        "replace the section",
        "generate revised protocol",
        "create revised protocol",
        "draft revised protocol",
        "apply changes",
        "make this safer",
        "mitigate this risk",
    )
    if any(trigger in normalized for trigger in direct_triggers):
        return True
    return bool(
        re.search(
            r"\b(?:please|can you|could you|should we|help me|write|draft|revise|rewrite|edit|modify|update|change|add|remove|replace)\b",
            normalized,
        )
        and re.search(r"\b(?:protocol|section|language|wording|text|criteria|dose|visit|monitoring|safety|risk)\b", normalized)
    )


def _normalize_revision_proposal(item: object) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    summary = str(item.get("summary", "")).strip()
    requested_change = str(item.get("requested_change", "")).strip()
    rationale = str(item.get("rationale", "")).strip()
    affected_sections_raw = item.get("affected_sections", [])
    if isinstance(affected_sections_raw, list):
        affected_sections = [str(value).strip() for value in affected_sections_raw if str(value).strip()][:8]
    else:
        affected_sections = []
    estimated_edits_raw = item.get("estimated_edits", 3)
    try:
        estimated_edits = int(estimated_edits_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        estimated_edits = 3
    estimated_edits = max(1, min(estimated_edits, 50))
    focus_raw = str(item.get("focus", "general")).strip().lower()
    valid_focus = {"health", "financial", "ethical", "eligibility", "intervention", "operational", "general"}
    focus = focus_raw if focus_raw in valid_focus else "general"

    if not summary and requested_change:
        summary = requested_change[:80].strip()
    if not requested_change and summary:
        requested_change = summary
    if not rationale:
        rationale = "This user-directed protocol change is intended to improve the protocol's risk profile."
    if len(summary) < 6 or len(requested_change) < 8 or len(rationale) < 12:
        return None
    return {
        "summary": summary,
        "requested_change": requested_change,
        "rationale": rationale,
        "affected_sections": affected_sections,
        "estimated_edits": estimated_edits,
        "focus": focus,
    }


def _fallback_revision_planner_payload(question: str) -> dict[str, Any]:
    question_text = str(question).strip() or "Protocol revision request"
    return {
        "intent": "revision_request",
        "assistant_reply": "I identified protocol edits for your review and can add them to the revision plan.",
        "proposals": [
            {
                "summary": "Protocol revision request",
                "requested_change": question_text,
                "rationale": "The user explicitly requested protocol modifications to improve the analyzed risk profile.",
                "affected_sections": ["Protocol section to update"],
                "estimated_edits": 3,
                "focus": "general",
            }
        ],
    }
