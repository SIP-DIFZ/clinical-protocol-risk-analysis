from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from protocol_risk_analyzer.chunking import DocumentChunk, chunk_pages
from protocol_risk_analyzer.io import PageText, read_document
from protocol_risk_analyzer.models import AnalysisResult, RiskAssessment


class ChatResponsePayload(BaseModel):
    can_answer: bool = True
    answer_markdown: str = Field(min_length=1)
    cited_source_ids: list[int] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ChatSource:
    source_id: int
    kind: Literal["analysis_risk", "protocol_excerpt", "executive_summary"]
    label: str
    page_start: int | None = None
    page_end: int | None = None
    risk_title: str = ""
    domain: str = ""
    excerpt: str = ""
    why_relevant: str = ""


@dataclass(frozen=True)
class ChatResult:
    payload: ChatResponsePayload
    sources: list[ChatSource]


@dataclass(frozen=True)
class ChatContext:
    analysis_title: str
    question: str
    analysis_json: str
    selected_protocol_excerpts: list[ChatSource]
    selected_risks: list[ChatSource]
    recent_history: list[dict[str, str]]


class ProtocolChatService:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        chunk_chars: int = 14_000,
        overlap_chars: int = 1_200,
        max_protocol_excerpts: int = 4,
        max_risk_details: int = 6,
        max_history_messages: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.max_protocol_excerpts = max_protocol_excerpts
        self.max_risk_details = max_risk_details
        self.max_history_messages = max_history_messages

    def answer_question(
        self,
        *,
        analysis: AnalysisResult,
        question: str,
        history: list[dict[str, str]] | None = None,
        pages: list[PageText] | None = None,
        chunks: list[DocumentChunk] | None = None,
        input_path: Path | None = None,
    ) -> ChatResult:
        context = self._build_context(
            analysis=analysis,
            question=question,
            history=history or [],
            pages=pages,
            chunks=chunks,
            input_path=input_path,
        )
        prompt = self._build_user_prompt(context)
        payload = self._complete_json(
            system_prompt=_CHAT_SYSTEM_PROMPT,
            user_prompt=prompt,
            expected_json_shape=_CHAT_RESPONSE_SHAPE,
            max_output_tokens=1200,
        )
        parsed = ChatResponsePayload.model_validate(payload)
        if not parsed.can_answer:
            parsed = parsed.model_copy(update={"cited_source_ids": [], "follow_up_questions": []})
            return ChatResult(payload=parsed, sources=[])
        if _is_conversational_message(question):
            parsed = parsed.model_copy(update={"cited_source_ids": []})
            return ChatResult(payload=parsed, sources=[])
        available_source_ids = {
            source.source_id for source in context.selected_protocol_excerpts + context.selected_risks
        }
        parsed = parsed.model_copy(
            update={"cited_source_ids": _sanitize_cited_source_ids(parsed.cited_source_ids, available_source_ids)}
        )
        return ChatResult(payload=parsed, sources=context.selected_protocol_excerpts + context.selected_risks)

    def _build_context(
        self,
        *,
        analysis: AnalysisResult,
        question: str,
        history: list[dict[str, str]],
        pages: list[PageText] | None,
        chunks: list[DocumentChunk] | None,
        input_path: Path | None,
    ) -> ChatContext:
        if pages is None:
            if input_path is None:
                raise ValueError("input_path is required when pages are not provided.")
            pages = read_document(input_path)
        if chunks is None:
            chunks = chunk_pages(pages, max_chars=self.chunk_chars, overlap_chars=self.overlap_chars)

        normalized_question = _normalize_text(question)
        risk_index = [_risk_index_entry(risk) for risk in analysis.risks]
        selected_risks = _select_chat_risk_sources(analysis.risks, normalized_question, self.max_risk_details)
        selected_excerpts = _select_protocol_excerpts(
            chunks=chunks,
            question=normalized_question,
            risks=selected_risks,
            max_items=self.max_protocol_excerpts,
        )
        selected_risks_sources = [
            ChatSource(
                source_id=idx + 1 + len(selected_excerpts),
                kind="analysis_risk",
                label=f"Risk: {item['title']}",
                risk_title=item["title"],
                domain=item["domain"],
                page_start=item.get("page_start"),
                page_end=item.get("page_end"),
                excerpt=item["summary"],
                why_relevant=item["why_relevant"],
            )
            for idx, item in enumerate(selected_risks)
        ]
        analysis_json = json.dumps(
            {
                "document_name": analysis.metadata.document_name,
                "executive_summary": analysis.executive_summary,
                "key_assumptions": analysis.key_assumptions,
                "risk_index": risk_index,
                "risk_details": selected_risks,
            },
            ensure_ascii=False,
        )
        return ChatContext(
            analysis_title=analysis.metadata.document_name,
            question=question,
            analysis_json=analysis_json,
            selected_protocol_excerpts=selected_excerpts,
            selected_risks=selected_risks_sources,
            recent_history=_trim_history(history, self.max_history_messages),
        )

    def _build_user_prompt(self, context: ChatContext) -> str:
        history_json = json.dumps(context.recent_history, ensure_ascii=False)
        source_registry = [
            {
                "source_id": source.source_id,
                "kind": source.kind,
                "label": source.label,
                "risk_title": source.risk_title,
                "domain": source.domain,
                "page_start": source.page_start,
                "page_end": source.page_end,
                "excerpt": source.excerpt,
                "why_relevant": source.why_relevant,
            }
            for source in context.selected_protocol_excerpts + context.selected_risks
        ]
        source_registry_json = json.dumps(source_registry, ensure_ascii=False)
        return f"""
Analysis title:
{context.analysis_title}

User question:
{context.question}

Conversation history:
{history_json}

Generated analysis JSON:
{context.analysis_json}

Relevant protocol excerpts and analysis sources:
{source_registry_json}

Rules:
- Use only the uploaded protocol, generated analysis, and conversation history above.
- Be friendly and conversational in tone.
- If the user sends a greeting or brief social message, respond warmly and briefly, then steer back to protocol/risk-analysis help.
- Do not answer unrelated general questions.
- For unrelated questions, set can_answer to false and reply with exactly one sentence: "I can only provide answers related to the clinical protocol and risk analysis content provided."
- If the answer is not present in the supplied materials, set can_answer to false and say that the information is not present.
- When can_answer is false, do not include citations or follow-up questions.
- Do not provide evidence excerpts or citations for regular conversational messages (for example greetings, thanks, or brief social chat).
- Evidence excerpts and citations are required only when answering protocol- or risk-analysis-related questions from the supplied materials.
- Keep answers tight: start with the direct answer, then add at most a few concise bullets when needed.
- Cite only source_id values above, and only when a source directly supports the statement being made.
- Prefer page numbers, quoted excerpts, and related risk titles when they make the answer more precise.
- Use no more than three citations.
- Organize the answer clearly with short headings or bullets only when they add clarity.

Return strict JSON with this shape:
{_CHAT_RESPONSE_SHAPE}
""".strip()

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
            repaired_text = self._repair_json_response(text, expected_json_shape)
            try:
                return _parse_json_response(repaired_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Model did not return valid JSON for chat response.") from exc

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

    def _repair_json_response(self, malformed_text: str, expected_json_shape: str) -> str:
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
            max_output_tokens=1500,
        )


def _select_chat_risk_sources(
    risks: list[RiskAssessment],
    question: str,
    max_items: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, RiskAssessment]] = []
    for risk in risks:
        score = _score_risk_for_question(risk, question)
        scored.append((score, risk))
    scored.sort(key=lambda item: (item[0], item[1].risk_score, item[1].confidence), reverse=True)
    selected = [risk for score, risk in scored[:max_items] if score > 0 or not scored]
    if not selected:
        selected = sorted(risks, key=lambda risk: (risk.risk_score, risk.confidence), reverse=True)[:max_items]
    entries: list[dict[str, Any]] = []
    for risk in selected:
        pages = _risk_pages(risk)
        summary = _risk_summary_text(risk)
        why_relevant = f"Relevant to {risk.domain.value} and category {risk.category}."
        entries.append(
            {
                "title": risk.title,
                "domain": risk.domain.value,
                "category": risk.category,
                "likelihood": risk.likelihood,
                "impact": risk.impact,
                "confidence": risk.confidence,
                "score": risk.risk_score,
                "mitigation_status": risk.mitigation_status,
                "page_start": pages[0] if pages else None,
                "page_end": pages[-1] if pages else None,
                "summary": summary,
                "why_relevant": why_relevant,
            }
        )
    return entries


def _select_protocol_excerpts(
    *,
    chunks: list[DocumentChunk],
    question: str,
    risks: list[dict[str, Any]],
    max_items: int,
) -> list[ChatSource]:
    evidence_pages = {
        page
        for risk in risks
        for page in _page_range(risk.get("page_start"), risk.get("page_end"))
    }
    scored: list[tuple[float, DocumentChunk]] = []
    for chunk in chunks:
        score = _score_chunk_for_question(chunk, question)
        if any(page in evidence_pages for page in range(chunk.start_page, chunk.end_page + 1)):
            score += 1.5
        scored.append((score, chunk))
    scored.sort(key=lambda item: (item[0], item[1].start_page, item[1].end_page), reverse=True)
    selected = [chunk for score, chunk in scored[:max_items] if score > 0]
    if not selected:
        selected = [chunk for _, chunk in scored[:max_items]]
    sources: list[ChatSource] = []
    for idx, chunk in enumerate(selected, start=1):
        excerpt = _truncate_excerpt(chunk.text)
        sources.append(
            ChatSource(
                source_id=idx,
                kind="protocol_excerpt",
                label=f"Protocol p.{chunk.start_page}-{chunk.end_page}",
                page_start=chunk.start_page,
                page_end=chunk.end_page,
                excerpt=excerpt,
                why_relevant="Selected by semantic overlap with the user's question and analysis evidence.",
            )
        )
    return sources


def _risk_index_entry(risk: RiskAssessment) -> dict[str, Any]:
    pages = _risk_pages(risk)
    return {
        "title": risk.title,
        "domain": risk.domain.value,
        "category": risk.category,
        "likelihood": risk.likelihood,
        "impact": risk.impact,
        "confidence": risk.confidence,
        "score": risk.risk_score,
        "mitigation_status": risk.mitigation_status,
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
    }


def _risk_pages(risk: RiskAssessment) -> list[int]:
    pages: set[int] = set()
    for snippet in risk.evidence:
        pages.update(range(snippet.page_start, snippet.page_end + 1))
    return sorted(pages)


def _risk_summary_text(risk: RiskAssessment) -> str:
    summary = risk.rationale.strip()
    if risk.evidence:
        summary += f" Evidence: {risk.evidence[0].excerpt.strip()}"
    return _truncate_excerpt(summary, 280)


def _score_risk_for_question(risk: RiskAssessment, question: str) -> float:
    query_tokens = set(_tokenize(question))
    if not query_tokens:
        return 0.0
    haystack = " ".join(
        [
            risk.title,
            risk.category,
            risk.rationale,
            " ".join(risk.mitigations),
            " ".join(risk.factor_triggers),
            " ".join(snippet.excerpt for snippet in risk.evidence),
        ]
    ).lower()
    overlap = len(query_tokens.intersection(_tokenize(haystack))) / max(len(query_tokens), 1)
    domain_boost = _domain_boost(question, risk.domain.value)
    title_similarity = SequenceMatcher(None, question.lower(), risk.title.lower()).ratio()
    if overlap == 0:
        title_similarity = 0.0
    return overlap * 2.0 + domain_boost + title_similarity


def _score_chunk_for_question(chunk: DocumentChunk, question: str) -> float:
    query_tokens = set(_tokenize(question))
    if not query_tokens:
        return 0.0
    haystack = chunk.text.lower()
    overlap = len(query_tokens.intersection(_tokenize(haystack))) / max(len(query_tokens), 1)
    phrase_bonus = 0.0
    for token in query_tokens:
        if len(token) > 4 and token in haystack:
            phrase_bonus += 0.1
    return overlap * 2.0 + phrase_bonus


def _domain_boost(question: str, domain: str) -> float:
    question_text = question.lower()
    keyword_map = {
        "health": ["safety", "patient", "adverse", "harm", "clinical"],
        "financial": ["cost", "budget", "site", "recruitment", "finance", "operational"],
        "ethical": ["regulatory", "consent", "ethic", "privacy", "governance", "rights"],
        "eligibility": ["eligibility", "inclusion", "exclusion", "biomarker", "lab"],
        "intervention": ["dose", "dosing", "administration", "schedule", "treatment", "procedure"],
    }
    return 0.8 if any(keyword in question_text for keyword in keyword_map.get(domain, [])) else 0.0


def _trim_history(history: list[dict[str, str]], max_messages: int) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for item in history[-max_messages:]:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            sanitized.append({"role": role, "content": content})
    return sanitized


def _is_conversational_message(question: str) -> bool:
    normalized = _normalize_text(question)
    if not normalized:
        return False
    protocol_keywords = {
        "analysis",
        "protocol",
        "risk",
        "risks",
        "safety",
        "patient",
        "regulatory",
        "eligibility",
        "intervention",
        "financial",
        "confidence",
        "score",
        "evidence",
        "citation",
        "recommendation",
        "mitigation",
        "section",
        "summary",
    }
    if any(keyword in normalized for keyword in protocol_keywords):
        return False
    direct_messages = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "got it",
        "how are you",
        "how is it going",
    }
    if normalized in direct_messages:
        return True
    tokens = _tokenize(normalized)
    if not tokens:
        return False
    social_tokens = {
        "hello",
        "hi",
        "hey",
        "greetings",
        "thanks",
        "thank",
        "morning",
        "afternoon",
        "evening",
        "good",
        "great",
        "nice",
    }
    return len(tokens) <= 5 and set(tokens).issubset(social_tokens)


def _tokenize(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "can",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "show",
        "summarize",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token and token not in stopwords]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _truncate_excerpt(text: str, max_chars: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _page_range(start: Any, end: Any) -> range:
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        return range(0)
    return range(start, end + 1)


def _sanitize_cited_source_ids(cited_source_ids: list[int], available_source_ids: set[int]) -> list[int]:
    sanitized: list[int] = []
    seen: set[int] = set()
    for source_id in cited_source_ids:
        if not isinstance(source_id, int):
            continue
        if source_id not in available_source_ids or source_id in seen:
            continue
        sanitized.append(source_id)
        seen.add(source_id)
        if len(sanitized) >= 3:
            break
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


_CHAT_SYSTEM_PROMPT = (
    "You are an experienced clinical protocol reviewer and AI assistant inside a protocol risk analysis application."
    " Be friendly and conversational."
    " Use only the supplied protocol excerpts, generated analysis, and conversation history."
    " Never invent facts or answer unrelated general questions."
    " If a user message is unrelated, refuse in exactly one sentence: 'I can only provide answers related to the clinical protocol and risk analysis content provided.'"
    " If the user sends a greeting or brief social message, respond warmly and briefly while staying in scope."
    " If the answer is not present in the supplied materials, state that clearly."
    " Do not provide evidence excerpts or citations for regular conversational messages."
    " Evidence excerpts and citations are required only for protocol- or risk-analysis-related answers grounded in supplied materials."
    " When can_answer is false, provide no citations or follow-up questions."
    " Keep answers tight: start with the direct answer, then add at most a few concise bullets when needed."
    " Start with a direct answer, then add concise evidence-backed detail and practical next steps when helpful."
    " Explain your reasoning clearly, reference evidence whenever possible, and organize answers with headings and bullets when helpful."
    " Return strict JSON only."
)

_CHAT_RESPONSE_SHAPE = """{
  "can_answer": true,
  "answer_markdown": "markdown answer",
  "cited_source_ids": [1, 2],
  "follow_up_questions": ["..."]
}"""
