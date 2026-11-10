from __future__ import annotations

import os
import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from inspect import Parameter, signature
from pathlib import Path
from time import monotonic
from typing import Callable

from rich.progress import track

from protocol_risk_analyzer.chunking import DocumentChunk
from protocol_risk_analyzer.llm import RiskLlmClient
from protocol_risk_analyzer.models import AnalysisResult, ProtocolMetadata, RiskAssessment
from protocol_risk_analyzer.rubric import (
    ELIGIBILITY_RISK_FACTORS,
    ETHICAL_RISK_FACTORS,
    FINANCIAL_RISK_FACTORS,
    HEALTH_RISK_FACTORS,
    INTERVENTION_RISK_FACTORS,
)

MAX_MITIGATIONS_PER_RISK = 10
MAX_EVIDENCE_SNIPPETS_PER_RISK = 10


def analyze_protocol(
    source_path: Path,
    chunks: list[DocumentChunk],
    llm: RiskLlmClient,
    progress_callback: Callable[..., None] | None = None,
    max_workers: int | None = None,
) -> AnalysisResult:
    analysis_started_at = monotonic()
    extraction_jobs = len(chunks) * len(_extractor_methods())
    judge_weight = 4
    summary_weight = 4
    total_steps = extraction_jobs + judge_weight + summary_weight
    all_risks: list[RiskAssessment] = _extract_risks_with_agents(
        chunks=chunks,
        llm=llm,
        progress_callback=progress_callback,
        max_workers=max_workers,
        total_steps=total_steps,
    )
    canonicalized = [_normalize_risk_category(risk) for risk in all_risks]
    deduped = aggregate_risks(canonicalized, include_domain=False)
    _emit_progress(
        progress_callback,
        extraction_jobs,
        total_steps,
        extraction_jobs / total_steps,
        _estimate_eta_seconds(monotonic() - analysis_started_at, extraction_jobs, total_steps),
        "Consolidating extracted risks...",
    )
    _emit_progress(
        progress_callback,
        extraction_jobs,
        total_steps,
        extraction_jobs / total_steps,
        _estimate_eta_seconds(monotonic() - analysis_started_at, extraction_jobs, total_steps),
        "Judge Agent reviewing consolidated risks...",
    )
    judged = llm.judge_aggregate_risks(deduped)
    judged_completed_steps = extraction_jobs + judge_weight
    _emit_progress(
        progress_callback,
        judged_completed_steps,
        total_steps,
        judged_completed_steps / total_steps,
        _estimate_eta_seconds(monotonic() - analysis_started_at, judged_completed_steps, total_steps),
        "Reconciling judged risks and generating summary...",
    )
    final_risks = sort_risks(
        aggregate_risks([_normalize_risk_category(risk) for risk in judged], include_domain=False)
    )
    final_risks = _harmonize_similar_categories(final_risks)
    final_risks = _reconcile_risk_domains(final_risks)
    final_risks = sort_risks(aggregate_risks([_normalize_risk_category(risk) for risk in final_risks]))
    if hasattr(llm, "classify_mitigation_statuses"):
        final_risks = llm.classify_mitigation_statuses(final_risks)
    _emit_progress(
        progress_callback,
        judged_completed_steps,
        total_steps,
        judged_completed_steps / total_steps,
        _estimate_eta_seconds(monotonic() - analysis_started_at, judged_completed_steps, total_steps),
        "Generating executive summary...",
    )
    summary = llm.summarize(final_risks)
    _emit_progress(progress_callback, total_steps, total_steps, 1.0, 0.0, "Analysis complete.")

    metadata = ProtocolMetadata(
        source_path=str(source_path),
        document_name=source_path.name,
        page_count=max(c.end_page for c in chunks),
        chunk_count=len(chunks),
        generated_at_utc=ProtocolMetadata.now_iso(),
    )
    return AnalysisResult(
        metadata=metadata,
        executive_summary=summary.executive_summary,
        key_assumptions=summary.key_assumptions,
        risks=final_risks,
    )


def _extract_risks_with_agents(
    chunks: list[DocumentChunk],
    llm: RiskLlmClient,
    progress_callback: Callable[..., None] | None,
    max_workers: int | None,
    total_steps: int,
) -> list[RiskAssessment]:
    all_risks: list[RiskAssessment] = []
    extractors = _extractor_methods()
    total_jobs = len(chunks) * len(extractors)
    if total_jobs == 0:
        return all_risks

    start_time = monotonic()
    _emit_progress(progress_callback, 0, total_steps, 0.0, 0.0, "Starting multi-agent extraction...")

    workers = _resolve_worker_count(max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: list[Future[list[RiskAssessment]]] = []
        for chunk in chunks:
            for extractor_name in extractors:
                extractor = getattr(llm, extractor_name)
                futures.append(executor.submit(extractor, chunk))

        completed = 0
        for future in track(as_completed(futures), total=len(futures), description="Running agents"):
            all_risks.extend(future.result())
            completed += 1
            elapsed_seconds = monotonic() - start_time
            progress_ratio = completed / total_steps
            eta_seconds = _estimate_eta_seconds(elapsed_seconds, completed, total_steps)
            _emit_progress(
                progress_callback,
                completed,
                total_steps,
                progress_ratio,
                eta_seconds,
                "Extracting risks across health, financial, ethical, eligibility, and intervention domains...",
            )
    return all_risks


def aggregate_risks(risks: list[RiskAssessment], include_domain: bool = True) -> list[RiskAssessment]:
    grouped: dict[tuple[str, ...], list[RiskAssessment]] = defaultdict(list)
    for risk in risks:
        canonical = _normalize_risk_category(risk)
        if include_domain:
            key = (
                canonical.domain.value,
                _normalize_key(canonical.title),
                _normalize_key(canonical.category),
            )
        else:
            key = (
                _normalize_key(canonical.title),
                _normalize_key(canonical.category),
            )
        grouped[key].append(canonical)

    merged: list[RiskAssessment] = []
    for _, group in grouped.items():
        if len(group) == 1:
            merged.append(_cap_risk_supporting_details(_normalize_risk_category(group[0])))
            continue

        head = max(group, key=lambda risk: (risk.confidence, risk.risk_score, len(risk.evidence)))
        evidences = []
        mitigations: list[str] = []
        seen_evidence = set()
        seen_mitigation = set()
        factor_triggers: list[str] = []
        seen_factor_trigger = set()
        confidence_weight = 0.0
        confidence_sum = 0.0
        likelihood_weight = 0.0
        impact_weight = 0.0

        for risk in group:
            w = max(risk.confidence, 0.01)
            confidence_weight += risk.confidence
            confidence_sum += w
            likelihood_weight += risk.likelihood * w
            impact_weight += risk.impact * w

            for ev in risk.evidence:
                ev_key = (ev.page_start, ev.page_end, ev.excerpt.strip().lower())
                if ev_key not in seen_evidence:
                    seen_evidence.add(ev_key)
                    evidences.append(ev)

            for mitigation in risk.mitigations:
                normalized = mitigation.strip().lower()
                if normalized not in seen_mitigation:
                    seen_mitigation.add(normalized)
                    mitigations.append(mitigation.strip())

            for factor in risk.factor_triggers:
                normalized_factor = factor.strip().lower()
                if normalized_factor not in seen_factor_trigger:
                    seen_factor_trigger.add(normalized_factor)
                    factor_triggers.append(factor.strip())

        merged.append(
            _cap_risk_supporting_details(
                RiskAssessment(
                    title=head.title,
                    domain=head.domain,
                    category=_canonicalize_category(head.category),
                    likelihood=_round_1_to_5(likelihood_weight / confidence_sum),
                    impact=_round_1_to_5(impact_weight / confidence_sum),
                    confidence=round(min(confidence_weight / len(group), 1.0), 2),
                    rationale=head.rationale,
                    mitigations=mitigations,
                    factor_triggers=factor_triggers,
                    evidence=evidences,
                )
            )
        )
    return merged


def _cap_risk_supporting_details(risk: RiskAssessment) -> RiskAssessment:
    return risk.model_copy(
        update={
            "mitigations": risk.mitigations[:MAX_MITIGATIONS_PER_RISK],
            "evidence": risk.evidence[:MAX_EVIDENCE_SNIPPETS_PER_RISK],
        }
    )


def sort_risks(risks: list[RiskAssessment]) -> list[RiskAssessment]:
    return sorted(risks, key=lambda r: r.risk_score, reverse=True)


def _normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _round_1_to_5(value: float) -> int:
    rounded = round(value)
    if rounded < 1:
        return 1
    if rounded > 5:
        return 5
    return rounded


def _estimate_eta_seconds(elapsed_seconds: float, processed_chunks: int, total_chunks: int) -> float:
    if processed_chunks <= 0:
        return 0.0
    remaining = max(total_chunks - processed_chunks, 0)
    average_time_per_chunk = elapsed_seconds / processed_chunks
    return max(remaining * average_time_per_chunk, 0.0)


def _emit_progress(
    progress_callback: Callable[..., None] | None,
    processed: int,
    total: int,
    ratio: float,
    eta_seconds: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    if _callback_accepts_message(progress_callback):
        progress_callback(processed, total, ratio, eta_seconds, message)
        return
    progress_callback(processed, total, ratio, eta_seconds)


def _callback_accepts_message(progress_callback: Callable[..., None]) -> bool:
    params = signature(progress_callback).parameters.values()
    positional = [
        p
        for p in params
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(p.kind == Parameter.VAR_POSITIONAL for p in params)
    return has_var_positional or len(positional) >= 5


def _resolve_worker_count(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, max_workers)

    env_workers = os.getenv("ANALYSIS_MAX_WORKERS")
    if env_workers and env_workers.isdigit():
        return max(1, int(env_workers))

    cpu_count = os.cpu_count() or 4
    return max(3, min(12, cpu_count))


def _normalize_risk_category(risk: RiskAssessment) -> RiskAssessment:
    return risk.model_copy(update={"category": _canonicalize_category(risk.category)})


def _canonicalize_category(category: str) -> str:
    normalized = category.strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+risk(s)?$", "", normalized).strip()
    normalized = _normalize_category_tokens(normalized)
    normalized = _apply_category_aliases(normalized)
    normalized = _map_to_broad_topic(normalized)
    normalized = _merge_close_category_topics(normalized)
    normalized = _generalize_category_phrase(normalized)
    if not normalized:
        return "General"
    return " ".join(word.capitalize() for word in normalized.split())


def _normalize_category_tokens(category: str) -> str:
    tokens = category.split()
    normalized_tokens = []
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def _apply_category_aliases(category: str) -> str:
    alias_patterns: list[tuple[str, str]] = [
        (r"unknown( or poorly characterized)? side effect", "unknown side effect"),
        (r"(unknown|uncharacterized) safety( profile)?", "unknown side effect"),
        (r"adverse reaction", "adverse event"),
        (r"adverse event(s)? management( cost| costs)?", "adverse event"),
        (r"adverse reaction(s)? management( cost| costs)?", "adverse event"),
        (r"allergic reaction", "allergic reaction"),
        (r"allergy|allergies", "allergic reaction"),
        (r"trial design( issue| concern)?", "trial design"),
        (r"study design( issue| concern)?", "trial design"),
        (r"protocol design( issue| concern)?", "trial design"),
        (r"(informed )?consent( quality| clarity| comprehension)?", "informed consent"),
        (r"(privacy|confidentiality|data privacy|data confidentiality)( concern| issue)?", "privacy and confidentiality"),
        (r"vulnerable population(s)?", "vulnerable populations"),
        (r"(undue|unduly influential?) compensation", "undue compensation"),
        (r"(deferred|emergency) consent", "deferred consent"),
        (r"(travel|lodging|reimbursement|out of pocket|out-of-pocket)( cost| burden)?", "cost and reimbursement"),
        (r"(visit|overnight stay|participation duration|participation period)( burden| cost)?", "participant logistics"),
        (r"(site startup|vendor coordination|supply chain|operational delay)( risk| issue)?", "operational complexity"),
        (r"(rare biomarker|biomarker requirement)", "eligibility criteria"),
        (r"(laboratory threshold|lab threshold|organ function requirement|performance status requirement)", "eligibility criteria"),
        (r"(age restriction|prior treatment requirement)", "eligibility criteria"),
        (r"(gene therapy|cell therapy|immunotherapy|radiation therapy|surgical intervention)", "intervention modality"),
        (r"(dose interruption|dose modification|dose escalation)", "dosing strategy"),
        (r"(administration schedule|infusion schedule|complex administration)", "administration complexity"),
    ]
    for pattern, canonical in alias_patterns:
        if re.fullmatch(pattern, category):
            return canonical
    return category


def _map_to_broad_topic(category: str) -> str:
    topic_patterns: list[tuple[str, str]] = [
        (r".*(unknown|poorly characterized).*(side effect|safety).*", "unknown side effects"),
        (r".*(safety uncertainty|safety profile uncertainty).*", "unknown side effects"),
        (r".*(allergic|anaphylaxis|hypersensitivity).*", "allergic reactions"),
        (r".*(adverse event|adverse reaction|toxicity).*", "adverse events"),
        (r".*(placebo|direct benefit|participant benefit).*", "participant benefit and placebo"),
        (r".*(trial design|study design|protocol design).*", "trial design"),
        (r".*(rare biomarker|laboratory threshold|lab threshold|organ function|performance status|age restriction|prior treatment).*", "eligibility criteria"),
        (r".*(eligibility|inclusion|exclusion|enrollment|recruitment).*", "eligibility and enrollment"),
        (r".*(dropout|retention|withdrawal).*", "dropout and retention"),
        (r".*(participant burden|invasive|procedure|blood draw|lumbar puncture|biopsy).*", "participant burden"),
        (r".*(monitoring|hospitalization|icu|intensive monitoring).*", "monitoring and hospitalization"),
        (r".*(medication restriction|concomitant medication|washout).*", "medication restrictions"),
        (r".*(reproductive|pregnancy|contraception).*", "reproductive safety"),
        (r".*(dose|dosing|dose escalation|first in human|phase i).*", "dosing strategy"),
        (r".*(travel|lodging|reimbursement|out of pocket|out-of-pocket).*", "cost and reimbursement"),
        (r".*(visit frequency|many visits|overnight stay|long participation).*", "participant logistics"),
        (r".*(cost|budget|financial|funding|economics).*", "financial feasibility"),
        (r".*(operational|site startup|vendor|logistics|supply chain).*", "operational complexity"),
        (r".*(informed consent|consent comprehension|consent clarity).*", "informed consent"),
        (r".*(privacy|confidentiality|data protection).*", "privacy and confidentiality"),
        (r".*(vulnerable population|undue influence|unduly influence|deferred consent|emergency consent).*", "ethical safeguards"),
        (r".*(gene therapy|cell therapy|immunotherapy|radiation therapy|surgical intervention|combination therap).*", "intervention modality"),
        (r".*(administration schedule|infusion schedule|dose interruption|dose modification).*", "administration complexity"),
    ]
    for pattern, broad_topic in topic_patterns:
        if re.match(pattern, category):
            return broad_topic
    return category


def _merge_close_category_topics(category: str) -> str:
    canonical_topics = [
        "unknown side effects",
        "allergic reactions",
        "adverse events",
        "participant benefit and placebo",
        "trial design",
        "eligibility and enrollment",
        "dropout and retention",
        "participant burden",
        "monitoring and hospitalization",
        "medication restrictions",
        "reproductive safety",
        "dosing strategy",
        "financial feasibility",
        "cost and reimbursement",
        "participant logistics",
        "operational complexity",
        "informed consent",
        "privacy and confidentiality",
        "ethical safeguards",
        "eligibility criteria",
        "intervention modality",
        "administration complexity",
    ]
    if category in canonical_topics:
        return category

    category_tokens = set(category.split())
    best_topic = category
    best_score = 0.0
    for topic in canonical_topics:
        token_overlap = 0.0
        topic_tokens = set(topic.split())
        if category_tokens and topic_tokens:
            token_overlap = len(category_tokens.intersection(topic_tokens)) / len(
                category_tokens.union(topic_tokens)
            )
        score = max(SequenceMatcher(None, category, topic).ratio(), token_overlap)
        if score > best_score:
            best_score = score
            best_topic = topic

    if best_score >= 0.72:
        return best_topic
    return category


def _generalize_category_phrase(category: str) -> str:
    phrase = re.sub(r"\s+", " ", category.strip())
    if not phrase:
        return phrase
    phrase = re.split(r"\b(with|for|due to|because of|caused by)\b", phrase, maxsplit=1)[0].strip()
    stopwords = {"and", "or", "of", "the", "to", "in", "on", "at", "for", "with", "from", "by"}
    tokens = [token for token in re.findall(r"[a-z0-9]+", phrase) if token and token not in stopwords]
    if not tokens:
        return ""
    if len(tokens) <= 2:
        return " ".join(tokens)
    if tokens[0] == "unknown":
        return f"{tokens[0]} {tokens[-1]}"
    return " ".join(tokens[:2])


def _harmonize_similar_categories(risks: list[RiskAssessment]) -> list[RiskAssessment]:
    if not risks:
        return []

    categories = sorted({risk.category for risk in risks})
    if len(categories) <= 1:
        return risks

    adjacency: dict[str, set[str]] = {category: set() for category in categories}
    for i, left in enumerate(categories):
        left_norm = _normalize_key(left)
        left_tokens = set(left_norm.split())
        left_driver = _shared_driver_topic(left_norm)
        for right in categories[i + 1 :]:
            right_norm = _normalize_key(right)
            similarity = SequenceMatcher(None, left_norm, right_norm).ratio()
            token_overlap = 0.0
            right_tokens = set(right_norm.split())
            right_driver = _shared_driver_topic(right_norm)
            if left_tokens and right_tokens:
                token_overlap = len(left_tokens.intersection(right_tokens)) / len(
                    left_tokens.union(right_tokens)
                )

            same_driver_topic = (
                left_driver is not None
                and right_driver is not None
                and left_driver == right_driver
            )

            if similarity >= 0.86 or token_overlap >= 0.8 or same_driver_topic:
                adjacency[left].add(right)
                adjacency[right].add(left)

    canonical_by_category: dict[str, str] = {}
    visited: set[str] = set()
    for category in categories:
        if category in visited:
            continue
        stack = [category]
        group: list[str] = []
        visited.add(category)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        group_drivers = {_shared_driver_topic(_normalize_key(member)) for member in group}
        group_drivers.discard(None)
        if len(group_drivers) == 1 and len(group) >= 2:
            canonical = _title_case_phrase(next(iter(group_drivers)))
        else:
            # Keep the shortest stable label for readability and consistency.
            canonical = min(group, key=lambda value: (len(value), value))
        for member in group:
            canonical_by_category[member] = canonical

    return [risk.model_copy(update={"category": canonical_by_category[risk.category]}) for risk in risks]


def _shared_driver_topic(normalized_category: str) -> str | None:
    tokens = normalized_category.split()
    if len(tokens) < 2:
        return None
    driver = tokens[0]
    aspect_token = tokens[1]
    subtopic_aspects = {
        "adherence",
        "compliance",
        "efficacy",
        "effectiveness",
        "quality",
        "delivery",
        "schedule",
        "interruption",
        "discontinuation",
        "uptake",
        "coverage",
    }
    if aspect_token not in subtopic_aspects:
        return None
    if driver in {"adverse", "allergic", "unknown", "general"}:
        return None
    return driver


def _title_case_phrase(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())


def _reconcile_risk_domains(risks: list[RiskAssessment]) -> list[RiskAssessment]:
    if not risks:
        return []

    category_domain_map = {
        "eligibility and enrollment": "eligibility",
        "eligibility enrollment": "eligibility",
        "eligibility criteria": "eligibility",
        "dosing strategy": "intervention",
        "intervention modality": "intervention",
        "administration complexity": "intervention",
        "financial feasibility": "financial",
        "cost and reimbursement": "financial",
        "participant logistics": "financial",
        "operational complexity": "financial",
        "informed consent": "ethical",
        "privacy and confidentiality": "ethical",
        "privacy confidentiality": "ethical",
        "ethical safeguards": "ethical",
        "adverse events": "health",
        "allergic reactions": "health",
        "unknown side effects": "health",
    }
    domain_keywords = {
        "eligibility": [
            "eligibility",
            "inclusion",
            "exclusion",
            "enrollment",
            "recruitment",
            "biomarker",
            "laboratory threshold",
            "lab threshold",
            "prior treatment",
            "age restriction",
            "performance status",
            "organ function",
        ],
        "intervention": [
            "first in human",
            "first-in-human",
            "gene therapy",
            "cell therapy",
            "combination therap",
            "dose escalation",
            "dose modification",
            "surgical intervention",
            "radiation therapy",
            "immunotherapy",
            "administration schedule",
        ],
        "ethical": [
            "informed consent",
            "vulnerable population",
            "fairness",
            "privacy",
            "confidentiality",
            "unduly influence",
            "undue influence",
            "deferred consent",
            "emergency consent",
            "participant selection",
        ],
        "financial": [
            "cost",
            "budget",
            "funding",
            "economic",
            "reimbursement",
            "out of pocket",
            "out-of-pocket",
            "travel",
            "lodging",
            "visit",
            "overnight",
            "long participation",
            "dropout",
            "retention",
            "operational",
            "site startup",
            "vendor",
            "logistics",
            "supply chain",
            "timeline",
        ],
        "health": [
            "adverse event",
            "adverse reaction",
            "toxicity",
            "side effect",
            "safety",
            "hospitalization",
            "monitoring",
            "allergic",
            "anaphylaxis",
            "pregnancy",
            "medication restriction",
        ],
    }
    factor_domain_map = {
        _normalize_key(label): "health" for label in HEALTH_RISK_FACTORS
    } | {
        _normalize_key(label): "financial" for label in FINANCIAL_RISK_FACTORS
    } | {
        _normalize_key(label): "ethical" for label in ETHICAL_RISK_FACTORS
    } | {
        _normalize_key(label): "eligibility" for label in ELIGIBILITY_RISK_FACTORS
    } | {
        _normalize_key(label): "intervention" for label in INTERVENTION_RISK_FACTORS
    }

    domain_values = {d.value: d for d in type(risks[0].domain)}
    reconciled: list[RiskAssessment] = []
    for risk in risks:
        scores = {key: 0 for key in domain_values.keys()}
        category_key = _normalize_key(risk.category)
        if category_key in category_domain_map:
            scores[category_domain_map[category_key]] += 5

        text_blob = _normalize_key(
            " ".join([risk.title, risk.category, risk.rationale, " ".join(risk.factor_triggers)])
        )
        for domain_name, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in text_blob:
                    scores[domain_name] += 1

        for trigger in risk.factor_triggers:
            mapped_domain = factor_domain_map.get(_normalize_key(trigger))
            if mapped_domain is not None:
                scores[mapped_domain] += 3

        best_domain_name = max(scores, key=scores.get)
        current_domain_name = risk.domain.value
        best_score = scores[best_domain_name]
        current_score = scores[current_domain_name]

        if best_score > 0 and best_domain_name != current_domain_name and best_score >= current_score + 2:
            reconciled.append(risk.model_copy(update={"domain": domain_values[best_domain_name]}))
            continue
        reconciled.append(risk)

    return reconciled


def _extractor_methods() -> list[str]:
    return [
        "extract_chunk_health_risks",
        "extract_chunk_financial_risks",
        "extract_chunk_ethical_risks",
        "extract_chunk_eligibility_risks",
        "extract_chunk_intervention_risks",
    ]
