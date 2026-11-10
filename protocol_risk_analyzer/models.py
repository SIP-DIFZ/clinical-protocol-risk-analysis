from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskDomain(str, Enum):
    HEALTH = "health"
    FINANCIAL = "financial"
    ETHICAL = "ethical"
    ELIGIBILITY = "eligibility"
    INTERVENTION = "intervention"


class EvidenceSnippet(BaseModel):
    excerpt: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class RiskAssessment(BaseModel):
    title: str = Field(min_length=3)
    domain: RiskDomain
    category: str = Field(min_length=2)
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=10)
    mitigation_status: Literal[
        "unmitigated",
        "partially_mitigated",
        "adequately_mitigated",
        "completely_mitigated",
        "mitigated",
    ] = "unmitigated"
    mitigation_status_reason: str = Field(default="")
    mitigations: list[str] = Field(default_factory=list)
    factor_triggers: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)

    @property
    def risk_score(self) -> float:
        base_score = self.likelihood * self.impact * self.confidence
        status_factor = {
            "unmitigated": 1.0,
            "partially_mitigated": 0.7,
            "adequately_mitigated": 0.45,
            "mitigated": 0.3,
            "completely_mitigated": 0.2,
        }.get(self.mitigation_status, 1.0)
        mitigation_discount = min(0.35, max(0, len(self.mitigations) - 1) * 0.04)
        adjusted_score = base_score * status_factor * max(0.45, 1.0 - mitigation_discount)
        return round(adjusted_score, 2)


class ProtocolMetadata(BaseModel):
    source_path: str
    document_name: str
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    generated_at_utc: str

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class AnalysisResult(BaseModel):
    metadata: ProtocolMetadata
    executive_summary: str = Field(min_length=20)
    key_assumptions: list[str] = Field(default_factory=list)
    risks: list[RiskAssessment] = Field(default_factory=list)


class ProtocolEditPatch(BaseModel):
    risk_title: str = Field(min_length=3)
    action: Literal["replace", "insert_before", "insert_after"]
    target_paragraph_id: str = Field(min_length=2)
    new_text: str = Field(min_length=20)
    reason: str = Field(min_length=10)
