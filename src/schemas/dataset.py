"""The records we publish, and the rules they must pass.

These are strict on purpose: a sample with a span that doesn't match its text,
an unknown label, or text that isn't NFC simply can't be built. (NFC matters
because Indic scripts can encode the same character in more than one way, which
throws character offsets off.)

The looser models the LLMs fill in are in agent_io.py.
"""

from __future__ import annotations

import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class Domain(StrEnum):
    # A new domain also needs labels in DOMAIN_ENTITY_LABELS and formats in prompts.TEXT_FORMATS.
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCE = "finance"


class Script(StrEnum):  # ISO 15924 codes
    DEVANAGARI = "Deva"
    LATIN = "Latn"
    BENGALI = "Beng"
    GURMUKHI = "Guru"
    GUJARATI = "Gujr"
    TAMIL = "Taml"
    TELUGU = "Telu"
    KANNADA = "Knda"
    MALAYALAM = "Mlym"


class QuarantineReason(StrEnum):
    LOW_QUALITY = "low_quality"  # still below the threshold after the last retry
    SCHEMA_VIOLATION = "schema_violation"  # passed review but failed strict validation
    PIPELINE_ERROR = "pipeline_error"  # an LLM/provider error aborted the sample


# The only labels allowed per domain; anything else is rejected at export.
DOMAIN_ENTITY_LABELS: Mapping[Domain, frozenset[str]] = {
    Domain.MEDICAL: frozenset(
        {
            "DRUG",
            "DOSAGE",
            "FREQUENCY",
            "ROUTE",
            "DURATION",
            "CONDITION",
            "SYMPTOM",
            "LAB_TEST",
            "BODY_PART",
            "PROCEDURE",
            "POPULATION",
        }
    ),
    Domain.LEGAL: frozenset(
        {
            "STATUTE",
            "PROVISION",
            "COURT",
            "PARTY",
            "LEGAL_ROLE",
            "JURISDICTION",
            "DATE",
            "PENALTY",
            "LEGAL_CONCEPT",
            "CASE_REFERENCE",
            "AUTHORITY",
        }
    ),
    Domain.FINANCE: frozenset(
        {
            "ORGANIZATION",
            "FINANCIAL_INSTRUMENT",
            "MONETARY_AMOUNT",
            "PERCENTAGE",
            "DATE",
            "REGULATOR",
            "FINANCIAL_METRIC",
            "TAX_CONCEPT",
            "ACCOUNT_TYPE",
            "MARKET_INDEX",
        }
    ),
}

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_nfc(value: str) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("text must be Unicode NFC-normalised before it is stored")
    return value


class EntityLabel(BaseModel):
    """A labelled span. Offsets are Python string indices, end-exclusive, so
    ``raw_text[start_char:end_char] == text``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, description="Exact surface form as it appears in raw_text.")
    label: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$", description="Domain taxonomy label.")
    start_char: int = Field(ge=0, description="Inclusive start offset (code points).")
    end_char: int = Field(gt=0, description="Exclusive end offset (code points).")

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError(f"end_char ({self.end_char}) must be > start_char ({self.start_char})")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError(
                f"span length {self.end_char - self.start_char} does not match "
                f"len(text)={len(self.text)} for entity {self.text!r}"
            )
        return self


class QualityBreakdown(BaseModel):
    """Per-dimension scores behind ``SyntheticSample.quality_score``."""

    model_config = ConfigDict(extra="forbid")

    semantic_fidelity: UnitInterval = Field(description="1.0 = no semantic drift from the source.")
    factual_consistency: UnitInterval = Field(description="1.0 = no hallucinated or altered facts.")
    entity_retention: UnitInterval = Field(description="Key entities preserved and correctly labelled.")
    fluency: UnitInterval = Field(description="Native naturalness; penalises translationese.")
    entity_coverage: UnitInterval = Field(description="Deterministic share of source entities aligned.")
    script_ratio: UnitInterval = Field(description="Deterministic share of letters in the target script.")


class Provenance(BaseModel):
    """Where a sample came from and which models made it."""

    model_config = ConfigDict(extra="forbid")

    seed_topic: str
    subtopic: str
    text_format: str
    source_context: str = Field(description="English grounding passage the sample was adapted from.")
    generator_model: str
    evaluator_model: str
    retry_count: int = Field(ge=0)
    created_at: AwareDatetime = Field(default_factory=_utc_now)


class SyntheticSample(BaseModel):
    """A sample that passed review. Building one fails if anything is inconsistent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"pg-{uuid.uuid4().hex}", min_length=1)
    target_language: str = Field(
        pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$", description="BCP-47 tag, e.g. 'hi', 'hi-Latn'."
    )
    script: Script
    raw_text: str = Field(min_length=1)
    entities: list[EntityLabel] = Field(default_factory=list)
    domain: Domain
    quality_score: UnitInterval
    feedback: str | None = Field(default=None, description="Final critic remarks for this sample.")
    quality_breakdown: QualityBreakdown | None = None
    provenance: Provenance | None = None

    _check_nfc = field_validator("raw_text")(_require_nfc)

    @model_validator(mode="after")
    def _check_entities(self) -> Self:
        allowed = DOMAIN_ENTITY_LABELS[self.domain]
        previous_end = -1
        for entity in self.entities:
            if entity.label not in allowed:
                raise ValueError(
                    f"label {entity.label!r} is not in the {self.domain} taxonomy: {sorted(allowed)}"
                )
            if entity.end_char > len(self.raw_text):
                raise ValueError(f"entity {entity.text!r} ends past the end of raw_text")
            if self.raw_text[entity.start_char : entity.end_char] != entity.text:
                raise ValueError(
                    f"entity {entity.text!r} does not match raw_text at "
                    f"[{entity.start_char}:{entity.end_char}]"
                )
            if entity.start_char < previous_end:
                raise ValueError("entities must be sorted by start_char and must not overlap")
            previous_end = entity.end_char
        return self


class QuarantinedSample(BaseModel):
    """A rejected sample, kept for error analysis. Loosely typed so that we can
    record it whatever went wrong."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"pgq-{uuid.uuid4().hex}")
    task_id: str
    target_language: str
    script: Script
    domain: Domain
    seed_topic: str
    subtopic: str
    reason: QuarantineReason
    last_text: str | None = None
    last_entities: list[dict[str, Any]] = Field(default_factory=list)
    quality_score: UnitInterval | None = None
    retry_count: int = Field(default=0, ge=0)
    feedback_history: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: AwareDatetime = Field(default_factory=_utc_now)


class DatasetBatch(BaseModel):
    """All verified samples from one run."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(default_factory=lambda: f"batch-{uuid.uuid4().hex[:12]}", min_length=1)
    timestamp: AwareDatetime = Field(default_factory=_utc_now)
    total_samples: int = Field(ge=0)
    samples: list[SyntheticSample]
    seed_topic: str | None = None
    domain: Domain | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.total_samples != len(self.samples):
            raise ValueError(
                f"total_samples={self.total_samples} but batch holds {len(self.samples)} samples"
            )
        ids = [s.id for s in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("sample ids must be unique within a batch")
        return self

    @classmethod
    def from_samples(cls, samples: Sequence[SyntheticSample], **fields: Any) -> DatasetBatch:
        return cls(samples=list(samples), total_samples=len(samples), **fields)
