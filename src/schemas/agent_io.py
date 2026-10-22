"""The shapes the LLMs fill in.

Careful when editing: the class docstrings and Field descriptions are sent to
the model as part of the schema, so they act as prompt text. These models are
forgiving (labels get normalised rather than rejected); the strict checks
happen later in the evaluator and in dataset.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


def _normalise_label(value: str) -> str:
    # "lab test" / "Lab-Test" -> "LAB_TEST"
    return "_".join(value.strip().upper().replace("-", " ").split())


class EntitySeed(BaseModel):
    """A key entity the architect expects every localisation to preserve."""

    text: str = Field(description="Entity exactly as written in domain_context (English).")
    label: str = Field(description="One label from the allowed label set.")

    _norm = field_validator("label")(_normalise_label)


class SubTopic(BaseModel):
    """One focused slice of the seed topic, with its English grounding passage."""

    title: str = Field(description="Short, specific sub-topic title.")
    text_format: str = Field(
        description="Realistic document type for this sub-topic, e.g. 'patient counselling dialogue'."
    )
    domain_context: str = Field(
        description=(
            "Self-contained, factually accurate English passage of 120-200 words written in the "
            "chosen text_format. Use only widely accepted, guideline-level facts."
        )
    )
    key_entities: list[EntitySeed] = Field(
        description="4-8 domain entities that appear verbatim in domain_context."
    )


class DomainBlueprint(BaseModel):
    """Architect output: the expanded topic plan."""

    subtopics: list[SubTopic] = Field(description="Distinct, non-overlapping sub-topics.")


class EntityMention(BaseModel):
    """An entity as it appears in the *localised* passage."""

    text: str = Field(
        description="Exact substring of the localised passage, copied character-for-character."
    )
    label: str = Field(description="One label from the allowed label set.")
    source_text: str = Field(
        description=(
            "The English key entity this corresponds to, copied exactly from the key entity list; "
            "'' only for an extra entity that is not in that list."
        ),
    )

    _norm = field_validator("label")(_normalise_label)


class LocalizedDraft(BaseModel):
    """Localizer output: the adapted passage plus its entity annotations."""

    text: str = Field(description="The complete localised passage only - no title, notes or gloss.")
    entities: list[EntityMention] = Field(description="Every domain entity in the passage.")
    adaptation_notes: str = Field(
        default="", description="Optional brief notes on non-obvious adaptation choices (English)."
    )


class CriticVerdict(BaseModel):
    """Evaluator output: a rubric score per dimension plus actionable feedback."""

    semantic_fidelity: float = Field(
        ge=0.0, le=1.0, description="1.0 = every source proposition preserved; no meaning drift."
    )
    factual_consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="1.0 = nothing added, removed or altered. Any changed number/dose/date <= 0.2.",
    )
    entity_retention: float = Field(
        ge=0.0, le=1.0, description="Key entities present, with correct spans and correct labels."
    )
    fluency: float = Field(
        ge=0.0, le=1.0, description="Native naturalness and register; penalise translationese."
    )
    issues: list[str] = Field(
        default_factory=list, description="Each concrete defect found, quoting the offending span."
    )
    feedback: str = Field(
        description="Actionable rewrite instructions for the localizer, or 'No issues.' if clean."
    )
