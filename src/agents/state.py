"""Graph state, plus the small models passed between nodes.

PipelineState belongs to the outer graph (one per run), SampleState to each
parallel worker. The list fields use operator.add so that parallel workers
append to them instead of overwriting each other.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.agent_io import EntityMention, SubTopic
from src.schemas.dataset import (
    DatasetBatch,
    Domain,
    EntityLabel,
    QualityBreakdown,
    QuarantinedSample,
    SyntheticSample,
)


class SampleTask(BaseModel):
    """One unit of work: adapt one sub-topic into one language."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    seed_topic: str
    domain: Domain
    language_key: str
    subtopic: SubTopic


class LocalizedCandidate(BaseModel):
    """A localizer draft after NFC normalisation and span alignment."""

    text: str
    entities: list[EntityLabel]  # aligned spans, ready for SyntheticSample
    mentions: list[EntityMention]  # raw model output, used for coverage checks
    unaligned: list[tuple[str, str]] = Field(default_factory=list)
    invalid_labels: list[str] = Field(default_factory=list)
    adaptation_notes: str = ""


class Evaluation(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: QualityBreakdown | None = None  # None if the rule checks failed (critic skipped)
    feedback: str
    issues: list[str] = Field(default_factory=list)
    judged_by: str  # "rules" or the critic's provider:model


class PipelineState(TypedDict, total=False):
    seed_topic: str
    domain: Domain
    language_keys: list[str]
    num_subtopics: int
    blueprint_subtopics: list[SubTopic]
    verified_samples: Annotated[list[SyntheticSample], operator.add]
    quarantined_samples: Annotated[list[QuarantinedSample], operator.add]
    batch: DatasetBatch


class WorkerInput(TypedDict):
    """What each Send() hands to a sample worker."""

    task: SampleTask


class SampleState(TypedDict, total=False):
    task: SampleTask
    attempts: int  # localizer attempts so far, so retries = attempts - 1
    candidate: LocalizedCandidate
    evaluation: Evaluation
    feedback_history: Annotated[list[str], operator.add]
    verified_samples: Annotated[list[SyntheticSample], operator.add]
    quarantined_samples: Annotated[list[QuarantinedSample], operator.add]
