"""Graph state definitions and the pipeline-internal models they carry.

Two state schemas exist because the workflow has two levels:

* :class:`PipelineState` - one per run: seed topic -> blueprint -> batch.
* :class:`SampleState` - one per (sub-topic x language) work item, owning the
  localize -> evaluate -> retry loop. Many run in parallel.

List fields use ``operator.add`` reducers so parallel branches append rather
than overwrite each other.
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

# --------------------------------------------------------------------------- #
# Internal models
# --------------------------------------------------------------------------- #


class SampleTask(BaseModel):
    """An immutable unit of work: adapt one sub-topic into one language."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    seed_topic: str
    domain: Domain
    language_key: str
    subtopic: SubTopic


class LocalizedCandidate(BaseModel):
    """A localizer draft after deterministic post-processing (NFC + alignment)."""

    text: str
    entities: list[EntityLabel]
    """Aligned spans, ready for :class:`SyntheticSample`."""
    mentions: list[EntityMention]
    """Raw mentions as returned by the model (kept for coverage and debugging)."""
    unaligned: list[tuple[str, str]] = Field(default_factory=list)
    invalid_labels: list[str] = Field(default_factory=list)
    adaptation_notes: str = ""


class Evaluation(BaseModel):
    """The Evaluator's verdict on one candidate."""

    score: float = Field(ge=0.0, le=1.0)
    breakdown: QualityBreakdown | None = None
    """``None`` when deterministic checks failed and the LLM critic was skipped."""
    feedback: str
    issues: list[str] = Field(default_factory=list)
    judged_by: str
    """``"rules"`` or the critic's ``provider:model`` label."""


# --------------------------------------------------------------------------- #
# Graph states
# --------------------------------------------------------------------------- #


class PipelineState(TypedDict, total=False):
    """Top-level graph state."""

    seed_topic: str
    domain: Domain
    language_keys: list[str]
    num_subtopics: int
    blueprint_subtopics: list[SubTopic]
    verified_samples: Annotated[list[SyntheticSample], operator.add]
    quarantined_samples: Annotated[list[QuarantinedSample], operator.add]
    batch: DatasetBatch


class WorkerInput(TypedDict):
    """Payload delivered to each parallel sample worker via ``Send``."""

    task: SampleTask


class SampleState(TypedDict, total=False):
    """Per-sample loop state (localize -> evaluate -> route)."""

    task: SampleTask
    attempts: int
    """Number of localization attempts so far; ``retry_count = attempts - 1``."""
    candidate: LocalizedCandidate
    evaluation: Evaluation
    feedback_history: Annotated[list[str], operator.add]
    verified_samples: Annotated[list[SyntheticSample], operator.add]
    quarantined_samples: Annotated[list[QuarantinedSample], operator.add]
