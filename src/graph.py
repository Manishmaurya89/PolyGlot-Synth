"""The LangGraph workflows.

There are two graphs. The outer one runs once per batch:

    domain_architect --Send per (sub-topic, language)--> sample_worker --> finalize_batch

Each sample_worker runs the inner graph, which loops until the sample passes
or runs out of retries:

    localization_adapter -> adversarial_evaluator -> export | retry | quarantine

A worker that throws (e.g. Ollama goes down) only quarantines its own sample;
the rest of the batch carries on.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.agents.nodes import AgentNodes, PipelineError
from src.agents.state import PipelineState, SampleState, SampleTask, WorkerInput
from src.config import AgentRole, QualityGate, Settings
from src.llm import build_role_llms
from src.schemas.dataset import DatasetBatch, Domain, QuarantineReason
from src.schemas.languages import get_language
from src.telemetry import UsageTracker

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineError",
    "build_pipeline",
    "build_pipeline_graph",
    "build_sample_graph",
    "initial_state",
    "route_after_evaluation",
]

Route = Literal["export", "retry", "quarantine"]


def route_after_evaluation(state: SampleState, gate: QualityGate) -> Route:
    """Export if the score clears the threshold, retry while retries are left,
    otherwise quarantine."""
    evaluation = state["evaluation"]
    retry_count = state.get("attempts", 1) - 1
    if evaluation.score >= gate.threshold:
        return "export"
    if retry_count < gate.max_retries:
        return "retry"
    return "quarantine"


def build_sample_graph(nodes: AgentNodes, gate: QualityGate) -> CompiledStateGraph:
    def router(state: SampleState) -> Route:
        return route_after_evaluation(state, gate)

    graph = StateGraph(SampleState)
    graph.add_node("localization_adapter", nodes.localization_adapter)
    graph.add_node("adversarial_evaluator", nodes.adversarial_evaluator)
    graph.add_node("export", nodes.export_sample)
    graph.add_node("quarantine", nodes.quarantine_sample)

    graph.add_edge(START, "localization_adapter")
    graph.add_edge("localization_adapter", "adversarial_evaluator")
    graph.add_conditional_edges(
        "adversarial_evaluator",
        router,
        {"export": "export", "retry": "localization_adapter", "quarantine": "quarantine"},
    )
    graph.add_edge("export", END)
    graph.add_edge("quarantine", END)
    return graph.compile(name="sample_worker")


def finalize_batch(state: PipelineState) -> dict[str, Any]:
    # Workers finish in any order; sort so the output file is stable between runs.
    samples = sorted(
        state.get("verified_samples", []),
        key=lambda s: (s.target_language, s.provenance.subtopic if s.provenance else "", s.id),
    )
    batch = DatasetBatch.from_samples(
        samples, seed_topic=state["seed_topic"], domain=Domain(state["domain"])
    )
    return {"batch": batch}


def build_pipeline_graph(nodes: AgentNodes, gate: QualityGate) -> CompiledStateGraph:
    sample_graph = build_sample_graph(nodes, gate)
    # Two steps per attempt plus the sink, with headroom.
    sample_config = {"recursion_limit": 2 * (gate.max_retries + 1) + 10}

    def sample_worker(payload: WorkerInput) -> dict[str, Any]:
        task = payload["task"]
        try:
            final: SampleState = sample_graph.invoke({"task": task, "attempts": 0}, sample_config)
        except Exception as exc:  # one bad sample shouldn't kill the batch
            logger.error(
                "Sample %s (%s | %s) aborted: %s",
                task.task_id, task.language_key, task.subtopic.title[:40], exc,
            )
            record = nodes.quarantine_record(
                task, reason=QuarantineReason.PIPELINE_ERROR, error=f"{type(exc).__name__}: {exc}"
            )
            return {"quarantined_samples": [record]}
        return {
            "verified_samples": final.get("verified_samples", []),
            "quarantined_samples": final.get("quarantined_samples", []),
        }

    def fan_out(state: PipelineState) -> list[Send]:
        return [
            Send(
                "sample_worker",
                WorkerInput(
                    task=SampleTask(
                        seed_topic=state["seed_topic"],
                        domain=state["domain"],
                        language_key=language_key,
                        subtopic=subtopic,
                    )
                ),
            )
            for subtopic in state["blueprint_subtopics"]
            for language_key in state["language_keys"]
        ]

    graph = StateGraph(PipelineState)
    graph.add_node("domain_architect", nodes.domain_architect)
    graph.add_node("sample_worker", sample_worker)
    graph.add_node("finalize_batch", finalize_batch)

    graph.add_edge(START, "domain_architect")
    graph.add_conditional_edges("domain_architect", fan_out, ["sample_worker"])
    graph.add_edge("sample_worker", "finalize_batch")
    graph.add_edge("finalize_batch", END)
    return graph.compile(name="polyglot_crew")


def build_pipeline(settings: Settings, tracker: UsageTracker) -> CompiledStateGraph:
    llms = build_role_llms(settings, tracker)
    nodes = AgentNodes(
        architect=llms[AgentRole.ARCHITECT],
        localizer=llms[AgentRole.LOCALIZER],
        evaluator=llms[AgentRole.EVALUATOR],
        quality=settings.quality,
    )
    return build_pipeline_graph(nodes, settings.quality)


def initial_state(
    seed_topic: str,
    domain: Domain | str,
    languages: Iterable[str],
    num_subtopics: int,
) -> PipelineState:
    """Check the run inputs and build the starting state. Raises ValueError on bad input."""
    if not seed_topic.strip():
        raise ValueError("seed_topic must not be empty")
    if num_subtopics < 1:
        raise ValueError("num_subtopics must be >= 1")
    language_keys = list(dict.fromkeys(get_language(lang).key for lang in languages))
    if not language_keys:
        raise ValueError("at least one target language is required")
    return PipelineState(
        seed_topic=seed_topic.strip(),
        domain=Domain(domain),
        language_keys=language_keys,
        num_subtopics=num_subtopics,
        verified_samples=[],
        quarantined_samples=[],
    )
