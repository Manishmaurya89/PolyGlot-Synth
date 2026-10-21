"""End-to-end LangGraph tests with scripted LLMs (no network)."""

from __future__ import annotations

import pytest

from src.agents.nodes import AgentNodes, PipelineError, aggregate_quality
from src.agents.state import Evaluation
from src.config import QualityGate
from src.graph import build_pipeline_graph, initial_state, route_after_evaluation
from src.llm import LLMCallError
from src.schemas.agent_io import (
    CriticVerdict,
    DomainBlueprint,
    EntityMention,
    EntitySeed,
    LocalizedDraft,
)
from src.schemas.dataset import QualityBreakdown, QuarantineReason
from tests.fakes import ScriptedGenerator, hindi_draft, hinglish_draft, make_subtopic, verdict

GATE = QualityGate()


def run(
    *,
    localizer: list,
    evaluator: list,
    blueprint: DomainBlueprint | None = None,
    languages: tuple[str, ...] = ("hindi",),
    num_subtopics: int = 1,
) -> tuple[dict, ScriptedGenerator, ScriptedGenerator]:
    blueprint = blueprint or DomainBlueprint(subtopics=[make_subtopic()])
    loc = ScriptedGenerator("fake:localizer", {LocalizedDraft: localizer})
    ev = ScriptedGenerator("fake:critic", {CriticVerdict: evaluator})
    nodes = AgentNodes(
        architect=ScriptedGenerator("fake:architect", {DomainBlueprint: [blueprint]}),
        localizer=loc,
        evaluator=ev,
        quality=GATE,
    )
    graph = build_pipeline_graph(nodes, GATE)
    state = initial_state("Clinical Drug Dosage Diagnostics", "medical", languages, num_subtopics)
    return graph.invoke(state, {"max_concurrency": 1}), loc, ev


def test_high_score_is_exported_with_aligned_entities() -> None:
    final, loc, ev = run(localizer=[hindi_draft()], evaluator=[verdict(0.95)])

    batch = final["batch"]
    assert batch.total_samples == 1 and not final["quarantined_samples"]
    sample = batch.samples[0]
    assert (sample.target_language, sample.script) == ("hi", "Deva")
    assert sample.quality_score == pytest.approx(0.95)
    assert sample.provenance.retry_count == 0
    assert sample.provenance.generator_model == "fake:localizer"
    assert {e.label for e in sample.entities} == {"DRUG", "DOSAGE", "FREQUENCY", "SYMPTOM"}
    for entity in sample.entities:
        assert sample.raw_text[entity.start_char : entity.end_char] == entity.text
    assert len(loc.calls) == len(ev.calls) == 1


def test_low_score_feeds_critic_feedback_back_to_localizer() -> None:
    final, loc, _ = run(
        localizer=[hindi_draft()],
        evaluator=[verdict(0.5, "Use the everyday word for fever."), verdict(0.92)],
    )

    sample = final["batch"].samples[0]
    assert sample.provenance.retry_count == 1
    retry_prompt = loc.calls[1]["user"]
    assert "REJECTED" in retry_prompt and "Use the everyday word for fever." in retry_prompt
    assert "REJECTED" not in loc.calls[0]["user"]


def test_exhausted_retries_quarantine_the_sample() -> None:
    final, loc, _ = run(localizer=[hindi_draft()], evaluator=[verdict(0.5)])

    assert final["batch"].total_samples == 0
    [record] = final["quarantined_samples"]
    assert record.reason is QuarantineReason.LOW_QUALITY
    assert record.retry_count == GATE.max_retries
    assert len(loc.calls) == GATE.max_retries + 1
    assert len(record.feedback_history) == GATE.max_retries + 1
    assert record.last_text is not None


def test_rule_failure_skips_llm_critic_and_explains_why() -> None:
    bad = hindi_draft(
        entities=[EntityMention(text="आइबुप्रोफ़ेन", label="DRUG", source_text="paracetamol")]
    )
    final, loc, ev = run(localizer=[bad, hindi_draft()], evaluator=[verdict(0.9)])

    assert final["batch"].total_samples == 1
    assert len(ev.calls) == 1  # first attempt never reached the critic
    assert "does not occur verbatim" in loc.calls[1]["user"]


def test_wrong_script_is_rejected_deterministically() -> None:
    final, loc, ev = run(localizer=[hinglish_draft(), hindi_draft()], evaluator=[verdict(0.9)])

    assert final["batch"].total_samples == 1
    assert len(ev.calls) == 1
    assert "Devanagari script" in loc.calls[1]["user"]


def test_changed_dose_is_rejected_before_the_critic() -> None:
    wrong_dose = hindi_draft(text=hindi_draft().text.replace("500 mg", "250 mg"))
    final, loc, ev = run(localizer=[wrong_dose, hindi_draft()], evaluator=[verdict(0.9)])

    assert final["batch"].total_samples == 1
    assert len(ev.calls) == 1
    assert "Numbers from the source are missing or changed: 500" in loc.calls[1]["user"]


def test_provider_error_on_one_sample_does_not_sink_the_batch() -> None:
    def localize(system: str, user: str) -> LocalizedDraft:
        if "Second topic" in user:
            raise LLMCallError("ollama:llama3 unavailable after 3 attempts")
        return hindi_draft()

    blueprint = DomainBlueprint(
        subtopics=[make_subtopic("First topic"), make_subtopic("Second topic")]
    )
    final, _, _ = run(localizer=[localize], evaluator=[verdict(0.95)], blueprint=blueprint, num_subtopics=2)

    assert final["batch"].total_samples == 1
    [record] = final["quarantined_samples"]
    assert record.reason is QuarantineReason.PIPELINE_ERROR
    assert record.subtopic == "Second topic"
    assert "unavailable" in (record.error or "")


def test_fan_out_covers_every_subtopic_language_pair() -> None:
    def localize(system: str, user: str) -> LocalizedDraft:
        return hinglish_draft() if "Hinglish" in system else hindi_draft()

    blueprint = DomainBlueprint(subtopics=[make_subtopic("Topic A"), make_subtopic("Topic B")])
    final, _, _ = run(
        localizer=[localize],
        evaluator=[verdict(0.95)],
        blueprint=blueprint,
        languages=("hindi", "hi-Latn"),
        num_subtopics=2,
    )

    batch = final["batch"]
    pairs = {(s.target_language, s.provenance.subtopic) for s in batch.samples}
    assert pairs == {(lang, t) for lang in ("hi", "hi-Latn") for t in ("Topic A", "Topic B")}
    assert [s.target_language for s in batch.samples] == sorted(s.target_language for s in batch.samples)


def test_architect_output_is_sanitised() -> None:
    noisy = make_subtopic("Noisy").model_copy(
        update={
            "key_entities": [
                EntitySeed(text="paracetamol", label="DRUG"),
                EntitySeed(text="aspirin", label="DRUG"),  # not in context
                EntitySeed(text="fever", label="STATUTE"),  # off-taxonomy
            ]
        }
    )
    blueprint = DomainBlueprint(subtopics=[noisy, make_subtopic("noisy")])  # duplicate title
    nodes = AgentNodes(
        architect=ScriptedGenerator("fake", {DomainBlueprint: [blueprint]}),
        localizer=ScriptedGenerator("fake", {}),
        evaluator=ScriptedGenerator("fake", {}),
        quality=GATE,
    )
    result = nodes.domain_architect(initial_state("t", "medical", ["hindi"], 5))

    [subtopic] = result["blueprint_subtopics"]
    assert [e.text for e in subtopic.key_entities] == ["paracetamol"]


def test_unusable_blueprint_aborts_the_run() -> None:
    empty = make_subtopic().model_copy(update={"key_entities": [EntitySeed(text="x", label="DRUG")]})
    with pytest.raises(PipelineError):
        run(localizer=[hindi_draft()], evaluator=[verdict(0.9)], blueprint=DomainBlueprint(subtopics=[empty]))


@pytest.mark.parametrize(
    ("score", "attempts", "expected"),
    [(0.85, 1, "export"), (0.84, 1, "retry"), (0.84, 3, "retry"), (0.84, 4, "quarantine")],
)
def test_router_thresholds(score: float, attempts: int, expected: str) -> None:
    state = {"evaluation": Evaluation(score=score, feedback="", judged_by="x"), "attempts": attempts}
    assert route_after_evaluation(state, GATE) == expected


def test_quality_score_is_capped_by_weakest_dimension() -> None:
    breakdown = QualityBreakdown(
        semantic_fidelity=1.0,
        factual_consistency=0.2,
        entity_retention=1.0,
        fluency=1.0,
        entity_coverage=1.0,
        script_ratio=1.0,
    )
    assert aggregate_quality(breakdown, GATE) == pytest.approx(0.2)
    clean = breakdown.model_copy(update={"factual_consistency": 0.9})
    assert aggregate_quality(clean, GATE) == pytest.approx(0.97)
