"""Agent node implementations.

``AgentNodes`` bundles the LangGraph node callables. Its LLM dependencies are
injected as :class:`~src.llm.StructuredGenerator` objects, so nodes are
provider-agnostic and unit-testable with scripted fakes.

Nodes:

* :meth:`AgentNodes.domain_architect` - seed topic -> sub-topics + English context.
* :meth:`AgentNodes.localization_adapter` - transcreates one sub-topic into one
  language and aligns entity spans deterministically.
* :meth:`AgentNodes.adversarial_evaluator` - rule-based gates, then an LLM
  critic; produces ``quality_score`` and feedback for the retry loop.
* :meth:`AgentNodes.export_sample` / :meth:`AgentNodes.quarantine_sample` -
  terminal sinks of the per-sample loop.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from src.agents import prompts
from src.agents.state import (
    Evaluation,
    LocalizedCandidate,
    PipelineState,
    SampleState,
    SampleTask,
)
from src.config import QualityGate
from src.linguistics import align_entities, extract_numbers, nfc, script_ratio
from src.llm import StructuredGenerator
from src.schemas.agent_io import (
    CriticVerdict,
    DomainBlueprint,
    EntitySeed,
    LocalizedDraft,
    SubTopic,
)
from src.schemas.dataset import (
    DOMAIN_ENTITY_LABELS,
    Domain,
    Provenance,
    QualityBreakdown,
    QuarantinedSample,
    QuarantineReason,
    SyntheticSample,
)
from src.schemas.languages import LanguageSpec, get_language

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 40


class PipelineError(RuntimeError):
    """A failure that makes the whole run unusable (e.g. no sub-topics)."""


def aggregate_quality(breakdown: QualityBreakdown, gate: QualityGate) -> float:
    """Combine rubric dimensions into a single ``quality_score``.

    Weighted mean of the critic's dimensions, capped at the weakest dimension
    whenever that dimension is below ``gate.dimension_floor``. The cap makes the
    score non-compensatory: excellent fluency cannot rescue a hallucinated dose.
    """
    dims = {name: float(getattr(breakdown, name)) for name in gate.weights}
    total_weight = sum(gate.weights.values())
    score = sum(dims[name] * weight for name, weight in gate.weights.items()) / total_weight
    weakest = min(dims.values())
    if weakest < gate.dimension_floor:
        score = min(score, weakest)
    return round(score, 4)


def _format_key_entities(subtopic: SubTopic) -> str:
    return prompts.bullet_list([f"{e.text} -> {e.label}" for e in subtopic.key_entities])


class AgentNodes:
    """LangGraph node callables with injected LLM dependencies."""

    def __init__(
        self,
        *,
        architect: StructuredGenerator,
        localizer: StructuredGenerator,
        evaluator: StructuredGenerator,
        quality: QualityGate,
    ) -> None:
        self._architect = architect
        self._localizer = localizer
        self._evaluator = evaluator
        self._quality = quality

    # ------------------------------------------------------------------ #
    # Domain Architect
    # ------------------------------------------------------------------ #

    def domain_architect(self, state: PipelineState) -> dict[str, Any]:
        """Expand the seed topic into grounded sub-topics.

        Raises:
            PipelineError: if no usable sub-topic survives sanitisation.
        """
        domain = Domain(state["domain"])
        requested = state["num_subtopics"]
        system = prompts.ARCHITECT_SYSTEM.format(
            domain=domain.value,
            num_subtopics=requested,
            formats="; ".join(prompts.TEXT_FORMATS[domain.value]),
            labels=", ".join(sorted(DOMAIN_ENTITY_LABELS[domain])),
        )
        blueprint = self._architect.generate(
            DomainBlueprint,
            system=system,
            user=prompts.ARCHITECT_USER.format(seed_topic=state["seed_topic"]),
            tag=f"seed={state['seed_topic'][:40]}",
        )
        subtopics = self._sanitise_blueprint(blueprint, domain)[:requested]
        if not subtopics:
            raise PipelineError(
                "Domain Architect returned no usable sub-topics (each needs a context passage and "
                "at least one key entity that appears verbatim in it with a valid label)."
            )
        if len(subtopics) < requested:
            logger.warning(
                "Domain Architect produced %d/%d usable sub-topics", len(subtopics), requested
            )
        logger.info(
            "Blueprint: %s", " | ".join(f"{s.title} [{s.text_format}]" for s in subtopics)
        )
        return {"blueprint_subtopics": subtopics}

    @staticmethod
    def _sanitise_blueprint(blueprint: DomainBlueprint, domain: Domain) -> list[SubTopic]:
        """Drop duplicate sub-topics and key entities that are off-taxonomy or
        not actually present in the context passage."""
        allowed = DOMAIN_ENTITY_LABELS[domain]
        seen_titles: set[str] = set()
        clean: list[SubTopic] = []
        for sub in blueprint.subtopics:
            title_key = sub.title.strip().casefold()
            context = sub.domain_context.strip()
            if not title_key or title_key in seen_titles or not context:
                continue
            context_cf = context.casefold()
            entities: list[EntitySeed] = []
            seen_entities: set[str] = set()
            for entity in sub.key_entities:
                key = entity.text.strip().casefold()
                if entity.label in allowed and key and key in context_cf and key not in seen_entities:
                    entities.append(EntitySeed(text=entity.text.strip(), label=entity.label))
                    seen_entities.add(key)
                else:
                    logger.debug("Dropping key entity %r (%s) from %r", entity.text, entity.label, sub.title)
            if not entities:
                logger.warning("Dropping sub-topic %r: no valid key entities", sub.title)
                continue
            seen_titles.add(title_key)
            clean.append(
                sub.model_copy(
                    update={"title": sub.title.strip(), "domain_context": context, "key_entities": entities}
                )
            )
        return clean

    # ------------------------------------------------------------------ #
    # Localization Adapter
    # ------------------------------------------------------------------ #

    def localization_adapter(self, state: SampleState) -> dict[str, Any]:
        """Transcreate the sub-topic into the target language.

        On a retry the previous draft and the critic's feedback are appended to
        the prompt, closing the adversarial loop.
        """
        task = state["task"]
        lang = get_language(task.language_key)
        attempts = state.get("attempts", 0) + 1
        sub = task.subtopic
        labels = DOMAIN_ENTITY_LABELS[task.domain]

        system = prompts.LOCALIZER_SYSTEM.format(
            language=lang.name,
            domain=task.domain.value,
            script=lang.script.name.title(),
            style_guide=lang.style_guide,
            labels=", ".join(sorted(labels)),
        )
        user = prompts.LOCALIZER_USER.format(
            title=sub.title,
            text_format=sub.text_format,
            context=sub.domain_context,
            key_entities=_format_key_entities(sub),
        )
        previous, evaluation = state.get("candidate"), state.get("evaluation")
        if previous is not None and evaluation is not None:
            user += prompts.LOCALIZER_RETRY.format(
                previous_text=previous.text, feedback=evaluation.feedback
            )

        draft = self._localizer.generate(
            LocalizedDraft,
            system=system,
            user=user,
            tag=f"{lang.code} | {sub.title[:40]} | attempt {attempts}",
        )
        return {"candidate": self._postprocess(draft, task, lang), "attempts": attempts}

    @staticmethod
    def _postprocess(draft: LocalizedDraft, task: SampleTask, lang: LanguageSpec) -> LocalizedCandidate:
        """NFC-normalise, filter off-taxonomy labels and align entity spans."""
        text = nfc(draft.text).strip()
        allowed = DOMAIN_ENTITY_LABELS[task.domain]
        valid = [m for m in draft.entities if m.label in allowed]
        invalid = sorted({m.label for m in draft.entities if m.label not in allowed})
        alignment = align_entities(
            text,
            [(m.text, m.label) for m in valid],
            strict_right_boundary=lang.strict_right_boundary,
        )
        return LocalizedCandidate(
            text=text,
            entities=alignment.entities,
            mentions=valid,
            unaligned=alignment.unaligned,
            invalid_labels=invalid,
            adaptation_notes=draft.adaptation_notes,
        )

    @staticmethod
    def _missing_key_entities(task: SampleTask, candidate: LocalizedCandidate) -> list[str]:
        """Key entities with no aligned counterpart in the candidate.

        A key entity counts as preserved if an aligned mention cites it as its
        ``source_text``, or if its English form appears verbatim in the text
        (common for Hinglish and for drug names kept in Latin script).
        """
        unaligned = {surface for surface, _ in candidate.unaligned}
        cited = {
            m.source_text.strip().casefold()
            for m in candidate.mentions
            if m.source_text.strip() and nfc(m.text).strip() not in unaligned
        }
        text_cf = candidate.text.casefold()
        return [
            e.text
            for e in task.subtopic.key_entities
            if e.text.casefold() not in cited and e.text.casefold() not in text_cf
        ]

    # ------------------------------------------------------------------ #
    # Adversarial Evaluator
    # ------------------------------------------------------------------ #

    def adversarial_evaluator(self, state: SampleState) -> dict[str, Any]:
        """Score the candidate.

        Stage 1 (rules, free): script purity, entity alignment, label taxonomy,
        key-entity coverage, numeric fidelity. Any failure scores 0.0 with
        precise feedback and skips the LLM call. Stage 2 (LLM critic): rubric
        scores for semantic drift, hallucination, entity retention and fluency.
        """
        task, candidate = state["task"], state["candidate"]
        lang = get_language(task.language_key)
        sub = task.subtopic

        ratio = script_ratio(candidate.text, lang.script)
        missing = self._missing_key_entities(task, candidate)
        n_keys = len(sub.key_entities)
        coverage = (n_keys - len(missing)) / n_keys if n_keys else 1.0
        missing_numbers = sorted(extract_numbers(sub.domain_context) - extract_numbers(candidate.text))
        failures = self._rule_failures(candidate, lang, ratio, coverage, missing, missing_numbers)

        if failures:
            evaluation = Evaluation(
                score=0.0,
                feedback="Automatic checks failed:\n" + prompts.bullet_list(failures),
                issues=failures,
                judged_by="rules",
            )
        else:
            verdict = self._evaluator.generate(
                CriticVerdict,
                system=prompts.EVALUATOR_SYSTEM.format(domain=task.domain.value, language=lang.name),
                user=prompts.EVALUATOR_USER.format(
                    language=lang.name,
                    script=lang.script.name.title(),
                    text_format=sub.text_format,
                    context=sub.domain_context,
                    key_entities=_format_key_entities(sub),
                    candidate=candidate.text,
                    spans=prompts.bullet_list([f"{e.text} -> {e.label}" for e in candidate.entities]),
                    script_ratio=ratio,
                    coverage=coverage,
                ),
                tag=f"{lang.code} | {sub.title[:40]} | attempt {state.get('attempts', 1)}",
            )
            breakdown = QualityBreakdown(
                semantic_fidelity=verdict.semantic_fidelity,
                factual_consistency=verdict.factual_consistency,
                entity_retention=verdict.entity_retention,
                fluency=verdict.fluency,
                entity_coverage=round(coverage, 4),
                script_ratio=round(ratio, 4),
            )
            evaluation = Evaluation(
                score=aggregate_quality(breakdown, self._quality),
                breakdown=breakdown,
                feedback=verdict.feedback.strip() or "No issues.",
                issues=verdict.issues,
                judged_by=self._evaluator.model_label,
            )

        logger.info(
            "[evaluator] %s | %s | attempt %d | score %.3f (%s)",
            lang.code, sub.title[:40], state.get("attempts", 1), evaluation.score, evaluation.judged_by,
        )
        return {"evaluation": evaluation, "feedback_history": [evaluation.feedback]}

    def _rule_failures(
        self,
        candidate: LocalizedCandidate,
        lang: LanguageSpec,
        ratio: float,
        coverage: float,
        missing: list[str],
        missing_numbers: list[str],
    ) -> list[str]:
        failures: list[str] = []
        script = lang.script.name.title()
        if len(candidate.text) < MIN_TEXT_CHARS:
            failures.append(f"The passage is empty or shorter than {MIN_TEXT_CHARS} characters.")
        if ratio < lang.min_script_ratio:
            failures.append(
                f"Only {ratio:.0%} of letters are in {script} script (minimum "
                f"{lang.min_script_ratio:.0%}). Write the passage in {script} script."
            )
        if candidate.invalid_labels:
            failures.append(
                f"Labels outside the allowed set were used: {', '.join(candidate.invalid_labels)}."
            )
        for surface, label in candidate.unaligned:
            failures.append(
                f"Entity '{surface}' ({label}) does not occur verbatim in the passage; copy entity "
                "text exactly from the passage."
            )
        if not candidate.entities:
            failures.append("No entities were tagged in the passage.")
        if coverage < self._quality.min_entity_coverage:
            failures.append(
                f"Only {coverage:.0%} of key entities were preserved. Missing: {', '.join(missing)}. "
                "Include each key entity in the passage and set its source_text to the English key "
                "entity it translates."
            )
        if missing_numbers:
            failures.append(
                f"Numbers from the source are missing or changed: {', '.join(missing_numbers)}. "
                "Every quantity must be preserved exactly, written as numerals."
            )
        return failures

    # ------------------------------------------------------------------ #
    # Sinks
    # ------------------------------------------------------------------ #

    def export_sample(self, state: SampleState) -> dict[str, Any]:
        """Promote an accepted candidate to a strict :class:`SyntheticSample`.

        Strict validation is the last gate: if it fails, the candidate is
        quarantined instead of crashing the batch.
        """
        task, candidate, evaluation = state["task"], state["candidate"], state["evaluation"]
        lang = get_language(task.language_key)
        try:
            sample = SyntheticSample(
                target_language=lang.code,
                script=lang.script,
                raw_text=candidate.text,
                entities=candidate.entities,
                domain=task.domain,
                quality_score=evaluation.score,
                feedback=evaluation.feedback,
                quality_breakdown=evaluation.breakdown,
                provenance=Provenance(
                    seed_topic=task.seed_topic,
                    subtopic=task.subtopic.title,
                    text_format=task.subtopic.text_format,
                    source_context=task.subtopic.domain_context,
                    generator_model=self._localizer.model_label,
                    evaluator_model=evaluation.judged_by,
                    retry_count=state.get("attempts", 1) - 1,
                ),
            )
        except ValidationError as exc:
            logger.error("Sample %s failed strict validation: %s", task.task_id, exc)
            return {
                "quarantined_samples": [
                    self.quarantine_record(
                        task, reason=QuarantineReason.SCHEMA_VIOLATION, state=state, error=str(exc)
                    )
                ]
            }
        return {"verified_samples": [sample]}

    def quarantine_sample(self, state: SampleState) -> dict[str, Any]:
        """Withhold a candidate that exhausted its retries."""
        task = state["task"]
        logger.warning(
            "Quarantined %s | %s after %d attempts",
            task.language_key, task.subtopic.title[:40], state.get("attempts", 0),
        )
        return {
            "quarantined_samples": [
                self.quarantine_record(task, reason=QuarantineReason.LOW_QUALITY, state=state)
            ]
        }

    @staticmethod
    def quarantine_record(
        task: SampleTask,
        *,
        reason: QuarantineReason,
        state: SampleState | None = None,
        error: str | None = None,
    ) -> QuarantinedSample:
        """Build a quarantine record from whatever state is available."""
        lang = get_language(task.language_key)
        state = state or {}
        candidate = state.get("candidate")
        evaluation = state.get("evaluation")
        return QuarantinedSample(
            task_id=task.task_id,
            target_language=lang.code,
            script=lang.script,
            domain=task.domain,
            seed_topic=task.seed_topic,
            subtopic=task.subtopic.title,
            reason=reason,
            last_text=candidate.text if candidate else None,
            last_entities=[m.model_dump() for m in candidate.mentions] if candidate else [],
            quality_score=evaluation.score if evaluation else None,
            retry_count=max(state.get("attempts", 1) - 1, 0),
            feedback_history=list(state.get("feedback_history", [])),
            error=error,
        )
