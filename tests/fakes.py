"""A fake LLM that returns scripted answers, plus some sample data.

Lets the tests run the real graph without a network or a model.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from src.schemas.agent_io import (
    CriticVerdict,
    EntityMention,
    EntitySeed,
    LocalizedDraft,
    SubTopic,
)

HINDI_TEXT = (
    "डॉक्टर ने मरीज़ को पैरासिटामोल 500 mg दिन में तीन बार लेने की सलाह दी। "
    "बुखार कम न हो तो दोबारा जाँच कराएँ।"
)

Response = BaseModel | Exception | Callable[[str, str], BaseModel]


class ScriptedGenerator:
    """Hands out queued responses per schema and repeats the last one.

    A response can be a model, an exception to raise, or a function f(system, user).
    """

    def __init__(self, label: str, script: dict[type[BaseModel], list[Response]]) -> None:
        self._label = label
        self._script = {schema: list(items) for schema, items in script.items()}
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []

    @property
    def model_label(self) -> str:
        return self._label

    def generate(self, schema: type[BaseModel], *, system: str, user: str, tag: str = "") -> Any:
        with self._lock:
            self.calls.append({"schema": schema, "system": system, "user": user, "tag": tag})
            queue = self._script[schema]
            item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item(system, user) if callable(item) and not isinstance(item, BaseModel) else item


def make_subtopic(title: str = "Paracetamol fever dosing") -> SubTopic:
    return SubTopic(
        title=title,
        text_format="patient counselling dialogue",
        domain_context=(
            "The doctor advised the patient to take paracetamol 500 mg three times a day. "
            "If the fever persists, get tested again."
        ),
        key_entities=[
            EntitySeed(text="paracetamol", label="DRUG"),
            EntitySeed(text="500 mg", label="DOSAGE"),
            EntitySeed(text="fever", label="SYMPTOM"),
        ],
    )


def hindi_draft(**overrides: Any) -> LocalizedDraft:
    fields: dict[str, Any] = {
        "text": HINDI_TEXT,
        "entities": [
            EntityMention(text="पैरासिटामोल", label="DRUG", source_text="paracetamol"),
            EntityMention(text="500 mg", label="DOSAGE", source_text="500 mg"),
            EntityMention(text="दिन में तीन बार", label="FREQUENCY", source_text="three times a day"),
            EntityMention(text="बुखार", label="SYMPTOM", source_text="fever"),
        ],
    }
    fields.update(overrides)
    return LocalizedDraft(**fields)


HINGLISH_TEXT = (
    "Doctor ne patient ko paracetamol 500 mg din mein teen baar lene ki salah di. "
    "Agar fever kam na ho to dobara test karwayein."
)


def hinglish_draft() -> LocalizedDraft:
    return LocalizedDraft(
        text=HINGLISH_TEXT,
        entities=[
            EntityMention(text="paracetamol", label="DRUG", source_text="paracetamol"),
            EntityMention(text="500 mg", label="DOSAGE", source_text="500 mg"),
            EntityMention(text="din mein teen baar", label="FREQUENCY", source_text=""),
            EntityMention(text="fever", label="SYMPTOM", source_text="fever"),
        ],
    )


def verdict(score: float, feedback: str = "No issues.") -> CriticVerdict:
    return CriticVerdict(
        semantic_fidelity=score,
        factual_consistency=score,
        entity_retention=score,
        fluency=score,
        issues=[] if score >= 0.85 else ["defect"],
        feedback=feedback,
    )
