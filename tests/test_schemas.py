from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import AgentRole, ConfigError, Provider, Settings
from src.schemas.dataset import DatasetBatch, EntityLabel, SyntheticSample

TEXT = "पैरासिटामोल 500 mg"


def sample(**overrides: object) -> SyntheticSample:
    fields: dict[str, object] = {
        "target_language": "hi",
        "script": "Deva",
        "raw_text": TEXT,
        "entities": [
            EntityLabel(text="पैरासिटामोल", label="DRUG", start_char=0, end_char=11),
            EntityLabel(text="500 mg", label="DOSAGE", start_char=12, end_char=18),
        ],
        "domain": "medical",
        "quality_score": 0.9,
    }
    fields.update(overrides)
    return SyntheticSample(**fields)


def test_valid_sample_round_trips_through_json() -> None:
    original = sample()
    restored = SyntheticSample.model_validate_json(original.model_dump_json())
    assert restored == original
    assert "पैरासिटामोल" in original.model_dump_json()  # no \\u escaping


def test_entity_span_length_must_match_text() -> None:
    with pytest.raises(ValidationError, match="span length"):
        EntityLabel(text="500 mg", label="DOSAGE", start_char=12, end_char=20)


def test_entity_offsets_must_match_raw_text() -> None:
    with pytest.raises(ValidationError, match="does not match raw_text"):
        sample(entities=[EntityLabel(text="500 mg", label="DOSAGE", start_char=0, end_char=6)])


def test_label_must_belong_to_domain_taxonomy() -> None:
    with pytest.raises(ValidationError, match="not in the medical taxonomy"):
        sample(entities=[EntityLabel(text="500 mg", label="STATUTE", start_char=12, end_char=18)])


def test_overlapping_entities_rejected() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        sample(
            raw_text="500 mg",
            entities=[
                EntityLabel(text="500 mg", label="DOSAGE", start_char=0, end_char=6),
                EntityLabel(text="mg", label="DOSAGE", start_char=4, end_char=6),
            ],
        )


def test_raw_text_must_be_nfc() -> None:
    with pytest.raises(ValidationError, match="NFC"):
        sample(raw_text="ज़", entities=[])  # precomposed ZA is not NFC


def test_quality_score_bounds() -> None:
    with pytest.raises(ValidationError):
        sample(quality_score=1.2)


def test_batch_total_must_match_samples() -> None:
    with pytest.raises(ValidationError, match="total_samples"):
        DatasetBatch(total_samples=2, samples=[sample()])


def test_batch_rejects_duplicate_ids() -> None:
    s = sample()
    with pytest.raises(ValidationError, match="unique"):
        DatasetBatch.from_samples([s, s])


def test_batch_from_samples_derives_count() -> None:
    batch = DatasetBatch.from_samples([sample(), sample()])
    assert batch.total_samples == 2
    assert batch.timestamp.tzinfo is not None


def test_settings_require_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        Settings.from_env("/nonexistent.env", provider="anthropic")


def test_role_override_mixes_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("EVALUATOR_LLM", "anthropic:claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    settings = Settings.from_env("/nonexistent.env")
    assert settings.spec_for(AgentRole.LOCALIZER).label == "ollama:llama3"
    evaluator = settings.spec_for(AgentRole.EVALUATOR)
    assert (evaluator.provider, evaluator.model) == (Provider.ANTHROPIC, "claude-sonnet-5-5")
    assert evaluator.temperature == 0.0


def test_malformed_role_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALIZER_LLM", "mistral")
    with pytest.raises(ConfigError, match="provider:model"):
        Settings.from_env("/nonexistent.env")
