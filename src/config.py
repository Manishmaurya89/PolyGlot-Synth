"""Settings, read from environment variables and .env.

Everything is validated at startup, so a typo in .env fails right away instead
of halfway through a run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator


class ConfigError(ValueError):
    pass


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class AgentRole(StrEnum):
    """Each role can use its own provider and model."""

    ARCHITECT = "architect"
    LOCALIZER = "localizer"
    EVALUATOR = "evaluator"


class LLMSpec(BaseModel):
    """The resolved model choice for one role."""

    model_config = ConfigDict(frozen=True)

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class QualityGate(BaseModel):
    """When a sample counts as good enough to publish."""

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(default=0.85, ge=0.0, le=1.0)  # min quality_score to export
    max_retries: int = Field(default=3, ge=0)  # retries after the first attempt
    # If any rubric score is below this, the final score is capped at it, so a
    # hallucination can't be averaged away by good fluency.
    dimension_floor: float = Field(default=0.6, ge=0.0, le=1.0)
    min_entity_coverage: float = Field(default=0.6, ge=0.0, le=1.0)  # share of key entities kept
    weights: Mapping[str, float] = Field(
        default_factory=lambda: {
            "semantic_fidelity": 0.30,
            "factual_consistency": 0.30,
            "entity_retention": 0.20,
            "fluency": 0.20,
        }
    )


_DEFAULT_TEMPERATURES: dict[AgentRole, float] = {
    AgentRole.ARCHITECT: 0.7,  # diversity of sub-topics
    AgentRole.LOCALIZER: 0.6,  # natural phrasing without drifting
    AgentRole.EVALUATOR: 0.0,  # consistent judgments
}


class Settings(BaseModel):
    """Use Settings.from_env() rather than building this directly."""

    model_config = ConfigDict(frozen=True)

    provider: Provider = Provider.OLLAMA
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_num_ctx: int = Field(default=8192, gt=0)
    # Optional "provider:model" per role, e.g. generate with Ollama, judge with Claude.
    role_overrides: dict[AgentRole, str] = Field(default_factory=dict)
    role_temperatures: dict[AgentRole, float | None] = Field(
        default_factory=lambda: dict(_DEFAULT_TEMPERATURES)
    )
    max_tokens: int = Field(default=4096, gt=0)
    request_timeout_s: float = Field(default=180.0, gt=0)
    llm_max_attempts: int = Field(default=3, ge=1)

    quality: QualityGate = Field(default_factory=QualityGate)
    max_concurrency: int = Field(default=4, ge=1)
    output_dir: Path = Path("data")

    hf_token: SecretStr | None = None
    hf_repo_id: str | None = None
    hf_private: bool = True

    def spec_for(self, role: AgentRole) -> LLMSpec:
        override = self.role_overrides.get(role)
        if override:
            provider_name, _, model = override.partition(":")
            provider = Provider(provider_name.strip().lower())
            model = model.strip()
        else:
            provider = self.provider
            model = self.anthropic_model if provider is Provider.ANTHROPIC else self.ollama_model
        return LLMSpec(
            provider=provider,
            model=model,
            temperature=self.role_temperatures.get(role),
            max_tokens=self.max_tokens,
        )

    @model_validator(mode="after")
    def _validate_roles(self) -> Self:
        for role, override in self.role_overrides.items():
            provider_name, sep, model = override.partition(":")
            if not sep or not model.strip():
                raise ValueError(
                    f"{role.upper()}_LLM must look like 'provider:model', got {override!r}"
                )
            if provider_name.strip().lower() not in {p.value for p in Provider}:
                raise ValueError(f"{role.upper()}_LLM: unknown provider {provider_name!r}")
        uses_anthropic = any(self.spec_for(r).provider is Provider.ANTHROPIC for r in AgentRole)
        if uses_anthropic and not (self.anthropic_api_key and self.anthropic_api_key.get_secret_value()):
            raise ValueError("ANTHROPIC_API_KEY is required when any role uses the anthropic provider")
        return self

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = None, **overrides: object) -> Settings:
        """Read settings from the environment (plus ``env_file`` if it exists).

        Keyword overrides, e.g. from CLI flags, win over the environment.
        Raises ConfigError if anything is missing or invalid.
        """
        load_dotenv(env_file or ".env", override=False)

        def env(name: str) -> str | None:
            value = os.getenv(name)
            return value.strip() if value and value.strip() else None

        def temperature(role: AgentRole) -> float | None:
            raw = os.getenv(f"{role.upper()}_TEMPERATURE")
            if raw is None:
                return _DEFAULT_TEMPERATURES[role]
            # An explicitly empty value means "let the provider decide".
            return float(raw) if raw.strip() else None

        try:
            fields: dict[str, object] = {
                "provider": env("LLM_PROVIDER") or Provider.OLLAMA,
                "anthropic_api_key": env("ANTHROPIC_API_KEY"),
                "anthropic_model": env("ANTHROPIC_MODEL") or cls.model_fields["anthropic_model"].default,
                "ollama_base_url": env("OLLAMA_BASE_URL") or cls.model_fields["ollama_base_url"].default,
                "ollama_model": env("OLLAMA_MODEL") or cls.model_fields["ollama_model"].default,
                "ollama_num_ctx": env("OLLAMA_NUM_CTX") or cls.model_fields["ollama_num_ctx"].default,
                "role_overrides": {
                    role: value for role in AgentRole if (value := env(f"{role.upper()}_LLM"))
                },
                "role_temperatures": {role: temperature(role) for role in AgentRole},
                "max_tokens": env("LLM_MAX_TOKENS") or cls.model_fields["max_tokens"].default,
                "request_timeout_s": env("LLM_TIMEOUT_S") or cls.model_fields["request_timeout_s"].default,
                "llm_max_attempts": env("LLM_MAX_ATTEMPTS") or cls.model_fields["llm_max_attempts"].default,
                "quality": QualityGate(
                    **{
                        key: value
                        for key, value in {
                            "threshold": env("QUALITY_THRESHOLD"),
                            "max_retries": env("MAX_RETRIES"),
                            "dimension_floor": env("DIMENSION_FLOOR"),
                            "min_entity_coverage": env("MIN_ENTITY_COVERAGE"),
                        }.items()
                        if value is not None
                    }
                ),
                "max_concurrency": env("MAX_CONCURRENCY") or cls.model_fields["max_concurrency"].default,
                "output_dir": env("OUTPUT_DIR") or cls.model_fields["output_dir"].default,
                "hf_token": env("HF_TOKEN"),
                "hf_repo_id": env("HF_REPO_ID"),
                "hf_private": (env("HF_PRIVATE") or "true").lower() in {"1", "true", "yes"},
            }
            fields.update({k: v for k, v in overrides.items() if v is not None})
            return cls.model_validate(fields)
        except (ValidationError, ValueError) as exc:
            raise ConfigError(f"Invalid configuration:\n{exc}") from exc
