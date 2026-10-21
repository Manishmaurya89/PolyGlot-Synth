from __future__ import annotations

import pytest

_CONFIG_VARS = (
    "LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
    "ARCHITECT_LLM", "LOCALIZER_LLM", "EVALUATOR_LLM", "QUALITY_THRESHOLD", "MAX_RETRIES",
    "ARCHITECT_TEMPERATURE", "LOCALIZER_TEMPERATURE", "EVALUATOR_TEMPERATURE",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's shell environment out of configuration tests."""
    for name in _CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
