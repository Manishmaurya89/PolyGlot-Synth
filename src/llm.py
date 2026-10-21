"""Provider-agnostic LLM layer.

The agents never touch a vendor SDK. They depend on the
:class:`StructuredGenerator` protocol: "give me an instance of this Pydantic
schema". :class:`StructuredLLM` implements it on top of any LangChain chat
model, adding:

* schema-validated outputs via ``with_structured_output(method="json_schema")``
  (Claude structured outputs / Ollama grammar-constrained decoding);
* bounded retries with exponential backoff and jitter for transient failures;
* self-correction: a schema-validation failure is fed back to the model on the
  next attempt;
* per-call token/latency telemetry.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from src.config import AgentRole, LLMSpec, Provider, Settings
from src.telemetry import LLMCallRecord, UsageTracker

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_TRANSIENT_NAME_HINTS = ("timeout", "connect", "ratelimit", "overloaded", "unavailable")


class LLMCallError(RuntimeError):
    """The provider could not produce a response (after retries if transient)."""


class StructuredOutputError(LLMCallError):
    """The model kept returning output that does not match the requested schema."""


class StructuredGenerator(Protocol):
    """What the agents need from an LLM. Implemented by :class:`StructuredLLM`
    in production and by scripted fakes in tests."""

    @property
    def model_label(self) -> str: ...

    def generate(self, schema: type[T], *, system: str, user: str, tag: str = "") -> T: ...


def build_chat_model(spec: LLMSpec, settings: Settings) -> BaseChatModel:
    """Instantiate the LangChain chat model for ``spec``.

    Provider packages are imported lazily so an Ollama-only install does not
    need Anthropic credentials (and vice versa).
    """
    if spec.provider is Provider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        if settings.anthropic_api_key is None:  # guarded by Settings validation
            raise LLMCallError("ANTHROPIC_API_KEY is not set")
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": spec.max_tokens,
            "timeout": settings.request_timeout_s,
            # The SDK retries 429/5xx itself and honours retry-after headers.
            "max_retries": 2,
            "api_key": settings.anthropic_api_key.get_secret_value(),
        }
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        return ChatAnthropic(**kwargs)

    if spec.provider is Provider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=spec.model,
            base_url=settings.ollama_base_url,
            temperature=spec.temperature,
            num_predict=spec.max_tokens,
            num_ctx=settings.ollama_num_ctx,
            client_kwargs={"timeout": settings.request_timeout_s},
        )

    raise LLMCallError(f"Unsupported provider: {spec.provider}")


def is_transient(exc: BaseException) -> bool:
    """Heuristically decide whether retrying ``exc`` can succeed.

    Uses HTTP status codes where the SDK exposes them and falls back to the
    exception class name, so it works across provider SDKs without importing
    them.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS
    name = type(exc).__name__.lower()
    return any(hint in name for hint in _TRANSIENT_NAME_HINTS)


class StructuredLLM:
    """A role-bound chat model that returns validated Pydantic objects."""

    def __init__(
        self,
        role: AgentRole,
        spec: LLMSpec,
        chat_model: BaseChatModel,
        tracker: UsageTracker,
        *,
        max_attempts: int = 3,
        backoff_base_s: float = 1.5,
    ) -> None:
        self.role = role
        self.spec = spec
        self._chat_model = chat_model
        self._tracker = tracker
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._runnables: dict[type[BaseModel], Runnable[Any, Any]] = {}

    @property
    def model_label(self) -> str:
        return self.spec.label

    def _runnable_for(self, schema: type[BaseModel]) -> Runnable[Any, Any]:
        if schema not in self._runnables:
            # Native constrained decoding on both backends: Claude structured
            # outputs (newer Claude models reject forced tool calling) and
            # Ollama's grammar-based `format`.
            self._runnables[schema] = self._chat_model.with_structured_output(
                schema, method="json_schema", include_raw=True
            )
        return self._runnables[schema]

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self._backoff_base_s * (2 ** (attempt - 1))
        time.sleep(delay + random.uniform(0, delay / 2))

    def _record(self, raw: Any, latency: float, success: bool, tag: str) -> None:
        usage = getattr(raw, "usage_metadata", None) or {}
        self._tracker.record(
            LLMCallRecord(
                role=self.role.value,
                model=self.model_label,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                latency_s=latency,
                success=success,
                tag=tag,
            )
        )
        logger.info(
            "[%s] %s | %s | %s in / %s out tokens | %.2fs%s",
            self.role.value,
            self.model_label,
            tag or "-",
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
            latency,
            "" if success else " | FAILED",
        )

    def generate(self, schema: type[T], *, system: str, user: str, tag: str = "") -> T:
        """Return a validated ``schema`` instance.

        Raises:
            LLMCallError: non-transient provider error, or transient errors
                persisting past ``max_attempts``.
            StructuredOutputError: the model never produced schema-valid output.
        """
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(user)]
        runnable = self._runnable_for(schema)
        last_problem = "no attempts made"

        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            try:
                result = runnable.invoke(messages)
            except Exception as exc:
                self._record(None, time.perf_counter() - started, False, tag)
                if not is_transient(exc):
                    raise LLMCallError(f"{self.model_label} request failed: {exc}") from exc
                last_problem = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[%s] transient error (attempt %d/%d): %s",
                    self.role.value, attempt, self._max_attempts, last_problem,
                )
                if attempt == self._max_attempts:
                    raise LLMCallError(
                        f"{self.model_label} unavailable after {attempt} attempts: {last_problem}"
                    ) from exc
                self._sleep_before_retry(attempt)
                continue

            raw: AIMessage | None = result.get("raw")
            parsed = result.get("parsed")
            error = result.get("parsing_error")
            if parsed is not None and error is None:
                try:
                    value = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
                except ValidationError as exc:
                    error = exc
                else:
                    self._record(raw, time.perf_counter() - started, True, tag)
                    return value

            self._record(raw, time.perf_counter() - started, False, tag)
            last_problem = str(error) if error else "model returned no structured output"
            logger.warning(
                "[%s] schema violation (attempt %d/%d): %s",
                self.role.value, attempt, self._max_attempts, last_problem[:300],
            )
            # Self-correction: show the model exactly why its output was rejected.
            messages = [
                SystemMessage(system),
                HumanMessage(
                    f"{user}\n\nYOUR PREVIOUS RESPONSE WAS REJECTED by schema validation:\n"
                    f"{last_problem[:1500]}\nReturn output that strictly matches the schema."
                ),
            ]

        raise StructuredOutputError(
            f"{self.model_label} did not return a valid {schema.__name__} after "
            f"{self._max_attempts} attempts: {last_problem[:500]}"
        )


def build_role_llms(settings: Settings, tracker: UsageTracker) -> dict[AgentRole, StructuredLLM]:
    """Create one :class:`StructuredLLM` per agent role from ``settings``."""
    llms: dict[AgentRole, StructuredLLM] = {}
    for role in AgentRole:
        spec = settings.spec_for(role)
        llms[role] = StructuredLLM(
            role,
            spec,
            build_chat_model(spec, settings),
            tracker,
            max_attempts=settings.llm_max_attempts,
        )
        logger.info("Role %-9s -> %s (temperature=%s)", role.value, spec.label, spec.temperature)
    return llms
