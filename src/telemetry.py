"""Thread-safe token and latency accounting for every LLM call."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """One physical request to a model (retries are recorded individually)."""

    role: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    success: bool
    tag: str = ""


class UsageTracker:
    """Collects :class:`LLMCallRecord` objects from concurrent graph branches."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[LLMCallRecord] = []

    def record(self, record: LLMCallRecord) -> None:
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> list[LLMCallRecord]:
        with self._lock:
            return list(self._records)

    def totals(self) -> dict[str, float]:
        """Aggregate counters across all roles."""
        records = self.records
        return {
            "calls": len(records),
            "failed_calls": sum(not r.success for r in records),
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "llm_time_s": round(sum(r.latency_s for r in records), 2),
        }

    def summary_frame(self) -> pd.DataFrame:
        """Per role/model usage table (empty frame if nothing was recorded)."""
        records = self.records
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(r) for r in records])
        grouped = df.groupby(["role", "model"], sort=False)
        summary = grouped.agg(
            calls=("success", "size"),
            failed=("success", lambda s: int((~s).sum())),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            mean_latency_s=("latency_s", "mean"),
            p95_latency_s=("latency_s", lambda s: s.quantile(0.95)),
        )
        return summary.round(2).reset_index()
