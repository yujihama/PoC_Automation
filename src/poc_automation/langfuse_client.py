"""Langfuse integration wrapper.

The wrapper is intentionally tolerant: if the Langfuse SDK is unavailable or
LANGFUSE_ENABLED is false, it becomes a no-op while preserving local registry
and artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .config import LangfuseConfig
from .models import EvaluationResult, NormalizedResult, TuningCandidate, to_jsonable


@dataclass(frozen=True)
class TraceHandle:
    trace_id: str | None
    enabled: bool


class LangfuseReporter:
    def __init__(self, config: LangfuseConfig | None = None):
        self.config = config or LangfuseConfig.from_env()
        self.enabled = self.config.enabled
        self.client: Any | None = None
        self._new_sdk = False
        if not self.enabled:
            return
        if self.config.host:
            os.environ.setdefault("LANGFUSE_HOST", self.config.host)
        if self.config.public_key:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self.config.public_key)
        if self.config.secret_key:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", self.config.secret_key)
        try:
            from langfuse import get_client  # type: ignore

            self.client = get_client()
            self._new_sdk = True
        except Exception:
            try:
                from langfuse import Langfuse  # type: ignore

                self.client = Langfuse(
                    public_key=self.config.public_key,
                    secret_key=self.config.secret_key,
                    host=self.config.host,
                )
                self._new_sdk = False
            except Exception:
                self.enabled = False
                self.client = None

    def start_trace(
        self,
        *,
        name: str,
        case_id: str,
        tuning_id: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceHandle:
        if not self.enabled or self.client is None:
            return TraceHandle(trace_id=None, enabled=False)
        trace_id = f"{name}:{case_id}:{tuning_id}"
        try:
            if hasattr(self.client, "trace"):
                self.client.trace(
                    id=trace_id,
                    name=name,
                    metadata={"case_id": case_id, "tuning_id": tuning_id, **(metadata or {})},
                )
            # Newer OpenTelemetry-first SDK creates traces through spans; scores
            # can still be emitted by trace_id when available.
        except Exception:
            return TraceHandle(trace_id=None, enabled=False)
        return TraceHandle(trace_id=trace_id, enabled=True)

    def record_output(
        self,
        *,
        trace: TraceHandle,
        output: NormalizedResult,
        candidate: TuningCandidate,
    ) -> None:
        if not trace.enabled or not self.client or not trace.trace_id:
            return
        try:
            if hasattr(self.client, "trace"):
                self.client.trace(
                    id=trace.trace_id,
                    output=to_jsonable(output),
                    metadata={"tuning_candidate": to_jsonable(candidate)},
                )
        except Exception:
            pass

    def record_scores(self, *, trace: TraceHandle, results: list[EvaluationResult]) -> None:
        if not trace.enabled or not self.client or not trace.trace_id:
            return
        for result in results:
            if result.score is None:
                continue
            try:
                if hasattr(self.client, "score"):
                    self.client.score(
                        trace_id=trace.trace_id,
                        name=result.evaluator_name,
                        value=result.score,
                        comment=result.comment,
                    )
            except Exception:
                continue

    def flush(self) -> None:
        if not self.enabled or not self.client:
            return
        for method in ("flush", "shutdown"):
            fn = getattr(self.client, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
