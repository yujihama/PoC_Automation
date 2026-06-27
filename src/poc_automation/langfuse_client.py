"""Langfuse integration wrapper.

Langfuse is the result-inspection UI for this prototype.  The local SQLite
registry remains the canonical ledger; this module is deliberately tolerant so
network or SDK failures never fail the search loop.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .config import LangfuseConfig
from .dataset import Dataset
from .models import (
    AppRunResult,
    Case,
    EvaluationResult,
    NormalizedResult,
    TuningCandidate,
    to_jsonable,
)

ScoreDataTypeName = Literal["NUMERIC", "CATEGORICAL", "BOOLEAN", "TEXT"]


@dataclass(frozen=True)
class TraceHandle:
    trace_id: str | None
    enabled: bool
    name: str = ""
    observation: Any | None = None


@dataclass(frozen=True)
class LangfuseDatasetRef:
    name: str
    snapshot_id: str
    item_count: int


@dataclass(frozen=True)
class LangfuseSyncResult:
    enabled: bool
    dataset_name: str | None = None
    created_items: int = 0
    skipped_items: int = 0
    failed_items: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreConfigSpec:
    name: str
    data_type: ScoreDataTypeName
    description: str
    min_value: float | None = None
    max_value: float | None = None
    categories: tuple[str, ...] = ()


SCORE_NAME_MAP = {
    "judgement_match": "judgement_score",
    "rationale_support": "rationale_score",
    "citation_quality": "citation_score",
    "format_valid": "format_score",
}

SCORE_CONFIG_SPECS: tuple[ScoreConfigSpec, ...] = (
    ScoreConfigSpec("total_score", "NUMERIC", "Overall evaluation score", 0.0, 1.0),
    ScoreConfigSpec("delta_vs_baseline", "NUMERIC", "Case-level delta against baseline", -1.0, 1.0),
    ScoreConfigSpec("judgement_score", "NUMERIC", "Judgement agreement score", 0.0, 1.0),
    ScoreConfigSpec("rationale_score", "NUMERIC", "Rationale support score", 0.0, 1.0),
    ScoreConfigSpec("citation_score", "NUMERIC", "Citation quality score", 0.0, 1.0),
    ScoreConfigSpec(
        "unsupported_claim_rate",
        "NUMERIC",
        "Rate of claims not supported by evidence",
        0.0,
        1.0,
    ),
    ScoreConfigSpec("format_score", "NUMERIC", "Output format validity score", 0.0, 1.0),
    ScoreConfigSpec("latency_ms", "NUMERIC", "Execution latency in milliseconds", 0.0, None),
    ScoreConfigSpec("dataset_case_count", "NUMERIC", "Number of cases in the synced dataset", 0.0, None),
    ScoreConfigSpec("input_tokens", "NUMERIC", "Input token count", 0.0, None),
    ScoreConfigSpec("output_tokens", "NUMERIC", "Output token count", 0.0, None),
    ScoreConfigSpec("total_tokens", "NUMERIC", "Total token count", 0.0, None),
    ScoreConfigSpec(
        "effect_label",
        "CATEGORICAL",
        "Candidate effect label",
        categories=("strongly_positive", "positive", "neutral", "negative", "risky"),
    ),
    ScoreConfigSpec(
        "candidate_status",
        "CATEGORICAL",
        "Candidate lifecycle status",
        categories=("draft", "candidate", "evaluated", "rejected", "promoted", "archived"),
    ),
    ScoreConfigSpec("duplicate_candidate", "BOOLEAN", "Whether the candidate was skipped as duplicate"),
    ScoreConfigSpec("has_regression", "BOOLEAN", "Whether the case regressed versus baseline"),
    ScoreConfigSpec("replicate_stable", "BOOLEAN", "Whether replicate checks were stable"),
    ScoreConfigSpec("replicate_mean_total", "NUMERIC", "Replicate mean total score", 0.0, 1.0),
    ScoreConfigSpec("replicate_worst_total", "NUMERIC", "Replicate worst total score", 0.0, 1.0),
    ScoreConfigSpec("replicate_std_total", "NUMERIC", "Replicate total-score standard deviation", 0.0, None),
    ScoreConfigSpec("replicate_worst_delta", "NUMERIC", "Worst replicate delta", -1.0, 1.0),
    ScoreConfigSpec("trial_mean_total", "NUMERIC", "Trial mean total score", 0.0, 1.0),
    ScoreConfigSpec("trial_delta_vs_baseline", "NUMERIC", "Trial mean delta against baseline", -1.0, 1.0),
    ScoreConfigSpec("trial_positive_case_count", "NUMERIC", "Positive cases in trial", 0.0, None),
    ScoreConfigSpec("trial_negative_case_count", "NUMERIC", "Negative cases in trial", 0.0, None),
    ScoreConfigSpec("trial_regression_count", "NUMERIC", "Regression cases in trial", 0.0, None),
    ScoreConfigSpec("trial_accepted", "BOOLEAN", "Whether the trial was accepted by the agent"),
    ScoreConfigSpec("trial_formal_gap", "NUMERIC", "Trial-to-formal score gap", -1.0, 1.0),
    ScoreConfigSpec(
        "promotion_decision",
        "CATEGORICAL",
        "Recorded promotion decision",
        categories=("none", "needs_more_validation", "promote_candidate", "rejected"),
    ),
    ScoreConfigSpec("needs_more_validation", "BOOLEAN", "Whether additional validation is required"),
)

SUMMARY_SCORE_NAMES = {
    "generated_candidates",
    "evaluated_candidates",
    "skipped_duplicate_candidates",
    "needs_more_validation_candidates",
    "positive_candidates",
    "promoted_candidates",
    "baseline_case_count",
}


class LangfuseReporter:
    def __init__(self, config: LangfuseConfig | None = None):
        self.config = config or LangfuseConfig.from_env()
        self.enabled = self.config.enabled
        self.client: Any | None = None
        self._new_sdk = False
        self._pending_dataset_run_items: list[dict[str, object]] = []
        self._score_config_ids: dict[str, str] | None = None
        if not self.enabled:
            return
        if self.config.host:
            os.environ.setdefault("LANGFUSE_HOST", self.config.host)
            os.environ.setdefault("LANGFUSE_BASE_URL", self.config.host)
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
                    base_url=self.config.host,
                    host=self.config.host,
                )
                self._new_sdk = False
            except TypeError:
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
            except Exception:
                self.enabled = False
                self.client = None

    def start_search_session(
        self,
        *,
        search_run_id: str,
        dataset: Dataset,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled or self.client is None:
            return
        session_metadata = {
            "search_run_id": search_run_id,
            "dataset_name": dataset.dataset_id,
            "dataset_snapshot_id": dataset.snapshot_id,
            "case_count": len(dataset.cases),
            **(metadata or {}),
        }
        self._create_session_score(
            session_id=search_run_id,
            name="dataset_case_count",
            value=float(len(dataset.cases)),
            metadata=session_metadata,
        )

    def sync_dataset(self, dataset: Dataset) -> LangfuseSyncResult:
        if self.config.dataset_mode != "hosted":
            return LangfuseSyncResult(enabled=False)
        return LangfuseDatasetSync(self).sync_dataset(dataset)

    def initialize_score_configs(self) -> dict[str, object]:
        return LangfuseScoreConfigInitializer(self).initialize()

    def start_trace(
        self,
        *,
        name: str,
        case_id: str,
        tuning_id: str,
        session_id: str | None = None,
        input: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        tags: Iterable[str] | None = None,
        as_type: str = "span",
    ) -> TraceHandle:
        if not self.enabled or self.client is None:
            return TraceHandle(trace_id=None, enabled=False, name=name)

        trace_metadata = {"case_id": case_id, "tuning_id": tuning_id, **(metadata or {})}
        seed = "|".join(
            [
                name,
                session_id or "",
                str(trace_metadata.get("experiment_id", "")),
                str(trace_metadata.get("run_type", "")),
                case_id,
                tuning_id,
                str(trace_metadata.get("replicate_index", "")),
            ]
        )
        trace_id = self._trace_id(seed)
        all_tags = self._merged_tags(tags)

        try:
            if hasattr(self.client, "start_observation"):
                observation = self._start_observation(
                    trace_id=trace_id,
                    session_id=session_id,
                    name=name,
                    as_type=as_type,
                    input=input,
                    metadata=trace_metadata,
                    tags=all_tags,
                )
                return TraceHandle(trace_id=trace_id, enabled=True, name=name, observation=observation)
            if hasattr(self.client, "trace"):
                self.client.trace(
                    id=trace_id,
                    name=name,
                    input=input,
                    metadata=trace_metadata,
                    session_id=session_id,
                    tags=all_tags,
                )
                return TraceHandle(trace_id=trace_id, enabled=True, name=name)
        except Exception:
            return TraceHandle(trace_id=None, enabled=False, name=name)

        return TraceHandle(trace_id=None, enabled=False, name=name)

    def emit_agent_iteration(
        self,
        *,
        search_run_id: str,
        search_iteration: int,
        draft_candidates: list[TuningCandidate],
        accepted_candidates: list[TuningCandidate],
        duplicate_skipped_count: int,
        failures: list[dict[str, object]],
        metadata: dict[str, object] | None = None,
    ) -> None:
        trace = self.start_trace(
            name="poc.agent_iteration",
            case_id=f"iteration_{search_iteration:04d}",
            tuning_id="agent_iteration",
            session_id=search_run_id,
            as_type="agent",
            input={
                "search_iteration": search_iteration,
                "failure_summaries": failures,
            },
            metadata={
                "search_run_id": search_run_id,
                "search_iteration": search_iteration,
                "agent_trial_round": search_iteration,
                "draft_count": len(draft_candidates),
                "accepted_count": len(accepted_candidates),
                "duplicate_skipped_count": duplicate_skipped_count,
                "run_type": "trial",
                **(metadata or {}),
            },
            tags=("search-run", "trial"),
        )
        if not trace.enabled:
            return
        payload = {
            "draft_candidates": [_candidate_summary(candidate) for candidate in draft_candidates],
            "accepted_candidates": [_candidate_summary(candidate) for candidate in accepted_candidates],
            "duplicate_skipped_count": duplicate_skipped_count,
        }
        self._finish_observation(trace, output=payload, metadata={"agent_iteration_saved": True})

    def emit_trial(
        self,
        *,
        search_run_id: str,
        trial_id: str,
        search_iteration: int,
        draft_index: int,
        instruction: str,
        hypothesis: str,
        case_ids: list[str],
        splits: list[str],
        summary: dict[str, object],
        case_results: list[dict[str, object]],
        status: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        trace = self.start_trace(
            name="poc.trial",
            case_id="trial",
            tuning_id=trial_id,
            session_id=search_run_id,
            input={
                "instruction": instruction,
                "hypothesis": hypothesis,
                "cases": case_ids,
            },
            metadata={
                "trial_id": trial_id,
                "search_run_id": search_run_id,
                "search_iteration": search_iteration,
                "agent_trial_round": search_iteration,
                "draft_index": draft_index,
                "candidate_instruction_hash": _stable_short_id(instruction),
                "cases": case_ids,
                "splits": splits,
                "trial_status": status,
                "run_type": "trial",
                **(metadata or {}),
            },
            tags=("trial",),
        )
        if not trace.enabled:
            return
        self._finish_observation(
            trace,
            output={"summary": summary, "case_results": case_results},
            metadata={"trial_saved": True},
        )
        self._record_scalar_scores(
            trace_id=trace.trace_id,
            scores={
                "trial_mean_total": summary.get("total_score_mean"),
                "trial_delta_vs_baseline": summary.get("delta_mean"),
                "trial_positive_case_count": summary.get("positive_count"),
                "trial_negative_case_count": summary.get("negative_count"),
                "trial_regression_count": summary.get("regression_count"),
                "trial_accepted": status == "succeeded",
            },
        )
        replicate_summary = summary.get("replicate_summary")
        if isinstance(replicate_summary, dict):
            self._record_scalar_scores(
                trace_id=trace.trace_id,
                scores={
                    "replicate_stable": replicate_summary.get("stable"),
                    "replicate_mean_total": replicate_summary.get("total_score_mean_avg"),
                    "replicate_worst_total": replicate_summary.get("total_score_mean_min"),
                    "replicate_worst_delta": replicate_summary.get("worst_case_delta"),
                },
            )

    def emit_replicate_run(
        self,
        *,
        search_run_id: str,
        trial_id: str,
        tuning_id: str,
        replicate_index: int,
        summary: dict[str, object],
        case_results: list[dict[str, object]],
        metadata: dict[str, object] | None = None,
    ) -> None:
        trace = self.start_trace(
            name="poc.replicate_run",
            case_id="replicate",
            tuning_id=tuning_id,
            session_id=search_run_id,
            input={"trial_id": trial_id, "replicate_index": replicate_index},
            metadata={
                "trial_id": trial_id,
                "search_run_id": search_run_id,
                "tuning_id": tuning_id,
                "replicate_index": replicate_index,
                "replicate_group_id": f"repl_{tuning_id}",
                "run_type": "replicate",
                **(metadata or {}),
            },
            tags=("replicate",),
        )
        if not trace.enabled:
            return
        self._finish_observation(
            trace,
            output={"summary": summary, "case_results": case_results},
            metadata={"replicate_saved": True},
        )
        self._record_scalar_scores(
            trace_id=trace.trace_id,
            scores={
                "replicate_mean_total": summary.get("total_score_mean"),
                "replicate_worst_delta": summary.get("delta_min"),
                "latency_ms": summary.get("latency_ms"),
            },
        )

    def record_output(
        self,
        *,
        trace: TraceHandle,
        output: NormalizedResult,
        candidate: TuningCandidate,
        app_result: AppRunResult | None = None,
    ) -> None:
        if not trace.enabled or not self.client or not trace.trace_id:
            return
        payload = to_jsonable(output)
        metadata = {"tuning_candidate": _candidate_summary(candidate)}
        try:
            if trace.observation is not None:
                self._record_target_generation(trace=trace, app_result=app_result)
                self._finish_observation(trace, output=payload, metadata=metadata)
            elif hasattr(self.client, "trace"):
                self.client.trace(id=trace.trace_id, output=payload, metadata=metadata)
        except Exception:
            pass

    def record_scores(
        self,
        *,
        trace: TraceHandle,
        results: list[EvaluationResult],
        extra_scores: dict[str, object] | None = None,
    ) -> None:
        if not trace.enabled or not self.client or not trace.trace_id:
            return
        for result in results:
            if result.score is None:
                continue
            score_name = SCORE_NAME_MAP.get(result.evaluator_name, result.evaluator_name)
            self._create_score(
                trace_id=trace.trace_id,
                name=score_name,
                value=float(result.score),
                data_type="NUMERIC",
                comment=result.comment,
                metadata={"source_evaluator_name": result.evaluator_name, **result.details},
            )
        if extra_scores:
            self._record_scalar_scores(trace_id=trace.trace_id, scores=extra_scores)

    def record_dataset_run_item(
        self,
        *,
        search_run_id: str,
        dataset: Dataset,
        case_id: str,
        run_type: str,
        split: str,
        tuning_id: str,
        trace_id: str | None,
        metadata: dict[str, object] | None = None,
    ) -> str | None:
        if self.config.dataset_mode != "hosted" or not self.enabled or not self.client or not trace_id:
            return None
        run_name = f"{search_run_id}__{run_type}__{split}__{tuning_id}"
        self._pending_dataset_run_items.append(
            {
                "run_name": run_name,
                "dataset_item_id": LangfuseDatasetSync.dataset_item_id(dataset.snapshot_id, case_id),
                "trace_id": trace_id,
                "metadata": {
                    "search_run_id": search_run_id,
                    "dataset_name": dataset.dataset_id,
                    "dataset_snapshot_id": dataset.snapshot_id,
                    "case_id": case_id,
                    "split": split,
                    "tuning_id": tuning_id,
                    "run_type": run_type,
                    **(metadata or {}),
                },
            }
        )
        return run_name

    def record_search_summary(
        self,
        *,
        search_run_id: str,
        summary: Any,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled or not self.client:
            return
        payload = to_jsonable(summary)
        if not isinstance(payload, dict):
            return
        for name in SUMMARY_SCORE_NAMES:
            value = payload.get(name)
            if isinstance(value, (int, float)):
                self._create_session_score(
                    session_id=search_run_id,
                    name=name,
                    value=float(value),
                    metadata={"search_run_id": search_run_id, **(metadata or {})},
                )

    def flush(self) -> None:
        if not self.enabled or not self.client:
            return
        flush = getattr(self.client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
        if self._pending_dataset_run_items:
            time.sleep(1.0)
        self._flush_dataset_run_items()
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
        shutdown = getattr(self.client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

    def _flush_dataset_run_items(self) -> None:
        if not self._pending_dataset_run_items or self.client is None:
            return
        api = getattr(self.client, "api", None)
        run_items = getattr(api, "dataset_run_items", None) if api is not None else None
        create = getattr(run_items, "create", None)
        if not callable(create):
            self._pending_dataset_run_items.clear()
            return
        pending = self._pending_dataset_run_items
        self._pending_dataset_run_items = []
        for item in pending:
            try:
                create(**item)
            except Exception:
                continue

    def _start_observation(
        self,
        *,
        trace_id: str,
        session_id: str | None,
        name: str,
        as_type: str,
        input: dict[str, object] | None,
        metadata: dict[str, object],
        tags: list[str],
    ) -> Any:
        context = nullcontext()
        try:
            from langfuse import propagate_attributes  # type: ignore

            context = propagate_attributes(
                session_id=session_id,
                metadata=metadata,
                tags=tags,
            )
        except Exception:
            pass
        with context:
            return self.client.start_observation(
                trace_context={"trace_id": trace_id},
                name=name,
                as_type=as_type,
                input=input,
                metadata=metadata,
            )

    def _finish_observation(
        self,
        trace: TraceHandle,
        *,
        output: object,
        metadata: dict[str, object] | None = None,
    ) -> None:
        observation = trace.observation
        if observation is None:
            return
        update = getattr(observation, "update", None)
        if callable(update):
            update(name=trace.name or None, output=output, metadata=metadata)
        end = getattr(observation, "end", None)
        if callable(end):
            end()

    def _record_target_generation(
        self,
        *,
        trace: TraceHandle,
        app_result: AppRunResult | None,
    ) -> None:
        if app_result is None or self.client is None or trace.trace_id is None or trace.observation is None:
            return
        start = getattr(self.client, "start_observation", None)
        if not callable(start):
            return
        usage = _usage_details(app_result.cost)
        metadata = {
            "app_run_id": app_result.app_run_id,
            "status": app_result.status,
            "latency_ms": app_result.latency_ms,
            "error_message": app_result.error_message,
        }
        try:
            generation = start(
                trace_context={
                    "trace_id": trace.trace_id,
                    "parent_span_id": getattr(trace.observation, "id", None),
                },
                name="target_agent_call",
                as_type="generation",
                input=None,
                output=to_jsonable(app_result.normalized_result),
                metadata=metadata,
                usage_details=usage or None,
            )
            end = getattr(generation, "end", None)
            if callable(end):
                end()
        except Exception:
            return

    def _record_scalar_scores(self, *, trace_id: str | None, scores: dict[str, object]) -> None:
        if not trace_id:
            return
        for name, value in scores.items():
            if value is None:
                continue
            data_type = _score_data_type(value)
            score_value = _score_value(value, data_type)
            if score_value is None:
                continue
            self._create_score(
                trace_id=trace_id,
                name=name,
                value=score_value,
                data_type=data_type,
            )

    def _create_score(
        self,
        *,
        name: str,
        value: float | str,
        data_type: ScoreDataTypeName | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
        dataset_run_id: str | None = None,
        comment: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled or self.client is None:
            return
        score_id = _stable_short_id("|".join([trace_id or session_id or dataset_run_id or "", name]))
        config_id = self._score_config_id(name)
        try:
            if hasattr(self.client, "create_score"):
                kwargs = {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "dataset_run_id": dataset_run_id,
                    "name": name,
                    "value": value,
                    "score_id": score_id,
                    "data_type": data_type,
                    "comment": comment,
                    "metadata": metadata,
                }
                if config_id:
                    kwargs["config_id"] = config_id
                try:
                    self.client.create_score(**kwargs)
                except TypeError:
                    kwargs.pop("config_id", None)
                    self.client.create_score(**kwargs)
            elif hasattr(self.client, "score"):
                kwargs = {
                    "name": name,
                    "value": value,
                    "comment": comment,
                }
                if trace_id is not None:
                    kwargs["trace_id"] = trace_id
                if session_id is not None:
                    kwargs["session_id"] = session_id
                self.client.score(**kwargs)
        except Exception:
            return

    def _create_session_score(
        self,
        *,
        session_id: str,
        name: str,
        value: float | str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._create_score(
            session_id=session_id,
            name=name,
            value=value,
            data_type="NUMERIC" if isinstance(value, (int, float)) else "TEXT",
            metadata=metadata,
        )

    def _trace_id(self, seed: str) -> str:
        create_trace_id = getattr(self.client, "create_trace_id", None) if self.client else None
        if callable(create_trace_id):
            try:
                return str(create_trace_id(seed=seed))
            except Exception:
                pass
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

    def _merged_tags(self, tags: Iterable[str] | None) -> list[str]:
        merged: list[str] = []
        for tag in [*self.config.tags, *(tags or [])]:
            if tag and tag not in merged:
                merged.append(tag)
        return merged

    def _score_config_id(self, name: str) -> str | None:
        if self._score_config_ids is None:
            self._score_config_ids = self._load_score_config_ids()
        return self._score_config_ids.get(name)

    def _load_score_config_ids(self) -> dict[str, str]:
        api = getattr(self.client, "api", None) if self.client else None
        score_configs = getattr(api, "score_configs", None) if api is not None else None
        get = getattr(score_configs, "get", None)
        if not callable(get):
            return {}
        try:
            response = get(limit=100)
        except Exception:
            return {}
        data = getattr(response, "data", None) or getattr(response, "items", None) or []
        config_ids: dict[str, str] = {}
        for item in data:
            item_name = getattr(item, "name", None)
            item_id = getattr(item, "id", None) or getattr(item, "config_id", None)
            if item_name and item_id:
                config_ids[str(item_name)] = str(item_id)
        return config_ids


class LangfuseDatasetSync:
    def __init__(self, reporter: LangfuseReporter):
        self.reporter = reporter

    def sync_dataset(self, dataset: Dataset) -> LangfuseSyncResult:
        client = self.reporter.client
        if not self.reporter.enabled or client is None:
            return LangfuseSyncResult(enabled=False)

        dataset_name = self.get_dataset_name(dataset.dataset_id, dataset.snapshot_id)
        errors: list[str] = []
        try:
            client.create_dataset(
                name=dataset_name,
                description="PoC_Automation tuning evaluation dataset snapshot",
                metadata={
                    "dataset_id": dataset.dataset_id,
                    "dataset_snapshot_id": dataset.snapshot_id,
                    **dataset.metadata,
                },
            )
        except Exception as exc:
            errors.append(f"dataset create skipped: {type(exc).__name__}")

        created = 0
        skipped = 0
        failed = 0
        for case in dataset.cases:
            item_id = self.dataset_item_id(dataset.snapshot_id, case.case_id)
            if self._dataset_item_exists(item_id):
                skipped += 1
                continue
            try:
                client.create_dataset_item(
                    dataset_name=dataset_name,
                    id=item_id,
                    input=self._item_input(case),
                    expected_output=self._expected_output(case),
                    metadata=self._item_metadata(dataset, case),
                )
                created += 1
            except Exception as exc:
                if _is_sdk_response_validation_error(exc):
                    created += 1
                    continue
                failed += 1
                errors.append(f"{case.case_id}: {type(exc).__name__}")

        return LangfuseSyncResult(
            enabled=True,
            dataset_name=dataset_name,
            created_items=created,
            skipped_items=skipped,
            failed_items=failed,
            errors=errors[:10],
        )

    @staticmethod
    def get_dataset_name(dataset_name: str, snapshot_id: str) -> str:
        return f"poc-tuning-{_slug(dataset_name)}-{snapshot_id}"

    @staticmethod
    def dataset_item_id(snapshot_id: str, case_id: str) -> str:
        return "lfdi_" + hashlib.sha256(f"{snapshot_id}:{case_id}".encode("utf-8")).hexdigest()[:24]

    def _dataset_item_exists(self, item_id: str) -> bool:
        api = getattr(self.reporter.client, "api", None) if self.reporter.client else None
        items = getattr(api, "dataset_items", None) if api is not None else None
        get = getattr(items, "get", None)
        if not callable(get):
            return False
        try:
            get(item_id)
            return True
        except Exception as exc:
            if _is_sdk_response_validation_error(exc):
                return True
            return False

    def _item_input(self, case: Case) -> dict[str, object]:
        return {
            "case_id": case.case_id,
            "procedure_id": case.metadata.get("procedure_id") or Path(case.procedure_csv_path).stem,
            "evidence_bundle_id": case.metadata.get("evidence_bundle_id") or Path(case.evidence_bundle_path).name,
            "split": case.split.value,
            "domain": case.metadata.get("domain"),
            "procedure_family": case.metadata.get("procedure_family"),
        }

    def _expected_output(self, case: Case) -> dict[str, object]:
        return {
            "judgement": case.expected_output.judgement,
            "required_keywords": list(case.expected_output.required_claim_keywords),
            "expected_citations": [to_jsonable(citation) for citation in case.expected_output.citations],
        }

    def _item_metadata(self, dataset: Dataset, case: Case) -> dict[str, object]:
        return {
            "dataset_snapshot_id": dataset.snapshot_id,
            "reference_visibility": "ref_visible" if case.human_result or case.human_result_text else "expected_only",
            "evidence_hash": _hash_path(case.evidence_bundle_path),
            "procedure_hash": _hash_path(case.procedure_csv_path),
            **case.metadata,
        }


class LangfuseScoreConfigInitializer:
    def __init__(self, reporter: LangfuseReporter):
        self.reporter = reporter

    def initialize(self) -> dict[str, object]:
        if not self.reporter.enabled or self.reporter.client is None:
            return {"enabled": False, "created": [], "skipped": [], "failed": []}
        api = getattr(self.reporter.client, "api", None)
        score_configs = getattr(api, "score_configs", None) if api is not None else None
        create = getattr(score_configs, "create", None)
        if not callable(create):
            return {"enabled": False, "created": [], "skipped": [], "failed": ["score_configs API unavailable"]}

        existing = self._existing_names(score_configs)
        created: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        for spec in SCORE_CONFIG_SPECS:
            if spec.name in existing:
                skipped.append(spec.name)
                continue
            try:
                kwargs = self._create_kwargs(spec)
                created_config = create(**kwargs)
                created_id = getattr(created_config, "id", None) or getattr(created_config, "config_id", None)
                if created_id:
                    if self.reporter._score_config_ids is None:
                        self.reporter._score_config_ids = {}
                    self.reporter._score_config_ids[spec.name] = str(created_id)
                created.append(spec.name)
            except Exception:
                failed.append(spec.name)
        return {"enabled": True, "created": created, "skipped": skipped, "failed": failed}

    def _existing_names(self, score_configs: Any) -> set[str]:
        get = getattr(score_configs, "get", None)
        if not callable(get):
            return set()
        try:
            response = get(limit=100)
        except Exception:
            return set()
        data = getattr(response, "data", None) or getattr(response, "items", None) or []
        return {str(getattr(item, "name", "")) for item in data if getattr(item, "name", None)}

    def _create_kwargs(self, spec: ScoreConfigSpec) -> dict[str, object]:
        from langfuse.api.commons.types.config_category import ConfigCategory  # type: ignore
        from langfuse.api.commons.types.score_config_data_type import ScoreConfigDataType  # type: ignore

        data_type = ScoreConfigDataType(spec.data_type)
        kwargs: dict[str, object] = {
            "name": spec.name,
            "data_type": data_type,
            "description": spec.description,
        }
        if spec.min_value is not None:
            kwargs["min_value"] = spec.min_value
        if spec.max_value is not None:
            kwargs["max_value"] = spec.max_value
        if spec.categories:
            kwargs["categories"] = [
                ConfigCategory(value=float(index), label=label)
                for index, label in enumerate(spec.categories, start=1)
            ]
        return kwargs


def _candidate_summary(candidate: TuningCandidate) -> dict[str, object]:
    return {
        "tuning_id": candidate.tuning_id,
        "fingerprint": candidate.labels.get("fingerprint"),
        "candidate_status": candidate.status.value,
        "instruction": candidate.instruction_text,
        "hypothesis": candidate.hypothesis,
        "generated_by": candidate.generated_by,
        "labels": candidate.labels,
        "risk_labels": candidate.risk_labels,
    }


def _score_data_type(value: object) -> ScoreDataTypeName:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, (int, float)):
        return "NUMERIC"
    return "CATEGORICAL"


def _score_value(value: object, data_type: ScoreDataTypeName) -> float | str | None:
    if data_type == "BOOLEAN":
        return 1.0 if bool(value) else 0.0
    if data_type == "NUMERIC":
        return float(value) if isinstance(value, (int, float)) else None
    return str(value)


def _usage_details(cost: dict[str, object] | None) -> dict[str, int]:
    usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "estimated_tokens"):
        value = (cost or {}).get(key)
        if isinstance(value, int):
            usage[key] = value
    return usage


def _stable_short_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _is_sdk_response_validation_error(exc: Exception) -> bool:
    return type(exc).__name__ == "ValidationError" and "media_references" in str(exc)


def _slug(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "dataset"


def _hash_path(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
    elif target.exists():
        for child in sorted(target.rglob("*")):
            if child.is_file():
                digest.update(str(child.relative_to(target)).encode("utf-8"))
                digest.update(child.read_bytes())
    else:
        digest.update(str(target).encode("utf-8"))
    return "sha256:" + digest.hexdigest()
