"""Automated tuning search orchestrator."""

from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .agents import TuningAgent
from .artifacts import LocalArtifactStore
from .config import SearchPolicy
from .csv_patch import CsvMaterializer, PatchValidator
from .dataset import Dataset
from .evidence import read_text_artifact
from .evaluators import EvaluatorSuite
from .generalization import TuningGeneralizer
from .langfuse_client import LangfuseReporter
from .models import (
    AppRunResult,
    Case,
    EvaluationResult,
    FailureSummary,
    Scope,
    Split,
    TuningCandidate,
    TuningStatus,
    to_jsonable,
    tuning_candidate_fingerprint,
)
from .registry import ExperimentRegistry
from .runner import PocAppRunner


@dataclass(frozen=True)
class SearchRunReport:
    iterations: int
    generated_candidates: int
    evaluated_candidates: int
    positive_candidates: int
    promoted_candidates: int
    skipped_duplicate_candidates: int = 0
    needs_more_validation_candidates: int = 0
    baseline_case_count: int = 0
    experiment_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseExecutionResult:
    case: Case
    result: AppRunResult
    materialized_path: str
    csv_hash: str
    eval_results: list[EvaluationResult]
    score_map: dict[str, float]


class SearchOrchestrator:
    def __init__(
        self,
        *,
        dataset: Dataset,
        registry: ExperimentRegistry,
        artifacts: LocalArtifactStore,
        runner: PocAppRunner,
        agent: TuningAgent,
        evaluator_suite: EvaluatorSuite,
        langfuse: LangfuseReporter | None = None,
        policy: SearchPolicy | None = None,
    ):
        self.dataset = dataset
        self.registry = registry
        self.artifacts = artifacts
        self.runner = runner
        self.agent = agent
        self.evaluator_suite = evaluator_suite
        self.langfuse = langfuse or LangfuseReporter()
        self.policy = policy or SearchPolicy()
        self.materializer = CsvMaterializer()
        self.validator = PatchValidator(max_instruction_chars=self.policy.max_instruction_chars)
        self.generalizer = TuningGeneralizer()
        self.random = random.Random(self.policy.random_seed)

    def run(self) -> SearchRunReport:
        generated_count = 0
        evaluated_count = 0
        positive_count = 0
        promoted_count = 0
        duplicate_count = 0
        needs_more_validation_count = 0
        experiment_ids: list[str] = []

        baseline = self._baseline_candidate()
        self.registry.add_candidate(baseline)
        baseline_results = self._evaluate_baseline_all_splits(baseline)
        for result in baseline_results.values():
            experiment_ids.append(str(result["experiment_id"]))
        baseline_case_count = sum(int(result["summary"]["case_count"]) for result in baseline_results.values())

        train_baseline = baseline_results.get(Split.TRAIN.value) or next(iter(baseline_results.values()), None)
        last_failures = self._failure_summaries(train_baseline["case_results"] if train_baseline else [])
        positive_candidates: list[TuningCandidate] = []
        seen_fingerprints = self._known_candidate_fingerprints()

        for iteration in range(1, self.policy.iterations + 1):
            row_selector = self._default_row_selector()
            base_csv_id = self._base_csv_id()
            self._bind_agent_runtime_context(
                iteration=iteration,
                base_csv_id=base_csv_id,
                row_selector=row_selector,
            )
            candidates = self.agent.propose_candidates(
                failures=last_failures,
                base_csv_id=base_csv_id,
                row_selector=row_selector,
                max_candidates=self.policy.candidates_per_iteration,
                parent_tuning_ids=[candidate.tuning_id for candidate in positive_candidates[-self.policy.beam_width :]],
            )
            generalized = self.generalizer.propose_generalized_candidates(
                positive_candidates=positive_candidates,
                base_csv_id=base_csv_id,
                row_selector=row_selector,
                min_cluster_size=2,
            )
            candidates.extend(generalized)
            generated_count += len(candidates)

            valid_candidates: list[TuningCandidate] = []
            for candidate in candidates:
                stored = self._validate_and_store_candidate(candidate, seen_fingerprints=seen_fingerprints)
                if stored is None:
                    duplicate_count += 1
                    continue
                valid_candidates.append(stored)

            cheap_cases = self._sample_cases(Split.TRAIN, self.policy.cheap_sample_size)
            scored: list[tuple[TuningCandidate, float, dict[str, object]]] = []
            for candidate in valid_candidates:
                cheap_result = self._evaluate_candidate(
                    candidate=candidate,
                    cases=cheap_cases,
                    split=Split.TRAIN.value,
                    search_iteration=iteration,
                )
                experiment_ids.append(str(cheap_result["experiment_id"]))
                evaluated_count += 1
                delta = float(cheap_result["summary"].get("delta_mean", 0.0))  # type: ignore[union-attr]
                self._save_effect(candidate, cheap_result, stage="cheap")
                scored.append((candidate, delta, cheap_result))

            scored.sort(key=lambda item: item[1], reverse=True)
            winners = [
                item
                for item in scored[: self.policy.beam_width]
                if self._should_probe_validation(item[1], int(item[2]["regression_count"]))
            ]
            for candidate, _, _ in winners:
                validation_cases = self._sample_cases(Split.VALIDATION, self.policy.validation_sample_size)
                if not validation_cases:
                    validation_cases = cheap_cases
                validation_result = self._evaluate_candidate(
                    candidate=candidate,
                    cases=validation_cases,
                    split=Split.VALIDATION.value,
                    search_iteration=iteration,
                )
                experiment_ids.append(str(validation_result["experiment_id"]))
                label = self._effect_label(
                    float(validation_result["summary"].get("delta_mean", 0.0)),  # type: ignore[union-attr]
                    int(validation_result["regression_count"]),
                )
                self._save_effect(candidate, validation_result, stage="validation", override_label=label)
                self.registry.update_candidate_status(candidate.tuning_id, TuningStatus.EVALUATED)

                should_holdout_probe = label in {"positive", "strongly_positive"} or self._should_probe_holdout(
                    validation_result
                )
                holdout_result: dict[str, object] | None = None
                if should_holdout_probe:
                    holdout_cases = self._sample_cases(Split.HOLDOUT, self.policy.holdout_sample_size)
                    if holdout_cases:
                        holdout_result = self._evaluate_candidate(
                            candidate=candidate,
                            cases=holdout_cases,
                            split=Split.HOLDOUT.value,
                            search_iteration=iteration,
                        )
                        experiment_ids.append(str(holdout_result["experiment_id"]))
                        self._save_effect(candidate, holdout_result, stage="holdout")

                if label in {"positive", "strongly_positive"}:
                    positive_candidates.append(candidate)
                    positive_count += 1

                    decision, reason = self._promotion_decision(candidate, validation_result, holdout_result)
                    if decision in {"promote_candidate", "needs_more_validation"}:
                        self.registry.save_promotion_decision(
                            tuning_id=candidate.tuning_id,
                            from_scope=candidate.scope.value,
                            to_scope=Scope.DOMAIN_COMMON.value,
                            decision=decision,
                            reason=reason,
                            policy_version="strict-v2",
                            validation_result=validation_result["summary"],
                            holdout_result=holdout_result["summary"] if holdout_result else {},
                        )
                    if decision == "promote_candidate":
                        self.registry.update_candidate_status(candidate.tuning_id, TuningStatus.PROMOTED)
                        promoted_count += 1
                    elif decision == "needs_more_validation":
                        needs_more_validation_count += 1

            if scored:
                # Feed only train-stage failures back into exploration. Validation
                # and holdout are deliberately not used to generate future patches.
                best_case_results = scored[0][2]["case_results"]
                last_failures = self._failure_summaries(best_case_results) or last_failures

        self.langfuse.flush()
        return SearchRunReport(
            iterations=self.policy.iterations,
            generated_candidates=generated_count,
            evaluated_candidates=evaluated_count,
            positive_candidates=positive_count,
            promoted_candidates=promoted_count,
            skipped_duplicate_candidates=duplicate_count,
            needs_more_validation_candidates=needs_more_validation_count,
            baseline_case_count=baseline_case_count,
            experiment_ids=experiment_ids,
        )

    def _evaluate_baseline_all_splits(self, baseline: TuningCandidate) -> dict[str, dict[str, object]]:
        if not self.policy.baseline_all_splits:
            cases = self._sample_cases(Split.TRAIN, self.policy.cheap_sample_size)
            return {
                Split.TRAIN.value: self._evaluate_candidate(
                    candidate=baseline,
                    cases=cases,
                    split=Split.TRAIN.value,
                    search_iteration=0,
                )
            }

        results: dict[str, dict[str, object]] = {}
        ordered_splits = [Split.TRAIN, Split.VALIDATION, Split.HOLDOUT, Split.LEAVE_PROCEDURE_OUT, Split.LEAVE_DOMAIN_OUT]
        for split in ordered_splits:
            cases = self.dataset.by_split(split)
            if not cases:
                continue
            results[split.value] = self._evaluate_candidate(
                candidate=baseline,
                cases=list(cases),
                split=split.value,
                search_iteration=0,
            )
        return results

    def _bind_agent_runtime_context(
        self,
        *,
        iteration: int,
        base_csv_id: str,
        row_selector: dict[str, object],
    ) -> None:
        setter = getattr(self.agent, "set_runtime_context", None)
        if not callable(setter):
            return
        setter(
            HumanReferenceSearchContext(
                orchestrator=self,
                iteration=iteration,
                base_csv_id=base_csv_id,
                row_selector=row_selector,
            )
        )

    def _should_probe_validation(self, train_delta: float, regression_count: int) -> bool:
        if regression_count != 0:
            return False
        threshold = 0.0 if self.policy.allow_neutral_train_probe else self.policy.min_delta_for_positive_label
        return train_delta >= threshold

    def _should_probe_holdout(self, validation_result: dict[str, object]) -> bool:
        if not self.policy.allow_neutral_train_probe:
            return False
        summary = validation_result["summary"]  # type: ignore[assignment]
        assert isinstance(summary, dict)
        delta = float(summary.get("delta_mean", 0.0))
        return int(validation_result["regression_count"]) == 0 and delta >= -self.policy.delta_epsilon

    def _validate_and_store_candidate(
        self,
        candidate: TuningCandidate,
        *,
        seen_fingerprints: set[str],
    ) -> TuningCandidate | None:
        fingerprint = tuning_candidate_fingerprint(candidate)
        candidate = self._candidate_with_fingerprint(candidate, fingerprint)
        if self.policy.deduplicate_candidates and fingerprint in seen_fingerprints:
            rejected = self._rejected_candidate(
                candidate,
                reason_code="duplicate_candidate",
                reason_message="同一targetと同一追加指示の候補が既に存在するため再評価をスキップしました。",
            )
            self.registry.add_candidate(rejected)
            return None

        base_csv = self._representative_csv_path()
        report = self.validator.validate(candidate.patch, base_csv_path=base_csv)
        if not report.valid:
            rejected = self._rejected_candidate(
                candidate,
                reason_code="validator_rejected",
                reason_message="patch validatorで不正または危険と判定されました。",
                extra_labels={"validation_issues": [to_jsonable(issue) for issue in report.issues]},
            )
            self.registry.add_candidate(rejected)
            seen_fingerprints.add(fingerprint)
            return None
        self.registry.add_candidate(candidate)
        seen_fingerprints.add(fingerprint)
        return candidate

    def _candidate_with_fingerprint(self, candidate: TuningCandidate, fingerprint: str) -> TuningCandidate:
        return TuningCandidate(
            tuning_id=candidate.tuning_id,
            patch=candidate.patch,
            scope=candidate.scope,
            parent_tuning_ids=candidate.parent_tuning_ids,
            hypothesis=candidate.hypothesis,
            generated_by=candidate.generated_by,
            generator_prompt_version=candidate.generator_prompt_version,
            labels={**candidate.labels, "fingerprint": fingerprint},
            risk_labels=candidate.risk_labels,
            status=candidate.status,
            created_at=candidate.created_at,
        )

    def _rejected_candidate(
        self,
        candidate: TuningCandidate,
        *,
        reason_code: str,
        reason_message: str,
        extra_labels: dict[str, object] | None = None,
    ) -> TuningCandidate:
        return TuningCandidate(
            tuning_id=candidate.tuning_id,
            patch=candidate.patch,
            scope=candidate.scope,
            parent_tuning_ids=candidate.parent_tuning_ids,
            hypothesis=candidate.hypothesis,
            generated_by=candidate.generated_by,
            generator_prompt_version=candidate.generator_prompt_version,
            labels={**candidate.labels, **(extra_labels or {})},
            risk_labels={**candidate.risk_labels, reason_code: True, "rejection_reason": reason_message},
            status=TuningStatus.REJECTED,
            created_at=candidate.created_at,
        )

    def _evaluate_candidate(
        self,
        *,
        candidate: TuningCandidate,
        cases: list[Case],
        split: str,
        search_iteration: int,
    ) -> dict[str, object]:
        experiment_id = self.registry.create_experiment_batch(
            search_iteration=search_iteration,
            dataset_snapshot_id=self.dataset.snapshot_id,
            split=split,
            notes=f"candidate={candidate.tuning_id}",
        )
        case_results: list[dict[str, object]] = []
        scores: list[float] = []
        deltas: list[float] = []
        metric_sums: dict[str, float] = {}
        metric_delta_sums: dict[str, float] = {}
        metric_delta_counts: dict[str, int] = {}
        baseline_by_case = self._baseline_case_metric_scores() if candidate.tuning_id != "baseline" else {}
        missing_baseline_case_ids: list[str] = []
        positive_count = 0
        negative_count = 0
        regression_count = 0
        failed_count = 0

        for case_execution in self._run_cases_for_candidate(candidate, cases):
            case = case_execution.case
            result = case_execution.result
            materialized_path = case_execution.materialized_path
            csv_hash = case_execution.csv_hash
            if result.status != "succeeded":
                failed_count += 1
            raw_artifact = self.artifacts.write_json(
                "raw_outputs",
                f"{experiment_id}_{case.case_id}_{candidate.tuning_id}.json",
                result.raw_output or to_jsonable(result.normalized_result),
            )
            trace = self.langfuse.start_trace(
                name="poc_tuning_run",
                case_id=case.case_id,
                tuning_id=candidate.tuning_id,
                metadata={
                    "experiment_id": experiment_id,
                    "split": split,
                    "dataset_snapshot_id": self.dataset.snapshot_id,
                    "materialized_csv_hash": csv_hash,
                    "candidate_generator": candidate.generated_by,
                    "candidate_fingerprint": candidate.labels.get("fingerprint"),
                },
            )
            eval_results = case_execution.eval_results
            self.langfuse.record_output(trace=trace, output=result.normalized_result, candidate=candidate)
            self.langfuse.record_scores(trace=trace, results=eval_results)
            run_id = self.registry.save_case_run(
                experiment_id=experiment_id,
                case_id=case.case_id,
                tuning_id=candidate.tuning_id,
                status=result.status,
                app_run_id=result.app_run_id,
                langfuse_trace_id=trace.trace_id,
                base_csv_hash=_hash_file(case.procedure_csv_path),
                materialized_csv_hash=csv_hash,
                evidence_bundle_hash=_hash_path(case.evidence_bundle_path),
                latency_ms=result.latency_ms,
                cost_json=result.cost,
                raw_output_uri=raw_artifact.uri,
                normalized_output_json=to_jsonable(result.normalized_result),
                error_message=result.error_message,
            )
            score_map = dict(case_execution.score_map)
            for evaluation in eval_results:
                self.registry.save_evaluation_result(run_id, evaluation)
            for metric_name, score in score_map.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + score
            total = float(score_map.get("total_score", 0.0))
            scores.append(total)

            baseline_metrics = baseline_by_case.get(case.case_id)
            if candidate.tuning_id == "baseline":
                delta = 0.0
            elif baseline_metrics is None:
                missing_baseline_case_ids.append(case.case_id)
                delta = 0.0
            else:
                delta = total - float(baseline_metrics.get("total_score", 0.0))
                for metric_name, metric_score in score_map.items():
                    if metric_name in baseline_metrics:
                        metric_delta_sums[metric_name] = metric_delta_sums.get(metric_name, 0.0) + (
                            metric_score - float(baseline_metrics[metric_name])
                        )
                        metric_delta_counts[metric_name] = metric_delta_counts.get(metric_name, 0) + 1
            deltas.append(delta)
            if delta > self.policy.delta_epsilon:
                positive_count += 1
            elif delta < -self.policy.delta_epsilon:
                negative_count += 1
                regression_count += 1
            case_results.append(
                {
                    "case_id": case.case_id,
                    "score_map": score_map,
                    "total_score": total,
                    "delta_vs_baseline": delta,
                    "normalized_output": to_jsonable(result.normalized_result),
                    "materialized_csv_path": str(materialized_path),
                    "domain": case.metadata.get("domain"),
                    "procedure_family": case.metadata.get("procedure_family"),
                }
            )

        case_count = len(scores)
        metric_means = {name: value / max(1, case_count) for name, value in metric_sums.items()}
        metric_delta_means = {
            name: metric_delta_sums[name] / max(1, metric_delta_counts.get(name, 0)) for name in metric_delta_sums
        }
        summary = {
            "total_score_mean": sum(scores) / len(scores) if scores else 0.0,
            "delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
            "case_count": case_count,
            "split": split,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "regression_count": regression_count,
            "regression_rate": regression_count / max(1, case_count),
            "failed_count": failed_count,
            "metric_means": metric_means,
            "metric_delta_means": metric_delta_means,
            "missing_baseline_case_ids": missing_baseline_case_ids,
            "domain_count": self._distinct_metadata_count(cases, "domain"),
            "procedure_family_count": self._distinct_metadata_count(cases, "procedure_family"),
        }
        return {
            "experiment_id": experiment_id,
            "summary": summary,
            "case_results": case_results,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "regression_count": regression_count,
            "failed_count": failed_count,
        }

    def _save_effect(
        self,
        candidate: TuningCandidate,
        result: dict[str, object],
        *,
        stage: str,
        override_label: str | None = None,
    ) -> None:
        summary = result["summary"]  # type: ignore[assignment]
        assert isinstance(summary, dict)
        delta = float(summary.get("delta_mean", 0.0))
        metric_delta_means = summary.get("metric_delta_means", {})
        if not isinstance(metric_delta_means, dict):
            metric_delta_means = {}
        regression_count = int(result["regression_count"])
        label = override_label or self._effect_label(delta, regression_count)
        self.registry.save_tuning_effect(
            tuning_id=candidate.tuning_id,
            experiment_id=str(result["experiment_id"]),
            split=str(summary.get("split", stage)),
            total_score_delta=delta,
            judgement_delta=float(metric_delta_means.get("judgement_match", 0.0) or 0.0),
            rationale_delta=float(metric_delta_means.get("rationale_support", 0.0) or 0.0),
            citation_delta=float(metric_delta_means.get("citation_quality", 0.0) or 0.0),
            regression_count=regression_count,
            positive_count=int(result["positive_count"]),
            negative_count=int(result["negative_count"]),
            generality_score=self._generality_score(candidate, result),
            overfit_risk=self._overfit_risk(candidate, result),
            effect_label=label,
            details={"stage": stage, "summary": summary},
        )

    def _run_materialized_case(self, candidate: TuningCandidate, case: Case) -> tuple[AppRunResult, str, str]:
        out_dir = Path(self.artifacts.root_dir) / "materialized_csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(case.procedure_csv_path).suffix or ".txt"
        output_path = out_dir / f"{case.case_id}_{candidate.tuning_id}{suffix}"
        materialized = self.materializer.materialize(case.procedure_csv_path, candidate, output_path)
        csv_hash = _hash_file(materialized.output_path)
        self.artifacts.write_text("csv_diffs", f"{case.case_id}_{candidate.tuning_id}.diff", materialized.diff)
        result = self.runner.run_case(case=case, materialized_csv_path=materialized.output_path)
        return result, str(materialized.output_path), csv_hash

    def _run_cases_for_candidate(self, candidate: TuningCandidate, cases: list[Case]) -> list[CaseExecutionResult]:
        if self.policy.runner_parallelism <= 1 or len(cases) <= 1:
            return [self._run_and_evaluate_case(candidate, case) for case in cases]
        workers = min(self.policy.runner_parallelism, len(cases))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="poc-case-runner") as executor:
            return list(executor.map(lambda case: self._run_and_evaluate_case(candidate, case), cases))

    def _run_and_evaluate_case(self, candidate: TuningCandidate, case: Case) -> CaseExecutionResult:
        result, materialized_path, csv_hash = self._run_materialized_case(candidate, case)
        eval_results = self.evaluator_suite.evaluate_case(
            case=case,
            output=result.normalized_result,
            candidate=candidate,
        )
        score_map = {
            evaluation.evaluator_name: float(evaluation.score)
            for evaluation in eval_results
            if evaluation.score is not None
        }
        return CaseExecutionResult(
            case=case,
            result=result,
            materialized_path=materialized_path,
            csv_hash=csv_hash,
            eval_results=eval_results,
            score_map=score_map,
        )

    def _baseline_candidate(self) -> TuningCandidate:
        return TuningCandidate(
            tuning_id="baseline",
            patch=None,
            scope=Scope.PROCEDURE_SPECIFIC,
            parent_tuning_ids=[],
            hypothesis="ベースラインCSVをそのまま実行する",
            generated_by="system",
            generator_prompt_version="baseline",
            labels={"tactic_type": ["baseline"]},
            risk_labels={},
            status=TuningStatus.EVALUATED,
        )

    def _sample_cases(self, split: Split, size: int) -> list[Case]:
        cases = self.dataset.by_split(split)
        if not cases:
            return []
        if size <= 0 or len(cases) <= size:
            return list(cases)
        return self.random.sample(cases, size)

    def _representative_csv_path(self) -> str:
        if not self.dataset.cases:
            raise ValueError("dataset has no cases")
        return self.dataset.cases[0].procedure_csv_path

    def _base_csv_id(self) -> str:
        first = Path(self._representative_csv_path())
        return first.stem

    def _default_row_selector(self) -> dict[str, object]:
        import csv

        representative_path = Path(self._representative_csv_path())
        if representative_path.suffix.lower() != ".csv":
            return {"document": "procedure_text"}

        with representative_path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            first = next(reader)
        if "step_id" in first:
            return {"step_id": first["step_id"]}
        key = next(iter(first.keys()))
        return {key: first[key]}

    def _failure_summaries(self, case_results: Iterable[dict[str, object]]) -> list[FailureSummary]:
        failures: list[FailureSummary] = []
        for item in case_results:
            score_map = item.get("score_map", {})
            if not isinstance(score_map, dict):
                continue
            metadata = {
                "delta_vs_baseline": item.get("delta_vs_baseline"),
                "domain": item.get("domain"),
                "procedure_family": item.get("procedure_family"),
            }
            if float(score_map.get("judgement_match", 1.0) or 0.0) < 1.0:
                failures.append(
                    FailureSummary(
                        case_id=str(item["case_id"]),
                        failure_mode="wrong_judgement",
                        summary="最終判定が人手回答と一致しませんでした。",
                        missing_capability="手続CSVの条件分岐、証跡不足、例外条件を判定前に確認する能力",
                        scores=score_map,
                        metadata=metadata,
                    )
                )
            if float(score_map.get("rationale_support", 1.0) or 0.0) < 0.8:
                failures.append(
                    FailureSummary(
                        case_id=str(item["case_id"]),
                        failure_mode="unsupported_rationale",
                        summary="根拠が期待論点を十分にカバーしていない、または証跡に支持されていません。",
                        missing_capability="証跡に明示された主張と推測の区別",
                        scores=score_map,
                        metadata=metadata,
                    )
                )
            if float(score_map.get("citation_quality", 1.0) or 0.0) < 0.8:
                failures.append(
                    FailureSummary(
                        case_id=str(item["case_id"]),
                        failure_mode="citation_mismatch",
                        summary="引用箇所が不足している、または期待引用と一致していません。",
                        missing_capability="根拠文と引用箇所の対応付け",
                        scores=score_map,
                        metadata=metadata,
                    )
                )
            if str(item.get("normalized_output", {})).find("判断不能") >= 0 and float(
                score_map.get("judgement_match", 1.0) or 0.0
            ) < 1.0:
                failures.append(
                    FailureSummary(
                        case_id=str(item["case_id"]),
                        failure_mode="insufficient_evidence",
                        summary="証跡不足時の判定が期待と一致していません。",
                        missing_capability="証跡不足時に推測で補わず判断不能を選ぶ能力",
                        scores=score_map,
                        metadata=metadata,
                    )
                )
        return failures[:20]

    def _baseline_case_metric_scores(self) -> dict[str, dict[str, float]]:
        with self.registry.connect() as conn:
            rows = conn.execute(
                """
                SELECT cr.case_id, er.evaluator_name, er.score
                FROM case_runs cr
                JOIN experiment_batches eb ON cr.experiment_id = eb.experiment_id
                JOIN evaluation_results er ON cr.run_id = er.run_id
                WHERE cr.tuning_id = 'baseline'
                  AND eb.dataset_snapshot_id = ?
                ORDER BY cr.created_at DESC
                """,
                (self.dataset.snapshot_id,),
            ).fetchall()
        scores: dict[str, dict[str, float]] = {}
        for row in rows:
            case_scores = scores.setdefault(row["case_id"], {})
            case_scores.setdefault(row["evaluator_name"], float(row["score"] or 0.0))
        return scores

    def _known_candidate_fingerprints(self) -> set[str]:
        fingerprints: set[str] = set()
        for candidate in self.registry.list_candidates(limit=10000):
            if candidate.tuning_id == "baseline":
                continue
            fingerprints.add(str(candidate.labels.get("fingerprint") or tuning_candidate_fingerprint(candidate)))
        return fingerprints

    def _effect_label(self, delta: float, regression_count: int) -> str:
        if regression_count > 0 and delta <= self.policy.min_delta_for_positive_label:
            return "risky"
        if delta >= 0.10 and regression_count == 0:
            return "strongly_positive"
        if delta >= self.policy.min_delta_for_positive_label and regression_count == 0:
            return "positive"
        if delta <= -0.03:
            return "negative"
        return "neutral"

    def _generality_score(self, candidate: TuningCandidate, result: dict[str, object]) -> float:
        summary = result["summary"]  # type: ignore[assignment]
        assert isinstance(summary, dict)
        case_count = int(summary.get("case_count", 0))
        positive = int(result["positive_count"])
        domain_count = int(summary.get("domain_count", 0))
        family_count = int(summary.get("procedure_family_count", 0))
        scope_bonus = {
            Scope.CASE_SPECIFIC: 0.0,
            Scope.PROCEDURE_SPECIFIC: 0.1,
            Scope.PROCEDURE_FAMILY: 0.2,
            Scope.DOMAIN_COMMON: 0.3,
            Scope.GLOBAL_COMMON: 0.4,
        }[candidate.scope]
        coverage_bonus = min(0.3, 0.05 * domain_count + 0.05 * family_count)
        return round(min(1.0, (positive / max(1, case_count)) + scope_bonus + coverage_bonus), 4)

    def _overfit_risk(self, candidate: TuningCandidate, result: dict[str, object]) -> float:
        summary = result["summary"]  # type: ignore[assignment]
        assert isinstance(summary, dict)
        case_count = int(summary.get("case_count", 0))
        domain_count = int(summary.get("domain_count", 0))
        specificity = 0.2 if candidate.scope in {Scope.CASE_SPECIFIC, Scope.PROCEDURE_SPECIFIC} else 0.0
        small_sample = 0.4 if case_count < self.policy.min_validation_cases_for_promotion else 0.0
        low_domain = 0.2 if domain_count < self.policy.min_domains_for_promotion else 0.0
        return round(min(1.0, specificity + small_sample + low_domain), 4)

    def _promotion_decision(
        self,
        candidate: TuningCandidate,
        validation_result: dict[str, object],
        holdout_result: dict[str, object] | None,
    ) -> tuple[str, str]:
        validation_summary = validation_result["summary"]  # type: ignore[assignment]
        assert isinstance(validation_summary, dict)
        holdout_summary = holdout_result["summary"] if holdout_result else {}  # type: ignore[index]
        assert isinstance(holdout_summary, dict)

        validation_cases = int(validation_summary.get("case_count", 0))
        holdout_cases = int(holdout_summary.get("case_count", 0)) if holdout_result else 0
        validation_delta = float(validation_summary.get("delta_mean", 0.0))
        holdout_delta = float(holdout_summary.get("delta_mean", 0.0)) if holdout_result else 0.0
        validation_regression_rate = float(validation_summary.get("regression_rate", 1.0))
        holdout_regression_rate = float(holdout_summary.get("regression_rate", 1.0)) if holdout_result else 1.0
        domain_count = max(
            int(validation_summary.get("domain_count", 0)),
            int(holdout_summary.get("domain_count", 0)) if holdout_result else 0,
        )
        family_count = max(
            int(validation_summary.get("procedure_family_count", 0)),
            int(holdout_summary.get("procedure_family_count", 0)) if holdout_result else 0,
        )
        metric_delta_means = validation_summary.get("metric_delta_means", {})
        holdout_metric_delta_means = holdout_summary.get("metric_delta_means", {}) if holdout_result else {}
        if not isinstance(metric_delta_means, dict):
            metric_delta_means = {}
        if not isinstance(holdout_metric_delta_means, dict):
            holdout_metric_delta_means = {}

        blockers: list[str] = []
        if validation_cases < self.policy.min_validation_cases_for_promotion:
            blockers.append(
                f"validation cases {validation_cases} < required {self.policy.min_validation_cases_for_promotion}"
            )
        if holdout_cases < self.policy.min_holdout_cases_for_promotion:
            blockers.append(f"holdout cases {holdout_cases} < required {self.policy.min_holdout_cases_for_promotion}")
        if domain_count < self.policy.min_domains_for_promotion:
            blockers.append(f"domains {domain_count} < required {self.policy.min_domains_for_promotion}")
        if family_count < self.policy.min_procedure_families_for_promotion:
            blockers.append(
                f"procedure families {family_count} < required {self.policy.min_procedure_families_for_promotion}"
            )
        if validation_delta < self.policy.min_delta_for_promotion:
            blockers.append(f"validation delta {validation_delta:.4f} < required {self.policy.min_delta_for_promotion:.4f}")
        if holdout_result and holdout_delta < 0:
            blockers.append(f"holdout delta {holdout_delta:.4f} < 0")
        if validation_regression_rate > self.policy.max_regression_rate_for_promotion:
            blockers.append(
                f"validation regression rate {validation_regression_rate:.4f} > allowed {self.policy.max_regression_rate_for_promotion:.4f}"
            )
        if holdout_result and holdout_regression_rate > self.policy.max_regression_rate_for_promotion:
            blockers.append(
                f"holdout regression rate {holdout_regression_rate:.4f} > allowed {self.policy.max_regression_rate_for_promotion:.4f}"
            )
        if self.policy.require_judgement_non_degradation:
            min_judgement_delta = min(
                float(metric_delta_means.get("judgement_match", 0.0) or 0.0),
                float(holdout_metric_delta_means.get("judgement_match", 0.0) or 0.0),
            )
            if min_judgement_delta < 0:
                blockers.append(f"judgement delta degraded: {min_judgement_delta:.4f}")
        if self.policy.require_citation_non_degradation:
            min_citation_delta = min(
                float(metric_delta_means.get("citation_quality", 0.0) or 0.0),
                float(holdout_metric_delta_means.get("citation_quality", 0.0) or 0.0),
            )
            if min_citation_delta < 0:
                blockers.append(f"citation delta degraded: {min_citation_delta:.4f}")

        if blockers:
            return "needs_more_validation", "昇格保留: " + "; ".join(blockers)
        return "promote_candidate", "strict-v2 policyの最小件数、case-level delta、非悪化条件を満たしました。"

    def _distinct_metadata_count(self, cases: list[Case], key: str) -> int:
        return len({str(case.metadata.get(key, "")) for case in cases if case.metadata.get(key)})


class HumanReferenceSearchContext:
    def __init__(
        self,
        *,
        orchestrator: SearchOrchestrator,
        iteration: int,
        base_csv_id: str,
        row_selector: dict[str, object],
    ):
        self.orchestrator = orchestrator
        self.iteration = iteration
        self.base_csv_id = base_csv_id
        self.row_selector = row_selector
        self.trial_count = 0

    def list_case_inventory(self) -> dict[str, object]:
        cases = []
        observation_splits = set(self.orchestrator.policy.agent_observation_splits)
        for case in self.orchestrator.dataset.cases:
            if case.split.value not in observation_splits:
                continue
            cases.append(
                {
                    "case_id": case.case_id,
                    "split": case.split.value,
                    "metadata": self._safe_metadata(case),
                    "human_result_visible": self._human_result_visible(case),
                }
            )
        payload = {
            "data_visibility_policy": self.orchestrator.policy.data_visibility_policy,
            "human_reference_splits": list(self.orchestrator.policy.human_reference_splits),
            "holdout_reference_visible": self.orchestrator.policy.holdout_reference_visible,
            "trial_budget_remaining": self.orchestrator.policy.per_case_trial_budget - self.trial_count,
            "cases": cases,
        }
        self.orchestrator.artifacts.write_json(
            "agent_observations",
            f"iteration_{self.iteration:04d}_case_inventory.json",
            payload,
        )
        return payload

    def read_case_input(self, case_id: str) -> dict[str, object]:
        case = self._case(case_id)
        payload = {
            "case_id": case.case_id,
            "split": case.split.value,
            "metadata": self._safe_metadata(case),
            "procedure_csv": read_text_artifact(case.procedure_csv_path, max_chars=12000),
            "evidence_bundle": read_text_artifact(case.evidence_bundle_path, max_chars=30000),
            "baseline_observation": self._baseline_observation(case.case_id),
        }
        self.orchestrator.artifacts.write_json(
            "agent_observations",
            f"iteration_{self.iteration:04d}_{case.case_id}_input.json",
            payload,
        )
        return payload

    def read_human_result(self, case_id: str) -> dict[str, object]:
        case = self._case(case_id)
        if not self._human_result_visible(case):
            payload = {
                "case_id": case.case_id,
                "split": case.split.value,
                "visible": False,
                "reason": "human_result is hidden for this split",
            }
        else:
            payload = {
                "case_id": case.case_id,
                "split": case.split.value,
                "visible": True,
                "human_result_text": case.human_result_text,
                "human_result": None
                if case.human_result_text
                else to_jsonable(case.human_result or case.expected_output),
            }
        self.orchestrator.artifacts.write_json(
            "agent_observations",
            f"iteration_{self.iteration:04d}_{case.case_id}_human_result.json",
            payload,
        )
        return payload

    def list_previous_trials(self) -> list[dict[str, object]]:
        with self.orchestrator.registry.connect() as conn:
            rows = conn.execute(
                """
                SELECT trial_id, search_iteration, draft_index, instruction_text,
                       hypothesis, case_ids_json, summary_json, status, error_message
                FROM agent_trial_observations
                ORDER BY search_iteration, draft_index, created_at
                LIMIT 100
                """
            ).fetchall()
        return [
            {
                "trial_id": row["trial_id"],
                "search_iteration": row["search_iteration"],
                "draft_index": row["draft_index"],
                "instruction": row["instruction_text"],
                "hypothesis": row["hypothesis"],
                "case_ids": json.loads(row["case_ids_json"] or "[]"),
                "summary": json.loads(row["summary_json"] or "{}"),
                "status": row["status"],
                "error_message": row["error_message"],
            }
            for row in rows
        ]

    def evaluate_draft_instruction(
        self,
        *,
        instruction: str,
        hypothesis: str = "",
        case_id: str | None = None,
    ) -> dict[str, object]:
        instruction = instruction.strip()
        hypothesis = hypothesis.strip()
        if self.trial_count >= self.orchestrator.policy.per_case_trial_budget:
            return {
                "status": "budget_exhausted",
                "trial_budget": self.orchestrator.policy.per_case_trial_budget,
                "trial_count": self.trial_count,
            }
        self.trial_count += 1
        draft_index = self.trial_count
        cases = self._trial_cases(case_id)
        if not instruction:
            trial_id = self.orchestrator.registry.save_agent_trial_observation(
                search_iteration=self.iteration,
                draft_index=draft_index,
                agent_name="deepagent-human-ref",
                tool_name="evaluate_draft_instruction",
                instruction_text=instruction,
                hypothesis=hypothesis,
                status="rejected",
                error_message="instruction is empty",
            )
            return {"status": "rejected", "trial_id": trial_id, "error": "instruction is empty"}

        candidate = TuningCandidate(
            tuning_id=f"trial_candidate_{self.iteration:04d}_{draft_index:02d}",
            parent_tuning_ids=[],
            scope=Scope.PROCEDURE_SPECIFIC,
            patch=self._patch(instruction),
            hypothesis=hypothesis or "human-reference draft trial",
            generated_by="deepagent-human-ref-trial",
            generator_prompt_version="deepagent-human-ref-v3",
            labels={"trial": True, "tactic_type": ["human_reference_trial"]},
            risk_labels={},
            status=TuningStatus.ARCHIVED,
        )
        validation = self.orchestrator.validator.validate(candidate.patch, base_csv_path=self.orchestrator._representative_csv_path())
        if not validation.valid:
            trial_id = self.orchestrator.registry.save_agent_trial_observation(
                search_iteration=self.iteration,
                draft_index=draft_index,
                agent_name="deepagent-human-ref",
                tool_name="evaluate_draft_instruction",
                instruction_text=instruction,
                hypothesis=hypothesis,
                case_ids=[case.case_id for case in cases],
                splits=sorted({case.split.value for case in cases}),
                status="rejected",
                error_message="; ".join(issue.message for issue in validation.errors()),
            )
            return {"status": "rejected", "trial_id": trial_id, "validation_issues": to_jsonable(validation.issues)}

        baseline_by_case = self.orchestrator._baseline_case_metric_scores()
        first_run = self._evaluate_trial_candidate_once(candidate, cases, baseline_by_case, replicate_index=1)
        summary = dict(first_run["summary"])
        case_results = list(first_run["case_results"])
        replicate_runs: list[dict[str, object]] = [first_run]
        if self._should_replicate_trial(summary):
            for replicate_index in range(2, self.orchestrator.policy.agent_trial_replicates + 1):
                replicate_runs.append(
                    self._evaluate_trial_candidate_once(candidate, cases, baseline_by_case, replicate_index=replicate_index)
                )
            summary["replicate_summary"] = self._summarize_trial_replicates(replicate_runs)
        trial_id = self.orchestrator.registry.save_agent_trial_observation(
            search_iteration=self.iteration,
            draft_index=draft_index,
            agent_name="deepagent-human-ref",
            tool_name="evaluate_draft_instruction",
            instruction_text=instruction,
            hypothesis=hypothesis,
            case_ids=[case.case_id for case in cases],
            splits=sorted({case.split.value for case in cases}),
            summary=summary,
            case_results=case_results,
            status="succeeded",
        )
        payload = {
            "status": "succeeded",
            "trial_id": trial_id,
            "instruction": instruction,
            "hypothesis": hypothesis,
            "summary": summary,
            "case_results": case_results,
            "replicate_runs": replicate_runs if len(replicate_runs) > 1 else [],
        }
        self.orchestrator.artifacts.write_json(
            "agent_observations",
            f"iteration_{self.iteration:04d}_{trial_id}.json",
            payload,
        )
        return payload

    def _evaluate_trial_candidate_once(
        self,
        candidate: TuningCandidate,
        cases: list[Case],
        baseline_by_case: dict[str, dict[str, float]],
        *,
        replicate_index: int,
    ) -> dict[str, object]:
        case_results: list[dict[str, object]] = []
        scores: list[float] = []
        deltas: list[float] = []
        regression_count = 0
        positive_count = 0
        negative_count = 0
        total_latency_ms = 0
        usage: dict[str, int] = {}
        for case_execution in self.orchestrator._run_cases_for_candidate(candidate, cases):
            case = case_execution.case
            result = case_execution.result
            materialized_path = case_execution.materialized_path
            total_latency_ms += int(result.latency_ms or 0)
            for key, value in (result.cost or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
            score_map = dict(case_execution.score_map)
            total = float(score_map.get("total_score", 0.0))
            baseline_total = float(baseline_by_case.get(case.case_id, {}).get("total_score", 0.0))
            delta = total - baseline_total
            scores.append(total)
            deltas.append(delta)
            if delta > self.orchestrator.policy.delta_epsilon:
                positive_count += 1
            elif delta < -self.orchestrator.policy.delta_epsilon:
                negative_count += 1
                regression_count += 1
            case_results.append(
                {
                    "replicate_index": replicate_index,
                    "case_id": case.case_id,
                    "split": case.split.value,
                    "status": result.status,
                    "total_score": total,
                    "delta_vs_baseline": delta,
                    "score_map": score_map,
                    "judgement": result.normalized_result.judgement,
                    "claim_text": result.normalized_result.claim_text(),
                    "materialized_csv_path": str(materialized_path),
                    "latency_ms": result.latency_ms,
                    "error_message": result.error_message,
                }
            )
        summary = {
            "replicate_index": replicate_index,
            "case_count": len(cases),
            "total_score_mean": sum(scores) / len(scores) if scores else 0.0,
            "delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
            "delta_min": min(deltas) if deltas else 0.0,
            "delta_max": max(deltas) if deltas else 0.0,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "regression_count": regression_count,
            "latency_ms": total_latency_ms,
            "usage": usage,
        }
        return {"replicate_index": replicate_index, "summary": summary, "case_results": case_results}

    def _should_replicate_trial(self, summary: dict[str, object]) -> bool:
        if self.orchestrator.policy.agent_trial_replicates <= 1:
            return False
        return (
            float(summary.get("delta_mean", 0.0) or 0.0)
            >= self.orchestrator.policy.agent_trial_replicate_min_delta_mean
            and int(summary.get("regression_count", 0) or 0)
            <= self.orchestrator.policy.agent_trial_replicate_max_regression_count
        )

    def _summarize_trial_replicates(self, replicate_runs: list[dict[str, object]]) -> dict[str, object]:
        summaries = [run.get("summary", {}) for run in replicate_runs if isinstance(run.get("summary"), dict)]
        delta_means = [float(summary.get("delta_mean", 0.0) or 0.0) for summary in summaries]
        total_means = [float(summary.get("total_score_mean", 0.0) or 0.0) for summary in summaries]
        regression_counts = [int(summary.get("regression_count", 0) or 0) for summary in summaries]
        all_case_deltas = [
            float(case_result.get("delta_vs_baseline", 0.0) or 0.0)
            for run in replicate_runs
            for case_result in run.get("case_results", [])
            if isinstance(case_result, dict)
        ]
        delta_mean_avg = sum(delta_means) / len(delta_means) if delta_means else 0.0
        delta_mean_min = min(delta_means) if delta_means else 0.0
        total_score_mean_avg = sum(total_means) / len(total_means) if total_means else 0.0
        total_score_mean_min = min(total_means) if total_means else 0.0
        worst_case_delta = min(all_case_deltas) if all_case_deltas else 0.0
        max_regression_count = max(regression_counts) if regression_counts else 0
        stable = (
            delta_mean_avg >= self.orchestrator.policy.agent_trial_replicate_min_delta_mean
            and delta_mean_min >= self.orchestrator.policy.agent_trial_replicate_min_worst_delta
            and worst_case_delta >= self.orchestrator.policy.agent_trial_replicate_min_worst_delta
            and max_regression_count <= self.orchestrator.policy.agent_trial_replicate_max_regression_count
        )
        return {
            "replicate_count": len(summaries),
            "stable": stable,
            "delta_mean_avg": delta_mean_avg,
            "delta_mean_min": delta_mean_min,
            "delta_mean_max": max(delta_means) if delta_means else 0.0,
            "total_score_mean_avg": total_score_mean_avg,
            "total_score_mean_min": total_score_mean_min,
            "worst_case_delta": worst_case_delta,
            "max_regression_count": max_regression_count,
            "regression_count_total": sum(regression_counts),
        }

    def synthesize_cross_case_tuning(self) -> dict[str, object]:
        trials = self.list_previous_trials()
        successful = [
            trial
            for trial in trials
            if trial.get("status") == "succeeded"
            and float(trial.get("summary", {}).get("delta_mean", 0.0) or 0.0) >= 0.0
            and int(trial.get("summary", {}).get("regression_count", 0) or 0) == 0
            and self._trial_is_replicate_stable(trial)
        ]
        return {
            "min_cases_for_generalized_tuning": self.orchestrator.policy.min_cases_for_generalized_tuning,
            "trial_replicates": self.orchestrator.policy.agent_trial_replicates,
            "successful_trial_count": len(successful),
            "recommended_source_trial_ids": [str(trial["trial_id"]) for trial in successful[:5]],
            "note": "Prefer replicated trials with non-negative delta, zero regressions, and stable worst-case behavior.",
        }

    def _trial_is_replicate_stable(self, trial: dict[str, object]) -> bool:
        summary = trial.get("summary", {})
        if not isinstance(summary, dict):
            return False
        replicate_summary = summary.get("replicate_summary")
        if not isinstance(replicate_summary, dict):
            return self.orchestrator.policy.agent_trial_replicates <= 1
        return bool(replicate_summary.get("stable"))

    def _trial_cases(self, case_id: str | None) -> list[Case]:
        cases: list[Case] = []
        allowed_splits = set(self.orchestrator.policy.agent_trial_eval_splits)
        for case in self.orchestrator.dataset.cases:
            if case.split.value in allowed_splits:
                cases.append(case)
        if case_id:
            focus = self._case(case_id)
            cases = [focus] + [case for case in cases if case.case_id != focus.case_id]
        return cases

    def _case(self, case_id: str) -> Case:
        for case in self.orchestrator.dataset.cases:
            if case.case_id == case_id:
                return case
        raise ValueError(f"unknown case_id: {case_id}")

    def _patch(self, instruction: str):
        from .models import PatchOperation, PatchTarget, TuningPatch

        return TuningPatch(
            operation=PatchOperation.APPEND_INSTRUCTION,
            target=PatchTarget(
                procedure_csv_base_id=self.base_csv_id,
                row_selector=self.row_selector,
                column="additional_instruction",
            ),
            text=instruction,
        )

    def _human_result_visible(self, case: Case) -> bool:
        if case.split == Split.HOLDOUT and not self.orchestrator.policy.holdout_reference_visible:
            return False
        return case.split.value in set(self.orchestrator.policy.human_reference_splits)

    def _safe_metadata(self, case: Case) -> dict[str, object]:
        blocked = {
            "expected_output",
            "human_result",
            "human_result_text",
            "reference_answer",
            "required_claim_keywords",
            "citations",
            "required_capability",
        }
        return {key: value for key, value in case.metadata.items() if key not in blocked}

    def _baseline_observation(self, case_id: str) -> dict[str, object]:
        with self.orchestrator.registry.connect() as conn:
            row = conn.execute(
                """
                SELECT cr.normalized_output_json,
                       MAX(CASE WHEN er.evaluator_name='total_score' THEN er.score END) AS total_score,
                       MAX(CASE WHEN er.evaluator_name='judgement_match' THEN er.score END) AS judgement_score,
                       MAX(CASE WHEN er.evaluator_name='rationale_support' THEN er.score END) AS rationale_score,
                       MAX(CASE WHEN er.evaluator_name='citation_quality' THEN er.score END) AS citation_score,
                       MAX(CASE WHEN er.evaluator_name='unsupported_claim_rate' THEN er.score END) AS unsupported_claim_rate
                FROM case_runs cr
                LEFT JOIN evaluation_results er ON er.run_id = cr.run_id
                WHERE cr.tuning_id = 'baseline'
                  AND cr.case_id = ?
                GROUP BY cr.run_id
                ORDER BY cr.created_at DESC
                LIMIT 1
                """,
                (case_id,),
            ).fetchone()
        if row is None:
            return {}
        return {
            "normalized_output": json.loads(row["normalized_output_json"] or "{}"),
            "scores": {
                "total_score": row["total_score"],
                "judgement_score": row["judgement_score"],
                "rationale_score": row["rationale_score"],
                "citation_score": row["citation_score"],
                "unsupported_claim_rate": row["unsupported_claim_rate"],
            },
        }


def _read_text_artifact(path: str | Path, *, max_chars: int) -> str:
    target = Path(path)
    if target.is_file():
        return target.read_text(encoding="utf-8-sig")[:max_chars]
    parts: list[str] = []
    for child in sorted(target.rglob("*")):
        if child.is_file():
            remaining = max_chars - sum(len(part) for part in parts)
            if remaining <= 0:
                break
            text = child.read_text(encoding="utf-8-sig", errors="replace")[:remaining]
            parts.append(f"# {child.relative_to(target)}\n{text}")
    return "\n\n".join(parts)


def _hash_file(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _hash_path(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return _hash_file(target)
    digest = hashlib.sha256()
    for child in sorted(target.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(target)).encode("utf-8"))
            digest.update(child.read_bytes())
    return "sha256:" + digest.hexdigest()
