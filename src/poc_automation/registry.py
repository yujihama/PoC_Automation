"""SQLite-backed experiment registry.

Langfuse is used for traces and score visualization.  This registry is the
canonical search ledger: candidate lineage, patch contents, batch metadata,
case-level outputs, labels, and promotion decisions.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .models import (
    EvaluationResult,
    ExperimentSummary,
    JsonDict,
    TuningCandidate,
    TuningStatus,
    to_jsonable,
    tuning_candidate_from_json,
    utcnow_iso,
)


@dataclass(frozen=True)
class CaseRunRecord:
    run_id: str
    experiment_id: str
    case_id: str
    tuning_id: str
    status: str
    normalized_output_json: JsonDict | None
    score_json: JsonDict


class ExperimentRegistry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tuning_candidates (
                  tuning_id TEXT PRIMARY KEY,
                  parent_tuning_ids TEXT NOT NULL DEFAULT '[]',
                  scope TEXT NOT NULL,
                  status TEXT NOT NULL,
                  patch_json TEXT,
                  instruction_text TEXT NOT NULL,
                  hypothesis TEXT,
                  generated_by TEXT,
                  generator_prompt_version TEXT,
                  created_at TEXT NOT NULL,
                  labels_json TEXT NOT NULL DEFAULT '{}',
                  risk_labels_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS experiment_batches (
                  experiment_id TEXT PRIMARY KEY,
                  search_iteration INTEGER NOT NULL,
                  dataset_snapshot_id TEXT NOT NULL,
                  split TEXT NOT NULL,
                  search_policy_version TEXT NOT NULL,
                  evaluator_policy_version TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  langfuse_dataset_run_id TEXT,
                  notes TEXT
                );

                CREATE TABLE IF NOT EXISTS case_runs (
                  run_id TEXT PRIMARY KEY,
                  experiment_id TEXT NOT NULL,
                  case_id TEXT NOT NULL,
                  tuning_id TEXT NOT NULL,
                  app_run_id TEXT,
                  langfuse_trace_id TEXT,
                  base_csv_hash TEXT,
                  materialized_csv_hash TEXT,
                  evidence_bundle_hash TEXT,
                  app_version TEXT,
                  model_version TEXT,
                  status TEXT NOT NULL,
                  latency_ms INTEGER,
                  cost_json TEXT NOT NULL DEFAULT '{}',
                  raw_output_uri TEXT,
                  normalized_output_json TEXT,
                  error_message TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(tuning_id) REFERENCES tuning_candidates(tuning_id)
                );

                CREATE TABLE IF NOT EXISTS evaluation_results (
                  eval_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  evaluator_name TEXT NOT NULL,
                  evaluator_version TEXT NOT NULL,
                  score REAL,
                  label TEXT,
                  comment TEXT,
                  details_json TEXT NOT NULL DEFAULT '{}',
                  langfuse_score_id TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(run_id) REFERENCES case_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS tuning_effects (
                  tuning_id TEXT NOT NULL,
                  experiment_id TEXT NOT NULL,
                  split TEXT NOT NULL,
                  total_score_delta REAL,
                  judgement_delta REAL,
                  rationale_delta REAL,
                  citation_delta REAL,
                  regression_count INTEGER,
                  positive_count INTEGER,
                  negative_count INTEGER,
                  generality_score REAL,
                  overfit_risk REAL,
                  effect_label TEXT,
                  details_json TEXT NOT NULL DEFAULT '{}',
                  PRIMARY KEY (tuning_id, experiment_id)
                );

                CREATE TABLE IF NOT EXISTS tuning_atoms (
                  atom_id TEXT PRIMARY KEY,
                  source_tuning_id TEXT NOT NULL,
                  text TEXT NOT NULL,
                  tactic_type TEXT,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS promotion_decisions (
                  decision_id TEXT PRIMARY KEY,
                  tuning_id TEXT NOT NULL,
                  from_scope TEXT NOT NULL,
                  to_scope TEXT NOT NULL,
                  decision TEXT NOT NULL,
                  reason TEXT,
                  policy_version TEXT NOT NULL,
                  validation_result_json TEXT NOT NULL DEFAULT '{}',
                  holdout_result_json TEXT NOT NULL DEFAULT '{}',
                  approved_by TEXT,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_trial_observations (
                  trial_id TEXT PRIMARY KEY,
                  search_iteration INTEGER NOT NULL,
                  draft_index INTEGER NOT NULL,
                  agent_name TEXT NOT NULL,
                  tool_name TEXT NOT NULL,
                  instruction_text TEXT NOT NULL,
                  hypothesis TEXT,
                  case_ids_json TEXT NOT NULL DEFAULT '[]',
                  splits_json TEXT NOT NULL DEFAULT '[]',
                  summary_json TEXT NOT NULL DEFAULT '{}',
                  case_results_json TEXT NOT NULL DEFAULT '[]',
                  status TEXT NOT NULL,
                  error_message TEXT,
                  created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_case_runs_experiment ON case_runs(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_case_runs_tuning ON case_runs(tuning_id);
                CREATE INDEX IF NOT EXISTS idx_eval_run ON evaluation_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_effect_label ON tuning_effects(effect_label);
                CREATE INDEX IF NOT EXISTS idx_agent_trials_iteration ON agent_trial_observations(search_iteration);
                """
            )

    def add_candidate(self, candidate: TuningCandidate) -> None:
        payload = to_jsonable(candidate)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tuning_candidates (
                  tuning_id, parent_tuning_ids, scope, status, patch_json,
                  instruction_text, hypothesis, generated_by, generator_prompt_version,
                  created_at, labels_json, risk_labels_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.tuning_id,
                    json.dumps(candidate.parent_tuning_ids, ensure_ascii=False),
                    candidate.scope.value,
                    candidate.status.value,
                    json.dumps(payload.get("patch"), ensure_ascii=False),
                    candidate.instruction_text,
                    candidate.hypothesis,
                    candidate.generated_by,
                    candidate.generator_prompt_version,
                    candidate.created_at,
                    json.dumps(candidate.labels, ensure_ascii=False),
                    json.dumps(candidate.risk_labels, ensure_ascii=False),
                ),
            )

    def get_candidate(self, tuning_id: str) -> TuningCandidate | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM tuning_candidates WHERE tuning_id = ?", (tuning_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_candidate(row)

    def list_candidates(
        self,
        *,
        status: TuningStatus | str | None = None,
        limit: int = 200,
    ) -> list[TuningCandidate]:
        params: list[object] = []
        query = "SELECT * FROM tuning_candidates"
        if status is not None:
            status_value = status.value if isinstance(status, TuningStatus) else status
            query += " WHERE status = ?"
            params.append(status_value)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def update_candidate_status(self, tuning_id: str, status: TuningStatus | str) -> None:
        value = status.value if isinstance(status, TuningStatus) else status
        with self.connect() as conn:
            conn.execute("UPDATE tuning_candidates SET status = ? WHERE tuning_id = ?", (value, tuning_id))

    def create_experiment_batch(
        self,
        *,
        search_iteration: int,
        dataset_snapshot_id: str,
        split: str,
        search_policy_version: str = "default",
        evaluator_policy_version: str = "default",
        notes: str | None = None,
    ) -> str:
        experiment_id = f"exp_{search_iteration:04d}_{split}_{uuid.uuid4().hex[:8]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_batches (
                  experiment_id, search_iteration, dataset_snapshot_id, split,
                  search_policy_version, evaluator_policy_version, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    search_iteration,
                    dataset_snapshot_id,
                    split,
                    search_policy_version,
                    evaluator_policy_version,
                    utcnow_iso(),
                    notes,
                ),
            )
        return experiment_id

    def save_case_run(
        self,
        *,
        experiment_id: str,
        case_id: str,
        tuning_id: str,
        status: str,
        app_run_id: str | None = None,
        langfuse_trace_id: str | None = None,
        base_csv_hash: str | None = None,
        materialized_csv_hash: str | None = None,
        evidence_bundle_hash: str | None = None,
        app_version: str | None = None,
        model_version: str | None = None,
        latency_ms: int | None = None,
        cost_json: JsonDict | None = None,
        raw_output_uri: str | None = None,
        normalized_output_json: JsonDict | None = None,
        error_message: str | None = None,
    ) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO case_runs (
                  run_id, experiment_id, case_id, tuning_id, app_run_id, langfuse_trace_id,
                  base_csv_hash, materialized_csv_hash, evidence_bundle_hash, app_version,
                  model_version, status, latency_ms, cost_json, raw_output_uri,
                  normalized_output_json, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_id,
                    case_id,
                    tuning_id,
                    app_run_id,
                    langfuse_trace_id,
                    base_csv_hash,
                    materialized_csv_hash,
                    evidence_bundle_hash,
                    app_version,
                    model_version,
                    status,
                    latency_ms,
                    json.dumps(cost_json or {}, ensure_ascii=False),
                    raw_output_uri,
                    json.dumps(normalized_output_json, ensure_ascii=False)
                    if normalized_output_json is not None
                    else None,
                    error_message,
                    utcnow_iso(),
                ),
            )
        return run_id

    def save_evaluation_result(self, run_id: str, result: EvaluationResult) -> str:
        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_results (
                  eval_id, run_id, evaluator_name, evaluator_version, score, label,
                  comment, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_id,
                    run_id,
                    result.evaluator_name,
                    result.evaluator_version,
                    result.score,
                    result.label,
                    result.comment,
                    json.dumps(result.details, ensure_ascii=False),
                    utcnow_iso(),
                ),
            )
        return eval_id

    def save_tuning_effect(
        self,
        *,
        tuning_id: str,
        experiment_id: str,
        split: str,
        total_score_delta: float,
        judgement_delta: float = 0.0,
        rationale_delta: float = 0.0,
        citation_delta: float = 0.0,
        regression_count: int = 0,
        positive_count: int = 0,
        negative_count: int = 0,
        generality_score: float = 0.0,
        overfit_risk: float = 0.0,
        effect_label: str = "neutral",
        details: JsonDict | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tuning_effects (
                  tuning_id, experiment_id, split, total_score_delta, judgement_delta,
                  rationale_delta, citation_delta, regression_count, positive_count,
                  negative_count, generality_score, overfit_risk, effect_label, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tuning_id,
                    experiment_id,
                    split,
                    total_score_delta,
                    judgement_delta,
                    rationale_delta,
                    citation_delta,
                    regression_count,
                    positive_count,
                    negative_count,
                    generality_score,
                    overfit_risk,
                    effect_label,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )

    def save_tuning_atom(self, source_tuning_id: str, text: str, tactic_type: str | None = None) -> str:
        atom_id = f"atom_{uuid.uuid4().hex[:12]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tuning_atoms (atom_id, source_tuning_id, text, tactic_type, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (atom_id, source_tuning_id, text, tactic_type, utcnow_iso()),
            )
        return atom_id

    def save_promotion_decision(
        self,
        *,
        tuning_id: str,
        from_scope: str,
        to_scope: str,
        decision: str,
        reason: str,
        policy_version: str = "default",
        validation_result: JsonDict | None = None,
        holdout_result: JsonDict | None = None,
        approved_by: str | None = None,
    ) -> str:
        decision_id = f"promo_{uuid.uuid4().hex[:12]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO promotion_decisions (
                  decision_id, tuning_id, from_scope, to_scope, decision, reason,
                  policy_version, validation_result_json, holdout_result_json, approved_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    tuning_id,
                    from_scope,
                    to_scope,
                    decision,
                    reason,
                    policy_version,
                    json.dumps(validation_result or {}, ensure_ascii=False),
                    json.dumps(holdout_result or {}, ensure_ascii=False),
                    approved_by,
                    utcnow_iso(),
                ),
            )
        return decision_id

    def save_agent_trial_observation(
        self,
        *,
        search_iteration: int,
        draft_index: int,
        agent_name: str,
        tool_name: str,
        instruction_text: str,
        hypothesis: str = "",
        case_ids: list[str] | None = None,
        splits: list[str] | None = None,
        summary: JsonDict | None = None,
        case_results: list[JsonDict] | None = None,
        status: str = "succeeded",
        error_message: str | None = None,
    ) -> str:
        trial_id = f"trial_{search_iteration:04d}_{draft_index:02d}_{uuid.uuid4().hex[:8]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_trial_observations (
                  trial_id, search_iteration, draft_index, agent_name, tool_name,
                  instruction_text, hypothesis, case_ids_json, splits_json,
                  summary_json, case_results_json, status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    search_iteration,
                    draft_index,
                    agent_name,
                    tool_name,
                    instruction_text,
                    hypothesis,
                    json.dumps(case_ids or [], ensure_ascii=False),
                    json.dumps(splits or [], ensure_ascii=False),
                    json.dumps(summary or {}, ensure_ascii=False),
                    json.dumps(case_results or [], ensure_ascii=False),
                    status,
                    error_message,
                    utcnow_iso(),
                ),
            )
        return trial_id

    def list_case_runs(self, experiment_id: str) -> list[CaseRunRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT cr.*, COALESCE(json_group_object(er.evaluator_name, er.score), '{}') AS score_json
                FROM case_runs cr
                LEFT JOIN evaluation_results er ON cr.run_id = er.run_id
                WHERE cr.experiment_id = ?
                GROUP BY cr.run_id
                ORDER BY cr.case_id
                """,
                (experiment_id,),
            ).fetchall()
        return [
            CaseRunRecord(
                run_id=row["run_id"],
                experiment_id=row["experiment_id"],
                case_id=row["case_id"],
                tuning_id=row["tuning_id"],
                status=row["status"],
                normalized_output_json=json.loads(row["normalized_output_json"])
                if row["normalized_output_json"]
                else None,
                score_json=json.loads(row["score_json"] or "{}"),
            )
            for row in rows
        ]

    def summarize_experiment(self, experiment_id: str, tuning_id: str, split: str) -> ExperimentSummary:
        runs = self.list_case_runs(experiment_id)
        total_scores = [float(run.score_json.get("total_score", 0.0) or 0.0) for run in runs]
        metric_names = sorted({key for run in runs for key in run.score_json.keys()})
        metric_means: JsonDict = {}
        for metric in metric_names:
            values = [float(run.score_json.get(metric, 0.0) or 0.0) for run in runs]
            metric_means[metric] = sum(values) / len(values) if values else 0.0
        return ExperimentSummary(
            experiment_id=experiment_id,
            tuning_id=tuning_id,
            split=split,
            total_score_mean=sum(total_scores) / len(total_scores) if total_scores else 0.0,
            case_count=len(total_scores),
            regression_count=0,
            positive_count=0,
            negative_count=0,
            metric_means=metric_means,
        )

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> TuningCandidate:
        data = {
            "tuning_id": row["tuning_id"],
            "parent_tuning_ids": json.loads(row["parent_tuning_ids"] or "[]"),
            "scope": row["scope"],
            "status": row["status"],
            "patch": json.loads(row["patch_json"]) if row["patch_json"] else None,
            "hypothesis": row["hypothesis"] or "",
            "generated_by": row["generated_by"] or "unknown",
            "generator_prompt_version": row["generator_prompt_version"] or "unknown",
            "labels": json.loads(row["labels_json"] or "{}"),
            "risk_labels": json.loads(row["risk_labels_json"] or "{}"),
            "created_at": row["created_at"],
        }
        return tuning_candidate_from_json(data)
