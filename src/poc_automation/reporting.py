"""Markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .dataset import Dataset, load_dataset_manifest
from .registry import ExperimentRegistry


def export_markdown_report(registry: ExperimentRegistry, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with registry.connect() as conn:
        candidates = conn.execute(
            """
            SELECT tc.tuning_id, tc.scope, tc.status, tc.hypothesis, tc.instruction_text,
                   COALESCE(MAX(te.total_score_delta), 0) AS best_delta,
                   COALESCE(MAX(te.effect_label), 'not_evaluated') AS effect_label
            FROM tuning_candidates tc
            LEFT JOIN tuning_effects te ON tc.tuning_id = te.tuning_id
            GROUP BY tc.tuning_id
            ORDER BY best_delta DESC, tc.created_at DESC
            """
        ).fetchall()
        promotions = conn.execute(
            "SELECT * FROM promotion_decisions ORDER BY created_at DESC"
        ).fetchall()

    lines = [
        "# PoCチューニング探索レポート",
        "",
        "## 候補ランキング",
        "",
        "| tuning_id | scope | status | best_delta | label | hypothesis |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        lines.append(
            f"| `{row['tuning_id']}` | {row['scope']} | {row['status']} | {float(row['best_delta']):.4f} | {row['effect_label']} | {row['hypothesis'] or ''} |"
        )
    lines.extend(["", "## 昇格判断", ""])
    if not promotions:
        lines.append("昇格判断はまだありません。")
    else:
        lines.append("| tuning_id | from | to | decision | reason | created_at |")
        lines.append("|---|---:|---:|---:|---|---:|")
        for row in promotions:
            lines.append(
                f"| `{row['tuning_id']}` | {row['from_scope']} | {row['to_scope']} | {row['decision']} | {row['reason']} | {row['created_at']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def export_full_run_report(
    registry: ExperimentRegistry,
    output_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    run_report: Mapping[str, object] | None = None,
) -> str:
    """Export a single Markdown file covering target data, all iterations, and final results."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_manifest(dataset_path) if dataset_path else None

    with registry.connect() as conn:
        overview = _fetch_overview(conn)
        experiments = conn.execute(_EXPERIMENT_SQL).fetchall()
        candidate_summaries = conn.execute(_CANDIDATE_SUMMARY_SQL).fetchall()
        promotions = conn.execute(_PROMOTION_SQL).fetchall()
        case_runs = conn.execute(_CASE_RUN_SQL).fetchall()
        effect_counts = conn.execute(_EFFECT_COUNTS_SQL).fetchall()
        agent_trials = conn.execute(_AGENT_TRIAL_SQL).fetchall()
        usage = conn.execute(_USAGE_SQL).fetchone()

    lines: list[str] = []
    lines.extend(_section_header("PoCチューニング探索 詳細レポート", level=1))
    lines.extend(_overview_lines(overview, run_report, usage, dataset))
    lines.extend(_target_lines(dataset))
    lines.extend(_procedure_lines(dataset))
    lines.extend(_autonomy_assessment_lines(agent_trials, candidate_summaries))
    lines.extend(_agent_trial_lines(agent_trials))
    lines.extend(_iteration_lines(experiments))
    lines.extend(_candidate_summary_lines(candidate_summaries))
    lines.extend(_promotion_lines(promotions))
    lines.extend(_effect_count_lines(effect_counts))
    lines.extend(_case_run_lines(case_runs))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path)


def _fetch_overview(conn) -> dict[str, object]:
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM tuning_candidates) AS candidate_count,
          (SELECT COUNT(*) FROM tuning_candidates WHERE tuning_id <> 'baseline') AS non_baseline_candidates,
          (SELECT COUNT(*) FROM tuning_candidates WHERE status = 'rejected' AND risk_labels_json LIKE '%duplicate_candidate%') AS duplicate_candidate_count,
          (SELECT COUNT(*) FROM experiment_batches) AS experiment_count,
          (SELECT COALESCE(MAX(search_iteration), 0) FROM experiment_batches) AS max_iteration,
          (SELECT COUNT(*) FROM case_runs) AS case_run_count,
          (SELECT COUNT(*) FROM case_runs WHERE status <> 'succeeded') AS failed_case_runs,
          (SELECT COUNT(DISTINCT case_id) FROM case_runs WHERE tuning_id = 'baseline') AS baseline_case_count,
          (SELECT COUNT(*) FROM promotion_decisions WHERE decision = 'promote_candidate') AS promotion_count,
          (SELECT COUNT(*) FROM promotion_decisions WHERE decision = 'needs_more_validation') AS needs_more_validation_count,
          (SELECT COUNT(*) FROM agent_trial_observations) AS agent_trial_count
        """
    ).fetchone()
    return dict(row) if row else {}


def _overview_lines(
    overview: Mapping[str, object],
    run_report: Mapping[str, object] | None,
    usage,
    dataset: Dataset | None,
) -> list[str]:
    generated = _optional_number(run_report, "generated_candidates")
    evaluated = _optional_number(run_report, "evaluated_candidates")
    positive = _optional_number(run_report, "positive_candidates")
    promoted = _optional_number(run_report, "promoted_candidates")
    skipped_duplicates = _optional_number(run_report, "skipped_duplicate_candidates")
    needs_more_validation = _optional_number(run_report, "needs_more_validation_candidates")
    baseline_case_count = _optional_number(run_report, "baseline_case_count")
    input_tokens = int(usage["input_tokens"] or 0) if usage else 0
    output_tokens = int(usage["output_tokens"] or 0) if usage else 0
    total_tokens = int(usage["total_tokens"] or 0) if usage else 0
    latency_ms = int(usage["latency_ms"] or 0) if usage else 0

    lines = _section_header("実行サマリ")
    lines.extend(
        [
            f"- dataset: `{_md(dataset.dataset_id if dataset else 'unknown')}`",
            f"- snapshot: `{_md(dataset.snapshot_id if dataset else 'unknown')}`",
            f"- agent: `{_md(run_report.get('agent', 'unknown') if run_report else 'unknown')}`",
            f"- runner: `{_md(run_report.get('runner', 'unknown') if run_report else 'unknown')}`",
            f"- provider: `{_md(run_report.get('provider', 'unknown') if run_report else 'unknown')}`",
            f"- model: `{_md(run_report.get('model', 'unknown') if run_report else 'unknown')}`",
            f"- candidate_provider: `{_md(run_report.get('candidate_provider', 'unknown') if run_report else 'unknown')}`",
            f"- candidate_model: `{_md(run_report.get('candidate_model', 'unknown') if run_report else 'unknown')}`",
            f"- max_iteration: `{int(overview.get('max_iteration') or 0)}`",
            f"- experiments: `{int(overview.get('experiment_count') or 0)}`",
            f"- candidates: `{int(overview.get('candidate_count') or 0)}` "
            f"(non-baseline `{int(overview.get('non_baseline_candidates') or 0)}`)",
            f"- duplicate_candidates_skipped: `{int(overview.get('duplicate_candidate_count') or 0)}`",
            f"- baseline_cases: `{int(overview.get('baseline_case_count') or 0)}`",
            f"- case_runs: `{int(overview.get('case_run_count') or 0)}` "
            f"(failed `{int(overview.get('failed_case_runs') or 0)}`)",
            f"- promotions: `{int(overview.get('promotion_count') or 0)}`",
            f"- needs_more_validation: `{int(overview.get('needs_more_validation_count') or 0)}`",
            f"- agent_trial_tool_calls: `{int(overview.get('agent_trial_count') or 0)}`",
        ]
    )
    if run_report:
        lines.extend(
            [
                f"- generated_candidates: `{generated}`",
                f"- evaluated_candidates: `{evaluated}`",
                f"- positive_candidates: `{positive}`",
                f"- promoted_candidates: `{promoted}`",
                f"- skipped_duplicate_candidates: `{skipped_duplicates}`",
                f"- needs_more_validation_candidates: `{needs_more_validation}`",
                f"- baseline_case_count: `{baseline_case_count}`",
                f"- data_visibility_policy: `{_md(run_report.get('data_visibility_policy', 'unknown'))}`",
                f"- human_reference_splits: `{_md(run_report.get('human_reference_splits', []))}`",
                f"- agent_trial_eval_splits: `{_md(run_report.get('agent_trial_eval_splits', []))}`",
                f"- per_case_trial_budget: `{_md(run_report.get('per_case_trial_budget', ''))}`",
            ]
        )
    lines.extend(
        [
            f"- OpenRouter input_tokens: `{input_tokens}`",
            f"- OpenRouter output_tokens: `{output_tokens}`",
            f"- OpenRouter total_tokens: `{total_tokens}`",
            f"- accumulated_latency_minutes: `{latency_ms / 60000:.2f}`",
            "",
        ]
    )
    return lines


def _target_lines(dataset: Dataset | None) -> list[str]:
    lines = _section_header("対象ケース")
    if dataset is None:
        lines.append("dataset path が指定されていないため、対象ケース詳細は出力できません。")
        lines.append("")
        return lines

    lines.extend(
        [
            "| case_id | split | domain | procedure_family | expected | required_keywords | expected_citations | evidence |",
            "|---|---:|---|---|---:|---|---|---|",
        ]
    )
    for case in dataset.cases:
        metadata = case.metadata
        citations = ", ".join(
            f"{citation.evidence_id}:{citation.page or ''}" for citation in case.expected_output.citations
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(case.case_id)}`",
                    _md(case.split.value),
                    _md(str(metadata.get("domain", ""))),
                    _md(str(metadata.get("procedure_family", ""))),
                    _md(case.expected_output.judgement),
                    _md(", ".join(case.expected_output.required_claim_keywords)),
                    _md(citations),
                    f"`{_md(case.evidence_bundle_path)}`",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _procedure_lines(dataset: Dataset | None) -> list[str]:
    lines = _section_header("評価手続")
    if dataset is None:
        lines.append("dataset path が指定されていないため、手続CSVは出力できません。")
        lines.append("")
        return lines

    procedure_paths = sorted({case.procedure_csv_path for case in dataset.cases})
    for procedure_path in procedure_paths:
        path = Path(procedure_path)
        lines.append(f"### `{_md(str(path))}`")
        if not path.is_file():
            lines.append("手続CSVが見つかりません。")
            lines.append("")
            continue
        lines.append("")
        lines.append("```csv")
        lines.append(path.read_text(encoding="utf-8-sig").rstrip())
        lines.append("```")
        lines.append("")
    return lines


def _autonomy_assessment_lines(agent_trials, candidate_summaries) -> list[str]:
    lines = _section_header("Autonomous Exploration Assessment")
    if not agent_trials:
        lines.extend(["- autonomous_trial_tool_calls: `0`", "- assessment: `not observed`", ""])
        return lines

    iterations = {int(row["search_iteration"]) for row in agent_trials}
    positive_trials = 0
    regression_trials = 0
    multi_case_trials = 0
    for row in agent_trials:
        summary = _json_obj(row["summary_json"])
        if float(summary.get("delta_mean", 0.0) or 0.0) > 0:
            positive_trials += 1
        if int(summary.get("regression_count", 0) or 0) > 0:
            regression_trials += 1
        if len(_json_list(row["case_ids_json"])) >= 2:
            multi_case_trials += 1

    accepted_with_trials = 0
    accepted_generalized = 0
    for row in candidate_summaries:
        if row["tuning_id"] == "baseline" or row["status"] == "rejected":
            continue
        labels = _json_obj(row["labels_json"])
        if labels.get("source_trial_ids"):
            accepted_with_trials += 1
        if labels.get("generalized_from_cases"):
            accepted_generalized += 1

    assessment = "observed" if agent_trials and accepted_with_trials else "partial"
    lines.extend(
        [
            f"- assessment: `{assessment}`",
            f"- autonomous_trial_tool_calls: `{len(agent_trials)}`",
            f"- iterations_with_trials: `{len(iterations)}`",
            f"- positive_trial_rows: `{positive_trials}`",
            f"- trial_rows_with_regressions: `{regression_trials}`",
            f"- multi_case_trial_rows: `{multi_case_trials}`",
            f"- accepted_candidates_linked_to_trials: `{accepted_with_trials}`",
            f"- accepted_candidates_with_generalized_cases: `{accepted_generalized}`",
        ]
    )
    if multi_case_trials == 0:
        lines.append(
            "- caveat: draft evaluations were case-level; cross-case behavior is represented by final candidate synthesis labels."
        )
    lines.append("")
    return lines


def _agent_trial_lines(rows) -> list[str]:
    lines = _section_header("Agent Trial Observations")
    if not rows:
        lines.append("No agent-internal draft trials were recorded.")
        lines.append("")
        return lines
    lines.append(
        f"Autonomous draft evaluation observed: `{'yes' if len(rows) > 0 else 'no'}` "
        f"({len(rows)} tool call rows)."
    )
    lines.append("")
    lines.extend(
        [
            "| iter | draft | trial_id | status | cases | mean_total | delta | pos | neg | reg | instruction | hypothesis |",
            "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        summary = _json_obj(row["summary_json"])
        case_ids = _json_list(row["case_ids_json"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["search_iteration"]),
                    str(row["draft_index"]),
                    f"`{_md(row['trial_id'])}`",
                    _md(row["status"]),
                    _md(", ".join(str(item) for item in case_ids)),
                    _fmt(summary.get("total_score_mean")),
                    _fmt(summary.get("delta_mean")),
                    str(summary.get("positive_count", "")),
                    str(summary.get("negative_count", "")),
                    str(summary.get("regression_count", "")),
                    _md(_short(row["instruction_text"] or "", 150)),
                    _md(_short(row["hypothesis"] or "", 120)),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _iteration_lines(rows) -> list[str]:
    lines = _section_header("全イテレーション結果")
    if not rows:
        lines.append("実験結果はまだありません。")
        lines.append("")
        return lines
    lines.extend(
        [
            "| iter | split | experiment | tuning_id | hypothesis | mean_total | delta | label | cases | pos | neg | reg | failed | instruction |",
            "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["search_iteration"]),
                    _md(row["split"]),
                    f"`{_md(row['experiment_id'])}`",
                    f"`{_md(row['tuning_id'] or '')}`",
                    _md(row["hypothesis"] or ""),
                    _fmt(row["mean_total_score"]),
                    _fmt(row["total_score_delta"]),
                    _md(row["effect_label"] or ""),
                    str(row["case_count"] or 0),
                    str(row["positive_count"] or 0),
                    str(row["negative_count"] or 0),
                    str(row["regression_count"] or 0),
                    str(row["failed_cases"] or 0),
                    _md(_short(row["instruction_text"] or "", 140)),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _candidate_summary_lines(rows) -> list[str]:
    lines = _section_header("候補別サマリ")
    if not rows:
        lines.append("候補はまだありません。")
        lines.append("")
        return lines
    lines.extend(
        [
            "| tuning_id | scope | status | generated_by | effects | best_delta | avg_delta | worst_delta | positive_effects | regressions | promoted | fingerprint | risk_labels | hypothesis | instruction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        labels = _json_obj(row["labels_json"])
        risk_labels = _json_obj(row["risk_labels_json"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(row['tuning_id'])}`",
                    _md(row["scope"]),
                    _md(row["status"]),
                    _md(row["generated_by"] or ""),
                    str(row["effect_rows"] or 0),
                    _fmt(row["best_delta"]),
                    _fmt(row["avg_delta"]),
                    _fmt(row["worst_delta"]),
                    str(row["positive_effects"] or 0),
                    str(row["regressions"] or 0),
                    str(row["promoted"] or 0),
                    f"`{_md(labels.get('fingerprint', ''))}`",
                    _md(_short(json.dumps(risk_labels, ensure_ascii=False, sort_keys=True), 120)),
                    _md(row["hypothesis"] or ""),
                    _md(_short(row["instruction_text"] or "", 160)),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _promotion_lines(rows) -> list[str]:
    lines = _section_header("最終結果 / 昇格判断")
    if not rows:
        lines.append("昇格判断はありません。")
        lines.append("")
        return lines
    lines.extend(
        [
            "| tuning_id | from | to | decision | validation_mean | holdout_mean | reason | instruction | created_at |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for row in rows:
        validation = _json_obj(row["validation_result_json"])
        holdout = _json_obj(row["holdout_result_json"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_md(row['tuning_id'])}`",
                    _md(row["from_scope"]),
                    _md(row["to_scope"]),
                    _md(row["decision"]),
                    _fmt(validation.get("total_score_mean")),
                    _fmt(holdout.get("total_score_mean")),
                    _md(row["reason"] or ""),
                    _md(_short(row["instruction_text"] or "", 160)),
                    _md(row["created_at"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _effect_count_lines(rows) -> list[str]:
    lines = _section_header("効果ラベル集計")
    if not rows:
        lines.append("効果ラベルはまだありません。")
        lines.append("")
        return lines
    lines.extend(
        [
            "| split | effect_label | count | avg_delta | regressions |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {_md(row['split'])} | {_md(row['effect_label'])} | {row['count']} | "
            f"{_fmt(row['avg_delta'])} | {row['regressions'] or 0} |"
        )
    lines.append("")
    return lines


def _case_run_lines(rows) -> list[str]:
    lines = _section_header("ケース別実行結果")
    if not rows:
        lines.append("ケース実行結果はまだありません。")
        lines.append("")
        return lines
    lines.extend(
        [
            "| iter | split | case_id | tuning_id | status | judgement | total | judgement_score | rationale | citation | unsupported | latency_ms | error |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        normalized = _json_obj(row["normalized_output_json"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["search_iteration"]),
                    _md(row["split"]),
                    f"`{_md(row['case_id'])}`",
                    f"`{_md(row['tuning_id'])}`",
                    _md(row["status"]),
                    _md(str(normalized.get("judgement", ""))),
                    _fmt(row["total_score"]),
                    _fmt(row["judgement_score"]),
                    _fmt(row["rationale_score"]),
                    _fmt(row["citation_score"]),
                    _fmt(row["unsupported_claim_rate"]),
                    str(row["latency_ms"] or ""),
                    _md(_short(row["error_message"] or "", 100)),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _section_header(title: str, *, level: int = 2) -> list[str]:
    return [f"{'#' * level} {title}", ""]


def _optional_number(source: Mapping[str, object] | None, key: str) -> object:
    if not source:
        return "unknown"
    return source.get(key, "unknown")


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _short(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1] + "…"


def _md(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _json_obj(value: object) -> dict[str, object]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_list(value: object) -> list[object]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


_EXPERIMENT_SQL = """
WITH run_scores AS (
  SELECT
    cr.run_id,
    cr.experiment_id,
    cr.tuning_id,
    cr.case_id,
    cr.status,
    cr.latency_ms,
    MAX(CASE WHEN er.evaluator_name='total_score' THEN er.score END) AS total_score
  FROM case_runs cr
  LEFT JOIN evaluation_results er ON er.run_id = cr.run_id
  GROUP BY cr.run_id
)
SELECT
  eb.search_iteration,
  eb.split,
  eb.experiment_id,
  COALESCE(rs.tuning_id, substr(eb.notes, 11)) AS tuning_id,
  tc.hypothesis,
  tc.instruction_text,
  COUNT(rs.run_id) AS case_count,
  SUM(CASE WHEN rs.status <> 'succeeded' THEN 1 ELSE 0 END) AS failed_cases,
  AVG(rs.total_score) AS mean_total_score,
  AVG(rs.latency_ms) AS avg_latency_ms,
  te.total_score_delta,
  te.effect_label,
  te.positive_count,
  te.negative_count,
  te.regression_count
FROM experiment_batches eb
LEFT JOIN run_scores rs ON rs.experiment_id = eb.experiment_id
LEFT JOIN tuning_candidates tc ON tc.tuning_id = COALESCE(rs.tuning_id, substr(eb.notes, 11))
LEFT JOIN tuning_effects te ON te.experiment_id = eb.experiment_id
  AND te.tuning_id = COALESCE(rs.tuning_id, substr(eb.notes, 11))
GROUP BY eb.experiment_id, COALESCE(rs.tuning_id, substr(eb.notes, 11))
ORDER BY eb.search_iteration, eb.created_at, eb.experiment_id
"""


_CANDIDATE_SUMMARY_SQL = """
SELECT
  tc.tuning_id,
  tc.scope,
  tc.status,
  tc.generated_by,
  tc.hypothesis,
  tc.instruction_text,
  tc.labels_json,
  tc.risk_labels_json,
  COUNT(te.experiment_id) AS effect_rows,
  MAX(te.total_score_delta) AS best_delta,
  AVG(te.total_score_delta) AS avg_delta,
  MIN(te.total_score_delta) AS worst_delta,
  SUM(CASE WHEN te.effect_label IN ('positive','strongly_positive') THEN 1 ELSE 0 END) AS positive_effects,
  SUM(COALESCE(te.regression_count, 0)) AS regressions,
  COUNT(DISTINCT pd.decision_id) AS promoted
FROM tuning_candidates tc
LEFT JOIN tuning_effects te ON te.tuning_id = tc.tuning_id
LEFT JOIN promotion_decisions pd ON pd.tuning_id = tc.tuning_id
GROUP BY tc.tuning_id
ORDER BY promoted DESC, best_delta DESC, avg_delta DESC, tc.created_at
"""


_PROMOTION_SQL = """
SELECT
  pd.*,
  tc.hypothesis,
  tc.instruction_text
FROM promotion_decisions pd
JOIN tuning_candidates tc ON tc.tuning_id = pd.tuning_id
ORDER BY pd.created_at
"""


_CASE_RUN_SQL = """
SELECT
  eb.search_iteration,
  eb.split,
  eb.created_at AS experiment_created_at,
  cr.case_id,
  cr.tuning_id,
  cr.status,
  cr.latency_ms,
  cr.normalized_output_json,
  cr.error_message,
  MAX(CASE WHEN er.evaluator_name='total_score' THEN er.score END) AS total_score,
  MAX(CASE WHEN er.evaluator_name='judgement_match' THEN er.score END) AS judgement_score,
  MAX(CASE WHEN er.evaluator_name='rationale_support' THEN er.score END) AS rationale_score,
  MAX(CASE WHEN er.evaluator_name='citation_quality' THEN er.score END) AS citation_score,
  MAX(CASE WHEN er.evaluator_name='unsupported_claim_rate' THEN er.score END) AS unsupported_claim_rate
FROM case_runs cr
JOIN experiment_batches eb ON eb.experiment_id = cr.experiment_id
LEFT JOIN evaluation_results er ON er.run_id = cr.run_id
GROUP BY cr.run_id
ORDER BY eb.search_iteration, eb.created_at, eb.split, cr.case_id, cr.tuning_id
"""


_EFFECT_COUNTS_SQL = """
SELECT
  split,
  effect_label,
  COUNT(*) AS count,
  AVG(total_score_delta) AS avg_delta,
  SUM(regression_count) AS regressions
FROM tuning_effects
GROUP BY split, effect_label
ORDER BY split, effect_label
"""


_AGENT_TRIAL_SQL = """
SELECT
  trial_id,
  search_iteration,
  draft_index,
  agent_name,
  tool_name,
  instruction_text,
  hypothesis,
  case_ids_json,
  splits_json,
  summary_json,
  status,
  error_message,
  created_at
FROM agent_trial_observations
ORDER BY search_iteration, draft_index, created_at
"""


_USAGE_SQL = """
SELECT
  SUM(CAST(json_extract(cost_json, '$.input_tokens') AS INTEGER)) AS input_tokens,
  SUM(CAST(json_extract(cost_json, '$.output_tokens') AS INTEGER)) AS output_tokens,
  SUM(CAST(json_extract(cost_json, '$.total_tokens') AS INTEGER)) AS total_tokens,
  SUM(latency_ms) AS latency_ms
FROM case_runs
"""
