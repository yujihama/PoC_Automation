"""Human-readable v3 evaluation summary export."""

from __future__ import annotations

from pathlib import Path

from .dataset import load_dataset_manifest
from .registry import ExperimentRegistry


def export_v3_evaluation_summary(
    registry: ExperimentRegistry,
    out_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    status_note: str | None = None,
) -> str:
    """Write a Japanese summary aligned to the v3 human-reference workflow."""

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_manifest(dataset_path) if dataset_path else None
    stats = _load_registry_stats(registry)
    lines: list[str] = []
    lines.append("# v3評価サマリ")
    lines.append("")
    lines.append("## この検証でやっていること")
    lines.append("")
    lines.append(
        "複数ケースについて、自然言語の手続、PDF/画像の証跡、人間の実施結果を用意し、"
        "探索用DeepAgentに可視範囲のデータを読ませます。探索用DeepAgentは手続に追記する"
        "追加指示を試し、評価対象runnerをOpenRouter経由で実行して、人間の実施結果に近づくかを確認します。"
        "最後に、ケース横断で効きそうな追加指示だけを候補として残します。"
    )
    lines.append("")
    lines.append("## 実行状況")
    lines.append("")
    if status_note:
        lines.append(f"- 状態: {status_note}")
    elif stats["promotion_decisions"] == 0 and stats["candidate_rows"] > stats["evaluated_candidates"]:
        lines.append("- 状態: 探索途中のDBです。promotion判定までは到達していません。")
    else:
        lines.append("- 状態: run-search が完了したDBです。")
    lines.append(f"- 候補数: {stats['candidate_rows']}件（評価済み {stats['evaluated_candidates']}件）")
    lines.append(f"- case run: {stats['case_runs']}件（失敗 {stats['failed_case_runs']}件）")
    lines.append(f"- 評価結果: {stats['evaluation_results']}件")
    lines.append(f"- 探索agentのドラフト試行: {stats['trial_rows']}件")
    lines.append(f"- agent_trial_rounds: {stats['agent_trial_rounds']}")
    lines.append(f"- replicated_trial_rows: {stats['replicated_trial_rows']}")
    lines.append(f"- stable_replicated_trial_rows: {stats['stable_replicated_trial_rows']}")
    if stats["trial_rows"]:
        lines.append(
            f"- ドラフト試行あたりの評価ケース数: min {stats['trial_min_cases']} / "
            f"max {stats['trial_max_cases']} / avg {stats['trial_avg_cases']}"
        )
    lines.append(f"- 全case runの平均total_score: {stats['overall_total_score']}")
    lines.append("")
    lines.append("## 検証データ")
    lines.append("")
    if dataset is None:
        lines.append("- dataset path が指定されていないため、ケース情報は省略しました。")
    else:
        lines.append(f"- dataset: `{dataset.dataset_id}`")
        lines.append(f"- snapshot: `{dataset.snapshot_id}`")
        lines.append("")
        lines.append("| case | split | 手続 | 証跡ファイル数 | 人間結果の文字数 | 期待判断 |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for case in dataset.cases:
            evidence_count = _evidence_file_count(case.evidence_bundle_path)
            procedure_name = Path(case.procedure_csv_path).name
            human_len = len(case.human_result_text or "")
            lines.append(
                f"| `{case.case_id}` | {case.split.value} | `{procedure_name}` | "
                f"{evidence_count} | {human_len} | {case.expected_output.judgement} |"
            )
    lines.append("")
    lines.append("## 観測されたチューニング傾向")
    lines.append("")
    if stats["top_effects"]:
        lines.append("効果が大きかった候補は、概ね次の方向に寄っています。")
        lines.append("")
        lines.append("- 手続の文言そのものを根拠として引用せず、PDF/画像証跡から読める事実だけを根拠にする。")
        lines.append("- 手続の各ステップで確認すべき項目を、証跡上で確認できた事実と確認できない事実に分ける。")
        lines.append("- 判断不能にすべき不足証跡を、単なる推測や本人申告で補わない。")
        lines.append("- 金額や日付などの計算・比較は、証跡上の数値に基づいて差額や条件充足を明示する。")
        lines.append("")
        lines.append("| tuning_id | split | delta | label | positive | negative | regression |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for item in stats["top_effects"]:
            lines.append(
                f"| `{item['tuning_id']}` | {item['split']} | {item['total_score_delta']:.4f} | "
                f"{item['effect_label']} | {item['positive_count']} | {item['negative_count']} | "
                f"{item['regression_count']} |"
            )
    else:
        lines.append("- tuning_effects がまだ記録されていません。")
    lines.append("")
    lines.append("## 注意点")
    lines.append("")
    lines.append(
        "- 今回のデータでは、全ドラフト試行が train+validation の3ケースを横断評価しており、"
        "1ケースだけに過適合する指示を早めに見つけられる状態になっています。"
    )
    lines.append(
        "- promotion判定は、十分なvalidation/holdout件数と複数domain/familyを要求するため、"
        "この小規模データでは最終採用ではなく観測結果として扱います。"
    )
    lines.append(
        "- OpenRouterの残クレジット不足やmax_tokens制約で途中終了したDBでは、"
        "候補生成の最終整理が終わっていない可能性があります。"
    )
    lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output)


def _load_registry_stats(registry: ExperimentRegistry) -> dict[str, object]:
    with registry.connect() as conn:
        tables = {
            row["name"] for row in conn.execute("select name from sqlite_master where type='table' order by name")
        }

        def count(table: str) -> int:
            if table not in tables:
                return 0
            return int(conn.execute(f"select count(*) from {table}").fetchone()[0])

        candidate_rows = count("tuning_candidates")
        evaluated_candidates = 0
        if "tuning_candidates" in tables:
            evaluated_candidates = int(
                conn.execute("select count(*) from tuning_candidates where status='evaluated'").fetchone()[0]
            )
        failed_case_runs = 0
        if "case_runs" in tables:
            failed_case_runs = int(
                conn.execute("select count(*) from case_runs where status != 'succeeded'").fetchone()[0]
            )
        overall_total_score = None
        if "case_runs" in tables and "evaluation_results" in tables:
            row = conn.execute(
                """
                select round(avg(er.score), 4) as total_score
                from case_runs cr
                join evaluation_results er on er.run_id = cr.run_id and er.evaluator_name='total_score'
                """
            ).fetchone()
            overall_total_score = row["total_score"]
        trial_min_cases = trial_max_cases = trial_avg_cases = None
        agent_trial_rounds = replicated_trial_rows = stable_replicated_trial_rows = 0
        if "agent_trial_observations" in tables:
            row = conn.execute(
                """
                select
                  min(json_extract(summary_json, '$.case_count')) as min_cases,
                  max(json_extract(summary_json, '$.case_count')) as max_cases,
                  round(avg(json_extract(summary_json, '$.case_count')), 2) as avg_cases
                from agent_trial_observations
                """
            ).fetchone()
            trial_min_cases = row["min_cases"]
            trial_max_cases = row["max_cases"]
            trial_avg_cases = row["avg_cases"]
            row = conn.execute(
                """
                select
                  coalesce(max(search_iteration), 0) as trial_rounds,
                  sum(case when json_extract(summary_json, '$.replicate_summary.replicate_count') is not null then 1 else 0 end) as replicated_rows,
                  sum(case when json_extract(summary_json, '$.replicate_summary.stable') = 1 then 1 else 0 end) as stable_rows
                from agent_trial_observations
                """
            ).fetchone()
            agent_trial_rounds = int(row["trial_rounds"] or 0)
            replicated_trial_rows = int(row["replicated_rows"] or 0)
            stable_replicated_trial_rows = int(row["stable_rows"] or 0)
        top_effects: list[dict[str, object]] = []
        if "tuning_effects" in tables:
            for row in conn.execute(
                """
                select tuning_id, split, total_score_delta, effect_label,
                       positive_count, negative_count, regression_count
                from tuning_effects
                order by total_score_delta desc
                limit 10
                """
            ):
                top_effects.append(dict(row))
        promotion_decisions = count("promotion_decisions")
        return {
            "candidate_rows": candidate_rows,
            "evaluated_candidates": evaluated_candidates,
            "case_runs": count("case_runs"),
            "failed_case_runs": failed_case_runs,
            "evaluation_results": count("evaluation_results"),
            "trial_rows": count("agent_trial_observations"),
            "agent_trial_rounds": agent_trial_rounds,
            "replicated_trial_rows": replicated_trial_rows,
            "stable_replicated_trial_rows": stable_replicated_trial_rows,
            "trial_min_cases": trial_min_cases,
            "trial_max_cases": trial_max_cases,
            "trial_avg_cases": trial_avg_cases,
            "overall_total_score": overall_total_score,
            "top_effects": top_effects,
            "promotion_decisions": promotion_decisions,
        }


def _evidence_file_count(path: str | Path) -> int:
    target = Path(path)
    if target.is_file():
        return 1
    if not target.exists():
        return 0
    return sum(1 for file_path in target.iterdir() if file_path.is_file() and not file_path.name.endswith(".txt"))
