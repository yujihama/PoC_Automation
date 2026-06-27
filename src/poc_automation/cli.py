"""Command-line interface for PoC Automation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .agents import build_agent
from .artifacts import LocalArtifactStore
from .config import RuntimeConfig, SearchPolicy
from .csv_patch import CsvMaterializer, PatchValidator
from .dataset import load_dataset_manifest
from .evaluators import EvaluatorSuite
from .langfuse_client import LangfuseReporter
from .models import TuningCandidate, tuning_candidate_from_json, to_jsonable
from .registry import ExperimentRegistry
from .reporting import export_full_run_report, export_markdown_report
from .runner import DeepAgentPocAppRunner, HttpPocAppRunner, MockPocAppRunner
from .search import SearchOrchestrator
from .v3_summary import export_v3_evaluation_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poc-auto", description="PoC tuning automation prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="initialize registry database")
    p_init.add_argument("--db", default=None)

    p_validate = sub.add_parser("validate-patch", help="validate a tuning patch JSON file")
    p_validate.add_argument("--patch", required=True)
    p_validate.add_argument("--base-csv", required=True)
    p_validate.add_argument("--max-chars", type=int, default=400)

    p_materialize = sub.add_parser("materialize-csv", help="apply a tuning candidate to a CSV")
    p_materialize.add_argument("--candidate", required=True)
    p_materialize.add_argument("--base-csv", required=True)
    p_materialize.add_argument("--out", required=True)

    p_run = sub.add_parser("run-search", help="run automated tuning search")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--db", default=None)
    p_run.add_argument("--artifact-dir", default=None)
    p_run.add_argument(
        "--agent",
        default=None,
        choices=[
            "heuristic",
            "deepagent",
            "deepagent-openrouter",
            "openrouter",
            "deepagent-human-ref",
            "human-reference",
            "human-ref",
            "deepagents-code",
            "dcode",
            "cline",
        ],
    )
    p_run.add_argument("--runner", default=None, choices=["mock", "http", "deepagent", "deepagent-openrouter"])
    p_run.add_argument("--iterations", type=int, default=None)
    p_run.add_argument("--candidates-per-iteration", type=int, default=None)
    p_run.add_argument(
        "--runner-parallelism",
        type=int,
        default=None,
        help="number of case runner calls to execute concurrently",
    )
    p_run.add_argument(
        "--trial-replicates",
        type=int,
        default=None,
        help="number of repeated runs for promising human-reference trial drafts",
    )
    p_run.add_argument("--report-out", default=None, help="write the full Markdown report here")
    p_run.add_argument("--no-report", action="store_true", help="skip the automatic full Markdown report")

    p_report = sub.add_parser("export-report", help="export markdown report from registry")
    p_report.add_argument("--db", required=True)
    p_report.add_argument("--out", required=True)

    p_full_report = sub.add_parser("export-full-report", help="export full run report from registry")
    p_full_report.add_argument("--db", required=True)
    p_full_report.add_argument("--out", required=True)
    p_full_report.add_argument("--dataset", default=None)

    p_demo = sub.add_parser("demo", help="run the bundled sample dataset through the full loop")
    p_demo.add_argument("--workspace", default=".tmp/demo")
    p_demo.add_argument("--iterations", type=int, default=2)

    args = parser.parse_args(argv)

    if args.command == "init-db":
        cfg = RuntimeConfig.from_env()
        db = args.db or cfg.db_path
        ExperimentRegistry(db)
        print(f"initialized registry: {db}")
        return 0

    if args.command == "validate-patch":
        candidate = _load_candidate(args.patch)
        report = PatchValidator(max_instruction_chars=args.max_chars).validate(
            candidate.patch, base_csv_path=args.base_csv
        )
        print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))
        return 0 if report.valid else 2

    if args.command == "materialize-csv":
        candidate = _load_candidate(args.candidate)
        result = CsvMaterializer().materialize(args.base_csv, candidate, args.out)
        print(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-search":
        cfg = RuntimeConfig.from_env()
        policy = cfg.search_policy
        if args.iterations is not None:
            policy = replace(policy, iterations=args.iterations)
        if args.candidates_per_iteration is not None:
            policy = replace(policy, candidates_per_iteration=args.candidates_per_iteration)
        if args.runner_parallelism is not None:
            policy = replace(policy, runner_parallelism=max(1, args.runner_parallelism))
        agent_name = args.agent or cfg.agent
        if agent_name in {"deepagent-human-ref", "human-reference", "human-ref"}:
            policy = replace(
                policy,
                allow_neutral_train_probe=True,
                data_visibility_policy="human_reference_v3_train_validation",
                agent_trial_replicates=max(policy.agent_trial_replicates, 3),
            )
        if args.trial_replicates is not None:
            policy = replace(policy, agent_trial_replicates=max(1, args.trial_replicates))
        db_path = args.db or cfg.db_path
        artifact_dir = args.artifact_dir or cfg.artifact_dir
        report = _run_search(
            dataset_path=args.dataset,
            db_path=db_path,
            artifact_dir=artifact_dir,
            agent_name=agent_name,
            runner_name=args.runner or cfg.runner,
            policy=policy,
        )
        runner_name = args.runner or cfg.runner
        provider = cfg.target_agent.provider if runner_name in {"deepagent", "deepagent-openrouter"} else "local"
        model = cfg.target_agent.model if runner_name in {"deepagent", "deepagent-openrouter"} else runner_name
        payload = to_jsonable(report)
        payload.update(
            {
                "agent": agent_name,
                "runner": runner_name,
                "provider": provider,
                "model": model,
                "candidate_provider": cfg.candidate_agent.provider
                if agent_name in {"deepagent", "deepagent-openrouter", "openrouter", "deepagent-human-ref", "human-reference", "human-ref"}
                else "local",
                "candidate_model": cfg.candidate_agent.model
                if agent_name in {"deepagent", "deepagent-openrouter", "openrouter", "deepagent-human-ref", "human-reference", "human-ref"}
                else agent_name,
                "data_visibility_policy": policy.data_visibility_policy,
                "human_reference_splits": list(policy.human_reference_splits),
                "agent_trial_eval_splits": list(policy.agent_trial_eval_splits),
                "per_case_trial_budget": policy.per_case_trial_budget,
                "agent_trial_replicates": policy.agent_trial_replicates,
                "agent_trial_replicate_min_delta_mean": policy.agent_trial_replicate_min_delta_mean,
                "agent_trial_replicate_min_worst_delta": policy.agent_trial_replicate_min_worst_delta,
                "agent_trial_replicate_max_regression_count": policy.agent_trial_replicate_max_regression_count,
                "allow_neutral_train_probe": policy.allow_neutral_train_probe,
                "runner_parallelism": policy.runner_parallelism,
                "db_path": db_path,
                "artifact_dir": artifact_dir,
                "dataset_path": args.dataset,
            }
        )
        if not args.no_report:
            report_path = args.report_out or str(Path(artifact_dir) / "run_report.md")
            payload["full_report"] = export_full_run_report(
                ExperimentRegistry(db_path),
                report_path,
                dataset_path=args.dataset,
                run_report=payload,
            )
            if agent_name in {"deepagent-human-ref", "human-reference", "human-ref"}:
                summary_path = str(Path(report_path).with_name("v3_evaluation_summary.md"))
                payload["v3_evaluation_summary"] = export_v3_evaluation_summary(
                    ExperimentRegistry(db_path),
                    summary_path,
                    dataset_path=args.dataset,
                )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-report":
        out = export_markdown_report(ExperimentRegistry(args.db), args.out)
        print(out)
        return 0

    if args.command == "export-full-report":
        out = export_full_run_report(
            ExperimentRegistry(args.db),
            args.out,
            dataset_path=args.dataset,
        )
        print(out)
        return 0

    if args.command == "demo":
        workspace = Path(args.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        db_path = workspace / "poc_automation.sqlite"
        artifact_dir = workspace / "artifacts"
        report = _run_search(
            dataset_path="examples/dataset.json",
            db_path=str(db_path),
            artifact_dir=str(artifact_dir),
            agent_name="heuristic",
            runner_name="mock",
            policy=SearchPolicy(iterations=args.iterations),
        )
        registry = ExperimentRegistry(db_path)
        report_path = export_markdown_report(registry, workspace / "report.md")
        demo_payload = to_jsonable(report)
        demo_payload.update(
            {
                "agent": "heuristic",
                "runner": "mock",
                "provider": "local",
                "model": "mock",
                "candidate_provider": "local",
                "candidate_model": "heuristic",
                "db_path": str(db_path),
                "artifact_dir": str(artifact_dir),
                "dataset_path": "examples/dataset.json",
            }
        )
        full_report_path = export_full_run_report(
            registry,
            artifact_dir / "run_report.md",
            dataset_path="examples/dataset.json",
            run_report=demo_payload,
        )
        print(
            json.dumps(
                {
                    "search_report": demo_payload,
                    "markdown_report": report_path,
                    "full_report": full_report_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    return 1


def _run_search(
    *,
    dataset_path: str,
    db_path: str,
    artifact_dir: str,
    agent_name: str,
    runner_name: str,
    policy: SearchPolicy,
):
    dataset = load_dataset_manifest(dataset_path)
    registry = ExperimentRegistry(db_path)
    artifacts = LocalArtifactStore(artifact_dir)
    cfg = RuntimeConfig.from_env()
    if runner_name == "mock":
        runner = MockPocAppRunner()
    elif runner_name == "http":
        runner = HttpPocAppRunner(cfg.app_api)
    elif runner_name in {"deepagent", "deepagent-openrouter"}:
        runner = DeepAgentPocAppRunner(cfg.target_agent)
    else:
        raise ValueError(f"unknown runner: {runner_name}")
    orchestrator = SearchOrchestrator(
        dataset=dataset,
        registry=registry,
        artifacts=artifacts,
        runner=runner,
        agent=build_agent(agent_name, cfg.candidate_agent),
        evaluator_suite=EvaluatorSuite(cfg.evaluator_policy),
        langfuse=LangfuseReporter(cfg.langfuse),
        policy=policy,
    )
    return orchestrator.run()


def _load_candidate(path: str) -> TuningCandidate:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuning_candidate_from_json(data)


if __name__ == "__main__":
    raise SystemExit(main())
