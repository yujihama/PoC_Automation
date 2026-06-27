from pathlib import Path

from poc_automation.agents import HeuristicTuningAgent
from poc_automation.artifacts import LocalArtifactStore
from poc_automation.config import SearchPolicy
from poc_automation.dataset import load_dataset_manifest
from poc_automation.evaluators import EvaluatorSuite
from poc_automation.registry import ExperimentRegistry
from poc_automation.reporting import export_full_run_report
from poc_automation.runner import MockPocAppRunner
from poc_automation.search import SearchOrchestrator


def test_full_run_report_includes_target_iterations_and_final_result(tmp_path: Path):
    dataset_path = "examples/dataset.json"
    dataset = load_dataset_manifest(dataset_path)
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    report = SearchOrchestrator(
        dataset=dataset,
        registry=registry,
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        runner=MockPocAppRunner(),
        agent=HeuristicTuningAgent(),
        evaluator_suite=EvaluatorSuite(),
        policy=SearchPolicy(iterations=1, candidates_per_iteration=2, cheap_sample_size=2),
    ).run()

    output_path = export_full_run_report(
        registry,
        tmp_path / "run_report.md",
        dataset_path=dataset_path,
        run_report={
            "generated_candidates": report.generated_candidates,
            "evaluated_candidates": report.evaluated_candidates,
            "positive_candidates": report.positive_candidates,
            "promoted_candidates": report.promoted_candidates,
        },
    )

    text = Path(output_path).read_text(encoding="utf-8")
    assert "## 対象ケース" in text
    assert "## 全イテレーション結果" in text
    assert "## 最終結果 / 昇格判断" in text
    assert "formal_evaluation_iterations" in text
    assert "agent_trial_rounds" in text
    assert "max_iteration:" not in text
    assert "case_001" in text
    assert "baseline" in text
