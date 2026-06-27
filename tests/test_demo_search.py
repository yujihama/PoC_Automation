from pathlib import Path

from poc_automation.agents import HeuristicTuningAgent
from poc_automation.artifacts import LocalArtifactStore
from poc_automation.config import SearchPolicy
from poc_automation.dataset import load_dataset_manifest
from poc_automation.evaluators import EvaluatorSuite
from poc_automation.registry import ExperimentRegistry
from poc_automation.runner import MockPocAppRunner
from poc_automation.search import SearchOrchestrator


def test_search_orchestrator_runs_sample_dataset(tmp_path: Path):
    dataset = load_dataset_manifest("examples/dataset.json")
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    orchestrator = SearchOrchestrator(
        dataset=dataset,
        registry=registry,
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        runner=MockPocAppRunner(),
        agent=HeuristicTuningAgent(),
        evaluator_suite=EvaluatorSuite(),
        policy=SearchPolicy(iterations=1, candidates_per_iteration=3, cheap_sample_size=2),
    )
    report = orchestrator.run()
    assert report.generated_candidates >= 1
    assert report.evaluated_candidates >= 1
    assert report.experiment_ids
