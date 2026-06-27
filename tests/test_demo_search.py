from pathlib import Path
from types import SimpleNamespace

from poc_automation.agents import HeuristicTuningAgent
from poc_automation.artifacts import LocalArtifactStore
from poc_automation.config import SearchPolicy
from poc_automation.dataset import load_dataset_manifest
from poc_automation.evaluators import EvaluatorSuite
from poc_automation.langfuse_client import TraceHandle
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


class CountingLangfuseReporter:
    def __init__(self):
        self.config = SimpleNamespace(send_evidence_text=False, host="http://localhost:3000", project="eom-poc-v2")
        self.enabled = True
        self.sync_dataset_calls = 0

    def start_search_session(self, **kwargs):
        return None

    def sync_dataset(self, dataset):
        self.sync_dataset_calls += 1
        return SimpleNamespace(dataset_name=dataset.dataset_id)

    def emit_agent_iteration(self, **kwargs):
        return None

    def start_trace(self, **kwargs):
        return TraceHandle(trace_id=None, enabled=False, name=kwargs.get("name", ""))

    def record_output(self, **kwargs):
        return None

    def record_scores(self, **kwargs):
        return None

    def record_dataset_run_item(self, **kwargs):
        return None

    def record_search_summary(self, **kwargs):
        return None

    def flush(self):
        return None


def test_search_orchestrator_syncs_langfuse_dataset_once(tmp_path: Path):
    dataset = load_dataset_manifest("examples/dataset.json")
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    langfuse = CountingLangfuseReporter()
    orchestrator = SearchOrchestrator(
        dataset=dataset,
        registry=registry,
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        runner=MockPocAppRunner(),
        agent=HeuristicTuningAgent(),
        evaluator_suite=EvaluatorSuite(),
        langfuse=langfuse,
        policy=SearchPolicy(iterations=1, candidates_per_iteration=1, cheap_sample_size=1),
    )

    orchestrator.run()

    assert langfuse.sync_dataset_calls == 1
