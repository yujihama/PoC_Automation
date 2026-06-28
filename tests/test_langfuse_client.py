from __future__ import annotations

import os
from dataclasses import dataclass

from poc_automation.config import LangfuseConfig
from poc_automation.dataset import load_dataset_manifest
from poc_automation.langfuse_client import (
    LangfuseReporter,
    LangfuseScoreConfigInitializer,
)
from poc_automation.models import AppRunResult, EvaluationResult, NormalizedResult, TuningCandidate


@dataclass(frozen=True)
class FakeScoreConfigItem:
    name: str
    id: str


class FakeObservation:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.id = f"obs_{trace_id[:8]}"
        self.updates = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return self

    def end(self):
        self.ended = True
        return self


class FakeDatasetItems:
    def __init__(self):
        self.items = {}

    def get(self, item_id):
        if item_id not in self.items:
            raise KeyError(item_id)
        return self.items[item_id]


class FakeScoreConfigs:
    def __init__(self):
        self.created = []
        self.existing = []

    def get(self, limit=100):
        return type("Response", (), {"data": self.existing})()

    def create(self, **kwargs):
        self.created.append(kwargs)
        created = FakeScoreConfigItem(name=kwargs["name"], id=f"cfg_{kwargs['name']}")
        self.existing.append(created)
        return created


@dataclass
class FakeApi:
    dataset_items: FakeDatasetItems
    score_configs: FakeScoreConfigs


class FakeClient:
    def __init__(self):
        self.observations = []
        self.scores = []
        self.datasets = []
        self.dataset_items = []
        self.api = FakeApi(FakeDatasetItems(), FakeScoreConfigs())

    def create_trace_id(self, *, seed=None):
        return ("trace_" + str(seed or "none"))[:32]

    def start_observation(self, **kwargs):
        observation = FakeObservation(kwargs["trace_context"]["trace_id"])
        self.observations.append({"kwargs": kwargs, "observation": observation})
        return observation

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def create_dataset(self, **kwargs):
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs):
        self.dataset_items.append(kwargs)
        self.api.dataset_items.items[kwargs["id"]] = kwargs


def _reporter() -> LangfuseReporter:
    reporter = LangfuseReporter(
        LangfuseConfig(
            enabled=False,
            dataset_mode="hosted",
            tags=("poc-tuning", "openrouter"),
        )
    )
    reporter.enabled = True
    reporter.client = FakeClient()
    return reporter


def test_reporter_sets_host_and_base_url_env(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

    LangfuseReporter(
        LangfuseConfig(
            enabled=True,
            host="http://localhost:3000",
            public_key="pk-test",
            secret_key="sk-test",
        )
    )

    assert os.environ["LANGFUSE_HOST"] == "http://localhost:3000"
    assert os.environ["LANGFUSE_BASE_URL"] == "http://localhost:3000"


def test_case_run_trace_uses_design_trace_name_metadata_and_score_names():
    reporter = _reporter()
    trace = reporter.start_trace(
        name="poc.case_run",
        case_id="case_001",
        tuning_id="tune_001",
        session_id="search_001",
        input={"materialized_instruction": "Check citations."},
        metadata={
            "search_run_id": "search_001",
            "dataset_name": "dataset",
            "dataset_snapshot_id": "ds_001",
            "split": "train",
            "run_type": "formal_evaluation",
        },
        tags=("formal_evaluation",),
    )
    candidate = TuningCandidate(tuning_id="tune_001", patch=None)
    reporter.record_output(
        trace=trace,
        output=NormalizedResult(judgement="ok"),
        candidate=candidate,
    )
    reporter.record_scores(
        trace=trace,
        results=[
            EvaluationResult(
                evaluator_name="judgement_match",
                evaluator_version="v1",
                score=1.0,
                comment="ok",
            )
        ],
        extra_scores={"delta_vs_baseline": 0.25, "effect_label": "positive"},
    )

    client = reporter.client
    assert isinstance(client, FakeClient)
    assert client.observations[0]["kwargs"]["name"] == "poc.case_run"
    assert client.observations[0]["kwargs"]["metadata"]["search_run_id"] == "search_001"
    score_names = {score["name"] for score in client.scores}
    assert "judgement_score" in score_names
    assert "delta_vs_baseline" in score_names
    assert "effect_label" in score_names


def test_record_output_creates_target_generation_with_usage_details():
    reporter = _reporter()
    trace = reporter.start_trace(
        name="poc.case_run",
        case_id="case_001",
        tuning_id="tune_001",
        session_id="search_001",
        input={"materialized_instruction": "Check citations."},
        metadata={"search_run_id": "search_001"},
    )
    candidate = TuningCandidate(tuning_id="tune_001", patch=None)

    reporter.record_output(
        trace=trace,
        output=NormalizedResult(judgement="ok"),
        candidate=candidate,
        app_result=AppRunResult(
            app_run_id="app_001",
            normalized_result=NormalizedResult(judgement="ok"),
            cost={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            latency_ms=123,
        ),
    )

    client = reporter.client
    assert isinstance(client, FakeClient)
    generation = [item for item in client.observations if item["kwargs"]["name"] == "target_agent_call"][0]
    assert generation["kwargs"]["as_type"] == "generation"
    assert generation["kwargs"]["usage_details"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_scores_use_matching_score_config_id_when_available():
    reporter = _reporter()
    client = reporter.client
    assert isinstance(client, FakeClient)
    client.api.score_configs.existing = [FakeScoreConfigItem(name="total_score", id="cfg_total_score")]
    trace = reporter.start_trace(
        name="poc.case_run",
        case_id="case_001",
        tuning_id="tune_001",
        session_id="search_001",
        input={},
        metadata={"search_run_id": "search_001"},
    )

    reporter.record_scores(
        trace=trace,
        results=[
            EvaluationResult(
                evaluator_name="total_score",
                evaluator_version="v1",
                score=0.9,
            )
        ],
    )

    total_score = [score for score in client.scores if score["name"] == "total_score"][0]
    assert total_score["config_id"] == "cfg_total_score"


def test_hosted_dataset_sync_creates_stable_dataset_items():
    reporter = _reporter()
    dataset = load_dataset_manifest("examples/dataset.json")

    result = reporter.sync_dataset(dataset)

    client = reporter.client
    assert isinstance(client, FakeClient)
    assert result.enabled is True
    assert result.created_items == len(dataset.cases)
    assert client.datasets[0]["name"].startswith("poc-tuning-")
    first_item = client.dataset_items[0]
    assert first_item["input"]["case_id"] == dataset.cases[0].case_id
    assert first_item["metadata"]["dataset_snapshot_id"] == dataset.snapshot_id
    assert first_item["id"].startswith("lfdi_")


def test_score_config_initializer_uses_score_config_api():
    reporter = _reporter()

    result = LangfuseScoreConfigInitializer(reporter).initialize()

    client = reporter.client
    assert isinstance(client, FakeClient)
    assert result["enabled"] is True
    created_names = {item["name"] for item in client.api.score_configs.created}
    assert "total_score" in created_names
    assert "judgement_score" in created_names
    assert "replicate_stable" in created_names
