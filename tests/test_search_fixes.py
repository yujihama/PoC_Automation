import threading
import time
from pathlib import Path

from poc_automation.agents import (
    DeepAgentTuningAgent,
    HeuristicTuningAgent,
    HumanReferenceDeepAgentTuningAgent,
    parse_candidate_json_response,
)
from poc_automation.artifacts import LocalArtifactStore
from poc_automation.config import SearchPolicy
from poc_automation.dataset import load_dataset_manifest
from poc_automation.evaluators import EvaluatorSuite
from poc_automation.models import FailureSummary
from poc_automation.registry import ExperimentRegistry
from poc_automation.runner import MockPocAppRunner
from poc_automation.search import SearchOrchestrator


def test_search_runs_baseline_for_all_splits_and_avoids_false_promotion(tmp_path: Path):
    dataset = load_dataset_manifest("examples/dataset.json")
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    orchestrator = SearchOrchestrator(
        dataset=dataset,
        registry=registry,
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        runner=MockPocAppRunner(),
        agent=HeuristicTuningAgent(),
        evaluator_suite=EvaluatorSuite(),
        policy=SearchPolicy(iterations=3, candidates_per_iteration=8, cheap_sample_size=2),
    )

    report = orchestrator.run()

    assert report.baseline_case_count == 4
    assert report.skipped_duplicate_candidates > 0
    assert report.promoted_candidates == 0

    with registry.connect() as conn:
        baseline_cases = conn.execute(
            "SELECT COUNT(DISTINCT case_id) AS count FROM case_runs WHERE tuning_id = 'baseline'"
        ).fetchone()["count"]
        duplicate_rejections = conn.execute(
            "SELECT COUNT(*) AS count FROM tuning_candidates WHERE risk_labels_json LIKE '%duplicate_candidate%'"
        ).fetchone()["count"]
        false_promotions = conn.execute(
            "SELECT COUNT(*) AS count FROM promotion_decisions WHERE decision = 'promote_candidate'"
        ).fetchone()["count"]

    assert baseline_cases == 4
    assert duplicate_rejections > 0
    assert false_promotions == 0


def test_strict_promotion_returns_needs_more_validation_for_small_samples(tmp_path: Path):
    dataset = load_dataset_manifest("examples/dataset.json")
    orchestrator = SearchOrchestrator(
        dataset=dataset,
        registry=ExperimentRegistry(tmp_path / "registry.sqlite"),
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        runner=MockPocAppRunner(),
        agent=HeuristicTuningAgent(),
        evaluator_suite=EvaluatorSuite(),
        policy=SearchPolicy(),
    )
    candidate = HeuristicTuningAgent().propose_candidates(
        failures=[],
        base_csv_id="procedure_base",
        row_selector={"step_id": "s1"},
        max_candidates=1,
    )[0]
    validation_result = {
        "summary": {
            "case_count": 1,
            "delta_mean": 0.2,
            "regression_rate": 0.0,
            "domain_count": 1,
            "procedure_family_count": 1,
            "metric_delta_means": {"judgement_match": 0.0, "citation_quality": 0.0},
        }
    }
    holdout_result = {
        "summary": {
            "case_count": 1,
            "delta_mean": 0.2,
            "regression_rate": 0.0,
            "domain_count": 1,
            "procedure_family_count": 1,
            "metric_delta_means": {"judgement_match": 0.0, "citation_quality": 0.0},
        }
    }

    decision, reason = orchestrator._promotion_decision(candidate, validation_result, holdout_result)

    assert decision == "needs_more_validation"
    assert "validation cases" in reason


class SlowMockRunner(MockPocAppRunner):
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run_case(self, *, case, materialized_csv_path):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().run_case(case=case, materialized_csv_path=materialized_csv_path)
        finally:
            with self.lock:
                self.active -= 1


def test_runner_parallelism_runs_case_evaluations_concurrently(tmp_path: Path):
    dataset = load_dataset_manifest("examples/dataset.json")
    runner = SlowMockRunner()
    orchestrator = SearchOrchestrator(
        dataset=dataset,
        registry=ExperimentRegistry(tmp_path / "registry.sqlite"),
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        runner=runner,
        agent=HeuristicTuningAgent(),
        evaluator_suite=EvaluatorSuite(),
        policy=SearchPolicy(runner_parallelism=2),
    )
    cases = dataset.by_split("train")

    results = orchestrator._run_cases_for_candidate(orchestrator._baseline_candidate(), cases)

    assert len(results) == len(cases)
    assert runner.max_active == 2


def test_candidate_json_parser_accepts_candidates_envelope():
    raw = '{"candidates":[{"instruction":"証跡不足時は判断不能とする。","hypothesis":"不足時の誤判定を減らす"}]}'
    items = parse_candidate_json_response(raw)
    assert items[0]["instruction"] == "証跡不足時は判断不能とする。"


def test_candidate_json_parser_accepts_single_candidate_object():
    raw = '{"instruction":"各根拠文に引用を付ける。","hypothesis":"引用不足を減らす"}'
    items = parse_candidate_json_response(raw)
    assert items[0]["instruction"] == "各根拠文に引用を付ける。"


class FakeDeepAgentTuningAgent(DeepAgentTuningAgent):
    def _prepare_openrouter_env(self) -> None:  # pragma: no cover
        return None

    def _invoke_agent(self, **kwargs):
        return {
            "messages": [
                {
                    "content": '[{"instruction":"証跡不足時は推測で補わず、判断不能とする。","hypothesis":"不足証跡時の過剰判定を抑える","target_failure_mode":["insufficient_evidence"],"tactic_type":["abstention_rule"],"scope":"procedure_specific"}]'
                }
            ]
        }


def test_deepagent_tuning_agent_normalizes_candidate_without_network():
    agent = FakeDeepAgentTuningAgent()
    candidates = agent.propose_candidates(
        failures=[
            FailureSummary(
                case_id="case_x",
                failure_mode="insufficient_evidence",
                summary="証跡不足で適合にしている",
                missing_capability="証跡不足時に判断不能を選ぶ能力",
            )
        ],
        base_csv_id="procedure_base",
        row_selector={"step_id": "s1"},
        max_candidates=3,
    )

    assert len(candidates) == 1
    assert candidates[0].generated_by == "deepagent-openrouter"
    assert candidates[0].labels["fingerprint"].startswith("fp_")
    assert candidates[0].patch.text.startswith("証跡不足時")


class FakeHumanReferenceContext:
    def list_case_inventory(self):
        return {"cases": []}

    def read_case_input(self, case_id):
        return {"case_id": case_id}

    def read_human_result(self, case_id):
        return {"case_id": case_id, "visible": True}

    def list_previous_trials(self):
        return [
            {
                "trial_id": "trial_0002_01_aaaabbbb",
                "instruction": "Require direct citation to the evidence item and mark missing facts inconclusive.",
                "hypothesis": "Direct citation should improve support.",
                "case_ids": ["case_a", "case_b"],
                "summary": {"delta_mean": 0.02, "total_score_mean": 0.80, "regression_count": 0},
                "status": "succeeded",
            },
            {
                "trial_id": "trial_0002_02_ccccdddd",
                "instruction": "Compare all available evidence before judging and explain contradictions.",
                "hypothesis": "Cross-evidence comparison should improve judgement.",
                "case_ids": ["case_a", "case_b", "case_c"],
                "summary": {"delta_mean": 0.08, "total_score_mean": 0.90, "regression_count": 0},
                "status": "succeeded",
            },
            {
                "trial_id": "trial_0002_03_eeeeffff",
                "instruction": "Regressing draft.",
                "hypothesis": "Should not be used.",
                "case_ids": ["case_a"],
                "summary": {"delta_mean": 0.30, "total_score_mean": 0.99, "regression_count": 1},
                "status": "succeeded",
            },
        ]

    def evaluate_draft_instruction(self, *, instruction, hypothesis="", case_id=None):
        return {}

    def synthesize_cross_case_tuning(self):
        return {}


class FakeHumanReferenceDeepAgent(HumanReferenceDeepAgentTuningAgent):
    def _invoke_human_reference_agent(self, **kwargs):
        return {
            "messages": [
                {
                    "content": (
                        "Based on the trials, best trial is trial_0002_02 with no regressions. "
                        "Returning prose instead of JSON."
                    )
                }
            ]
        }


def test_human_reference_agent_recovers_candidate_from_trials_when_final_json_is_missing():
    agent = FakeHumanReferenceDeepAgent()
    agent.set_runtime_context(FakeHumanReferenceContext())

    candidates = agent.propose_candidates(
        failures=[],
        base_csv_id="procedure_base",
        row_selector={"step_id": "s1"},
        max_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates[0].generated_by == "deepagent-human-ref"
    assert candidates[0].patch.text.startswith("Compare all available evidence")
    assert candidates[0].labels["source_trial_ids"] == ["trial_0002_02_ccccdddd"]


class FakeUnstableTrialContext(FakeHumanReferenceContext):
    def list_previous_trials(self):
        return [
            {
                "trial_id": "trial_0003_01_badbad00",
                "instruction": "Unstable but initially attractive instruction.",
                "hypothesis": "Should be skipped after replicate checks.",
                "case_ids": ["case_a", "case_b"],
                "summary": {
                    "delta_mean": 0.50,
                    "total_score_mean": 0.95,
                    "regression_count": 0,
                    "replicate_summary": {
                        "replicate_count": 3,
                        "stable": False,
                        "delta_mean_min": -0.20,
                        "worst_case_delta": -0.60,
                    },
                },
                "status": "succeeded",
            },
            {
                "trial_id": "trial_0003_02_good1111",
                "instruction": "Stable instruction chosen after replicate checks.",
                "hypothesis": "Should be promoted to final candidate.",
                "case_ids": ["case_a", "case_b"],
                "summary": {
                    "delta_mean": 0.10,
                    "total_score_mean": 0.82,
                    "regression_count": 0,
                    "replicate_summary": {
                        "replicate_count": 3,
                        "stable": True,
                        "delta_mean_min": 0.05,
                        "worst_case_delta": 0.0,
                    },
                },
                "status": "succeeded",
            },
        ]


class FakeUnstableTrialHumanReferenceDeepAgent(HumanReferenceDeepAgentTuningAgent):
    def _invoke_human_reference_agent(self, **kwargs):
        return {
            "messages": [
                {
                    "content": (
                        "Best trial is trial_0003_01_badbad00, but the final answer is not JSON."
                    )
                }
            ]
        }


def test_human_reference_agent_skips_unstable_replicated_trial_when_recovering_candidate():
    agent = FakeUnstableTrialHumanReferenceDeepAgent()
    agent.set_runtime_context(FakeUnstableTrialContext())

    candidates = agent.propose_candidates(
        failures=[],
        base_csv_id="procedure_base",
        row_selector={"step_id": "s1"},
        max_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates[0].patch.text.startswith("Stable instruction")
    assert candidates[0].labels["source_trial_ids"] == ["trial_0003_02_good1111"]
