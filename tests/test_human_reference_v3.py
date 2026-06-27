import json
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


class ContextUsingAgent:
    def __init__(self):
        self.context = None
        self.trial_result = None
        self.budget_result = None

    def set_runtime_context(self, context):
        self.context = context

    def propose_candidates(
        self,
        *,
        failures,
        base_csv_id,
        row_selector,
        max_candidates,
        parent_tuning_ids=None,
    ):
        inventory = self.context.list_case_inventory()
        assert len(inventory["cases"]) >= 3
        assert self.context.read_human_result("case_004")["visible"] is False

        case_input = self.context.read_case_input("case_001")
        serialized_input = json.dumps(case_input, ensure_ascii=False)
        assert "expected_output" not in serialized_input
        assert "required_claim_keywords" not in serialized_input
        assert "required_capability" not in serialized_input

        self.trial_result = self.context.evaluate_draft_instruction(
            instruction="Cite only facts directly shown in the evidence and mark missing required evidence as inconclusive.",
            hypothesis="A stricter evidence rule should move outputs closer to human results.",
            case_id="case_003",
        )
        self.budget_result = self.context.evaluate_draft_instruction(
            instruction="Second draft should not run because the budget is one.",
            hypothesis="budget check",
        )
        return HeuristicTuningAgent().propose_candidates(
            failures=failures,
            base_csv_id=base_csv_id,
            row_selector=row_selector,
            max_candidates=max_candidates,
            parent_tuning_ids=parent_tuning_ids,
        )


def test_human_reference_context_records_autonomous_trial_and_hides_holdout(tmp_path: Path):
    dataset = load_dataset_manifest("examples/dataset.json")
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    agent = ContextUsingAgent()
    orchestrator = SearchOrchestrator(
        dataset=dataset,
        registry=registry,
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
        runner=MockPocAppRunner(),
        agent=agent,
        evaluator_suite=EvaluatorSuite(),
        policy=SearchPolicy(
            iterations=1,
            candidates_per_iteration=1,
            cheap_sample_size=2,
            per_case_trial_budget=1,
            agent_trial_replicates=2,
            agent_trial_replicate_min_delta_mean=-999,
            agent_trial_replicate_min_worst_delta=-999,
            agent_trial_replicate_max_regression_count=99,
            allow_neutral_train_probe=True,
            data_visibility_policy="human_reference_v3_train_validation",
        ),
    )

    orchestrator.run()

    assert agent.trial_result["status"] == "succeeded"
    assert agent.trial_result["summary"]["case_count"] == 3
    assert agent.trial_result["summary"]["replicate_summary"]["replicate_count"] == 2
    assert agent.trial_result["summary"]["replicate_summary"]["stable"] is True
    assert agent.budget_result["status"] == "budget_exhausted"
    with registry.connect() as conn:
        trial_count = conn.execute("SELECT COUNT(*) AS count FROM agent_trial_observations").fetchone()["count"]
    assert trial_count == 1

    report_path = export_full_run_report(
        registry,
        tmp_path / "run_report.md",
        dataset_path="examples/dataset.json",
        run_report={"agent": "deepagent-human-ref", "runner": "mock"},
    )
    report_text = Path(report_path).read_text(encoding="utf-8")
    assert "Agent Trial Observations" in report_text
    assert "Autonomous draft evaluation observed: `yes`" in report_text
