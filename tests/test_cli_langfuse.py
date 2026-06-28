from __future__ import annotations

from poc_automation import cli as cli_mod
from poc_automation.search import SearchRunReport


def test_run_search_cli_langfuse_flags_survive_into_reporter(monkeypatch, tmp_path):
    captured = {}

    class FakeLangfuseReporter:
        def __init__(self, config):
            self.config = config
            captured["langfuse_config"] = config

    class FakeSearchOrchestrator:
        def __init__(self, **kwargs):
            captured["orchestrator_langfuse_config"] = kwargs["langfuse"].config

        def run(self):
            return SearchRunReport(
                iterations=1,
                generated_candidates=0,
                evaluated_candidates=0,
                positive_candidates=0,
                promoted_candidates=0,
                search_run_id="search_test",
            )

    monkeypatch.setenv("POC_LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("POC_LANGFUSE_DATASET_MODE", "local")
    monkeypatch.setattr(cli_mod, "LangfuseReporter", FakeLangfuseReporter)
    monkeypatch.setattr(cli_mod, "SearchOrchestrator", FakeSearchOrchestrator)

    rc = cli_mod.main(
        [
            "run-search",
            "--dataset",
            "examples/dataset.json",
            "--db",
            str(tmp_path / "registry.sqlite"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--agent",
            "heuristic",
            "--runner",
            "mock",
            "--iterations",
            "1",
            "--langfuse-enabled",
            "--langfuse-dataset-mode",
            "hosted",
            "--no-report",
        ]
    )

    assert rc == 0
    assert captured["langfuse_config"].enabled is True
    assert captured["langfuse_config"].dataset_mode == "hosted"
    assert captured["orchestrator_langfuse_config"].enabled is True
    assert captured["orchestrator_langfuse_config"].dataset_mode == "hosted"
