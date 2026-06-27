from poc_automation.config import TargetAgentConfig
from poc_automation.dataset import load_dataset_manifest
from poc_automation.runner import (
    DeepAgentPocAppRunner,
    build_deepagent_prompt,
    normalize_app_response,
    parse_agent_json_response,
)


def test_target_agent_defaults_to_qwen_model():
    cfg = TargetAgentConfig(api_key="test")
    assert cfg.provider == "openrouter"
    assert cfg.model.startswith("qwen/")


def test_parse_agent_json_accepts_markdown_fence():
    raw = """```json
{"result":{"judgement":"適合","rationale_items":[],"warnings":[]}}
```"""
    parsed = parse_agent_json_response(raw)
    assert parsed["result"]["judgement"] == "適合"


def test_parse_agent_json_extracts_embedded_object():
    raw = '前置き {"result":{"judgement":"判断不能","rationale_items":[],"warnings":["x"]}} 後置き'
    parsed = parse_agent_json_response(raw)
    normalized = normalize_app_response(parsed)
    assert normalized.judgement == "判断不能"
    assert normalized.warnings == ["x"]


def test_deepagent_prompt_does_not_include_expected_output():
    dataset = load_dataset_manifest("examples/dataset.json")
    case = dataset.cases[0]
    prompt = build_deepagent_prompt(case)
    assert case.case_id in prompt
    assert "expected_output" not in prompt
    assert "human_reference" not in prompt
    assert "required_capability" not in prompt


class FakeDeepAgentRunner(DeepAgentPocAppRunner):
    def _prepare_openrouter_env(self) -> None:  # pragma: no cover - not used by fake
        return None

    def _invoke_agent(self, *, case, materialized_csv_path):
        return {
            "messages": [
                {
                    "content": '{"result":{"judgement":"適合","rationale_items":[{"claim":"住所が一致している","citations":[{"evidence_id":"doc_identity_001","page":1,"span":"住所一致"}]}],"warnings":[]}}',
                    "usage_metadata": {"input_tokens": 12, "output_tokens": 34, "total_tokens": 46},
                }
            ]
        }


def test_deepagent_runner_can_be_exercised_without_network() -> None:
    dataset = load_dataset_manifest("examples/dataset.json")
    case = dataset.cases[0]
    runner = FakeDeepAgentRunner()

    result = runner.run_case(case=case, materialized_csv_path=case.procedure_csv_path)

    assert result.status == "succeeded"
    assert result.normalized_result.judgement == "適合"
    assert result.normalized_result.citation_count() == 1
    assert result.cost["total_tokens"] == 46

def test_target_agent_env_parses_openrouter_routing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROVIDER_JSON", '{"data_collection":"deny"}')
    monkeypatch.setenv("OPENROUTER_ROUTE", "fallback")
    monkeypatch.setenv("POC_TARGET_AGENT_USE_DEEPAGENT_TOOLS", "true")

    cfg = TargetAgentConfig.from_env()

    assert cfg.openrouter_provider == {"data_collection": "deny"}
    assert cfg.route == "fallback"
    assert cfg.use_deepagent_tools is True
