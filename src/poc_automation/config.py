"""Configuration helpers.

The project intentionally supports JSON config files first. YAML can be added
later if the target environment already standardizes on it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_QWEN_MODEL = "qwen/qwen3-max"
_DOTENV_LOADED = False


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load local .env values without overriding the process environment."""

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name.removeprefix("export ").strip()
        if not name or name in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _env_int(*names: str, default: int) -> int:
    value = _env_first(*names)
    return default if value is None else int(value)


def _env_float(*names: str, default: float) -> float:
    value = _env_first(*names)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> tuple[str, ...]:
    value = os.getenv(name)
    if value in (None, ""):
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_json_mapping(*names: str) -> JsonDict | None:
    value = _env_first(*names)
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object for {names[0]}, got {type(parsed).__name__}")
    return parsed


@dataclass(frozen=True)
class SearchPolicy:
    iterations: int = 2
    candidates_per_iteration: int = 8
    cheap_sample_size: int = 3
    validation_sample_size: int = 10
    holdout_sample_size: int = 10
    beam_width: int = 4
    max_instruction_chars: int = 400
    min_delta_for_positive_label: float = 0.03
    # Baseline and delta handling
    baseline_all_splits: bool = True
    delta_epsilon: float = 0.001
    deduplicate_candidates: bool = True
    # Promotion guardrails. The defaults are intentionally strict so a tiny
    # sample dataset yields `needs_more_validation`, not a false promotion.
    min_delta_for_promotion: float = 0.01
    max_regression_rate_for_promotion: float = 0.02
    min_validation_cases_for_promotion: int = 10
    min_holdout_cases_for_promotion: int = 10
    min_domains_for_promotion: int = 2
    min_procedure_families_for_promotion: int = 3
    require_judgement_non_degradation: bool = True
    require_citation_non_degradation: bool = True
    # v3 human-reference exploration. These defaults are inactive for the
    # existing agents unless the CLI selects the human-reference agent.
    human_reference_splits: tuple[str, ...] = ("train", "validation")
    holdout_reference_visible: bool = False
    per_case_trial_budget: int = 3
    agent_trial_replicates: int = 1
    agent_trial_replicate_min_delta_mean: float = 0.0
    agent_trial_replicate_min_worst_delta: float = 0.0
    agent_trial_replicate_max_regression_count: int = 0
    cross_case_synthesis_enabled: bool = True
    min_cases_for_generalized_tuning: int = 2
    allow_neutral_train_probe: bool = False
    agent_observation_splits: tuple[str, ...] = ("train", "validation", "holdout")
    agent_trial_eval_splits: tuple[str, ...] = ("train", "validation")
    data_visibility_policy: str = "failure_summary_only_v2"
    runner_parallelism: int = 1
    random_seed: int = 42

    @classmethod
    def from_mapping(cls, data: JsonDict | None) -> "SearchPolicy":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        values = {key: value for key, value in data.items() if key in allowed}
        for key in ("human_reference_splits", "agent_observation_splits", "agent_trial_eval_splits"):
            if key not in values:
                continue
            value = values[key]
            if isinstance(value, str):
                values[key] = tuple(part.strip() for part in value.split(",") if part.strip())
            elif isinstance(value, (list, tuple)):
                values[key] = tuple(str(part) for part in value)
        return cls(**values)

    @classmethod
    def from_env(cls) -> "SearchPolicy":
        _load_dotenv()
        return cls(
            runner_parallelism=max(
                1,
                _env_int(
                    "POC_SEARCH_RUNNER_PARALLELISM",
                    "POC_AUTOMATION_RUNNER_PARALLELISM",
                    default=cls.runner_parallelism,
                ),
            ),
            agent_trial_replicates=max(1, _env_int("POC_AGENT_TRIAL_REPLICATES", default=cls.agent_trial_replicates)),
            agent_trial_replicate_min_delta_mean=_env_float(
                "POC_AGENT_TRIAL_REPLICATE_MIN_DELTA_MEAN",
                default=cls.agent_trial_replicate_min_delta_mean,
            ),
            agent_trial_replicate_min_worst_delta=_env_float(
                "POC_AGENT_TRIAL_REPLICATE_MIN_WORST_DELTA",
                default=cls.agent_trial_replicate_min_worst_delta,
            ),
            agent_trial_replicate_max_regression_count=max(
                0,
                _env_int(
                    "POC_AGENT_TRIAL_REPLICATE_MAX_REGRESSION_COUNT",
                    default=cls.agent_trial_replicate_max_regression_count,
                ),
            ),
        )


@dataclass(frozen=True)
class EvaluatorPolicy:
    judgement_weight: float = 0.35
    rationale_weight: float = 0.25
    citation_weight: float = 0.20
    format_weight: float = 0.10
    generality_weight: float = 0.10
    unsupported_claim_penalty: float = 0.20
    leakage_penalty: float = 0.15
    regression_penalty: float = 0.10
    cost_latency_penalty: float = 0.05

    @classmethod
    def from_mapping(cls, data: JsonDict | None) -> "EvaluatorPolicy":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(frozen=True)
class AppApiConfig:
    base_url: str = "http://localhost:8080"
    api_key: str | None = None
    timeout_seconds: int = 120
    poll_interval_seconds: float = 2.0
    max_polls: int = 60

    @classmethod
    def from_env(cls) -> "AppApiConfig":
        _load_dotenv()
        return cls(
            base_url=os.getenv("POC_APP_BASE_URL", cls.base_url),
            api_key=os.getenv("POC_APP_API_KEY"),
            timeout_seconds=int(os.getenv("POC_APP_TIMEOUT_SECONDS", "120")),
            poll_interval_seconds=float(os.getenv("POC_APP_POLL_INTERVAL_SECONDS", "2")),
            max_polls=int(os.getenv("POC_APP_MAX_POLLS", "60")),
        )


TARGET_AGENT_SYSTEM_PROMPT = """\
あなたは、手続CSVと証跡を入力にして、業務評価結果・根拠・引用を生成する評価対象AIエージェントです。

守るべき方針:
- 手続CSVに書かれた条件、手順、追加指示を最優先する。
- 証跡に明示された内容だけを根拠にする。推測、一般知識、期待されそうな結論を根拠にしない。
- 各根拠文には、その根拠を直接支持する引用を付ける。
- 証跡不足や証跡間矛盾で判定できない場合は、無理に適合・不適合にせず「判断不能」とする。
- 出力は必ずJSONオブジェクトのみとし、Markdownや説明文を付けない。
"""


@dataclass(frozen=True)
class TargetAgentConfig:
    """Configuration for the OpenRouter-backed target PoC runner."""

    provider: str = "openrouter"
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    model: str = DEFAULT_OPENROUTER_QWEN_MODEL
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    max_retries: int = 2
    timeout_seconds: int = 120
    app_url: str | None = None
    app_title: str = "PoC Automation"
    openrouter_provider: JsonDict | None = None
    route: str | None = None
    max_csv_chars: int = 12000
    max_evidence_chars: int = 30000
    system_prompt: str = TARGET_AGENT_SYSTEM_PROMPT

    @classmethod
    def from_mapping(cls, data: JsonDict | None) -> "TargetAgentConfig":
        base = cls.from_env()
        if not data:
            return base
        provider_value = data.get("openrouter_provider", base.openrouter_provider)
        if isinstance(provider_value, str) and provider_value:
            provider_value = json.loads(provider_value)
        return cls(
            provider=str(data.get("provider", base.provider)),
            base_url=str(data.get("base_url", base.base_url)),
            model=str(data.get("model", base.model)),
            api_key=data.get("api_key", base.api_key),
            temperature=float(data.get("temperature", base.temperature)),
            max_tokens=int(data.get("max_tokens", base.max_tokens)),
            max_retries=int(data.get("max_retries", base.max_retries)),
            timeout_seconds=int(data.get("timeout_seconds", base.timeout_seconds)),
            app_url=data.get("app_url", base.app_url),
            app_title=str(data.get("app_title", base.app_title)),
            openrouter_provider=provider_value if isinstance(provider_value, dict) else None,
            route=data.get("route", base.route),
            max_csv_chars=int(data.get("max_csv_chars", base.max_csv_chars)),
            max_evidence_chars=int(data.get("max_evidence_chars", base.max_evidence_chars)),
            system_prompt=str(data.get("system_prompt", base.system_prompt)),
        )

    @classmethod
    def from_env(cls) -> "TargetAgentConfig":
        _load_dotenv()
        return cls(
            provider=(
                os.getenv("POC_TARGET_AGENT_PROVIDER")
                or os.getenv("TARGET_AGENT_PROVIDER")
                or os.getenv("OPENROUTER_PROVIDER")
                or cls.provider
            ),
            base_url=(
                os.getenv("POC_TARGET_AGENT_BASE_URL")
                or os.getenv("TARGET_AGENT_BASE_URL")
                or os.getenv("OPENROUTER_BASE_URL")
                or cls.base_url
            ),
            model=(
                os.getenv("POC_TARGET_AGENT_MODEL")
                or os.getenv("TARGET_AGENT_MODEL")
                or os.getenv("OPENROUTER_MODEL")
                or cls.model
            ),
            api_key=(
                os.getenv("POC_TARGET_AGENT_API_KEY")
                or os.getenv("TARGET_AGENT_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
            ),
            temperature=float(
                os.getenv("POC_TARGET_AGENT_TEMPERATURE")
                or os.getenv("TARGET_AGENT_TEMPERATURE")
                or os.getenv("OPENROUTER_TEMPERATURE")
                or "0"
            ),
            max_tokens=int(
                os.getenv("POC_TARGET_AGENT_MAX_TOKENS")
                or os.getenv("TARGET_AGENT_MAX_TOKENS")
                or os.getenv("OPENROUTER_MAX_TOKENS")
                or "2048"
            ),
            max_retries=int(
                os.getenv("POC_TARGET_AGENT_MAX_RETRIES")
                or os.getenv("TARGET_AGENT_MAX_RETRIES")
                or os.getenv("OPENROUTER_MAX_RETRIES")
                or "2"
            ),
            timeout_seconds=int(
                os.getenv("POC_TARGET_AGENT_TIMEOUT_SECONDS")
                or os.getenv("TARGET_AGENT_TIMEOUT_SECONDS")
                or os.getenv("OPENROUTER_TIMEOUT_SECONDS")
                or "120"
            ),
            app_url=(
                os.getenv("POC_TARGET_AGENT_APP_URL")
                or os.getenv("TARGET_AGENT_APP_URL")
                or os.getenv("OPENROUTER_HTTP_REFERER")
                or os.getenv("OPENROUTER_APP_URL")
            ),
            app_title=(
                os.getenv("POC_TARGET_AGENT_APP_TITLE")
                or os.getenv("TARGET_AGENT_APP_TITLE")
                or os.getenv("OPENROUTER_APP_TITLE")
                or cls.app_title
            ),
            openrouter_provider=_env_json_mapping(
                "POC_TARGET_AGENT_OPENROUTER_PROVIDER_JSON",
                "TARGET_AGENT_OPENROUTER_PROVIDER_JSON",
                "OPENROUTER_PROVIDER_JSON",
            ),
            route=(
                os.getenv("POC_TARGET_AGENT_OPENROUTER_ROUTE")
                or os.getenv("TARGET_AGENT_OPENROUTER_ROUTE")
                or os.getenv("OPENROUTER_ROUTE")
                or cls.route
            ),
            max_csv_chars=int(
                os.getenv("POC_TARGET_AGENT_MAX_CSV_CHARS")
                or os.getenv("TARGET_AGENT_MAX_CSV_CHARS")
                or "12000"
            ),
            max_evidence_chars=int(
                os.getenv("POC_TARGET_AGENT_MAX_EVIDENCE_CHARS")
                or os.getenv("TARGET_AGENT_MAX_EVIDENCE_CHARS")
                or "30000"
            ),
            system_prompt=os.getenv("POC_TARGET_AGENT_SYSTEM_PROMPT")
            or os.getenv("TARGET_AGENT_SYSTEM_PROMPT")
            or TARGET_AGENT_SYSTEM_PROMPT,
        )


CANDIDATE_AGENT_SYSTEM_PROMPT = """\
あなたは、業務PoCの手続CSVチューニング探索エージェントです。
失敗要約だけを材料に、CSVの追加指示として試す小さなpatch候補を生成します。

守るべき方針:
- 人手正解、期待出力、ケース固有の値を推測して指示に入れない。
- 1候補は1つの意図に絞り、短く原子的にする。
- 証跡グラウンディング、引用精度、条件分岐、証跡不足時の判断不能、矛盾処理などの汎用能力を改善する。
- 出力は必ずJSON配列のみとし、Markdownや説明文を付けない。
"""


@dataclass(frozen=True)
class CandidateAgentConfig:
    """Configuration for the OpenRouter-backed tuning candidate generator."""

    provider: str = "openrouter"
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    model: str = DEFAULT_OPENROUTER_QWEN_MODEL
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    max_retries: int = 2
    timeout_seconds: int = 120
    app_url: str | None = None
    app_title: str = "PoC Automation Candidate Generator"
    openrouter_provider: JsonDict | None = None
    route: str | None = None
    system_prompt: str = CANDIDATE_AGENT_SYSTEM_PROMPT

    @classmethod
    def from_mapping(cls, data: JsonDict | None) -> "CandidateAgentConfig":
        base = cls.from_env()
        if not data:
            return base
        provider_value = data.get("openrouter_provider", base.openrouter_provider)
        if isinstance(provider_value, str) and provider_value:
            provider_value = json.loads(provider_value)
        return cls(
            provider=str(data.get("provider", base.provider)),
            base_url=str(data.get("base_url", base.base_url)),
            model=str(data.get("model", base.model)),
            api_key=data.get("api_key", base.api_key),
            temperature=float(data.get("temperature", base.temperature)),
            max_tokens=int(data.get("max_tokens", base.max_tokens)),
            max_retries=int(data.get("max_retries", base.max_retries)),
            timeout_seconds=int(data.get("timeout_seconds", base.timeout_seconds)),
            app_url=data.get("app_url", base.app_url),
            app_title=str(data.get("app_title", base.app_title)),
            openrouter_provider=provider_value if isinstance(provider_value, dict) else None,
            route=data.get("route", base.route),
            system_prompt=str(data.get("system_prompt", base.system_prompt)),
        )

    @classmethod
    def from_env(cls) -> "CandidateAgentConfig":
        _load_dotenv()
        return cls(
            provider=(
                os.getenv("POC_CANDIDATE_AGENT_PROVIDER")
                or os.getenv("CANDIDATE_AGENT_PROVIDER")
                or os.getenv("OPENROUTER_PROVIDER")
                or cls.provider
            ),
            base_url=(
                os.getenv("POC_CANDIDATE_AGENT_BASE_URL")
                or os.getenv("CANDIDATE_AGENT_BASE_URL")
                or os.getenv("OPENROUTER_BASE_URL")
                or cls.base_url
            ),
            model=(
                os.getenv("POC_CANDIDATE_AGENT_MODEL")
                or os.getenv("CANDIDATE_AGENT_MODEL")
                or os.getenv("OPENROUTER_MODEL")
                or cls.model
            ),
            api_key=(
                os.getenv("POC_CANDIDATE_AGENT_API_KEY")
                or os.getenv("CANDIDATE_AGENT_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
            ),
            temperature=float(
                os.getenv("POC_CANDIDATE_AGENT_TEMPERATURE")
                or os.getenv("CANDIDATE_AGENT_TEMPERATURE")
                or os.getenv("OPENROUTER_TEMPERATURE")
                or "0.2"
            ),
            max_tokens=int(
                os.getenv("POC_CANDIDATE_AGENT_MAX_TOKENS")
                or os.getenv("CANDIDATE_AGENT_MAX_TOKENS")
                or os.getenv("OPENROUTER_MAX_TOKENS")
                or "4096"
            ),
            max_retries=int(
                os.getenv("POC_CANDIDATE_AGENT_MAX_RETRIES")
                or os.getenv("CANDIDATE_AGENT_MAX_RETRIES")
                or os.getenv("OPENROUTER_MAX_RETRIES")
                or "2"
            ),
            timeout_seconds=int(
                os.getenv("POC_CANDIDATE_AGENT_TIMEOUT_SECONDS")
                or os.getenv("CANDIDATE_AGENT_TIMEOUT_SECONDS")
                or os.getenv("OPENROUTER_TIMEOUT_SECONDS")
                or "120"
            ),
            app_url=(
                os.getenv("POC_CANDIDATE_AGENT_APP_URL")
                or os.getenv("CANDIDATE_AGENT_APP_URL")
                or os.getenv("OPENROUTER_HTTP_REFERER")
                or os.getenv("OPENROUTER_APP_URL")
            ),
            app_title=(
                os.getenv("POC_CANDIDATE_AGENT_APP_TITLE")
                or os.getenv("CANDIDATE_AGENT_APP_TITLE")
                or os.getenv("OPENROUTER_APP_TITLE")
                or cls.app_title
            ),
            openrouter_provider=_env_json_mapping(
                "POC_CANDIDATE_AGENT_OPENROUTER_PROVIDER_JSON",
                "CANDIDATE_AGENT_OPENROUTER_PROVIDER_JSON",
                "OPENROUTER_PROVIDER_JSON",
            ),
            route=(
                os.getenv("POC_CANDIDATE_AGENT_OPENROUTER_ROUTE")
                or os.getenv("CANDIDATE_AGENT_OPENROUTER_ROUTE")
                or os.getenv("OPENROUTER_ROUTE")
                or cls.route
            ),
            system_prompt=os.getenv("POC_CANDIDATE_AGENT_SYSTEM_PROMPT")
            or os.getenv("CANDIDATE_AGENT_SYSTEM_PROMPT")
            or CANDIDATE_AGENT_SYSTEM_PROMPT,
        )


@dataclass(frozen=True)
class LangfuseConfig:
    enabled: bool = False
    host: str | None = None
    public_key: str | None = None
    secret_key: str | None = None
    project: str | None = None
    dataset_mode: str = "local"
    session_by_search_run: bool = True
    send_raw_output: bool = True
    send_evidence_text: bool = False
    send_evidence_files: bool = False
    tags: tuple[str, ...] = ("poc-tuning",)

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        _load_dotenv()
        enabled = _env_bool("POC_LANGFUSE_ENABLED", _env_bool("LANGFUSE_ENABLED", False))
        tags = _env_list("POC_LANGFUSE_TAGS") or ("poc-tuning", "openrouter", "deepagent")
        return cls(
            enabled=enabled,
            host=os.getenv("LANGFUSE_HOST"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            project=os.getenv("LANGFUSE_PROJECT") or os.getenv("POC_LANGFUSE_PROJECT"),
            dataset_mode=os.getenv("POC_LANGFUSE_DATASET_MODE", "local"),
            session_by_search_run=_env_bool("POC_LANGFUSE_SESSION_BY_SEARCH_RUN", True),
            send_raw_output=_env_bool("POC_LANGFUSE_SEND_RAW_OUTPUT", True),
            send_evidence_text=_env_bool("POC_LANGFUSE_SEND_EVIDENCE_TEXT", False),
            send_evidence_files=_env_bool("POC_LANGFUSE_SEND_EVIDENCE_FILES", False),
            tags=tuple(tags),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: str = ".tmp/poc_automation.sqlite"
    artifact_dir: str = ".tmp/artifacts"
    agent: str = "heuristic"
    runner: str = "mock"
    search_policy: SearchPolicy = field(default_factory=SearchPolicy)
    evaluator_policy: EvaluatorPolicy = field(default_factory=EvaluatorPolicy)
    app_api: AppApiConfig = field(default_factory=AppApiConfig.from_env)
    target_agent: TargetAgentConfig = field(default_factory=TargetAgentConfig.from_env)
    candidate_agent: CandidateAgentConfig = field(default_factory=CandidateAgentConfig.from_env)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig.from_env)

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        _load_dotenv()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            db_path=data.get("db_path", cls.db_path),
            artifact_dir=data.get("artifact_dir", cls.artifact_dir),
            agent=data.get("agent", os.getenv("POC_AUTOMATION_AGENT", "heuristic")),
            runner=data.get("runner", os.getenv("POC_AUTOMATION_RUNNER", "mock")),
            search_policy=SearchPolicy.from_mapping(data.get("search_policy")),
            evaluator_policy=EvaluatorPolicy.from_mapping(data.get("evaluator_policy")),
            app_api=AppApiConfig.from_env(),
            target_agent=TargetAgentConfig.from_mapping(data.get("target_agent")),
            candidate_agent=CandidateAgentConfig.from_mapping(data.get("candidate_agent")),
            langfuse=LangfuseConfig.from_env(),
        )

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        _load_dotenv()
        return cls(
            db_path=os.getenv("POC_AUTOMATION_DB", cls.db_path),
            artifact_dir=os.getenv("POC_AUTOMATION_ARTIFACT_DIR", cls.artifact_dir),
            agent=os.getenv("POC_AUTOMATION_AGENT", "heuristic"),
            runner=os.getenv("POC_AUTOMATION_RUNNER", "mock"),
            search_policy=SearchPolicy.from_env(),
            target_agent=TargetAgentConfig.from_env(),
            candidate_agent=CandidateAgentConfig.from_env(),
        )
