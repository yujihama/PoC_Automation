"""Tuning agents.

The orchestrator depends on the `TuningAgent` protocol.  DeepAgent/OpenRouter is
used for LLM-based tuning exploration, while the deterministic heuristic agent
keeps the prototype runnable in isolated environments and tests.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .config import CandidateAgentConfig
from .models import (
    FailureSummary,
    JsonDict,
    PatchOperation,
    PatchTarget,
    Scope,
    TuningCandidate,
    TuningPatch,
    TuningStatus,
    tuning_candidate_from_json,
    tuning_candidate_fingerprint,
)


class TuningAgent(Protocol):
    def propose_candidates(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str] | None = None,
    ) -> list[TuningCandidate]: ...


class HumanReferenceRuntimeContext(Protocol):
    def list_case_inventory(self) -> JsonDict: ...

    def read_case_input(self, case_id: str) -> JsonDict: ...

    def read_human_result(self, case_id: str) -> JsonDict: ...

    def list_previous_trials(self) -> list[JsonDict]: ...

    def evaluate_draft_instruction(
        self,
        *,
        instruction: str,
        hypothesis: str = "",
        case_id: str | None = None,
    ) -> JsonDict: ...

    def synthesize_cross_case_tuning(self) -> JsonDict: ...


class HeuristicTuningAgent:
    """Template-based candidate generator for deterministic exploration."""

    generator_prompt_version = "heuristic-v2"

    TEMPLATE_MAP: dict[str, list[tuple[str, str, list[str]]]] = {
        "unsupported_rationale": [
            (
                "根拠は証跡に明示された内容のみを使用する。証跡にない推測・一般知識・補完は根拠に含めない。",
                "証跡に支持されない根拠文を減らす",
                ["evidence_grounding"],
            ),
            (
                "各根拠文には、それを直接支持する証跡の引用を少なくとも1つ付ける。引用できない主張は出力しない。",
                "根拠と引用の対応を強制する",
                ["citation_rule", "evidence_grounding"],
            ),
        ],
        "citation_mismatch": [
            (
                "根拠文と引用を1対1で対応させ、引用箇所が当該根拠文を直接支持している場合のみ引用として採用する。",
                "引用ずれを減らす",
                ["citation_precision"],
            ),
            (
                "引用には証跡ID、ページ、該当項目名または該当文言を含める。主張を支持しない箇所は引用しない。",
                "引用の粒度を上げる",
                ["citation_precision"],
            ),
        ],
        "wrong_judgement": [
            (
                "判断前に、手続CSVの必須条件、任意条件、例外条件を分けて確認し、未確認の条件を適合根拠にしない。",
                "条件分岐の読み落としを減らす",
                ["decision_checklist", "condition_branching"],
            ),
            (
                "判定は、手続CSVに書かれた条件をすべて照合してから行う。矛盾する証跡がある場合は矛盾点を明示する。",
                "判定前の条件照合を徹底する",
                ["condition_branching", "contradiction_handling"],
            ),
        ],
        "insufficient_evidence": [
            (
                "必要な証跡が不足している場合は推測で補わず、判断不能とし、不足している証跡または項目を列挙する。",
                "証跡不足時の過剰判断を抑制する",
                ["abstention_rule"],
            ),
            (
                "証跡に確認対象項目が存在しない場合、その項目は未確認として扱い、適合・不適合の根拠にしない。",
                "未確認項目の扱いを明確化する",
                ["abstention_rule", "evidence_grounding"],
            ),
        ],
        "format_violation": [
            (
                "出力は評価結果、根拠、引用の3要素を必ず含め、根拠ごとに対応する引用を付ける。",
                "出力形式を安定させる",
                ["schema_enforcement"],
            )
        ],
    }

    FALLBACKS = [
        (
            "判断根拠は手続CSVと証跡に明示された情報のみに限定し、引用できない主張は出力しない。",
            "汎用的な証跡グラウンディングを強化する",
            ["evidence_grounding", "citation_rule"],
        )
    ]

    def propose_candidates(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str] | None = None,
    ) -> list[TuningCandidate]:
        seen_texts: set[str] = set()
        candidates: list[TuningCandidate] = []
        failure_modes = [failure.failure_mode for failure in failures] or ["unsupported_rationale"]
        for failure_mode in failure_modes:
            templates = self.TEMPLATE_MAP.get(failure_mode, self.FALLBACKS)
            for text, hypothesis, tactics in templates:
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                candidates.append(
                    self._candidate(
                        base_csv_id=base_csv_id,
                        row_selector=row_selector,
                        text=text,
                        hypothesis=hypothesis,
                        failure_mode=failure_mode,
                        tactics=tactics,
                        parent_tuning_ids=parent_tuning_ids or [],
                    )
                )
                if len(candidates) >= max_candidates:
                    return candidates
        for text, hypothesis, tactics in self.FALLBACKS:
            if len(candidates) >= max_candidates:
                break
            if text in seen_texts:
                continue
            candidates.append(
                self._candidate(
                    base_csv_id=base_csv_id,
                    row_selector=row_selector,
                    text=text,
                    hypothesis=hypothesis,
                    failure_mode="generic",
                    tactics=tactics,
                    parent_tuning_ids=parent_tuning_ids or [],
                )
            )
        return candidates

    def _candidate(
        self,
        *,
        base_csv_id: str,
        row_selector: dict[str, object],
        text: str,
        hypothesis: str,
        failure_mode: str,
        tactics: list[str],
        parent_tuning_ids: list[str],
    ) -> TuningCandidate:
        tuning_id = f"tune_{uuid.uuid4().hex[:12]}"
        candidate = TuningCandidate(
            tuning_id=tuning_id,
            parent_tuning_ids=parent_tuning_ids,
            scope=Scope.PROCEDURE_SPECIFIC,
            patch=TuningPatch(
                operation=PatchOperation.APPEND_INSTRUCTION,
                target=PatchTarget(
                    procedure_csv_base_id=base_csv_id,
                    row_selector=row_selector,
                    column="additional_instruction",
                ),
                text=text,
            ),
            hypothesis=hypothesis,
            generated_by="heuristic",
            generator_prompt_version=self.generator_prompt_version,
            labels={"target_failure_mode": [failure_mode], "tactic_type": tactics},
            risk_labels={},
            status=TuningStatus.CANDIDATE,
        )
        return _with_fingerprint(candidate)


class DeepAgentTuningAgent:
    """LLM-based tuning candidate generator using LangChain Deep Agents.

    The candidate generator only receives failure summaries and patch context. It
    does not receive human reference answers or expected outputs.
    """

    generator_prompt_version = "deepagent-openrouter-v2"

    def __init__(self, config: CandidateAgentConfig | None = None):
        self.config = config or CandidateAgentConfig.from_env()

    def propose_candidates(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str] | None = None,
    ) -> list[TuningCandidate]:
        result = self._invoke_agent(
            failures=failures,
            base_csv_id=base_csv_id,
            row_selector=row_selector,
            max_candidates=max_candidates,
            parent_tuning_ids=parent_tuning_ids or [],
        )
        text = _extract_agent_text(result)
        items = parse_candidate_json_response(text)
        candidates = [
            self._candidate_from_item(
                item,
                base_csv_id=base_csv_id,
                row_selector=row_selector,
                parent_tuning_ids=parent_tuning_ids or [],
            )
            for item in items
        ]
        return candidates[:max_candidates]

    def _invoke_agent(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str],
    ) -> Any:
        self._prepare_openrouter_env()
        prompt = build_candidate_agent_prompt(
            failures=failures,
            base_csv_id=base_csv_id,
            row_selector=row_selector,
            max_candidates=max_candidates,
            parent_tuning_ids=parent_tuning_ids,
        )
        if not self.config.use_deepagent_tools:
            return self._invoke_openrouter_http_messages(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

        try:
            from deepagents import create_deep_agent  # type: ignore
            from langchain_openrouter import ChatOpenRouter  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "DeepAgentTuningAgent requires optional dependencies. "
                "Install with: pip install -e .[target-agent]"
            ) from exc

        model_kwargs: dict[str, object] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_retries": self.config.max_retries,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.openrouter_provider:
            model_kwargs["openrouter_provider"] = self.config.openrouter_provider
        if self.config.route:
            model_kwargs["route"] = self.config.route
        if self.config.app_url:
            model_kwargs["app_url"] = self.config.app_url
        if self.config.app_title:
            model_kwargs["app_title"] = self.config.app_title

        model = ChatOpenRouter(**model_kwargs)
        agent = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=self.config.system_prompt,
        )
        return agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    def _invoke_openrouter_http_messages(self, messages: list[dict[str, str]]) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.openrouter_provider:
            payload["provider"] = self.config.openrouter_provider
        if self.config.route:
            payload["route"] = self.config.route

        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY or POC_CANDIDATE_AGENT_API_KEY is required for --agent deepagent")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        app_url = self.config.app_url or os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_APP_URL")
        if app_url:
            headers["HTTP-Referer"] = app_url
        app_title = self.config.app_title or os.getenv("OPENROUTER_APP_TITLE")
        if app_title:
            headers["X-Title"] = app_title

        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:1000]}") from exc

        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"OpenRouter response did not include choices: {str(raw)[:500]}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenRouter response did not include message content: {str(raw)[:500]}")
        raw_usage = raw.get("usage") if isinstance(raw, dict) else None
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        return {
            "messages": [
                {
                    "content": content,
                    "usage_metadata": {
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                }
            ],
            "openrouter_http": True,
            "raw_response": raw,
        }

    def _prepare_openrouter_env(self) -> None:
        if self.config.provider != "openrouter":
            raise ValueError(f"Unsupported candidate agent provider: {self.config.provider}")
        if self.config.api_key:
            os.environ.setdefault("OPENROUTER_API_KEY", self.config.api_key)
        if self.config.base_url:
            os.environ.setdefault("OPENROUTER_BASE_URL", self.config.base_url)
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY or POC_CANDIDATE_AGENT_API_KEY is required for --agent deepagent")
        if self.config.app_url:
            os.environ.setdefault("OPENROUTER_HTTP_REFERER", self.config.app_url)
            os.environ.setdefault("OPENROUTER_APP_URL", self.config.app_url)
        if self.config.app_title:
            os.environ.setdefault("OPENROUTER_APP_TITLE", self.config.app_title)
            os.environ.setdefault("OPENROUTER_X_TITLE", self.config.app_title)

    def _candidate_from_item(
        self,
        item: JsonDict,
        *,
        base_csv_id: str,
        row_selector: dict[str, object],
        parent_tuning_ids: list[str],
    ) -> TuningCandidate:
        # Accept either the full TuningCandidate JSON shape or a compact shape.
        if "patch" in item and "tuning_id" in item:
            candidate = tuning_candidate_from_json(item)
            candidate = TuningCandidate(
                tuning_id=candidate.tuning_id or f"tune_{uuid.uuid4().hex[:12]}",
                patch=candidate.patch,
                scope=candidate.scope,
                parent_tuning_ids=candidate.parent_tuning_ids or parent_tuning_ids,
                hypothesis=candidate.hypothesis,
                generated_by="deepagent-openrouter",
                generator_prompt_version=self.generator_prompt_version,
                labels=candidate.labels,
                risk_labels=candidate.risk_labels,
                status=TuningStatus.CANDIDATE,
                created_at=candidate.created_at,
            )
            return _with_fingerprint(candidate)

        text = str(item.get("instruction") or item.get("text") or item.get("patch_text") or "").strip()
        if not text:
            raise ValueError(f"candidate item did not include instruction text: {item}")
        tactics = item.get("tactic_type") or item.get("tactics") or item.get("labels", {}).get("tactic_type", [])
        if isinstance(tactics, str):
            tactics = [tactics]
        if not isinstance(tactics, list):
            tactics = ["generic"]
        failure_modes = item.get("target_failure_mode") or item.get("failure_mode") or []
        if isinstance(failure_modes, str):
            failure_modes = [failure_modes]
        if not isinstance(failure_modes, list):
            failure_modes = []
        candidate = TuningCandidate(
            tuning_id=str(item.get("tuning_id") or f"tune_{uuid.uuid4().hex[:12]}"),
            parent_tuning_ids=parent_tuning_ids,
            scope=Scope(str(item.get("scope", Scope.PROCEDURE_SPECIFIC.value))),
            patch=TuningPatch(
                operation=PatchOperation.APPEND_INSTRUCTION,
                target=PatchTarget(
                    procedure_csv_base_id=base_csv_id,
                    row_selector=row_selector,
                    column=str(item.get("column", "additional_instruction")),
                ),
                text=text,
            ),
            hypothesis=str(item.get("hypothesis") or "DeepAgentが提案したCSV追加指示候補"),
            generated_by="deepagent-openrouter",
            generator_prompt_version=self.generator_prompt_version,
            labels={
                "target_failure_mode": failure_modes or ["generic"],
                "tactic_type": [str(value) for value in tactics] or ["generic"],
            },
            risk_labels=dict(item.get("risk_labels", {})) if isinstance(item.get("risk_labels"), dict) else {},
            status=TuningStatus.CANDIDATE,
        )
        return _with_fingerprint(candidate)


class HumanReferenceDeepAgentTuningAgent(DeepAgentTuningAgent):
    """Data-aware candidate generator that can trial draft instructions.

    The agent is intentionally search-context aware: SearchOrchestrator injects
    a runtime context before each iteration, and the Deep Agent receives tools
    for reading case inputs, reading visible human references, and evaluating
    draft instructions before it returns final candidates.
    """

    generator_prompt_version = "deepagent-human-ref-v3"
    generated_by = "deepagent-human-ref"

    def __init__(self, config: CandidateAgentConfig | None = None):
        super().__init__(config)
        self.runtime_context: HumanReferenceRuntimeContext | None = None

    def set_runtime_context(self, context: HumanReferenceRuntimeContext) -> None:
        self.runtime_context = context

    def propose_candidates(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str] | None = None,
    ) -> list[TuningCandidate]:
        if self.runtime_context is None:
            raise RuntimeError("deepagent-human-ref requires a search runtime context")
        result = self._invoke_human_reference_agent(
            failures=failures,
            base_csv_id=base_csv_id,
            row_selector=row_selector,
            max_candidates=max_candidates,
            parent_tuning_ids=parent_tuning_ids or [],
        )
        text = _extract_agent_text(result)
        try:
            items = parse_candidate_json_response(text)
        except RuntimeError:
            items = self._fallback_items_from_trials(text, max_candidates=max_candidates)
            if not items:
                raise
        items = self._filter_items_by_stable_source_trials(items)
        candidates = [
            self._candidate_from_item(
                item,
                base_csv_id=base_csv_id,
                row_selector=row_selector,
                parent_tuning_ids=parent_tuning_ids or [],
            )
            for item in items
        ]
        return candidates[:max_candidates]

    def _filter_items_by_stable_source_trials(self, items: list[JsonDict]) -> list[JsonDict]:
        if self.runtime_context is None:
            return items
        trials = {str(trial.get("trial_id")): trial for trial in self.runtime_context.list_previous_trials()}
        filtered: list[JsonDict] = []
        for item in items:
            source_trial_ids = item.get("source_trial_ids", [])
            if isinstance(source_trial_ids, str):
                source_trial_ids = [source_trial_ids]
            if not source_trial_ids:
                if not trials:
                    filtered.append(item)
                continue
            source_trials = [trials.get(str(trial_id)) for trial_id in source_trial_ids]
            if source_trials and all(trial is not None and _trial_replicate_stable(trial) for trial in source_trials):
                filtered.append(item)
        return filtered

    def _fallback_items_from_trials(self, text: str, *, max_candidates: int) -> list[JsonDict]:
        if self.runtime_context is None:
            return []
        trials = [
            trial
            for trial in self.runtime_context.list_previous_trials()
            if trial.get("status") == "succeeded"
            and str(trial.get("instruction") or "").strip()
            and int(trial.get("summary", {}).get("regression_count", 0) or 0) == 0
            and _trial_replicate_stable(trial)
        ]
        if not trials:
            return []

        mentioned_trial_ids = re.findall(r"trial_\d{4}_\d{2}_[0-9a-fA-F]+|trial_\d{4}_\d{2}", text)
        mentioned_order = {trial_id: index for index, trial_id in enumerate(mentioned_trial_ids)}

        def trial_rank(trial: JsonDict) -> tuple[int, float, float]:
            trial_id = str(trial.get("trial_id") or "")
            mention_index = min(
                [index for key, index in mentioned_order.items() if trial_id.startswith(key)],
                default=10_000,
            )
            summary = trial.get("summary", {})
            return (
                mention_index,
                -float(summary.get("delta_mean", 0.0) or 0.0),
                -float(summary.get("total_score_mean", 0.0) or 0.0),
            )

        items: list[JsonDict] = []
        for trial in sorted(trials, key=trial_rank)[:max_candidates]:
            case_ids = [str(case_id) for case_id in trial.get("case_ids", [])]
            items.append(
                {
                    "instruction": str(trial.get("instruction") or "").strip(),
                    "hypothesis": str(trial.get("hypothesis") or "Recovered from a successful autonomous draft trial."),
                    "target_failure_mode": ["wrong_judgement", "unsupported_rationale", "citation_mismatch"],
                    "tactic_type": [
                        "evidence_grounding",
                        "citation_rule",
                        "condition_branching",
                        "cross_case_generalization",
                    ],
                    "scope": "procedure_specific",
                    "source_trial_ids": [str(trial.get("trial_id") or "")],
                    "generalized_from_cases": case_ids,
                    "synthesis_notes": "Recovered from successful draft trials because final agent response was not JSON.",
                    "risk_labels": {"overfitting": "medium", "answer_leakage": "medium"},
                }
            )
        return items

    def _invoke_human_reference_agent(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str],
    ) -> Any:
        if self.runtime_context is None:
            raise RuntimeError("deepagent-human-ref runtime context is not configured")
        self._prepare_openrouter_env()
        if not self.config.use_deepagent_tools:
            return self._invoke_human_reference_http_loop(
                failures=failures,
                base_csv_id=base_csv_id,
                row_selector=row_selector,
                max_candidates=max_candidates,
                parent_tuning_ids=parent_tuning_ids,
            )

        try:
            from deepagents import create_deep_agent  # type: ignore
            from langchain_openrouter import ChatOpenRouter  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "HumanReferenceDeepAgentTuningAgent requires optional dependencies. "
                "Install with: pip install -e .[target-agent]"
            ) from exc

        context = self.runtime_context

        def list_case_inventory() -> str:
            """List available case ids, splits, domains, and reference visibility."""
            return json.dumps(context.list_case_inventory(), ensure_ascii=False, indent=2)

        def read_case_input(case_id: str) -> str:
            """Read one case's procedure, evidence, and safe metadata."""
            return json.dumps(context.read_case_input(case_id), ensure_ascii=False, indent=2)

        def read_human_result(case_id: str) -> str:
            """Read the visible human result for a train/validation case."""
            return json.dumps(context.read_human_result(case_id), ensure_ascii=False, indent=2)

        def list_previous_trials() -> str:
            """List earlier draft-instruction trials and their score summaries."""
            return json.dumps(context.list_previous_trials(), ensure_ascii=False, indent=2)

        def evaluate_draft_instruction(instruction: str, hypothesis: str = "", case_id: str = "") -> str:
            """Evaluate one draft on all visible trial cases; case_id only changes ordering/focus."""
            return json.dumps(
                context.evaluate_draft_instruction(
                    instruction=instruction,
                    hypothesis=hypothesis,
                    case_id=case_id or None,
                ),
                ensure_ascii=False,
                indent=2,
            )

        def synthesize_cross_case_tuning() -> str:
            """Summarize which draft instructions look reusable across cases."""
            return json.dumps(context.synthesize_cross_case_tuning(), ensure_ascii=False, indent=2)

        model_kwargs: dict[str, object] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_retries": self.config.max_retries,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.openrouter_provider:
            model_kwargs["openrouter_provider"] = self.config.openrouter_provider
        if self.config.route:
            model_kwargs["route"] = self.config.route
        if self.config.app_url:
            model_kwargs["app_url"] = self.config.app_url
        if self.config.app_title:
            model_kwargs["app_title"] = self.config.app_title

        model = ChatOpenRouter(**model_kwargs)
        agent = create_deep_agent(
            model=model,
            tools=[
                list_case_inventory,
                read_case_input,
                read_human_result,
                list_previous_trials,
                evaluate_draft_instruction,
                synthesize_cross_case_tuning,
            ],
            system_prompt=self.config.system_prompt,
        )
        prompt = build_human_reference_agent_prompt(
            failures=failures,
            base_csv_id=base_csv_id,
            row_selector=row_selector,
            max_candidates=max_candidates,
            parent_tuning_ids=parent_tuning_ids,
        )
        return agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    def _invoke_human_reference_http_loop(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str],
    ) -> dict[str, object]:
        if self.runtime_context is None:
            raise RuntimeError("deepagent-human-ref runtime context is not configured")
        context = self.runtime_context
        inventory = context.list_case_inventory()
        visible_cases = [
            case
            for case in inventory.get("cases", [])
            if isinstance(case, dict) and case.get("human_result_visible")
        ][:3]
        case_payloads: list[JsonDict] = []
        for case_info in visible_cases:
            case_id = str(case_info.get("case_id"))
            case_payloads.append(
                {
                    "case_input": context.read_case_input(case_id),
                    "human_result": context.read_human_result(case_id),
                }
            )

        draft_request = {
            "task": "propose_draft_instructions_for_human_reference_tuning",
            "base_csv_id": base_csv_id,
            "row_selector": row_selector,
            "max_drafts": max(1, min(3, max_candidates + 1)),
            "parent_tuning_ids": parent_tuning_ids,
            "failure_summaries": [
                {
                    "case_id": failure.case_id,
                    "failure_mode": failure.failure_mode,
                    "summary": failure.summary,
                    "missing_capability": failure.missing_capability,
                    "scores": failure.scores,
                    "metadata": failure.metadata,
                }
                for failure in failures
            ],
            "visible_cases": case_payloads,
            "previous_trials": context.list_previous_trials()[-10:],
            "required_tactic_pool": [
                "missing_information_detection",
                "numeric_delta_tolerance_check",
                "condition_precondition_check",
                "exception_priority_ordering",
                "cross_evidence_contradiction_handling",
                "output_schema_stabilization",
                "inconclusive_trigger",
                "citation_granularity_control",
                "avoid_over_pruning_relevant_evidence",
                "fact_vs_inference_separation",
            ],
            "rules": [
                "Return JSON only.",
                "Do not copy exact case-specific values, dates, amounts, names, file names, or evidence ids.",
                "Each item must be a short reusable instruction to append to additional_instruction.",
                "Return drafts that cover different tactic_type values; do not return only citation or evidence-grounding variants.",
                "Actively explore numeric comparison, missing evidence, condition branching, exception priority, contradiction handling, schema stability, and inconclusive triggers when the cases support them.",
            ],
            "output_schema": [
                {
                    "instruction": "short reusable draft instruction",
                    "hypothesis": "why it should move outputs closer to human results",
                    "target_failure_mode": [
                        "wrong_judgement",
                        "unsupported_rationale",
                        "citation_mismatch",
                        "insufficient_evidence",
                        "numeric_mismatch",
                        "condition_branching_error",
                    ],
                    "tactic_type": [
                        "missing_information_detection",
                        "numeric_delta_tolerance_check",
                        "condition_precondition_check",
                    ],
                    "scope": "procedure_specific",
                }
            ],
        }
        response = self._invoke_openrouter_http_messages(
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": json.dumps(draft_request, ensure_ascii=False, indent=2)},
            ]
        )
        draft_items = parse_candidate_json_response(_extract_agent_text(response))
        for draft in draft_items[:3]:
            instruction = str(draft.get("instruction") or draft.get("text") or draft.get("patch_text") or "").strip()
            if not instruction:
                continue
            context.evaluate_draft_instruction(
                instruction=instruction,
                hypothesis=str(draft.get("hypothesis") or "OpenRouter HTTP draft trial."),
            )

        synthesis = context.synthesize_cross_case_tuning()
        final_items = self._fallback_items_from_trials(json.dumps(synthesis, ensure_ascii=False), max_candidates=max_candidates)
        return {
            "messages": [
                {
                    "content": json.dumps(final_items, ensure_ascii=False),
                    "usage_metadata": response.get("messages", [{}])[-1].get("usage_metadata", {})
                    if isinstance(response.get("messages"), list)
                    else {},
                }
            ],
            "openrouter_http_loop": True,
        }

    def _candidate_from_item(
        self,
        item: JsonDict,
        *,
        base_csv_id: str,
        row_selector: dict[str, object],
        parent_tuning_ids: list[str],
    ) -> TuningCandidate:
        candidate = super()._candidate_from_item(
            item,
            base_csv_id=base_csv_id,
            row_selector=row_selector,
            parent_tuning_ids=parent_tuning_ids,
        )
        labels = {
            **candidate.labels,
            "human_reference_mode": True,
            "source_trial_ids": item.get("source_trial_ids", []),
            "generalized_from_cases": item.get("generalized_from_cases", []),
            "synthesis_notes": item.get("synthesis_notes", ""),
        }
        risk_labels = dict(candidate.risk_labels)
        if "answer_leakage" not in risk_labels:
            risk_labels["answer_leakage"] = item.get("answer_leakage", "medium")
        return _with_fingerprint(
            TuningCandidate(
                tuning_id=candidate.tuning_id,
                patch=candidate.patch,
                scope=candidate.scope,
                parent_tuning_ids=candidate.parent_tuning_ids,
                hypothesis=candidate.hypothesis,
                generated_by=self.generated_by,
                generator_prompt_version=self.generator_prompt_version,
                labels=labels,
                risk_labels=risk_labels,
                status=TuningStatus.CANDIDATE,
                created_at=candidate.created_at,
            )
        )


@dataclass(frozen=True)
class SubprocessAgent:
    command: list[str]
    generated_by: str
    timeout_seconds: int = 120

    def propose_candidates(
        self,
        *,
        failures: list[FailureSummary],
        base_csv_id: str,
        row_selector: dict[str, object],
        max_candidates: int,
        parent_tuning_ids: list[str] | None = None,
    ) -> list[TuningCandidate]:
        payload = {
            "task": "propose_csv_tuning_patches",
            "base_csv_id": base_csv_id,
            "row_selector": row_selector,
            "max_candidates": max_candidates,
            "parent_tuning_ids": parent_tuning_ids or [],
            "failures": [failure.__dict__ for failure in failures],
            "output_schema": "list[TuningCandidate JSON]",
            "rules": [
                "Do not include human reference answers.",
                "Return JSON only.",
                "Create small atomic CSV patch candidates.",
            ],
        }
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        completed = subprocess.run(  # noqa: S603 - command is configured by trusted runtime env
            self.command + [prompt],
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        return [_with_fingerprint(candidate) for candidate in self._parse_candidates(completed.stdout)]

    def _parse_candidates(self, stdout: str) -> list[TuningCandidate]:
        items = parse_candidate_json_response(stdout)
        candidates = [tuning_candidate_from_json(item) for item in items]
        return [
            TuningCandidate(
                tuning_id=candidate.tuning_id,
                patch=candidate.patch,
                scope=candidate.scope,
                parent_tuning_ids=candidate.parent_tuning_ids,
                hypothesis=candidate.hypothesis,
                generated_by=self.generated_by,
                generator_prompt_version=candidate.generator_prompt_version,
                labels=candidate.labels,
                risk_labels=candidate.risk_labels,
                status=candidate.status,
                created_at=candidate.created_at,
            )
            for candidate in candidates
        ]


def build_candidate_agent_prompt(
    *,
    failures: list[FailureSummary],
    base_csv_id: str,
    row_selector: dict[str, object],
    max_candidates: int,
    parent_tuning_ids: list[str] | None = None,
) -> str:
    payload = {
        "task": "propose_csv_tuning_patches",
        "base_csv_id": base_csv_id,
        "row_selector": row_selector,
        "max_candidates": max_candidates,
        "parent_tuning_ids": parent_tuning_ids or [],
        "failure_summaries": [
            {
                "case_id": failure.case_id,
                "failure_mode": failure.failure_mode,
                "summary": failure.summary,
                "missing_capability": failure.missing_capability,
                "scores": failure.scores,
                "metadata": failure.metadata,
            }
            for failure in failures
        ],
        "rules": [
            "Do not use human reference answers. You are not given them.",
            "Do not include case-specific values, file names, addresses, dates, amounts, or customer names.",
            "Each candidate must be an atomic instruction suitable for appending to the additional_instruction column.",
            "Prefer generally reusable tactics over fixing one case.",
            "Diversify tactic_type across candidates instead of returning only citation/evidence-grounding variants.",
            "Consider missing information detection, numeric delta/tolerance checks, condition preconditions, exception priority, cross-evidence contradiction handling, output schema stabilization, inconclusive triggers, citation granularity, and avoiding over-pruning of relevant evidence.",
            "Return JSON only. No markdown.",
        ],
        "output_schema": [
            {
                "instruction": "短い追加指示。400文字以内。",
                "hypothesis": "この指示で改善する失敗モードと理由。",
                "target_failure_mode": ["wrong_judgement | unsupported_rationale | citation_mismatch | insufficient_evidence | evidence_conflict | format_violation"],
                "tactic_type": [
                    "evidence_grounding | citation_rule | condition_branching | abstention_rule | contradiction_handling | schema_enforcement | missing_information_detection | numeric_delta_tolerance_check | condition_precondition_check | exception_priority_ordering | cross_evidence_contradiction_handling | output_schema_stabilization | inconclusive_trigger | citation_granularity_control | avoid_over_pruning_relevant_evidence"
                ],
                "scope": "procedure_specific",
                "risk_labels": {"overfitting": "low|medium|high", "answer_leakage": "low|medium|high"},
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_human_reference_agent_prompt(
    *,
    failures: list[FailureSummary],
    base_csv_id: str,
    row_selector: dict[str, object],
    max_candidates: int,
    parent_tuning_ids: list[str] | None = None,
) -> str:
    payload = {
        "task": "human_reference_supervised_csv_tuning_search_v3",
        "base_csv_id": base_csv_id,
        "row_selector": row_selector,
        "max_final_candidates": max_candidates,
        "parent_tuning_ids": parent_tuning_ids or [],
        "initial_failure_summaries": [
            {
                "case_id": failure.case_id,
                "failure_mode": failure.failure_mode,
                "summary": failure.summary,
                "missing_capability": failure.missing_capability,
                "scores": failure.scores,
                "metadata": failure.metadata,
            }
            for failure in failures
        ],
        "required_workflow": [
            "Use list_case_inventory first.",
            "Inspect at least two visible cases when available.",
            "Use read_case_input and read_human_result for the inspected train/validation cases.",
            "Call evaluate_draft_instruction at least once before the final answer.",
            "Treat each draft evaluation as a cross-case check. Passing a case_id only marks the focus case; the tool still evaluates the visible trial set.",
            "Use synthesize_cross_case_tuning before the final answer when two or more trials/cases are available.",
            "Return final JSON only. No markdown.",
        ],
        "rules": [
            "The goal is to make target-agent outputs closer to visible human results.",
            "Do not copy case_id, evidence_id, dates, names, amounts, or exact case-specific strings into a generalized instruction.",
            "Prefer short reusable instructions for the additional_instruction column.",
            "Use case_specific only for a deliberately non-general candidate.",
            "Use procedure_specific or procedure_family for cross-case candidates.",
            "If a draft regresses another case, explain the risk and avoid promoting it as generalized.",
            "Keep tactic diversity: do not only produce citation/evidence-grounding variants.",
            "Explore missing information detection, numeric delta/tolerance checks, condition preconditions, exception priority, cross-evidence contradictions, schema stability, inconclusive triggers, citation granularity, and rules that prevent over-pruning relevant evidence.",
        ],
        "output_schema": [
            {
                "instruction": "short instruction to append to additional_instruction",
                "hypothesis": "why this should move outputs closer to human results",
                "target_failure_mode": [
                    "wrong_judgement",
                    "unsupported_rationale",
                    "citation_mismatch",
                    "insufficient_evidence",
                    "condition_branching",
                    "numeric_mismatch",
                    "missing_information",
                ],
                "tactic_type": [
                    "evidence_grounding",
                    "citation_rule",
                    "condition_branching",
                    "abstention_rule",
                    "cross_case_generalization",
                    "missing_information_detection",
                    "numeric_delta_tolerance_check",
                    "condition_precondition_check",
                    "exception_priority_ordering",
                    "cross_evidence_contradiction_handling",
                    "output_schema_stabilization",
                    "inconclusive_trigger",
                    "citation_granularity_control",
                    "avoid_over_pruning_relevant_evidence",
                ],
                "scope": "procedure_specific",
                "source_trial_ids": ["trial id returned by evaluate_draft_instruction"],
                "generalized_from_cases": ["case ids used for synthesis"],
                "synthesis_notes": "brief cross-case rationale",
                "risk_labels": {"overfitting": "low|medium|high", "answer_leakage": "low|medium|high"},
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_candidate_json_response(text: str) -> list[JsonDict]:
    """Parse a candidate-generator response with markdown-fence tolerance."""

    stripped = _strip_code_fence(text.strip())
    value: Any
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", stripped, flags=re.DOTALL)
        if fenced:
            value = json.loads(fenced.group(1))
        else:
            decoder = json.JSONDecoder()
            for index, char in enumerate(stripped):
                if char not in "[{":
                    continue
                try:
                    value, _ = decoder.raw_decode(stripped[index:])
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise RuntimeError(f"candidate agent did not return JSON: {text[:500]}") from None
    if isinstance(value, dict):
        for key in ("candidates", "items", "result"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
            if isinstance(value.get(key), dict):
                nested = value[key]
                if any(candidate_key in nested for candidate_key in ("instruction", "text", "patch_text", "patch")):
                    value = [nested]
                    break
        else:
            if any(candidate_key in value for candidate_key in ("instruction", "text", "patch_text", "patch")):
                value = [value]
    if not isinstance(value, list):
        raise RuntimeError(f"candidate agent JSON response must be a list: {type(value).__name__}")
    return [item for item in value if isinstance(item, dict)]


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _trial_replicate_stable(trial: JsonDict) -> bool:
    summary = trial.get("summary")
    if not isinstance(summary, dict):
        return False
    replicate_summary = summary.get("replicate_summary")
    if replicate_summary is None:
        return True
    return isinstance(replicate_summary, dict) and bool(replicate_summary.get("stable"))


def _extract_agent_text(result: object) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                text = _message_content(message)
                if text.strip():
                    return text
        for key in ("content", "output", "text"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    text = _message_content(result)
    if text.strip():
        return text
    raise RuntimeError("DeepAgent candidate response did not include final message content")


def _message_content(message: object) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value is not None:
                    parts.append(str(value))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _with_fingerprint(candidate: TuningCandidate) -> TuningCandidate:
    fingerprint = tuning_candidate_fingerprint(candidate)
    return TuningCandidate(
        tuning_id=candidate.tuning_id,
        patch=candidate.patch,
        scope=candidate.scope,
        parent_tuning_ids=candidate.parent_tuning_ids,
        hypothesis=candidate.hypothesis,
        generated_by=candidate.generated_by,
        generator_prompt_version=candidate.generator_prompt_version,
        labels={**candidate.labels, "fingerprint": fingerprint},
        risk_labels=candidate.risk_labels,
        status=candidate.status,
        created_at=candidate.created_at,
    )


def build_agent(name: str | None = None, config: CandidateAgentConfig | None = None) -> TuningAgent:
    agent_name = (name or os.getenv("POC_AUTOMATION_AGENT", "heuristic")).lower()
    if agent_name == "heuristic":
        return HeuristicTuningAgent()
    if agent_name in {"deepagent", "deepagent-openrouter", "openrouter"}:
        return DeepAgentTuningAgent(config)
    if agent_name in {"deepagent-human-ref", "human-reference", "human-ref"}:
        return HumanReferenceDeepAgentTuningAgent(config)
    if agent_name in {"deepagents-code", "dcode"}:
        binary = os.getenv("DEEPAGENTS_CODE_BIN", "dcode")
        return SubprocessAgent(
            command=[binary, "--skill", "poc-tuning", "--non-interactive", "--quiet"],
            generated_by="deepagents-code",
        )
    if agent_name == "cline":
        binary = os.getenv("CLINE_BIN", "cline")
        return SubprocessAgent(command=[binary, "--headless"], generated_by="cline")
    raise ValueError(f"unknown agent: {agent_name}")
