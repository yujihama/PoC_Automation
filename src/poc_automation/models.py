"""Core data structures used by the tuning search prototype.

The package deliberately uses dataclasses and the standard library so that the
prototype can run in restricted environments.  Integrations such as Langfuse,
Deep Agents Code, Cline, and the real PoC application are isolated behind thin
adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Iterable

JsonDict = dict[str, Any]


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Scope(str, Enum):
    CASE_SPECIFIC = "case_specific"
    PROCEDURE_SPECIFIC = "procedure_specific"
    PROCEDURE_FAMILY = "procedure_family"
    DOMAIN_COMMON = "domain_common"
    GLOBAL_COMMON = "global_common"


class TuningStatus(str, Enum):
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    ARCHIVED = "archived"


class PatchOperation(str, Enum):
    APPEND_INSTRUCTION = "append_instruction"
    PREPEND_INSTRUCTION = "prepend_instruction"
    SET_INSTRUCTION = "set_instruction"
    REPLACE_TEXT = "replace_text"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"
    LEAVE_PROCEDURE_OUT = "leave_procedure_out"
    LEAVE_DOMAIN_OUT = "leave_domain_out"


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    page: int | None = None
    span: str | None = None
    claim: str | None = None


@dataclass(frozen=True)
class RationaleItem:
    claim: str
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedResult:
    judgement: str
    rationale_items: list[RationaleItem] = field(default_factory=list)
    raw_output: JsonDict | str | None = None
    warnings: list[str] = field(default_factory=list)

    def claim_text(self) -> str:
        return "\n".join(item.claim for item in self.rationale_items)

    def citation_count(self) -> int:
        return sum(len(item.citations) for item in self.rationale_items)


@dataclass(frozen=True)
class ExpectedOutput:
    judgement: str
    required_claim_keywords: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class Case:
    case_id: str
    split: Split
    procedure_csv_path: str
    evidence_bundle_path: str
    expected_output: ExpectedOutput
    metadata: JsonDict = field(default_factory=dict)
    human_result: ExpectedOutput | None = None
    human_result_text: str | None = None


@dataclass(frozen=True)
class PatchTarget:
    procedure_csv_base_id: str
    row_selector: JsonDict
    column: str = "additional_instruction"


@dataclass(frozen=True)
class TuningPatch:
    operation: PatchOperation
    target: PatchTarget
    text: str
    replace_from: str | None = None


@dataclass(frozen=True)
class TuningCandidate:
    tuning_id: str
    patch: TuningPatch | None
    scope: Scope = Scope.PROCEDURE_SPECIFIC
    parent_tuning_ids: list[str] = field(default_factory=list)
    hypothesis: str = ""
    generated_by: str = "heuristic"
    generator_prompt_version: str = "local"
    labels: JsonDict = field(default_factory=dict)
    risk_labels: JsonDict = field(default_factory=dict)
    status: TuningStatus = TuningStatus.CANDIDATE
    created_at: str = field(default_factory=utcnow_iso)

    @property
    def instruction_text(self) -> str:
        return "" if self.patch is None else self.patch.text


def normalize_instruction_text(text: str) -> str:
    """Normalize instruction text for duplicate detection.

    The goal is conservative deduplication across iterations: Japanese/English
    whitespace, common punctuation variants, and repeated spaces are collapsed,
    but semantic rewriting is intentionally not attempted.
    """

    import re

    value = text.strip().lower()
    value = value.replace("，", "、").replace(",", "、")
    value = value.replace("．", "。").replace(".", "。")
    value = value.replace("・", "、")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[。]+", "。", value)
    return value


def tuning_candidate_fingerprint(candidate: TuningCandidate) -> str:
    """Return a stable fingerprint for exact/near-exact duplicate patches."""

    import hashlib
    import json

    if candidate.patch is None:
        basis = {"kind": "baseline", "tuning_id": candidate.tuning_id}
    else:
        basis = {
            "operation": candidate.patch.operation.value,
            "target": {
                "procedure_csv_base_id": candidate.patch.target.procedure_csv_base_id,
                "row_selector": candidate.patch.target.row_selector,
                "column": candidate.patch.target.column,
            },
            "text": normalize_instruction_text(candidate.patch.text),
            "replace_from": normalize_instruction_text(candidate.patch.replace_from or ""),
        }
    payload = json.dumps(basis, ensure_ascii=False, sort_keys=True)
    return "fp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]


@dataclass(frozen=True)
class AppRunResult:
    app_run_id: str
    normalized_result: NormalizedResult
    raw_output_uri: str | None = None
    raw_output: JsonDict | str | None = None
    latency_ms: int | None = None
    cost: JsonDict = field(default_factory=dict)
    status: str = "succeeded"
    error_message: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    evaluator_name: str
    evaluator_version: str
    score: float | None
    label: str | None = None
    comment: str = ""
    details: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class FailureSummary:
    case_id: str
    failure_mode: str
    summary: str
    missing_capability: str
    scores: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    tuning_id: str
    split: str
    total_score_mean: float
    case_count: int
    regression_count: int
    positive_count: int
    negative_count: int
    metric_means: JsonDict = field(default_factory=dict)


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and enums to JSON-serializable structures."""

    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(_enum_to_value(key)): to_jsonable(val) for key, val in value.items()}
    return value


def citations_from_json(items: Iterable[JsonDict] | None) -> list[Citation]:
    if not items:
        return []
    return [Citation(**item) for item in items]


def normalized_result_from_json(data: JsonDict) -> NormalizedResult:
    rationale_items: list[RationaleItem] = []
    for item in data.get("rationale_items", []):
        rationale_items.append(
            RationaleItem(
                claim=item.get("claim", ""),
                citations=citations_from_json(item.get("citations", [])),
            )
        )
    return NormalizedResult(
        judgement=str(data.get("judgement", "")),
        rationale_items=rationale_items,
        raw_output=data.get("raw_output"),
        warnings=list(data.get("warnings", [])),
    )


def expected_output_from_json(data: JsonDict | None) -> ExpectedOutput:
    data = data or {}
    return ExpectedOutput(
        judgement=str(data.get("judgement", "")),
        required_claim_keywords=list(data.get("required_claim_keywords", [])),
        citations=citations_from_json(data.get("citations", [])),
        notes=data.get("notes"),
    )


def case_from_json(data: JsonDict, base_dir: str | None = None) -> Case:
    from pathlib import Path

    def resolve(path_value: str) -> str:
        if base_dir is None:
            return path_value
        p = Path(path_value)
        return str(p if p.is_absolute() else Path(base_dir) / p)

    expected_data = data.get("expected_output") or data.get("human_result") or {}
    human_data = data.get("human_result") or expected_data
    human_text = data.get("human_result_text")
    return Case(
        case_id=str(data["case_id"]),
        split=Split(str(data.get("split", Split.TRAIN.value))),
        procedure_csv_path=resolve(str(data["procedure_csv_path"])),
        evidence_bundle_path=resolve(str(data["evidence_bundle_path"])),
        expected_output=expected_output_from_json(expected_data),
        metadata=dict(data.get("metadata", {})),
        human_result=expected_output_from_json(human_data) if human_data else None,
        human_result_text=str(human_text) if human_text else None,
    )


def tuning_candidate_from_json(data: JsonDict) -> TuningCandidate:
    patch_data = data.get("patch")
    patch: TuningPatch | None = None
    if patch_data:
        target_data = patch_data["target"]
        patch = TuningPatch(
            operation=PatchOperation(patch_data["operation"]),
            target=PatchTarget(
                procedure_csv_base_id=target_data["procedure_csv_base_id"],
                row_selector=dict(target_data.get("row_selector", {})),
                column=target_data.get("column", "additional_instruction"),
            ),
            text=patch_data.get("text", ""),
            replace_from=patch_data.get("replace_from"),
        )

    return TuningCandidate(
        tuning_id=data["tuning_id"],
        patch=patch,
        scope=Scope(data.get("scope", Scope.PROCEDURE_SPECIFIC.value)),
        parent_tuning_ids=list(data.get("parent_tuning_ids", [])),
        hypothesis=data.get("hypothesis", ""),
        generated_by=data.get("generated_by", "unknown"),
        generator_prompt_version=data.get("generator_prompt_version", "unknown"),
        labels=dict(data.get("labels", {})),
        risk_labels=dict(data.get("risk_labels", {})),
        status=TuningStatus(data.get("status", TuningStatus.CANDIDATE.value)),
        created_at=data.get("created_at", utcnow_iso()),
    )
