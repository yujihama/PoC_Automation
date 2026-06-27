"""Procedure tuning patch validation and materialization.

Historically this module only handled CSV procedure files.  v3 experiments also
use natural-language procedure text, so the public class names are kept for
compatibility while the implementation handles both CSV and plain text inputs.
"""

from __future__ import annotations

import csv
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import (
    PatchOperation,
    TuningCandidate,
    TuningPatch,
    ValidationIssue,
    ValidationReport,
)


@dataclass(frozen=True)
class MaterializationResult:
    output_path: str
    matched_rows: int
    diff: str


class CsvPatchError(ValueError):
    pass


class PatchValidator:
    """Checks that a patch is syntactically safe and not obviously overfit."""

    DEFAULT_BANNED_PATTERNS = [
        r"case[_-]?\d+",
        r"ev[_-]?\d+",
        r"doc[_-]?\d+",
        r"golden",
        r"reference answer",
    ]

    def __init__(
        self,
        max_instruction_chars: int = 400,
        banned_patterns: Iterable[str] | None = None,
    ):
        self.max_instruction_chars = max_instruction_chars
        self.banned_patterns = [re.compile(pat, re.IGNORECASE) for pat in (banned_patterns or [])]
        self.banned_patterns.extend(re.compile(pat, re.IGNORECASE) for pat in self.DEFAULT_BANNED_PATTERNS)

    def validate(
        self,
        patch: TuningPatch | None,
        *,
        base_csv_path: str | Path | None = None,
        reference_texts: Iterable[str] | None = None,
        case_specific_values: Iterable[str] | None = None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if patch is None:
            return ValidationReport(valid=True, issues=[])

        if not patch.text.strip():
            issues.append(ValidationIssue("error", "empty_instruction", "Instruction text is empty."))

        if len(patch.text) > self.max_instruction_chars:
            issues.append(
                ValidationIssue(
                    "warning",
                    "instruction_too_long",
                    f"Instruction is long: {len(patch.text)} chars > {self.max_instruction_chars} chars",
                )
            )

        for pattern in self.banned_patterns:
            if pattern.search(patch.text):
                issues.append(
                    ValidationIssue(
                        "error",
                        "banned_pattern",
                        f"Instruction contains a banned case-specific pattern: {pattern.pattern}",
                    )
                )

        for value in case_specific_values or []:
            normalized = value.strip()
            if len(normalized) >= 4 and normalized in patch.text:
                issues.append(
                    ValidationIssue(
                        "error",
                        "case_specific_value",
                        f"Instruction contains a case-specific value: {normalized[:40]}",
                    )
                )

        for ref in reference_texts or []:
            for phrase in _extract_reference_phrases(ref):
                if phrase and phrase in patch.text:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "reference_leakage",
                            f"Instruction may leak reference-answer wording: {phrase[:40]}",
                        )
                    )

        if base_csv_path is not None:
            issues.extend(_validate_patch_target_for_base(patch, Path(base_csv_path)))

        return ValidationReport(valid=not any(issue.severity == "error" for issue in issues), issues=issues)


class CsvMaterializer:
    def materialize(
        self,
        base_csv_path: str | Path,
        candidate: TuningCandidate,
        output_path: str | Path,
    ) -> MaterializationResult:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        base_path = Path(base_csv_path)
        before = base_path.read_text(encoding="utf-8-sig")
        if candidate.patch is None:
            output.write_text(before, encoding="utf-8")
            return MaterializationResult(output_path=str(output), matched_rows=0, diff="")

        if _is_csv_path(base_path):
            return self._materialize_csv(base_path, candidate.patch, output, before)
        return self._materialize_text(base_path, candidate.patch, output, before)

    def _materialize_csv(
        self,
        base_path: Path,
        patch: TuningPatch,
        output: Path,
        before: str,
    ) -> MaterializationResult:
        rows, fieldnames = _read_csv(base_path)
        matched_indexes = _select_row_indexes(rows, patch.target.row_selector)
        if not matched_indexes:
            raise CsvPatchError(f"row_selector did not match: {patch.target.row_selector}")
        if patch.target.column not in fieldnames:
            raise CsvPatchError(f"target column missing: {patch.target.column}")

        for idx in matched_indexes:
            row = rows[idx]
            row[patch.target.column] = _apply_operation(
                current=row.get(patch.target.column, ""),
                patch=patch,
            )

        with output.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        after = output.read_text(encoding="utf-8")
        return MaterializationResult(
            output_path=str(output),
            matched_rows=len(matched_indexes),
            diff=_make_diff(before, after, fromfile=str(base_path), tofile=str(output)),
        )

    def _materialize_text(
        self,
        base_path: Path,
        patch: TuningPatch,
        output: Path,
        before: str,
    ) -> MaterializationResult:
        after = _apply_text_operation(before, patch)
        output.write_text(after, encoding="utf-8")
        return MaterializationResult(
            output_path=str(output),
            matched_rows=1,
            diff=_make_diff(before, after, fromfile=str(base_path), tofile=str(output)),
        )


def _validate_patch_target_for_base(patch: TuningPatch, base_path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not base_path.exists():
        return [ValidationIssue("error", "base_not_found", f"Base procedure not found: {base_path}")]

    if not _is_csv_path(base_path):
        try:
            base_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            issues.append(ValidationIssue("error", "text_validation_failed", str(exc)))
        return issues

    try:
        rows, fieldnames = _read_csv(base_path)
        if patch.target.column not in fieldnames:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_column",
                    f"Target column is missing from CSV: {patch.target.column}",
                )
            )
        matched = _select_rows(rows, patch.target.row_selector)
        if not matched:
            issues.append(
                ValidationIssue(
                    "error",
                    "row_selector_no_match",
                    f"row_selector did not match any rows: {patch.target.row_selector}",
                )
            )
    except Exception as exc:  # noqa: BLE001 - validation should collect details
        issues.append(ValidationIssue("error", "csv_validation_failed", str(exc)))
    return issues


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def _select_rows(rows: list[dict[str, str]], selector: dict[str, object]) -> list[dict[str, str]]:
    return [rows[idx] for idx in _select_row_indexes(rows, selector)]


def _select_row_indexes(rows: list[dict[str, str]], selector: dict[str, object]) -> list[int]:
    indexes: list[int] = []
    for idx, row in enumerate(rows):
        matched = True
        for key, expected in selector.items():
            if str(row.get(key, "")) != str(expected):
                matched = False
                break
        if matched:
            indexes.append(idx)
    return indexes


def _apply_operation(current: str, patch: TuningPatch) -> str:
    text = patch.text.strip()
    if patch.operation == PatchOperation.APPEND_INSTRUCTION:
        return _join_instruction(current, text)
    if patch.operation == PatchOperation.PREPEND_INSTRUCTION:
        return _join_instruction(text, current)
    if patch.operation == PatchOperation.SET_INSTRUCTION:
        return text
    if patch.operation == PatchOperation.REPLACE_TEXT:
        if not patch.replace_from:
            raise CsvPatchError("replace_text requires replace_from")
        return current.replace(patch.replace_from, text)
    raise CsvPatchError(f"unsupported operation: {patch.operation}")


def _apply_text_operation(current: str, patch: TuningPatch) -> str:
    text = patch.text.strip()
    if patch.operation == PatchOperation.APPEND_INSTRUCTION:
        return _join_text_blocks(current, "Additional instruction:\n" + text)
    if patch.operation == PatchOperation.PREPEND_INSTRUCTION:
        return _join_text_blocks("Additional instruction:\n" + text, current)
    if patch.operation == PatchOperation.SET_INSTRUCTION:
        return text
    if patch.operation == PatchOperation.REPLACE_TEXT:
        if not patch.replace_from:
            raise CsvPatchError("replace_text requires replace_from")
        return current.replace(patch.replace_from, text)
    raise CsvPatchError(f"unsupported operation: {patch.operation}")


def _join_instruction(first: str, second: str) -> str:
    parts = [part.strip() for part in [first, second] if part and part.strip()]
    return "\n".join(parts)


def _join_text_blocks(first: str, second: str) -> str:
    parts = [part.strip() for part in [first, second] if part and part.strip()]
    return "\n\n".join(parts) + ("\n" if parts else "")


def _make_diff(before: str, after: str, *, fromfile: str, tofile: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def _extract_reference_phrases(text: str) -> list[str]:
    # Keep this intentionally conservative. It catches obvious leakage without
    # blocking generic instructions that share common business words.
    normalized = re.sub(r"\s+", "", text)
    if len(normalized) < 12:
        return []
    phrases: list[str] = []
    for width in (24, 32, 48):
        if len(normalized) >= width:
            phrases.append(normalized[:width])
    return phrases


def _is_csv_path(path: Path) -> bool:
    return path.suffix.lower() == ".csv"
