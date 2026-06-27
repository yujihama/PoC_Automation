"""Deterministic evaluator suite for PoC tuning experiments.

LLM-as-judge can be added behind the same interface, but the first prototype
keeps core metrics deterministic so that search results remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import EvaluatorPolicy
from .models import Case, EvaluationResult, NormalizedResult, TuningCandidate


class Evaluator(Protocol):
    name: str
    version: str

    def evaluate(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> EvaluationResult: ...


@dataclass(frozen=True)
class JudgementEvaluator:
    name: str = "judgement_match"
    version: str = "1.0"

    def evaluate(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> EvaluationResult:
        score = 1.0 if normalize_label(output.judgement) == normalize_label(case.expected_output.judgement) else 0.0
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=score,
            label="match" if score == 1.0 else "mismatch",
            comment="最終判定が人手回答と一致" if score == 1.0 else "最終判定が人手回答と不一致",
            details={"expected": case.expected_output.judgement, "actual": output.judgement},
        )


@dataclass(frozen=True)
class RationaleEvaluator:
    name: str = "rationale_support"
    version: str = "1.0"

    def evaluate(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> EvaluationResult:
        required = [keyword for keyword in case.expected_output.required_claim_keywords if keyword]
        claims = output.claim_text()
        if not required:
            score = 1.0 if claims.strip() else 0.0
            matched: list[str] = []
        else:
            matched = [keyword for keyword in required if keyword in claims]
            score = len(matched) / len(required)
        unsupported_penalty = _unsupported_claim_penalty(output)
        adjusted = max(0.0, score - unsupported_penalty)
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=adjusted,
            label="supported" if adjusted >= 0.8 else "weak",
            comment=f"根拠キーワード一致 {len(matched)}/{len(required)}; unsupported_penalty={unsupported_penalty:.2f}",
            details={"required": required, "matched": matched, "raw_score": score, "unsupported_penalty": unsupported_penalty},
        )


@dataclass(frozen=True)
class CitationEvaluator:
    name: str = "citation_quality"
    version: str = "1.0"

    def evaluate(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> EvaluationResult:
        expected = case.expected_output.citations
        actual_pairs = {
            (citation.evidence_id, citation.page)
            for item in output.rationale_items
            for citation in item.citations
        }
        if not expected:
            score = 1.0 if output.citation_count() > 0 else 0.0
            matched = []
        else:
            expected_pairs = {(citation.evidence_id, citation.page) for citation in expected}
            matched = sorted(actual_pairs.intersection(expected_pairs))
            score = len(matched) / len(expected_pairs) if expected_pairs else 0.0
            if output.rationale_items and output.citation_count() == 0:
                score = 0.0
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=score,
            label="good" if score >= 0.8 else "poor",
            comment=f"期待引用との一致 {len(matched)}/{len(expected) if expected else 'n/a'}",
            details={"actual_pairs": sorted(actual_pairs), "matched_pairs": matched},
        )


@dataclass(frozen=True)
class FormatEvaluator:
    name: str = "format_valid"
    version: str = "1.0"

    def evaluate(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> EvaluationResult:
        has_judgement = bool(output.judgement.strip())
        has_rationale = bool(output.rationale_items)
        score = 1.0 if has_judgement and has_rationale else 0.0
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=score,
            label="valid" if score == 1.0 else "invalid",
            comment="必須項目を満たしています" if score == 1.0 else "必須項目が不足しています",
            details={"has_judgement": has_judgement, "has_rationale": has_rationale},
        )


@dataclass(frozen=True)
class UnsupportedClaimEvaluator:
    name: str = "unsupported_claim_rate"
    version: str = "1.0"

    def evaluate(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> EvaluationResult:
        if not output.rationale_items:
            rate = 1.0
        else:
            unsupported = sum(1 for item in output.rationale_items if not item.citations)
            rate = unsupported / len(output.rationale_items)
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=rate,
            label="low" if rate <= 0.1 else "high",
            comment=f"引用のない根拠文の割合: {rate:.2f}",
            details={},
        )


@dataclass(frozen=True)
class LeakageRiskEvaluator:
    name: str = "leakage_risk"
    version: str = "1.0"

    def evaluate(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> EvaluationResult:
        text = candidate.instruction_text if candidate else ""
        high_risk_terms = [case.case_id, case.expected_output.judgement]
        hits = [term for term in high_risk_terms if term and len(term) >= 4 and term in text]
        score = 1.0 if hits else 0.0
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_version=self.version,
            score=score,
            label="risky" if hits else "none",
            comment="リーク疑いあり" if hits else "明確なリーク疑いなし",
            details={"hits": hits},
        )


class EvaluatorSuite:
    def __init__(self, policy: EvaluatorPolicy | None = None):
        self.policy = policy or EvaluatorPolicy()
        self.evaluators: list[Evaluator] = [
            JudgementEvaluator(),
            RationaleEvaluator(),
            CitationEvaluator(),
            FormatEvaluator(),
            UnsupportedClaimEvaluator(),
            LeakageRiskEvaluator(),
        ]

    def evaluate_case(
        self,
        *,
        case: Case,
        output: NormalizedResult,
        candidate: TuningCandidate | None = None,
    ) -> list[EvaluationResult]:
        results = [evaluator.evaluate(case=case, output=output, candidate=candidate) for evaluator in self.evaluators]
        scores = {result.evaluator_name: result.score for result in results if result.score is not None}
        total = self.total_score(scores)
        results.append(
            EvaluationResult(
                evaluator_name="total_score",
                evaluator_version="1.0",
                score=total,
                label="pass" if total >= 0.75 else "fail",
                comment="重み付き総合スコア",
                details={"component_scores": scores},
            )
        )
        return results

    def total_score(self, scores: dict[str, float | None]) -> float:
        judgement = float(scores.get("judgement_match") or 0.0)
        rationale = float(scores.get("rationale_support") or 0.0)
        citation = float(scores.get("citation_quality") or 0.0)
        fmt = float(scores.get("format_valid") or 0.0)
        unsupported = float(scores.get("unsupported_claim_rate") or 0.0)
        leakage = float(scores.get("leakage_risk") or 0.0)
        total = (
            self.policy.judgement_weight * judgement
            + self.policy.rationale_weight * rationale
            + self.policy.citation_weight * citation
            + self.policy.format_weight * fmt
            + self.policy.generality_weight * 0.5
            - self.policy.unsupported_claim_penalty * unsupported
            - self.policy.leakage_penalty * leakage
        )
        return round(max(0.0, min(1.0, total)), 4)


def normalize_label(value: str) -> str:
    return value.strip().replace(" ", "").replace("　", "").lower()


def _unsupported_claim_penalty(output: NormalizedResult) -> float:
    if not output.rationale_items:
        return 0.3
    missing_citation_count = sum(1 for item in output.rationale_items if not item.citations)
    return min(0.3, missing_citation_count / max(1, len(output.rationale_items)) * 0.3)
