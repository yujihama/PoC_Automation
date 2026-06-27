"""Generalization utilities for converting local tuning wins into reusable rules."""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass

from .models import (
    PatchOperation,
    PatchTarget,
    Scope,
    TuningCandidate,
    TuningPatch,
    TuningStatus,
)


@dataclass(frozen=True)
class TuningAtom:
    source_tuning_id: str
    text: str
    tactic_type: str


TACTIC_KEYWORDS = {
    "evidence_grounding": ["証跡", "明示", "推測", "根拠"],
    "citation_rule": ["引用", "証跡ID", "ページ", "該当"],
    "condition_branching": ["条件", "分岐", "必須", "任意", "例外"],
    "abstention_rule": ["判断不能", "不足", "未確認"],
    "contradiction_handling": ["矛盾", "不一致", "補助証跡"],
    "schema_enforcement": ["出力", "形式", "項目"],
}


class TuningGeneralizer:
    def atomize(self, candidates: list[TuningCandidate]) -> list[TuningAtom]:
        atoms: list[TuningAtom] = []
        for candidate in candidates:
            if candidate.patch is None:
                continue
            fragments = [frag.strip() for frag in re.split(r"[。\n]+", candidate.patch.text) if frag.strip()]
            for fragment in fragments:
                atoms.append(
                    TuningAtom(
                        source_tuning_id=candidate.tuning_id,
                        text=fragment + "。",
                        tactic_type=self.classify_tactic(fragment),
                    )
                )
        return atoms

    def classify_tactic(self, text: str) -> str:
        scores: dict[str, int] = {}
        for tactic, keywords in TACTIC_KEYWORDS.items():
            scores[tactic] = sum(1 for keyword in keywords if keyword in text)
        best = max(scores.items(), key=lambda item: item[1])
        return best[0] if best[1] > 0 else "generic"

    def propose_generalized_candidates(
        self,
        *,
        positive_candidates: list[TuningCandidate],
        base_csv_id: str,
        row_selector: dict[str, object],
        min_cluster_size: int = 2,
    ) -> list[TuningCandidate]:
        atoms = self.atomize(positive_candidates)
        clusters: dict[str, list[TuningAtom]] = defaultdict(list)
        for atom in atoms:
            clusters[atom.tactic_type].append(atom)

        generalized: list[TuningCandidate] = []
        for tactic, group in clusters.items():
            if len({atom.source_tuning_id for atom in group}) < min_cluster_size:
                continue
            instruction = self._instruction_for_tactic(tactic, group)
            generalized.append(
                TuningCandidate(
                    tuning_id=f"tune_global_{uuid.uuid4().hex[:12]}",
                    patch=TuningPatch(
                        operation=PatchOperation.APPEND_INSTRUCTION,
                        target=PatchTarget(
                            procedure_csv_base_id=base_csv_id,
                            row_selector=row_selector,
                            column="additional_instruction",
                        ),
                        text=instruction,
                    ),
                    scope=Scope.DOMAIN_COMMON if tactic != "generic" else Scope.PROCEDURE_FAMILY,
                    parent_tuning_ids=sorted({atom.source_tuning_id for atom in group}),
                    hypothesis=f"{tactic} の成功パターンを共通指示として統合する",
                    generated_by="generalizer",
                    generator_prompt_version="generalizer-v1",
                    labels={"tactic_type": [tactic], "expected_scope": "domain_common"},
                    risk_labels={"overfitting": "medium"},
                    status=TuningStatus.CANDIDATE,
                )
            )
        return generalized

    def _instruction_for_tactic(self, tactic: str, atoms: list[TuningAtom]) -> str:
        if tactic == "evidence_grounding":
            return "根拠は証跡に明示された内容のみを使用する。証跡にない推測、一般知識、補完は根拠に含めない。"
        if tactic == "citation_rule":
            return "各根拠文には、その主張を直接支持する証跡ID、ページ、該当箇所を引用として付ける。引用できない主張は出力しない。"
        if tactic == "condition_branching":
            return "判定前に、手続CSVの必須条件、任意条件、例外条件を分けて照合し、未確認の条件を適合根拠にしない。"
        if tactic == "abstention_rule":
            return "必要な証跡または確認対象項目が不足している場合は推測で補わず、判断不能とし、不足項目を列挙する。"
        if tactic == "contradiction_handling":
            return "申請情報、主証跡、補助証跡の間に不一致または矛盾がある場合は、矛盾点と採用した根拠を引用付きで明示する。"
        if tactic == "schema_enforcement":
            return "出力は評価結果、根拠、引用の3要素を必ず含め、根拠ごとに対応する引用を付ける。"
        # Conservative fallback: combine short unique atoms.
        seen: list[str] = []
        for atom in atoms:
            if atom.text not in seen:
                seen.append(atom.text)
            if len(seen) >= 2:
                break
        return "".join(seen)
