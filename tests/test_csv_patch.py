from pathlib import Path

from poc_automation.csv_patch import CsvMaterializer, PatchValidator
from poc_automation.models import PatchOperation, PatchTarget, TuningCandidate, TuningPatch


def test_materialize_appends_instruction_to_csv(tmp_path: Path):
    base = tmp_path / "base.csv"
    base.write_text("step_id,additional_instruction\ns1,existing\n", encoding="utf-8")
    candidate = TuningCandidate(
        tuning_id="t1",
        patch=TuningPatch(
            operation=PatchOperation.APPEND_INSTRUCTION,
            target=PatchTarget(procedure_csv_base_id="base", row_selector={"step_id": "s1"}),
            text="added",
        ),
    )

    result = CsvMaterializer().materialize(base, candidate, tmp_path / "out.csv")
    materialized = Path(result.output_path).read_text(encoding="utf-8")

    assert result.matched_rows == 1
    assert "existing" in materialized
    assert "added" in materialized
    assert "added" in result.diff


def test_materialize_appends_instruction_to_text_procedure(tmp_path: Path):
    base = tmp_path / "procedure.txt"
    base.write_text("Step 1. Compare the application and evidence.\nStep 2. Record the result.\n", encoding="utf-8")
    candidate = TuningCandidate(
        tuning_id="t-text",
        patch=TuningPatch(
            operation=PatchOperation.APPEND_INSTRUCTION,
            target=PatchTarget(procedure_csv_base_id="procedure", row_selector={"document": "procedure_text"}),
            text="If evidence is missing, mark the case inconclusive and cite the missing item.",
        ),
    )

    result = CsvMaterializer().materialize(base, candidate, tmp_path / "procedure.out.txt")
    materialized = Path(result.output_path).read_text(encoding="utf-8")

    assert result.matched_rows == 1
    assert "Step 1." in materialized
    assert "Additional instruction:" in materialized
    assert "mark the case inconclusive" in materialized
    assert "Additional instruction:" in result.diff


def test_validator_rejects_case_specific_pattern(tmp_path: Path):
    base = tmp_path / "base.csv"
    base.write_text("step_id,additional_instruction\ns1,\n", encoding="utf-8")
    patch = TuningPatch(
        operation=PatchOperation.APPEND_INSTRUCTION,
        target=PatchTarget(procedure_csv_base_id="base", row_selector={"step_id": "s1"}),
        text="case_001 should pass",
    )

    report = PatchValidator().validate(patch, base_csv_path=base)

    assert not report.valid
    assert any(issue.code == "banned_pattern" for issue in report.issues)


def test_validator_accepts_text_procedure_without_csv_target_columns(tmp_path: Path):
    base = tmp_path / "procedure.txt"
    base.write_text("Review the procedure text.", encoding="utf-8")
    patch = TuningPatch(
        operation=PatchOperation.APPEND_INSTRUCTION,
        target=PatchTarget(
            procedure_csv_base_id="procedure",
            row_selector={"document": "procedure_text"},
            column="additional_instruction",
        ),
        text="Focus on direct evidence only.",
    )

    report = PatchValidator().validate(patch, base_csv_path=base)

    assert report.valid
