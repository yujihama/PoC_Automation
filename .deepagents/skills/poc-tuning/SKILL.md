---
name: poc-tuning
description: Generate safe CSV tuning patches for the PoC Automation search loop.
---

# Goal

Generate safe, small, testable CSV tuning patches for an AI-agent PoC evaluation application.

# Hard Rules

- Return JSON only.
- Do not include human reference answer text.
- Do not include case IDs, evidence file names, customer names, exact addresses, exact amounts, or other case-specific values.
- Do not modify entire CSV files. Produce patch objects only.
- Keep each patch atomic and easy to evaluate.
- The patch must target an allowed `additional_instruction` column.
- Include hypothesis, target failure mode, tactic type, expected scope, and risks.
- Do not use holdout information.

# Candidate Shape

```json
{
  "tuning_id": "tune_generated_id",
  "scope": "procedure_specific",
  "parent_tuning_ids": [],
  "patch": {
    "operation": "append_instruction",
    "target": {
      "procedure_csv_base_id": "procedure_base",
      "row_selector": {"step_id": "s1"},
      "column": "additional_instruction"
    },
    "text": "根拠は証跡に明示された内容のみを使用する。"
  },
  "hypothesis": "証跡に支持されない根拠文を減らす",
  "generated_by": "deepagents-code",
  "generator_prompt_version": "poc-tuning-v1",
  "labels": {
    "target_failure_mode": ["unsupported_rationale"],
    "tactic_type": ["evidence_grounding"],
    "expected_scope": "procedure_specific"
  },
  "risk_labels": {
    "overfitting": "low"
  },
  "status": "candidate"
}
```
