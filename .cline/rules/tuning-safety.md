# PoC Tuning Safety Rules

When proposing CSV tuning patches:

- Return machine-readable JSON only when called by automation.
- Never include human reference answer wording.
- Never include case IDs, evidence file names, exact customer values, addresses, amounts, or dates.
- Propose small patches, not full CSV rewrites.
- Prefer instructions that improve evidence grounding, citation precision, condition checking, abstention, or output schema stability.
- Do not use holdout result details to generate new candidates.
- Include hypothesis, target failure mode, tactic type, expected scope, and risks.
