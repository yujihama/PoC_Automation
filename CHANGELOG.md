# Changelog

## 0.3.0 - v3 human-reference DeepAgent search
- Added `--agent deepagent-human-ref` for supervised exploration with multiple cases, procedure CSVs, evidence bundles, and visible human results.
- Added runtime-context methods for case inspection, visible human-result reading, draft-instruction evaluation, previous-trial lookup, and cross-case synthesis.
- Added `agent_trial_observations` to record autonomous draft evaluations separately from final candidate evaluations.
- Added report output for agent-internal draft trials so autonomous exploration can be audited from `run_report.md`.
- Added v3 policy knobs for visible reference splits, trial budget, neutral train probing, and trial evaluation splits.

## 0.2.0 - 探索側修正版

- `--agent deepagent` / `--agent deepagent-openrouter` を追加し、チューニング候補生成側も OpenRouter HTTP + Qwen で実行可能にした。
- `CandidateAgentConfig` を追加し、`POC_CANDIDATE_AGENT_*` 環境変数で候補生成モデルを評価対象runnerと分離可能にした。
- baselineをtrainだけでなく、validation / holdoutを含む全split・全caseで事前実行するようにした。
- 効果差分をsplit平均ではなく、同一caseのbaselineとの差分で計算するようにした。
- `tuning_candidate_fingerprint` を追加し、同一target・同一追加指示の重複候補を再評価しないようにした。
- 重複候補は `rejected` としてRegistryに保存し、`duplicate_candidate` ラベルを付けるようにした。
- promotion判定をstrict-v2 policyへ変更し、validation/holdout件数、domain数、procedure family数が不足する場合は `promote_candidate` ではなく `needs_more_validation` にした。
- レポートにbaseline件数、重複候補数、fingerprint、needs_more_validationの理由を表示するようにした。
- 探索側DeepAgent、case-level baseline delta、重複排除、false promotion防止のテストを追加した。
