# DeepAgent探索Agent

この版では、評価対象runnerだけでなく、CSVチューニング候補を生成する探索側も OpenRouter HTTP + Qwen で実行できます。

## 実行例

```bash
pip install -e .

export OPENROUTER_API_KEY=sk-or-v1-...
export POC_CANDIDATE_AGENT_MODEL=qwen/qwen3-max
export POC_CANDIDATE_AGENT_TEMPERATURE=0.2
export POC_TARGET_AGENT_MODEL=qwen/qwen3-max
export POC_TARGET_AGENT_TEMPERATURE=0

poc-auto run-search \
  --dataset examples/dataset.json \
  --db .tmp/poc_automation.sqlite \
  --artifact-dir .tmp/artifacts \
  --agent deepagent \
  --runner deepagent \
  --iterations 3
```

責務は以下です。

```text
--agent deepagent
  失敗要約を読み、手続CSVのadditional_instructionに追記するpatch候補を生成する。

--runner deepagent
  materialized CSVと証跡を読み、評価結果・根拠・引用を生成する評価対象アプリを代替する。
```

## 候補生成Agentに渡す情報

探索Agentには、Evaluatorが作成した失敗要約だけを渡します。人手回答やexpected_outputは渡しません。

```text
- failure_mode
- 失敗要約
- 不足している能力
- 評価スコア
- patch対象のbase_csv_id / row_selector
- 親候補ID
```

渡さないものは以下です。

```text
- 人手回答の文章
- expected_output
- required_claim_keywords
- 期待引用の詳細
- 証跡本文そのもの
- holdoutの結果を元にした失敗要約
```

## 出力形式

探索AgentはJSON配列のみを返します。

```json
[
  {
    "instruction": "必要な証跡が不足している場合は推測で補わず、判断不能とし、不足項目を列挙する。",
    "hypothesis": "証跡不足時の過剰判定を減らす",
    "target_failure_mode": ["insufficient_evidence"],
    "tactic_type": ["abstention_rule"],
    "scope": "procedure_specific",
    "risk_labels": {
      "overfitting": "low",
      "answer_leakage": "low"
    }
  }
]
```

Orchestrator側で、これを `TuningCandidate` と `TuningPatch` に正規化します。

## 重複排除

同じtargetに同じ追加指示を出した場合は、`tuning_candidate_fingerprint` で検出し、再評価しません。重複候補はRegistryに `rejected` として保存され、`duplicate_candidate` ラベルが付きます。

fingerprintは以下を元にします。

```text
- patch operation
- procedure_csv_base_id
- row_selector
- target column
- 正規化済みinstruction text
- replace_from
```

## baselineとdelta

探索開始時に、train / validation / holdout を含む全splitのbaselineを先に実行します。チューニング効果はsplit平均との比較ではなく、同一caseのbaselineとの差分として計算します。

```text
delta(case_i, tune_j) = score(case_i, tune_j) - score(case_i, baseline)
```

このため、validationやholdoutのdeltaがtrain baseline平均に引きずられることを避けます。

## 昇格判定

`domain_common` への昇格は、strict-v2 policyで判定します。デフォルトでは、validation / holdoutの件数、domain数、procedure family数が不足する場合は `promote_candidate` ではなく `needs_more_validation` になります。

主な条件は以下です。

```text
min_validation_cases_for_promotion = 10
min_holdout_cases_for_promotion = 10
min_domains_for_promotion = 2
min_procedure_families_for_promotion = 3
min_delta_for_promotion = 0.01
max_regression_rate_for_promotion = 0.02
require_judgement_non_degradation = true
require_citation_non_degradation = true
```

サンプルデータのようにvalidation 1件、holdout 1件しかない場合は、候補が良いスコアを出しても昇格ではなく追加検証待ちになります。
