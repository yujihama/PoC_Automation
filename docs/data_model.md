# データモデル

## TuningCandidate

チューニング候補は、CSV全体ではなくpatchとして保存します。

```json
{
  "tuning_id": "tune_xxx",
  "parent_tuning_ids": ["tune_parent"],
  "scope": "procedure_specific",
  "patch": {
    "operation": "append_instruction",
    "target": {
      "procedure_csv_base_id": "procedure_base",
      "row_selector": {"step_id": "s1"},
      "column": "additional_instruction"
    },
    "text": "根拠は証跡に明示された内容のみを使用する。"
  },
  "hypothesis": "証跡にない根拠を減らす",
  "labels": {
    "target_failure_mode": ["unsupported_rationale"],
    "tactic_type": ["evidence_grounding"]
  }
}
```

## Scope

```text
case_specific
procedure_specific
procedure_family
domain_common
global_common
```

探索の目標は、`case_specific` ではなく `domain_common` または `global_common` に昇格できるチューニングを見つけることです。

## Dataset

`examples/dataset.json` と同じ形式です。

```json
{
  "dataset_id": "sample_poc_tuning_dataset",
  "cases": [
    {
      "case_id": "case_001",
      "split": "train",
      "procedure_csv_path": "procedure_base.csv",
      "evidence_bundle_path": "evidence/case_001",
      "expected_output": {
        "judgement": "適合",
        "required_claim_keywords": ["住所", "一致"],
        "citations": [{"evidence_id": "doc_identity_001", "page": 1}]
      },
      "metadata": {
        "domain": "本人確認",
        "procedure_family": "住所照合",
        "required_capability": "evidence_grounding"
      }
    }
  ]
}
```

## Registry Tables

### tuning_candidates

候補そのもの、patch、仮説、ラベル、ステータスを保存します。

### experiment_batches

探索iteration、split、dataset snapshot、評価ポリシーを保存します。

### case_runs

ケースごとの実行結果、materialized CSV hash、証跡bundle hash、raw output URI、Langfuse trace IDを保存します。

### evaluation_results

評価器ごとのscore、label、commentを保存します。

### tuning_effects

候補ごとの効果を保存します。

```text
total_score_delta
regression_count
positive_count
negative_count
generality_score
overfit_risk
effect_label
```

### tuning_atoms

効果のあった候補を原子指示へ分解した結果を保存します。

### promotion_decisions

scope昇格の判断を保存します。

## Artifact

以下はDBに直接入れず、artifact storeに保存します。

```text
materialized_csv/
csv_diffs/
raw_outputs/
reports/
```

DBにはURIとhashだけを保存します。
