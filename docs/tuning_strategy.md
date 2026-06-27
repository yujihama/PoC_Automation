# チューニング探索戦略

## 基本方針

探索は、Agentに自由に最終回答を作らせるのではなく、CSVに追記する小さな追加指示を生成させ、その効果を固定評価器で測ります。

```text
失敗要約 → 候補生成 → patch検証 → 実行 → 評価 → ラベル → 汎用化
```

## 失敗モード

現時点の実装では、以下の失敗モードを扱います。

```text
wrong_judgement
unsupported_rationale
citation_mismatch
insufficient_evidence
format_violation
```

### wrong_judgement

最終判定が人手回答と異なる失敗です。

候補例:

```text
判断前に、手続CSVの必須条件、任意条件、例外条件を分けて確認し、未確認の条件を適合根拠にしない。
```

### unsupported_rationale

根拠に証跡で支持されない主張が混ざる失敗です。

候補例:

```text
根拠は証跡に明示された内容のみを使用する。証跡にない推測・一般知識・補完は根拠に含めない。
```

### citation_mismatch

引用が不足する、または主張を支持していない箇所を引用する失敗です。

候補例:

```text
根拠文と引用を1対1で対応させ、引用箇所が当該根拠文を直接支持している場合のみ引用として採用する。
```

### insufficient_evidence

証跡不足にもかかわらず推測で判定してしまう失敗です。

候補例:

```text
必要な証跡が不足している場合は推測で補わず、判断不能とし、不足している証跡または項目を列挙する。
```

## 探索アルゴリズム

### 1. Baseline

ベースCSVをそのまま実行して、失敗ケースを抽出します。

### 2. Candidate Generation

失敗要約をAgentに渡して候補を生成します。人手回答本文は渡しません。

### 3. Patch Validation

以下のような危険な候補を落とします。

- ケースIDを含む
- 人手回答という語を含む
- 正解を直接誘導する
- row_selectorがCSVに一致しない
- 対象列がない
- 指示が長すぎる

### 4. Cheap Evaluation

小さいtrain subsetで候補を評価します。

### 5. Validation Evaluation

cheap evaluationで有望だった候補だけvalidationへ進めます。

### 6. Holdout Evaluation

positive候補のうち昇格可能性があるものだけholdoutで確認します。holdoutの結果は次の候補生成に戻しません。

### 7. Generalization

positive候補を原子化し、tactic別にクラスタリングし、共通指示候補を生成します。

## ラベル

```text
strongly_positive
positive
neutral
negative
risky
```

`total_score_delta` と `regression_count` をもとに自動付与します。

## 汎用化

個別候補は、以下の条件を満たす場合に共通候補へ昇格させます。

- 複数ケースで改善している
- regression rateが小さい
- validationで一定以上のscore
- holdoutで非悪化
- リーク疑いがない
- 指示が特定ケース固有ではない

## 探索予算

`configs/search_policy.json` で制御します。

```json
{
  "iterations": 3,
  "candidates_per_iteration": 8,
  "cheap_sample_size": 20,
  "validation_sample_size": 50,
  "holdout_sample_size": 50,
  "beam_width": 4
}
```
