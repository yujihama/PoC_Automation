# セキュリティ・過学習対策

## リスク

PoCチューニング自動化で最も危険なのは、AgentがPoCデータセットだけに効く指示を作ることです。

代表的なリスクは以下です。

```text
- 人手回答の文言をCSV指示へ埋め込む
- case_idや証跡ファイル名に依存する
- 特定金額、住所、顧客名などの固有値に依存する
- holdout結果を探索ループに戻す
- 指示が長文化し、別ケースで悪化する
- 評価結果だけを上げ、根拠や引用品質を落とす
```

## 対策

### 1. 人手回答の隔離

候補生成Agentには人手回答本文を渡しません。渡すのはEvaluatorが作成した失敗要約です。

```json
{
  "failure_mode": "citation_mismatch",
  "summary": "引用箇所が不足している",
  "missing_capability": "根拠文と引用箇所の対応付け"
}
```

### 2. Patch Validator

`PatchValidator` は以下を検出します。

```text
case_001
human reference
正解
人手回答
golden
case_specific_values
reference_texts由来の長い一致フレーズ
```

### 3. Holdout隔離

holdoutは昇格判断だけに使います。holdoutの失敗要約は次の候補生成に戻しません。

### 4. Scope管理

チューニングは以下のscopeで管理します。

```text
case_specific
procedure_specific
procedure_family
domain_common
global_common
```

`global_common`へ昇格するには、複数domain・複数procedureで非悪化であることを確認します。

### 5. Negative Resultの保存

効かなかった候補、悪化した候補、validatorで落ちた候補も保存します。これにより、Agentが同じ失敗を繰り返すことを防ぎます。

### 6. Agent権限制御

Agentに許可する操作は限定します。

```text
propose_tuning_patch
validate_tuning_patch
materialize_csv
get_candidate
```

許可しない操作:

```text
- 任意SQL実行
- 本番API実行
- holdout正解の閲覧
- Object Storageの任意ファイル読み出し
- approvedでないglobal tuningの本番反映
```

## 機密データ

証跡ファイルそのものは、初期設計ではLangfuseに送らず、artifact storeまたは社内Object Storageに保存します。Langfuseにはhash、case_id、tuning_id、scoreを中心に送ります。
