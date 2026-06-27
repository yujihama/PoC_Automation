# Langfuse連携

## 位置づけ

Langfuseは、探索結果の観測・比較・評価可視化に使います。一方で、探索の正本は自前のExperiment Registryに保存します。

この分担にしている理由は、LangfuseはTrace、Score、Dataset、Prompt、Dashboardに強く、チューニング候補の親子関係、CSV patch、昇格判定、失敗候補の履歴といった探索台帳は自前DBで持つ方が扱いやすいためです。

## Langfuseへ送る情報

各ケース実行を1 traceとして扱います。

```text
trace: poc_tuning_run
  metadata:
    experiment_id
    case_id
    tuning_id
    split
    dataset_snapshot_id
    materialized_csv_hash

  output:
    judgement
    rationale_items
    citations

  scores:
    judgement_match
    rationale_support
    citation_quality
    format_valid
    unsupported_claim_rate
    leakage_risk
    total_score
```

## 有効化

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

SDKが入っていない場合や接続に失敗した場合、`LangfuseReporter` はno-opになります。これにより、ネットワークが閉じている環境でもRegistryとartifact保存は継続できます。

## self-hostの構成

開発・低スケール検証では `infra/docker-compose.langfuse.yml` をテンプレートとして使います。

```bash
cd infra
cp ../.env.example ../.env
# Langfuse用secretを.envに追加してから起動
# docker compose -f docker-compose.langfuse.yml up -d
```

本番または大量探索では、Langfuseの公式self-hostドキュメントに従い、Postgres、ClickHouse、Redis/Valkey、S3/Blob Storeを永続化・監視してください。

## Prompt Managementの使い方

このプロトタイプでは、まず以下のプロンプトをLangfuse Prompt Managementで管理する想定です。

- tuning-candidate-generator
- failure-analyzer
- patch-reviewer
- generalizer
- promotion-reviewer
- rationale-judge
- citation-judge

CSVに書き込む追加指示そのものは、まず自前Registryの `tuning_candidates` に保存します。`global_common` や `domain_common` に昇格したものだけ、必要に応じてLangfuse Promptへ同期します。

## 注意点

- holdoutの評価結果はLangfuseで見える化してよいが、探索Agentの入力には戻さない。
- Langfuse Datasetのバージョンだけに依存せず、自前の `dataset_snapshot_id` とartifact hashをRegistryに保存する。
- 証跡ファイルそのものをLangfuseに送るかは、機密性と運用方針に合わせて決める。初期設計ではObject Storageに置き、LangfuseにはhashとIDを送る。
