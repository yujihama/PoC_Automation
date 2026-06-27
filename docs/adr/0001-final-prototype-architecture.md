# ADR 0001: 最終形プロトタイプを一気通貫で実装する

## Status

Accepted

## Context

PoCチューニングは、単発のプロンプト調整ではなく、大量実験、評価、ラベル付け、汎用化探索を含む。段階的に小さく作るより、最初から最終形の細いパイプラインを通す方が、データ構造と責務分離の妥当性を早く検証できる。

LangSmithは利用できないため、観測・評価UIにはLangfuseを使う。

## Decision

以下の構成を採用する。

```text
Search Orchestrator: Python
Agent: Heuristic / Deep Agents Code / Cline adapter
Observability: Langfuse
Registry: SQLite, 将来PostgreSQL
Artifact Store: local filesystem, 将来S3互換
Runner: Mock / HTTP API
```

Agentは候補生成に限定し、API実行、評価、保存、昇格判定は固定コードで行う。

## Consequences

良い点:

- ローカルdemoで全体像を検証できる
- LangfuseがなくてもRegistryで再現できる
- Agent差し替えが容易
- 過学習・リーク対策を最初から組み込める

注意点:

- 初期コード量は増える
- 本番化時にはSQLiteからPostgreSQLへの移行が必要
- 実PoCアプリAPIに合わせてRunner調整が必要
