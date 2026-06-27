# 運用Runbook

## 1. 初期化

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
poc-auto init-db --db .tmp/poc_automation.sqlite
```

## 2. サンプル実行

```bash
poc-auto demo --workspace .tmp/demo --iterations 2
```

成果物:

```text
.tmp/demo/poc_automation.sqlite
.tmp/demo/artifacts/
.tmp/demo/report.md
```

## 3. 実データセット実行

```bash
poc-auto run-search \
  --dataset datasets/your_dataset.json \
  --db .tmp/poc_automation.sqlite \
  --artifact-dir .tmp/artifacts \
  --runner http \
  --agent heuristic \
  --iterations 3
```

## 4. Langfuse有効化

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

その後、通常通り `poc-auto run-search` を実行します。

## 5. OpenRouter human-reference探索へ切り替え

```bash
export POC_AUTOMATION_AGENT=deepagent-human-ref
export POC_AUTOMATION_RUNNER=deepagent
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=qwen/qwen3.6-flash

poc-auto run-search \
  --dataset datasets/your_dataset.json \
  --runner deepagent
```

## 6. OpenRouter設定確認

```bash
export POC_AUTOMATION_AGENT=deepagent-human-ref
export POC_AUTOMATION_RUNNER=deepagent

poc-auto run-search \
  --dataset datasets/your_dataset.json \
  --runner http
```

## 7. レポート出力

```bash
poc-auto export-report \
  --db .tmp/poc_automation.sqlite \
  --out reports/tuning_report.md
```

## 8. よく見るSQL

候補ランキング:

```sql
SELECT tc.tuning_id, tc.scope, tc.status, tc.hypothesis,
       te.total_score_delta, te.effect_label
FROM tuning_candidates tc
LEFT JOIN tuning_effects te ON tc.tuning_id = te.tuning_id
ORDER BY te.total_score_delta DESC;
```

悪化候補:

```sql
SELECT *
FROM tuning_effects
WHERE regression_count > 0 OR effect_label IN ('negative', 'risky')
ORDER BY regression_count DESC;
```

昇格判断:

```sql
SELECT *
FROM promotion_decisions
ORDER BY created_at DESC;
```

## 9. 失敗時の確認

### CSV patchが適用されない

```bash
poc-auto validate-patch --patch candidate.json --base-csv procedure.csv
```

row_selectorと対象列を確認してください。

### HTTP Runnerが失敗する

環境変数を確認します。

```bash
echo $POC_APP_BASE_URL
echo $POC_APP_API_KEY
```

API形状が違う場合は `HttpPocAppRunner` を調整してください。

### LangfuseにTraceが出ない

- `LANGFUSE_ENABLED=true` になっているか
- SDKが入っているか
- `LANGFUSE_HOST` に到達できるか
- public/secret keyが正しいか
- 実行終了時にflushされているか

このプロトタイプはLangfuse接続に失敗しても処理を継続します。Registry側に結果が残っているかを確認してください。
