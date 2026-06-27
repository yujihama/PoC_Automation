# PoCアプリAPI Runner

## 概要

`src/poc_automation/runner.py` に、実PoCアプリ接続用の `HttpPocAppRunner`、ローカル検証用の `MockPocAppRunner`、OpenRouter HTTP + Qwenで評価対象アプリを代替する `DeepAgentPocAppRunner` があります。


## DeepAgent runner

既存のPoCアプリAPIが未接続でも、実LLMの評価対象Agentを使って探索できます。

```bash
pip install -e .
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=qwen/qwen3-max

poc-auto run-search \
  --dataset examples/dataset.json \
  --agent heuristic \
  --runner deepagent
```

`DeepAgentPocAppRunner` は、materialized CSVと証跡bundleをOpenRouterのpromptに含め、以下の `NormalizedResult` に変換します。人手回答は評価対象runnerへ渡しません。詳細は [`target_deepagent_runner.md`](target_deepagent_runner.md) を参照してください。

## 標準API想定

```text
POST /evidence
POST /procedure-csv
POST /runs
GET  /runs/{run_id}
```

### POST /evidence

リクエスト例:

```json
{
  "case_id": "case_001",
  "evidence_bundle_path": "s3://.../case_001"
}
```

レスポンス例:

```json
{
  "id": "uploaded_evidence_123"
}
```

### POST /procedure-csv

リクエスト例:

```json
{
  "case_id": "case_001",
  "csv_path": "/tmp/materialized.csv"
}
```

レスポンス例:

```json
{
  "id": "uploaded_csv_123"
}
```

### POST /runs

リクエスト例:

```json
{
  "case_id": "case_001",
  "uploaded_evidence_id": "uploaded_evidence_123",
  "uploaded_csv_id": "uploaded_csv_123"
}
```

レスポンス例:

```json
{
  "run_id": "run_123"
}
```

### GET /runs/{run_id}

レスポンス例:

```json
{
  "status": "succeeded",
  "result": {
    "judgement": "適合",
    "rationale_items": [
      {
        "claim": "住所が一致している",
        "citations": [
          {
            "evidence_id": "doc_identity_001",
            "page": 1,
            "span": "住所が申請住所と一致"
          }
        ]
      }
    ]
  },
  "cost": {
    "input_tokens": 1000,
    "output_tokens": 200
  }
}
```

## API形状が異なる場合

`HttpEndpointMap` を変更してください。

```python
from poc_automation.runner import HttpEndpointMap, HttpPocAppRunner

runner = HttpPocAppRunner(
    config,
    endpoints=HttpEndpointMap(
        upload_evidence="/api/files/evidence",
        upload_csv="/api/files/csv",
        execute="/api/executions",
        fetch_result_template="/api/executions/{run_id}"
    )
)
```

リクエスト・レスポンスの形式が大きく違う場合は、`HttpPocAppRunner` を継承して `run_case` を上書きしてください。
