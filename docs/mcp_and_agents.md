# MCP / Deep Agents Code / Cline連携

## 方針

Agentに任せるのは、以下に限定します。

```text
- 失敗要約の分析
- CSV patch候補生成
- patchの改善案作成
- positive候補の原子化・汎用化案作成
```

以下は通常コードで固定します。

```text
- API実行
- 評価
- DB保存
- Langfuse送信
- holdout管理
- 昇格判定
```

## Deep Agents Code

`.deepagents/skills/poc-tuning/SKILL.md` に、CSV tuning用のskillを置いています。

実行例:

```bash
poc-auto run-search \
  --dataset examples/dataset.json \
  --agent deepagents-code \
  --runner mock
```

内部的には以下のようなコマンドを呼び出します。

```bash
dcode --skill poc-tuning --non-interactive --quiet '<JSON payload>'
```

## Cline

`.cline/rules/tuning-safety.md` に、Cline用の安全ルールを置いています。

実行例:

```bash
poc-auto run-search \
  --dataset examples/dataset.json \
  --agent cline \
  --runner mock
```

## Tool Server

`src/poc_automation/mcp_server.py` は依存なしで動くJSON-RPC風tool serverです。完全なMCP serverではありませんが、実際のMCP serverで公開すべきtool境界を示します。

起動例:

```bash
PYTHONPATH=src python -m poc_automation.mcp_server
```

リクエスト例:

```json
{"tool":"get_candidate","args":{"tuning_id":"baseline"}}
```

## Agent出力の期待形式

Agentは `TuningCandidate` のJSON配列だけを返します。

```json
[
  {
    "tuning_id": "tune_xxx",
    "scope": "procedure_specific",
    "parent_tuning_ids": [],
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
]
```
