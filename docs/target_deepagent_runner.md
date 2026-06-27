# DeepAgent評価対象Runner

このリポジトリには、評価対象アプリケーションを実行するためのrunnerが3種類あります。

| runner | 用途 |
| --- | --- |
| `mock` | 外部APIなしで探索ループを検証する決定論的な疑似runner |
| `http` | 既存の自社PoCアプリAPIを呼び出すrunner |
| `deepagent` / `deepagent-openrouter` | LangChain Deep Agents + OpenRouter/Qwenで評価対象アプリを代替するrunner |

`mock` は実LLMではありません。CSV追加指示に含まれる「証跡」「引用」「条件」「判断不能」などの語を見て、改善・失敗を疑似的に発生させます。探索基盤のテストには有効ですが、評価対象エージェントとしての振る舞いを検証するものではありません。

`deepagent` は、手続CSVと証跡bundleをDeep Agentに渡し、評価結果・根拠・引用をJSONで生成します。人手回答はpromptに含めません。人手回答はEvaluatorだけが参照します。

## 依存関係

```bash
pip install -e '.[target-agent]'
```

このoptional extraは以下を入れます。

```text
deepagents
langchain-openrouter
```

## 環境変数

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=qwen/qwen3-max
export OPENROUTER_TEMPERATURE=0
export OPENROUTER_MAX_TOKENS=2048
export OPENROUTER_MAX_RETRIES=2
export OPENROUTER_TIMEOUT_SECONDS=120
export OPENROUTER_APP_URL=http://localhost/poc-automation
export OPENROUTER_APP_TITLE='PoC Automation'
export POC_TARGET_AGENT_MAX_CSV_CHARS=12000
export POC_TARGET_AGENT_MAX_EVIDENCE_CHARS=30000
```

`OPENROUTER_MODEL` または `POC_TARGET_AGENT_MODEL` にはOpenRouter上のQwen model slugを指定します。既定値は以下です。

```text
qwen/qwen3-max
```

必要に応じて、別のQwenモデルへ差し替えられます。

## 実行例

チューニング候補生成はローカルheuristic、評価対象アプリはDeepAgent/Qwenにする例です。

```bash
poc-auto run-search \
  --dataset examples/dataset.json \
  --db .tmp/poc_automation.sqlite \
  --artifact-dir .tmp/artifacts \
  --agent heuristic \
  --runner deepagent \
  --iterations 1
```

チューニング候補生成もDeep Agents Codeに任せる場合は、`--agent` と `--runner` を分けて指定します。

```bash
poc-auto run-search \
  --dataset examples/dataset.json \
  --agent deepagents-code \
  --runner deepagent
```

このときの責務分担は次のとおりです。

```text
--agent deepagents-code
  CSVチューニング候補を生成する探索Agent

--runner deepagent
  materialized CSVと証跡を入力に評価結果・根拠・引用を出す評価対象Agent
```

## 入力prompt

`DeepAgentPocAppRunner` は、以下だけをDeepAgentに渡します。

```text
- case_id
- case metadata。ただし正解や期待値に相当する項目は除外
- materialized procedure CSV
- evidence bundle
- 出力JSON schema
- 判定・根拠・引用のルール
```

以下は渡しません。

```text
- expected_output
- 人手回答の判定
- required_claim_keywords
- 期待引用
- human_reference / reference_answer
```

これにより、候補生成側だけでなく、評価対象Agent側にも正解リークが起きにくい構成にしています。

## 出力schema

DeepAgentには、JSONオブジェクトのみを返すよう指示します。

```json
{
  "result": {
    "judgement": "適合",
    "rationale_items": [
      {
        "claim": "申請住所と本人確認書類の住所が一致している。",
        "citations": [
          {
            "evidence_id": "doc_identity_001",
            "page": 1,
            "span": "本人確認書類の住所が申請住所と一致している。",
            "claim": "住所一致"
          }
        ]
      }
    ],
    "warnings": []
  }
}
```

Runner側では、Markdownコードフェンス付きのJSONが返った場合も抽出します。JSON抽出に失敗した場合は、`deepagent_runner_failed` warningを付け、探索全体は継続します。

## 実装箇所

```text
src/poc_automation/runner.py
  DeepAgentPocAppRunner
  build_target_agent_prompt_payload
  build_target_agent_prompt
  parse_deepagent_json_response / extract_json_object
  normalize_deepagent_response

src/poc_automation/config.py
  TargetAgentConfig
  TARGET_AGENT_SYSTEM_PROMPT
```

## 注意点

`deepagent` runnerはOpenRouter APIを呼び出すため、実行ごとにコストとレイテンシが発生します。大量探索では、`search_policy` の `cheap_sample_size`、`candidates_per_iteration`、`iterations` を小さめに設定してから広げてください。

また、OpenRouter上のmodel slugは変わる可能性があるため、固定値を前提にせず、`OPENROUTER_MODEL` または `POC_TARGET_AGENT_MODEL` で切り替える前提にしています。
