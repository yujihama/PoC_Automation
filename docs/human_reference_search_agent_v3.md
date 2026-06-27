# v3 人間実施結果参照型 DeepAgent 探索

`--agent deepagent-human-ref` は、探索用 DeepAgent に評価対象データと可視範囲内の人間実施結果を見せ、追加指示を試行錯誤させる探索モードです。

このモードで検証したい流れは次のとおりです。

1. 複数ケースに、それぞれ自然言語の手続、PDF/画像の証跡、人間の実施結果を用意する。
2. 探索エージェントは各ケースの入力と人間実施結果を読み、手続に追記する追加指示案を作る。
3. `evaluate_draft_instruction` でドラフト指示を対象ケース横断で実行し、人間結果に近づいたかを見る。
4. 試行結果を比較し、最後にケース横断で使える汎用的なチューニング案を候補として返す。

## 見せるデータ

探索エージェントが読めるもの:

- case id、split、安全化済み metadata
- 自然言語手続またはCSV手続
- PDF/画像証跡から抽出したテキスト
- baseline 出力とスコア概要
- `human_reference_splits` に含まれるケースの人間実施結果
- 過去のドラフト評価結果

探索エージェントに見せないもの:

- holdout の人間実施結果
- `expected_output`
- `required_claim_keywords`
- `citations`
- `required_capability`
- `human_result_text` を metadata に入れた場合の値

## PDF/画像証跡

PDFや画像そのものは `examples/v3_multimodal_human_ref/evidence/...` に置きます。LLM runner に渡すテキストは、同名の sidecar から読みます。

- `identity_summary.pdf` に対して `identity_summary.pdf.txt`
- `application_capture.bmp` に対して `application_capture.bmp.txt`

sidecar がないPDF/画像は、証跡ファイル名だけが渡され、内容は読めません。実運用でOCRやPDF抽出を追加する場合も、この境界に接続します。

## ドラフト評価

`evaluate_draft_instruction` は、`case_id` を渡してもその1ケースだけを評価しません。`case_id` は注目ケースの順序付けに使い、実際には `agent_trial_eval_splits` に含まれるケース全体で評価します。

これにより、探索エージェントが「case A には効くが case B で壊れる」指示を早い段階で観測できます。

## 新しい検証データの作成

```bash
python scripts/create_v3_multimodal_fixture.py
```

生成される主なファイル:

- `examples/v3_multimodal_human_ref/dataset.json`
- `examples/v3_multimodal_human_ref/procedures/*.txt`
- `examples/v3_multimodal_human_ref/evidence/*/*.pdf`
- `examples/v3_multimodal_human_ref/evidence/*/*.bmp`
- `examples/v3_multimodal_human_ref/evidence/*/*.txt`

各ケースの人間実施結果は、500〜1000字の非構造テキストです。人間がどの証跡のどの箇所を見て、なぜその判断にしたかを本文で説明します。

## 実行例

```bash
poc-auto run-search \
  --dataset examples/v3_multimodal_human_ref/dataset.json \
  --db .tmp/openrouter-v3-multimodal.sqlite \
  --artifact-dir .tmp/openrouter-v3-multimodal-artifacts \
  --agent deepagent-human-ref \
  --runner deepagent \
  --iterations 3 \
  --candidates-per-iteration 2
```

OpenRouter を使う場合は、`.env` に `OPENROUTER_API_KEY` または `POC_TARGET_AGENT_API_KEY`、および candidate/target agent の model 設定を置きます。キー値はログやレポートに出しません。
