# Langfuse Dashboard Setup for PoC Tuning Results

This repository does not build a separate result UI.  Langfuse self-host is the
inspection UI, and PoC_Automation sends Sessions, Traces, Scores, Datasets,
Dataset Run links, metadata, and tags to Langfuse.

## Prerequisites

1. Start Langfuse self-host v3.  The local template is `infra/docker-compose.langfuse.yml`.
2. Configure credentials in `.env` or the process environment:

```bash
POC_LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PROJECT=poc-tuning
POC_LANGFUSE_DATASET_MODE=hosted
POC_LANGFUSE_SESSION_BY_SEARCH_RUN=true
POC_LANGFUSE_SEND_RAW_OUTPUT=true
POC_LANGFUSE_SEND_EVIDENCE_TEXT=false
POC_LANGFUSE_SEND_EVIDENCE_FILES=false
```

3. Initialize score configs:

```bash
poc-auto langfuse init-score-configs
```

4. Sync the evaluation dataset snapshot:

```bash
poc-auto langfuse sync-dataset \
  --dataset examples/v3_multimodal_human_ref/dataset.json
```

5. Run a search with Langfuse enabled:

```bash
poc-auto run-search \
  --dataset examples/v3_multimodal_human_ref/dataset.json \
  --agent deepagent-human-ref \
  --runner deepagent \
  --langfuse-enabled \
  --langfuse-dataset-mode hosted
```

For local smoke tests without LLM calls, use `--agent heuristic --runner mock`.

## Trace Model

Use Langfuse filters by `session_id = search_run_id`.

Standard trace names:

- `poc.case_run`: one case, tuning, run type, and optional replicate index.
- `poc.agent_iteration`: one search round and its draft/accepted candidates.
- `poc.trial`: one draft instruction trial across cases.
- `poc.replicate_run`: one replicate pass for trial stability checks.

Common metadata:

- `search_run_id`
- `dataset_name`
- `dataset_snapshot_id`
- `case_id`
- `split`
- `domain`
- `procedure_family`
- `tuning_id`
- `fingerprint`
- `experiment_id`
- `run_type`
- `agent_mode`
- `runner`
- `provider`
- `model`
- `candidate_provider`
- `candidate_model`
- `candidate_status`
- `promotion_decision`

Common tags:

- `poc-tuning`
- `search-run`
- `baseline`
- `trial`
- `replicate`
- `formal_evaluation`
- `deepagent-human-ref`
- `openrouter`
- `qwen`

Keep high-cardinality values such as `case_id`, `tuning_id`, and `trial_id` in
metadata rather than tags.

## Dashboard: Search Run Overview

Purpose: inspect one complete tuning search run.

Suggested widgets:

| Widget | Data source | Metric | Group / Filter |
|---|---|---|---|
| Generated candidates | Scores | sum `generated_candidates` | group by `search_run_id` |
| Evaluated candidates | Scores | sum `evaluated_candidates` | group by `search_run_id` |
| Duplicate skipped | Scores | sum `skipped_duplicate_candidates` | group by `search_run_id` |
| Needs more validation | Scores | sum `needs_more_validation_candidates` | group by `search_run_id` |
| Baseline cases | Scores | sum `baseline_case_count` | group by `search_run_id` |
| Case run failures | Traces | count where status failed | group by `search_run_id` |
| Total tokens | Scores | sum `total_tokens` | group by `search_run_id` |
| Latency | Scores | sum `latency_ms` | group by `search_run_id` |

## Dashboard: Candidate Quality

Purpose: compare candidate effects.

Suggested widgets:

| Widget | Data source | Metric | Group / Filter |
|---|---|---|---|
| Mean total by tuning | Scores | avg `total_score` | group by `tuning_id` |
| Delta by tuning | Scores | avg `delta_vs_baseline` | group by `tuning_id` |
| Worst delta by tuning | Scores | min `delta_vs_baseline` | group by `tuning_id` |
| Effect label distribution | Scores | count `effect_label` | group by `effect_label` |
| Regression count | Scores | count where `has_regression=true` | group by `tuning_id` |
| Candidate status count | Scores / metadata | count | group by `candidate_status` |

## Dashboard: Case Result Matrix

Purpose: identify which cases improved or regressed.

Langfuse Custom Dashboards may not provide a native heatmap in every self-host
setup.  Use a table or grouped bar chart when heatmap rendering is not available.

Suggested widgets:

| Widget | Data source | Metric | Group / Filter |
|---|---|---|---|
| Total score matrix | Scores | avg `total_score` | group by `case_id`, `tuning_id` |
| Delta matrix | Scores | avg `delta_vs_baseline` | group by `case_id`, `tuning_id` |
| Judgement failures | Scores | count where `judgement_score < 1` | group by `case_id` |
| Unsupported claims | Scores | avg `unsupported_claim_rate` | group by `case_id`, `tuning_id` |
| Citation quality | Scores | avg `citation_score` | group by `case_id`, `tuning_id` |

## Dashboard: Trial / Formal / Replicate Stability

Purpose: find candidates that looked good in trial but fail formal evaluation or
show unstable replicate behavior.

Suggested widgets:

| Widget | Data source | Metric | Group / Filter |
|---|---|---|---|
| Trial mean total | Scores | avg `trial_mean_total` | group by `trial_id` |
| Trial formal gap | Scores | avg `trial_formal_gap` | group by `tuning_id` |
| Replicate stable rate | Scores | ratio `replicate_stable=true` | group by `tuning_id` |
| Replicate std | Scores | avg `replicate_std_total` | group by `tuning_id` |
| Replicate worst delta | Scores | min `replicate_worst_delta` | group by `tuning_id` |
| Unstable trials | Scores | count where `replicate_stable=false` | group by `tuning_id` |

## Dashboard: Cost / Latency

Purpose: inspect OpenRouter/Qwen usage and search cost proxies.

Suggested widgets:

| Widget | Data source | Metric | Group / Filter |
|---|---|---|---|
| Input tokens | Scores / observations | sum `input_tokens` | by `search_run_id`, `model` |
| Output tokens | Scores / observations | sum `output_tokens` | by `search_run_id`, `model` |
| Total tokens | Scores / observations | sum `total_tokens` | by `search_run_id`, `model` |
| Avg latency | Scores / observations | avg `latency_ms` | by `run_type`, `model` |
| Longest case runs | Traces | max latency | group by `case_id` |

## Security Defaults

By default, PoC_Automation sends procedure text, candidate instructions,
normalized outputs, citation spans, scores, token counts, and latency.  It does
not send evidence PDF or image files.  Evidence text is disabled unless
`POC_LANGFUSE_SEND_EVIDENCE_TEXT=true`.
