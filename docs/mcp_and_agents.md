# MCP tool server

This repository keeps the current tuning path in the CLI:

```bash
poc-auto run-search \
  --dataset examples/v3_multimodal_human_ref/dataset.json \
  --agent deepagent-human-ref \
  --runner deepagent
```

`src/poc_automation/mcp_server.py` is a small JSON-RPC-style helper for local
tool experiments. It is not the main v3 OpenRouter execution path.

Start it with:

```bash
PYTHONPATH=src python -m poc_automation.mcp_server
```

Example request:

```json
{"tool":"get_candidate","args":{"tuning_id":"baseline"}}
```
