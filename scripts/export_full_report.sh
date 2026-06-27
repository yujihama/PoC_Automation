#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DB="${1:-.tmp/poc_automation.sqlite}"
OUT="${2:-reports/full_run_report.md}"
DATASET="${3:-examples/dataset.json}"
PYTHONPATH=src python -m poc_automation export-full-report --db "$DB" --out "$OUT" --dataset "$DATASET"
