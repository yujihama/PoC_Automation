#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DB="${1:-.tmp/poc_automation.sqlite}"
OUT="${2:-reports/tuning_report.md}"
PYTHONPATH=src python -m poc_automation export-report --db "$DB" --out "$OUT"
